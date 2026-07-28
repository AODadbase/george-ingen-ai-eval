"""
IGuide-Inspired RAG Pipeline
==============================

Architecture mirrors the IGuide production RAG system:
  - Append-only document log (list, not a mutable index)
  - Composite ranking = TF-IDF score + semantic similarity score
    Weights: TF-IDF 0.4, semantic 0.6  (documented below)
  - Persona-vector augmentation layer: a domain-specific descriptor
    string is prepended to the user query before embedding, providing
    context-aware retrieval. This layer is toggled via `use_persona_vector`.

Composite Ranking Weight Rationale
------------------------------------
  TF-IDF (0.4): Strong for exact keyword matching (e.g., drug names,
    policy terms) which are common in eldercare/education documents.
  Semantic similarity (0.6): Stronger weight for semantic understanding
    so paraphrased queries still retrieve relevant documents.
  Both scores are normalised to [0, 1] before combining.

Persona Vectors
---------------
  Domain descriptors injected before the query to bias retrieval toward
  domain-specific content. Defined in PERSONA_VECTORS below.
"""

import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — document these for the ablation self-check
# ---------------------------------------------------------------------------

TFIDF_WEIGHT: float = 0.4          # weight for TF-IDF score in composite ranking
SEMANTIC_WEIGHT: float = 0.6       # weight for semantic similarity (must sum to 1.0)
assert abs(TFIDF_WEIGHT + SEMANTIC_WEIGHT - 1.0) < 1e-9, "Weights must sum to 1.0"

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # sentence-transformers model

# --- Week 3 ablation additions: chunking + reranking ---
RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Candidate pool size for reranking = top_k * this multiplier (capped at corpus size).
# The composite-score stage over-retrieves so the cross-encoder has real candidates
# to re-order rather than just re-scoring the same top_k it would've returned anyway.
RERANK_CANDIDATE_MULTIPLIER: int = 4
RERANKING_MODES: tuple[str, ...] = ("none", "cross-encoder")

# Persona vectors: domain-specific descriptor prepended to query when
# use_persona_vector=True. Held constant across all runs for reproducibility.
PERSONA_VECTORS: dict[str, str] = {
    "fari": (
        "Context: Eldercare companion robot assisting elderly residents in a care facility. "
        "Domain: medication management, daily routines, emergency response, nutrition. "
        "Query: "
    ),
    "senpai": (
        "Context: Educational robot tutoring K-12 students in academic subjects. "
        "Domain: adaptive tutoring, curriculum guidance, study planning, subject mastery. "
        "Query: "
    ),
}

# Knowledge base file paths (relative to this file)
_HERE = Path(__file__).parent
KB_PATHS: dict[str, Path] = {
    "fari": _HERE / "knowledge_base_fari.json",
    "senpai": _HERE / "knowledge_base_senpai.json",
}


# ---------------------------------------------------------------------------
# Document model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    doc_id: str
    domain: str
    text: str
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunking (Week 3 ablation addition)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """
    Split text into whitespace-token windows of at most chunk_size tokens.

    Approximation note: "token" here is a whitespace-separated word, not an
    LLM subword token (no tokenizer dependency was already established in
    this module for chunking). This under-counts true token length somewhat,
    which makes chunk_size a conservative (slightly generous) bound.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def _chunk_documents(docs: list["Document"], chunk_size: int) -> list["Document"]:
    """
    Split each document's text into chunk_size-token windows, each becoming
    its own retrievable Document. A document shorter than chunk_size is left
    as a single chunk with its original doc_id unchanged; a document that
    actually splits gets suffixed doc_ids (f"{doc_id}#chunk{i}") so retrieval
    results stay traceable to their source passage.
    """
    chunked: list[Document] = []
    for doc in docs:
        pieces = _chunk_text(doc.text, chunk_size)
        if len(pieces) == 1:
            chunked.append(doc)
            continue
        for i, piece in enumerate(pieces):
            chunked.append(Document(
                doc_id=f"{doc.doc_id}#chunk{i}",
                domain=doc.domain,
                text=piece,
                tags=doc.tags,
            ))
    return chunked


# ---------------------------------------------------------------------------
# TF-IDF helpers (in-memory, inspectable)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_tfidf_index(docs: list[Document]) -> tuple[dict, dict]:
    """
    Build TF-IDF index over the document corpus.

    Returns:
        tf_index: {doc_id: {term: tf_score}}
        idf_index: {term: idf_score}
    """
    N = len(docs)
    tf_index: dict[str, dict[str, float]] = {}
    df_counter: dict[str, int] = {}

    for doc in docs:
        tokens = _tokenize(doc.text)
        token_counts: dict[str, int] = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        max_freq = max(token_counts.values()) if token_counts else 1
        tf_index[doc.doc_id] = {
            t: count / max_freq for t, count in token_counts.items()
        }
        for t in token_counts:
            df_counter[t] = df_counter.get(t, 0) + 1

    idf_index: dict[str, float] = {
        t: math.log((N + 1) / (df + 1)) + 1  # smoothed IDF
        for t, df in df_counter.items()
    }
    return tf_index, idf_index


def _tfidf_score(
    query: str,
    doc_id: str,
    tf_index: dict,
    idf_index: dict,
) -> float:
    """Compute TF-IDF dot product between query and document."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    doc_tf = tf_index.get(doc_id, {})
    score = sum(
        doc_tf.get(t, 0.0) * idf_index.get(t, 0.0)
        for t in query_tokens
    )
    return score


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    IGuide-inspired in-memory RAG pipeline.

    Parameters
    ----------
    domain : "fari" or "senpai"
    top_k : number of documents to retrieve
    use_persona_vector : if True, prepend domain persona descriptor to query
    embedding_model : sentence-transformers model name
    chunk_size : if set, split each KB document into windows of at most this
        many whitespace tokens before indexing (Week 3 ablation dimension).
        None (default) preserves the original whole-document retrieval unit
        — existing callers (e.g. Week 2's persona-vector ablation) are
        unaffected.
    reranking : "none" (default, original behavior) or "cross-encoder" —
        over-retrieve a candidate pool by composite score, then re-order it
        with a cross-encoder before taking the final top_k.
    """

    def __init__(
        self,
        domain: str,
        top_k: int = 3,
        use_persona_vector: bool = True,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: Optional[int] = None,
        reranking: str = "none",
    ) -> None:
        if domain not in KB_PATHS:
            raise ValueError(f"Unknown domain '{domain}'. Must be one of {list(KB_PATHS)}")
        if reranking not in RERANKING_MODES:
            raise ValueError(f"Unknown reranking mode '{reranking}'. Must be one of {RERANKING_MODES}")

        self.domain = domain
        self.top_k = top_k
        self.use_persona_vector = use_persona_vector
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.reranking = reranking

        # Load append-only document log, then chunk if requested
        self.docs: list[Document] = self._load_docs(KB_PATHS[domain])
        if chunk_size is not None:
            self.docs = _chunk_documents(self.docs, chunk_size)
        logger.info(
            "[RAG] Loaded %d retrievable units for domain=%s (chunk_size=%s)",
            len(self.docs), domain, chunk_size,
        )

        # Build TF-IDF index
        self._tf_index, self._idf_index = _build_tfidf_index(self.docs)

        # Load sentence-transformer model
        logger.info("[RAG] Loading embedding model: %s", embedding_model)
        self._embedder = SentenceTransformer(embedding_model)

        # Pre-compute document embeddings (corpus is small enough for in-memory)
        self._doc_embeddings: np.ndarray = self._embedder.encode(
            [doc.text for doc in self.docs],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        logger.info("[RAG] Document embeddings computed: shape=%s", self._doc_embeddings.shape)

        # Cross-encoder is loaded lazily (only when actually needed) since it's
        # an extra model download/load that most configs (reranking="none") never use.
        self._cross_encoder: Optional["object"] = None
        if reranking == "cross-encoder":
            logger.info("[RAG] Loading cross-encoder reranker: %s", RERANK_MODEL)
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(RERANK_MODEL)

    @staticmethod
    def _load_docs(path: Path) -> list[Document]:
        """Load the append-only document log from JSON."""
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return [
            Document(
                doc_id=d["doc_id"],
                domain=d["domain"],
                text=d["text"],
                tags=d.get("tags", []),
            )
            for d in raw
        ]

    def _augment_query(self, query: str) -> str:
        """Prepend persona vector to query if layer is enabled."""
        if self.use_persona_vector:
            persona = PERSONA_VECTORS.get(self.domain, "")
            return persona + query
        return query

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """
        Retrieve top-k documents using composite ranking.

        Returns list of (Document, composite_score) tuples, sorted descending.
        """
        augmented_query = self._augment_query(query)

        # --- Semantic similarity scores ---
        query_embedding: np.ndarray = self._embedder.encode(
            augmented_query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # Cosine similarity (both vectors normalised → dot product)
        semantic_scores: np.ndarray = self._doc_embeddings @ query_embedding

        # --- TF-IDF scores (using ORIGINAL query for keyword matching) ---
        # The persona-vector prefix (augmented_query) must NOT be passed here.
        # TF-IDF keyword matching should be sensitive only to the user's actual
        # query terms, not to the persona descriptor. Only the semantic embedding
        # path uses augmented_query. Passing augmented_query here would cause
        # persona-domain keywords (e.g. "medication", "emergency") to pollute
        # TF-IDF scores systematically, violating the single-variable ablation.
        raw_tfidf = np.array([
            _tfidf_score(query, doc.doc_id, self._tf_index, self._idf_index)
            for doc in self.docs
        ])

        # --- Normalise TF-IDF to [0, 1] ---
        tfidf_max = raw_tfidf.max()
        if tfidf_max > 0:
            norm_tfidf = raw_tfidf / tfidf_max
        else:
            norm_tfidf = raw_tfidf

        # Semantic similarity is already in [-1, 1] (unit vectors);
        # shift to [0, 1] for comparability
        norm_semantic = (semantic_scores + 1.0) / 2.0

        # --- Composite score ---
        composite = TFIDF_WEIGHT * norm_tfidf + SEMANTIC_WEIGHT * norm_semantic

        if self.reranking == "none":
            ranked_indices = np.argsort(composite)[::-1][: self.top_k]
            return [(self.docs[i], float(composite[i])) for i in ranked_indices]

        # --- Cross-encoder reranking ---
        # Over-retrieve a candidate pool by composite score, then re-order it
        # with the cross-encoder before taking the final top_k. Reranking
        # scores (query, doc.text) pairs directly — the persona-vector prefix
        # is retrieval-stage-only, same rationale as the TF-IDF path above.
        pool_size = min(len(self.docs), max(self.top_k * RERANK_CANDIDATE_MULTIPLIER, self.top_k))
        pool_indices = np.argsort(composite)[::-1][:pool_size]
        pairs = [(query, self.docs[i].text) for i in pool_indices]
        cross_scores = self._cross_encoder.predict(pairs)
        reranked_order = np.argsort(cross_scores)[::-1][: self.top_k]
        return [
            (self.docs[pool_indices[i]], float(cross_scores[i]))
            for i in reranked_order
        ]

    def format_context(self, retrieved: list[tuple[Document, float]]) -> str:
        """Format retrieved documents as a context string for the LLM."""
        parts = []
        for i, (doc, score) in enumerate(retrieved, 1):
            parts.append(
                f"[Document {i} | id={doc.doc_id} | score={score:.3f}]\n{doc.text}"
            )
        return "\n\n".join(parts)

    def pipeline_config(self) -> dict:
        """
        Return a dict of all parameters that must be held constant in the ablation.
        Used by the ablation guard in rag_eval.py.
        """
        return {
            "domain": self.domain,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model_name,
            "tfidf_weight": TFIDF_WEIGHT,
            "semantic_weight": SEMANTIC_WEIGHT,
            "chunk_size": self.chunk_size,
            "reranking": self.reranking,
            "use_persona_vector": self.use_persona_vector,
        }
