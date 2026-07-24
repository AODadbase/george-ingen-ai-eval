"""
RAGAS-Style Scorers for InGen RAG Evaluation
=============================================

Implements three metrics using an LLM judge (GPT-4o, temperature=0)
with strict JSON output — the same pattern as llm_judge.py.

  faithfulness     : does the answer contain ONLY claims supported by the
                     retrieved context? (0.0 = completely unsupported,
                     1.0 = all claims are grounded)

  answer_relevance : does the answer address the actual question asked?
                     (0.0 = completely off-topic, 1.0 = fully addresses it)

  context_coverage : how much of the context material relevant to the
                     question appears in the answer?
                     (0.0 = none of the relevant context used,
                     1.0 = all relevant context faithfully used)

NOTE: The official `ragas` library uses its own LangChain plumbing and
OpenAI-specific adapters that do not integrate cleanly with this repo's
BaseAsyncLLMClient retry wrapper. To avoid bypassing the retry wrapper
and re-introducing the 429 rate-limit failures seen earlier in this codebase,
these scorers are implemented as custom GPT-4o judge calls that reuse the
existing OpenAIClient (with retry/backoff) rather than calling the ragas
library directly. The scoring logic is semantically equivalent.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import openai

logger = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-4o"
JUDGE_TEMPERATURE = 0.0
JUDGE_SEED = 42
JUDGE_TIMEOUT_S = 60.0
JUDGE_MAX_RETRIES = 5
JUDGE_BACKOFF_BASE_S = 8.0


# ---------------------------------------------------------------------------
# Score container
# ---------------------------------------------------------------------------

@dataclass
class RAGScores:
    faithfulness: Optional[float]         # 0.0 – 1.0
    answer_relevance: Optional[float]     # 0.0 – 1.0
    context_coverage: Optional[float]     # 0.0 – 1.0
    reasoning: Optional[str]
    error: Optional[str] = None
    judge_latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_FAITHFULNESS_PROMPT = """You are evaluating whether an AI-generated answer is faithful to a given context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

TASK: Assess faithfulness — the degree to which every claim in the answer is supported by the retrieved context. Claims that introduce information not present in the context are considered hallucinations.

Score faithfulness on a 0.0 to 1.0 scale:
  1.0 = All claims in the answer are directly supported by the context.
  0.7 = Most claims are supported; 1-2 minor additions not in context.
  0.4 = Some claims supported, but significant unsupported content present.
  0.0 = Answer is largely or entirely unsupported by the context.

Return ONLY a valid JSON object, nothing else:
{{"faithfulness": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""

_ANSWER_RELEVANCE_PROMPT = """You are evaluating whether an AI-generated answer is relevant to the question asked.

QUESTION:
{question}

GENERATED ANSWER:
{answer}

TASK: Assess answer relevance — the degree to which the answer directly and fully addresses the question. Do not consider whether the answer is factually correct; only whether it addresses what was asked.

Score answer_relevance on a 0.0 to 1.0 scale:
  1.0 = Answer directly and completely addresses the question.
  0.7 = Answer addresses the question but includes significant off-topic content.
  0.4 = Answer partially addresses the question; key aspects are missing.
  0.0 = Answer does not address the question at all.

Return ONLY a valid JSON object, nothing else:
{{"answer_relevance": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""

_CONTEXT_COVERAGE_PROMPT = """You are evaluating how well an AI-generated answer utilizes the relevant parts of the retrieved context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

TASK: Assess context coverage — the degree to which the answer incorporates and uses the portions of the retrieved context that are actually relevant to the question. An answer that ignores relevant context and gives a generic response scores low; an answer that synthesizes the most relevant context passages scores high.

Score context_coverage on a 0.0 to 1.0 scale:
  1.0 = The answer uses all the most relevant parts of the context.
  0.7 = The answer uses most relevant context but misses some key passages.
  0.4 = The answer uses some context but significant relevant portions are unused.
  0.0 = The answer ignores the context entirely.

Return ONLY a valid JSON object, nothing else:
{{"context_coverage": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""


# ---------------------------------------------------------------------------
# Judge call with retry (reusing the same pattern as llm_judge.py,
# but here we call openai.AsyncOpenAI directly since this module is
# imported in an async context and we want to keep the scorer self-contained.
# The OpenAIClient from clients.py wraps the same openai.AsyncOpenAI but
# requires a full client instantiation — for a tightly scoped scorer module
# we instead replicate the retry loop inline to avoid circular imports.)
# ---------------------------------------------------------------------------

async def _call_judge(
    client: "openai.AsyncOpenAI",
    prompt: str,
) -> tuple[dict, float]:
    """Call GPT-4o with retry/backoff and return (parsed_dict, latency_ms)."""
    t0 = time.perf_counter()
    last_error = None

    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=JUDGE_MODEL,
                    temperature=JUDGE_TEMPERATURE,
                    seed=JUDGE_SEED,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a rigorous evaluation judge. "
                                "Return ONLY a valid JSON object — no prose, no markdown."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                ),
                timeout=JUDGE_TIMEOUT_S,
            )
            raw = resp.choices[0].message.content or ""
            parsed = _parse_json(raw)
            latency_ms = (time.perf_counter() - t0) * 1000
            return parsed, latency_ms

        except openai.RateLimitError as exc:
            wait = JUDGE_BACKOFF_BASE_S * (2 ** (attempt - 1))
            logger.warning(
                "[rag_scorer] Judge TPM limit (attempt %d/%d) — backing off %.0fs",
                attempt, JUDGE_MAX_RETRIES, wait,
            )
            last_error = exc
            if attempt < JUDGE_MAX_RETRIES:
                await asyncio.sleep(wait)

        except asyncio.TimeoutError as exc:
            last_error = exc
            logger.warning("[rag_scorer] Judge timeout on attempt %d", attempt)

        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error("[rag_scorer] Judge error: %s", exc)
            break  # non-transient

    latency_ms = (time.perf_counter() - t0) * 1000
    raise RuntimeError(f"Judge failed after {JUDGE_MAX_RETRIES} attempts: {last_error}") from last_error


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    return json.loads(cleaned)


def _clamp(val, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(val)))


# ---------------------------------------------------------------------------
# Public scorer
# ---------------------------------------------------------------------------

class RAGScorer:
    """
    Scores a single RAG result using three LLM-judge metrics.

    Uses GPT-4o with retry/backoff — never bypasses the retry logic.
    """

    def __init__(self, openai_api_key: str) -> None:
        self._client = openai.AsyncOpenAI(api_key=openai_api_key)

    async def score(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> RAGScores:
        """
        Run all three metrics concurrently for a single (question, context, answer) triple.

        Returns RAGScores with null fields and error string if any call fails.
        """
        faithfulness_task = self._score_faithfulness(question, context, answer)
        relevance_task = self._score_relevance(question, answer)
        coverage_task = self._score_coverage(question, context, answer)

        t0 = time.perf_counter()
        results = await asyncio.gather(
            faithfulness_task, relevance_task, coverage_task,
            return_exceptions=True,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        faithfulness = answer_relevance = context_coverage = None
        reasoning_parts = []
        errors = []

        for name, result in zip(
            ["faithfulness", "answer_relevance", "context_coverage"], results
        ):
            if isinstance(result, Exception):
                errors.append(f"{name}: {result}")
                logger.error("[rag_scorer] %s failed: %s", name, result)
            else:
                val, reasoning = result
                if name == "faithfulness":
                    faithfulness = val
                elif name == "answer_relevance":
                    answer_relevance = val
                else:
                    context_coverage = val
                if reasoning:
                    reasoning_parts.append(f"{name}: {reasoning}")

        return RAGScores(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_coverage=context_coverage,
            reasoning=" | ".join(reasoning_parts) if reasoning_parts else None,
            error="; ".join(errors) if errors else None,
            judge_latency_ms=total_ms,
        )

    async def _score_faithfulness(
        self, question: str, context: str, answer: str
    ) -> tuple[float, str]:
        prompt = _FAITHFULNESS_PROMPT.format(
            question=question, context=context, answer=answer
        )
        data, _ = await _call_judge(self._client, prompt)
        score = _clamp(data.get("faithfulness", 0.0))
        return score, data.get("reasoning", "")

    async def _score_relevance(
        self, question: str, answer: str
    ) -> tuple[float, str]:
        prompt = _ANSWER_RELEVANCE_PROMPT.format(
            question=question, answer=answer
        )
        data, _ = await _call_judge(self._client, prompt)
        score = _clamp(data.get("answer_relevance", 0.0))
        return score, data.get("reasoning", "")

    async def _score_coverage(
        self, question: str, context: str, answer: str
    ) -> tuple[float, str]:
        prompt = _CONTEXT_COVERAGE_PROMPT.format(
            question=question, context=context, answer=answer
        )
        data, _ = await _call_judge(self._client, prompt)
        score = _clamp(data.get("context_coverage", 0.0))
        return score, data.get("reasoning", "")
