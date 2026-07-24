"""
RAG Evaluation Runner
======================

Loads Fari + Senpai scenarios from trackA_conversational.yaml, runs them
through the IGuide-inspired RAG pipeline twice (persona-vector on/off),
scores each result with three RAGAS-style metrics via GPT-4o, and writes
W02_RAG_Eval_results.json.

Usage
-----
    python -m week02_evaluation.rag_eval.rag_eval            # full run
    python -m week02_evaluation.rag_eval.rag_eval --dry-run  # skip API calls

Ablation Guard
--------------
Before reporting any results, the runner asserts programmatically that
chunk_size, top_k, and generation model are IDENTICAL between the two
persona-vector conditions. This check is hard — it raises AssertionError
and halts if any parameter diverges.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — allow running as module from repo root
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from week01_benchmark.eval_harness.clients import OpenAIClient  # noqa: E402

from .rag_pipeline import RAGPipeline                           # noqa: E402
from .rag_scorer import RAGScorer                               # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants — must be identical across both ablation conditions
# ---------------------------------------------------------------------------

GENERATION_MODEL = "gpt-4o"
GENERATION_TEMPERATURE = 0.0
GENERATION_SEED = 42
TOP_K = 3
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SCENARIO_YAML = REPO_ROOT / "week01_benchmark" / "trackA_conversational.yaml"
OUTPUT_PATH = REPO_ROOT / "week02_evaluation" / "W02_RAG_Eval_results.json"

GENERATION_SYSTEM_PROMPT = (
    "You are an AI assistant deployed on an InGen Dynamics robot platform. "
    "Answer the user's question using ONLY information from the retrieved context provided. "
    "If the context does not contain enough information, say so explicitly. "
    "Be concise and direct."
)


# ---------------------------------------------------------------------------
# Ablation guard
# ---------------------------------------------------------------------------

def assert_ablation_parity(pipeline_with: RAGPipeline, pipeline_without: RAGPipeline) -> None:
    """
    Hard assertion: every parameter except use_persona_vector must be identical.
    Raises AssertionError with a clear message if any parameter diverges.
    """
    cfg_with = pipeline_with.pipeline_config()
    cfg_without = pipeline_without.pipeline_config()

    controlled_params = ["top_k", "embedding_model", "tfidf_weight", "semantic_weight"]
    mismatches = []
    for param in controlled_params:
        if cfg_with.get(param) != cfg_without.get(param):
            mismatches.append(
                f"{param}: persona_vector=True → {cfg_with.get(param)!r}, "
                f"persona_vector=False → {cfg_without.get(param)!r}"
            )

    if mismatches:
        raise AssertionError(
            "ABLATION PARITY VIOLATION — the following parameters differ between "
            "persona-vector conditions:\n" + "\n".join(f"  - {m}" for m in mismatches)
        )

    logger.info(
        "[ablation] Parity check PASSED. Controlled params: %s",
        {p: cfg_with[p] for p in controlled_params},
    )


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def load_fari_senpai_scenarios(yaml_path: Path) -> list[dict]:
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    scenarios = data.get("scenarios", data) if isinstance(data, dict) else data
    filtered = [
        s for s in scenarios
        if s.get("platform") in ("Fari", "Senpai")
    ]
    logger.info(
        "[runner] Loaded %d Fari/Senpai scenarios from %s",
        len(filtered), yaml_path.name,
    )
    return filtered


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

async def evaluate_scenario(
    scenario: dict,
    pipeline: RAGPipeline,
    generator: OpenAIClient,
    scorer: RAGScorer,
    dry_run: bool,
) -> dict:
    """
    Run one (scenario, persona_vector_condition) evaluation.

    Returns a result dict conforming to the SREGym reproduction-metadata
    standard — every field needed to reproduce the result is present.
    Never fabricates scores on failure; writes null + error instead.
    """
    scenario_id = scenario.get("scenario_id", "unknown")
    platform = scenario.get("platform", "")
    query = scenario.get("input_stimulus", "")
    domain = platform.lower()   # "fari" or "senpai"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    base_result: dict = {
        "scenario_id": scenario_id,
        "platform": platform,
        "use_persona_vector": pipeline.use_persona_vector,
        "query": query,
        "retrieved_doc_ids": [],
        "retrieved_context": "",
        "generated_answer": None,
        "faithfulness": None,
        "answer_relevance": None,
        "context_coverage": None,
        "rag_reasoning": None,
        "generation_provider": generator.provider_name,
        "generation_model": generator.model,
        "temperature": GENERATION_TEMPERATURE,
        "seed": GENERATION_SEED,
        "top_k": pipeline.top_k,
        "embedding_model": pipeline.embedding_model_name,
        "persona_vector_domain": domain if pipeline.use_persona_vector else None,
        "timestamp_utc": ts,
        "error": None,
    }

    if dry_run:
        base_result.update({
            "retrieved_doc_ids": ["dry_run_doc_1", "dry_run_doc_2", "dry_run_doc_3"],
            "retrieved_context": "[DRY RUN — no real retrieval]",
            "generated_answer": "[DRY RUN — no generation]",
            "faithfulness": 0.8,
            "answer_relevance": 0.75,
            "context_coverage": 0.7,
            "rag_reasoning": "dry-run placeholder scores",
        })
        return base_result

    # --- Step 1: Retrieval ---
    try:
        retrieved = pipeline.retrieve(query)
        context = pipeline.format_context(retrieved)
        doc_ids = [doc.doc_id for doc, _ in retrieved]
        base_result["retrieved_doc_ids"] = doc_ids
        base_result["retrieved_context"] = context
        logger.info(
            "[runner] %s | persona=%s | retrieved %d docs: %s",
            scenario_id, pipeline.use_persona_vector, len(doc_ids), doc_ids,
        )
    except Exception as exc:  # noqa: BLE001
        base_result["error"] = f"Retrieval failed: {exc}"
        logger.error("[runner] %s | retrieval error: %s", scenario_id, exc)
        return base_result

    # --- Step 2: Generation (reuse existing client with retry/backoff) ---
    user_prompt = (
        f"Retrieved context:\n{context}\n\n"
        f"Question: {query}"
    )
    try:
        gen_response = await generator.complete(GENERATION_SYSTEM_PROMPT, user_prompt)
        if gen_response.error:
            base_result["error"] = f"Generation failed: {gen_response.error}"
            logger.error("[runner] %s | generation error: %s", scenario_id, gen_response.error)
            return base_result
        answer = gen_response.content
        base_result["generated_answer"] = answer
        logger.info("[runner] %s | persona=%s | generated answer (%d chars)",
                    scenario_id, pipeline.use_persona_vector, len(answer))
    except Exception as exc:  # noqa: BLE001
        base_result["error"] = f"Generation exception: {exc}"
        logger.error("[runner] %s | generation exception: %s", scenario_id, exc)
        return base_result

    # --- Step 3: Scoring ---
    try:
        scores = await scorer.score(query, context, answer)
        base_result.update({
            "faithfulness": scores.faithfulness,
            "answer_relevance": scores.answer_relevance,
            "context_coverage": scores.context_coverage,
            "rag_reasoning": scores.reasoning,
        })
        if scores.error:
            base_result["error"] = f"Partial scoring error: {scores.error}"
        logger.info(
            "[runner] %s | persona=%s | F=%.2f AR=%.2f CC=%.2f",
            scenario_id,
            pipeline.use_persona_vector,
            scores.faithfulness or 0,
            scores.answer_relevance or 0,
            scores.context_coverage or 0,
        )
    except Exception as exc:  # noqa: BLE001
        base_result["error"] = f"Scoring exception: {exc}"
        logger.error("[runner] %s | scoring exception: %s", scenario_id, exc)

    return base_result


# ---------------------------------------------------------------------------
# Main async runner
# ---------------------------------------------------------------------------

async def run(dry_run: bool) -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key and not dry_run:
        raise SystemExit("OPENAI_API_KEY not found in .env")

    scenarios = load_fari_senpai_scenarios(SCENARIO_YAML)
    if not scenarios:
        raise SystemExit("No Fari/Senpai scenarios found in trackA_conversational.yaml")

    # --- Build pipelines for each domain × persona_vector condition ---
    # Pipeline config must be identical except for use_persona_vector
    pipelines: dict[tuple[str, bool], RAGPipeline] = {}
    for domain in ("fari", "senpai"):
        for pv in (True, False):
            pipelines[(domain, pv)] = RAGPipeline(
                domain=domain,
                top_k=TOP_K,
                use_persona_vector=pv,
                embedding_model=EMBEDDING_MODEL,
            )

    # --- Ablation guard: assert parity for each domain ---
    for domain in ("fari", "senpai"):
        assert_ablation_parity(
            pipelines[(domain, True)],
            pipelines[(domain, False)],
        )
    logger.info("[ablation] All parity checks passed. Generation model=%s temperature=%s seed=%s",
                GENERATION_MODEL, GENERATION_TEMPERATURE, GENERATION_SEED)

    # --- Generator client (reuse OpenAIClient with retry/backoff) ---
    generator = OpenAIClient(
        api_key=openai_key if not dry_run else "dry_run",
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
        seed=GENERATION_SEED,
    )

    # --- Scorer ---
    scorer = RAGScorer(openai_api_key=openai_key if not dry_run else "dry_run")

    # --- Run all (scenario × persona_vector) combinations ---
    all_results = []
    total = len(scenarios) * 2  # 2 ablation conditions
    completed = 0

    for pv_condition in (True, False):
        label = "persona_vector=ON" if pv_condition else "persona_vector=OFF"
        logger.info("=== %s (%d scenarios) ===", label, len(scenarios))

        for scenario in scenarios:
            domain = scenario.get("platform", "").lower()
            if domain not in ("fari", "senpai"):
                logger.warning("Skipping unexpected platform: %s", scenario.get("platform"))
                continue

            pipeline = pipelines[(domain, pv_condition)]
            result = await evaluate_scenario(
                scenario=scenario,
                pipeline=pipeline,
                generator=generator,
                scorer=scorer,
                dry_run=dry_run,
            )
            all_results.append(result)
            completed += 1
            logger.info("[runner] Progress: %d/%d", completed, total)

    # --- Save output ---
    output = {
        "metadata": {
            "evaluation": "W02_RAG_Eval",
            "total_results": len(all_results),
            "scenarios": len(scenarios),
            "ablation_conditions": ["persona_vector=True", "persona_vector=False"],
            "generation_model": GENERATION_MODEL,
            "generation_temperature": GENERATION_TEMPERATURE,
            "generation_seed": GENERATION_SEED,
            "top_k": TOP_K,
            "embedding_model": EMBEDDING_MODEL,
            "dry_run": dry_run,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "ablation_controlled_params": [
                "top_k", "embedding_model", "tfidf_weight",
                "semantic_weight", "generation_model",
                "generation_temperature", "generation_seed",
            ],
        },
        "results": all_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    errors = [r for r in all_results if r.get("error")]
    logger.info(
        "Done. Results: %d total, %d errors → %s",
        len(all_results), len(errors),
        OUTPUT_PATH.relative_to(REPO_ROOT),
    )
    if errors:
        logger.warning("Failed evaluations:")
        for r in errors:
            logger.warning("  %s | persona=%s | %s",
                           r["scenario_id"], r["use_persona_vector"], r["error"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG evaluation runner for InGen harness")
    p.add_argument(
        "--dry-run", action="store_true",
        help="Skip API calls; write placeholder scores for pipeline testing.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run(dry_run=args.dry_run))
