"""
RAG Configuration Ablation — chunk_size x top_k x reranking
==============================================================

Runs the Senpai scenario subset (from trackA_conversational.yaml) through
12 RAGPipeline configurations and scores each with the same RAGAS-style
judge used in Week 2 (week02_evaluation/rag_eval/rag_scorer.py).

SCOPE NOTE — resolving a real ambiguity in the plan
-----------------------------------------------------
The plan text says "12 configurations (3x2x2, varying one dimension at a
time)" for chunk_size in [256, 512, 1024] x top_k in [1, 3, 5] x reranking
in [none, cross-encoder]. That is a contradiction: chunk_size x top_k x
reranking is a 3x3x2 = 18-cell full factorial, not 3x2x2=12, and neither a
strict one-factor-at-a-time sweep from a single baseline (which gives only
6 unique configs here) reproduces 12 either. Rather than silently pick one
reading, or drop a stated level of any dimension (which would silently
narrow scope on something the plan explicitly specifies), this module runs
a deliberate 12-config two-stage design:

  Stage A — full chunk_size x top_k grid at reranking="none" (3x3 = 9 configs).
            Finds the best chunk_size/top_k combination under the pipeline's
            original (no-reranking) retrieval.
  Stage B — reranking="cross-encoder" at top_k=3 (the pipeline's existing
            default, already the fixed point used throughout Week 2's
            persona-vector ablation) across all 3 chunk_size values (3 configs).
            Checks whether reranking changes the picture at a representative
            top_k, without paying for the full 18-cell factorial.

Total: 9 + 3 = 12 configs. Every level of every stated dimension
(chunk_size in {256,512,1024}, top_k in {1,3,5}, reranking in {none,
cross-encoder}) appears in at least one config; nothing specified is
silently dropped.

This design also gives three clean "sweeps" for the self-check requirement
(every pairwise comparison within a sweep differs in exactly one parameter):
  - chunk_size sweep: the 3 configs at (top_k=3, reranking=none)
  - top_k sweep: the 3 configs at (chunk_size=512, reranking=none)
  - reranking sweep: 3 pairs, one per chunk_size, each comparing
    (chunk_size=X, top_k=3, none) against (chunk_size=X, top_k=3, cross-encoder)

use_persona_vector is held constant at True for every config (it is not one
of the three dimensions under test here — that ablation was Week 2's).
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from week01_benchmark.eval_harness.clients import OpenAIClient          # noqa: E402
from week02_evaluation.rag_eval.rag_pipeline import RAGPipeline         # noqa: E402
from week02_evaluation.rag_eval.rag_scorer import RAGScorer             # noqa: E402
from week02_evaluation.rag_eval.rag_eval import evaluate_scenario       # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

SCENARIO_YAML = REPO_ROOT / "week01_benchmark" / "trackA_conversational.yaml"
OUTPUT_PATH = Path(__file__).resolve().parent / "W03_RAG_Ablation_results.json"

GENERATION_MODEL = "gpt-4o"
GENERATION_TEMPERATURE = 0.0
GENERATION_SEED = 42
USE_PERSONA_VECTOR = True  # held constant — not a dimension under test here

# Parameters that must be identical across every config in this ablation
# (everything except chunk_size, top_k, reranking).
ALWAYS_CONSTANT_PARAMS = ["domain", "embedding_model", "tfidf_weight", "semantic_weight", "use_persona_vector"]


@dataclass
class AblationConfig:
    config_id: str
    chunk_size: Optional[int]
    top_k: int
    reranking: str


# ---------------------------------------------------------------------------
# The 12 configurations — see module docstring for the design rationale.
# ---------------------------------------------------------------------------

CONFIGS: list[AblationConfig] = [
    # Stage A: full chunk_size x top_k grid, reranking=none (9 configs)
    AblationConfig("cs256_tk1_none",  256, 1, "none"),
    AblationConfig("cs256_tk3_none",  256, 3, "none"),
    AblationConfig("cs256_tk5_none",  256, 5, "none"),
    AblationConfig("cs512_tk1_none",  512, 1, "none"),
    AblationConfig("cs512_tk3_none",  512, 3, "none"),  # baseline — shared by both edge sweeps below
    AblationConfig("cs512_tk5_none",  512, 5, "none"),
    AblationConfig("cs1024_tk1_none", 1024, 1, "none"),
    AblationConfig("cs1024_tk3_none", 1024, 3, "none"),
    AblationConfig("cs1024_tk5_none", 1024, 5, "none"),
    # Stage B: reranking=cross-encoder at top_k=3, across all 3 chunk_sizes (3 configs)
    AblationConfig("cs256_tk3_cross",  256, 3, "cross-encoder"),
    AblationConfig("cs512_tk3_cross",  512, 3, "cross-encoder"),
    AblationConfig("cs1024_tk3_cross", 1024, 3, "cross-encoder"),
]

assert len(CONFIGS) == 12, f"Expected exactly 12 configs, got {len(CONFIGS)}"
assert len({c.config_id for c in CONFIGS}) == 12, "config_ids must be unique"

# ---------------------------------------------------------------------------
# Sweep membership — explicit id lists, not a per-config label. The baseline
# point (cs512_tk3_none) legitimately belongs to BOTH edge sweeps below, and
# 4 of the 9 Stage-A grid cells (e.g. cs256_tk1_none) belong to NEITHER edge
# sweep — they exist to complete the interaction grid, not as single-variable
# comparisons. A per-config "sweep" label can't express that; explicit lists
# can.
# ---------------------------------------------------------------------------

# Single-variable comparison: top_k=3, reranking=none fixed; chunk_size varies.
CHUNK_SIZE_SWEEP: list[str] = ["cs256_tk3_none", "cs512_tk3_none", "cs1024_tk3_none"]

# Single-variable comparison: chunk_size=512, reranking=none fixed; top_k varies.
TOP_K_SWEEP: list[str] = ["cs512_tk1_none", "cs512_tk3_none", "cs512_tk5_none"]

# Single-variable comparisons: chunk_size + top_k=3 fixed per pair; reranking varies.
RERANKING_PAIRS: list[tuple[str, str]] = [
    ("cs256_tk3_none", "cs256_tk3_cross"),
    ("cs512_tk3_none", "cs512_tk3_cross"),
    ("cs1024_tk3_none", "cs1024_tk3_cross"),
]


# ---------------------------------------------------------------------------
# Self-check: every pairwise comparison within a sweep differs in exactly
# one parameter. Callable from the notebook's final self-check cell too.
# ---------------------------------------------------------------------------

def verify_single_variable_sweeps(configs: list[AblationConfig]) -> list[str]:
    """
    Returns a list of violation strings (empty if every sweep is clean).
    Does NOT raise — the caller decides whether to assert on the result, so
    this can be reused both as an internal build-time check and as the
    notebook's printable self-check.
    """
    by_id = {c.config_id: c for c in configs}
    violations: list[str] = []

    def _diffs(a: AblationConfig, b: AblationConfig) -> list[str]:
        return [
            dim for dim in ("chunk_size", "top_k", "reranking")
            if getattr(a, dim) != getattr(b, dim)
        ]

    for sweep_name, member_ids, allowed_diff in [
        ("chunk_size", CHUNK_SIZE_SWEEP, "chunk_size"),
        ("top_k", TOP_K_SWEEP, "top_k"),
    ]:
        members = [by_id[cid] for cid in member_ids]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                diffs = _diffs(a, b)
                if diffs != [allowed_diff]:
                    violations.append(
                        f"sweep={sweep_name}: {a.config_id} vs {b.config_id} "
                        f"differ in {diffs} (expected exactly ['{allowed_diff}'])"
                    )

    for id_a, id_b in RERANKING_PAIRS:
        a, b = by_id[id_a], by_id[id_b]
        diffs = _diffs(a, b)
        if diffs != ["reranking"]:
            violations.append(
                f"sweep=reranking: {id_a} vs {id_b} differ in {diffs} "
                f"(expected exactly ['reranking'])"
            )

    return violations


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def load_senpai_scenarios(yaml_path: Path) -> list[dict]:
    with yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    scenarios = data.get("scenarios", data) if isinstance(data, dict) else data
    filtered = [s for s in scenarios if s.get("platform") == "Senpai"]
    logger.info("[ablation] Loaded %d Senpai scenarios from %s", len(filtered), yaml_path.name)
    return filtered


# ---------------------------------------------------------------------------
# Main async runner
# ---------------------------------------------------------------------------

async def run(dry_run: bool = False) -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key and not dry_run:
        raise SystemExit("OPENAI_API_KEY not found in .env")

    sweep_violations = verify_single_variable_sweeps(CONFIGS)
    if sweep_violations:
        raise AssertionError(
            "Sweep design is not single-variable clean:\n" + "\n".join(sweep_violations)
        )
    logger.info("[ablation] Sweep design verified: every within-sweep pair differs in exactly one parameter.")

    scenarios = load_senpai_scenarios(SCENARIO_YAML)
    if not scenarios:
        raise SystemExit("No Senpai scenarios found in trackA_conversational.yaml")

    generator = OpenAIClient(
        api_key=openai_key if not dry_run else "dry_run",
        model=GENERATION_MODEL,
        temperature=GENERATION_TEMPERATURE,
        seed=GENERATION_SEED,
    )
    scorer = RAGScorer(openai_api_key=openai_key if not dry_run else "dry_run")

    all_results: list[dict] = []
    total = len(CONFIGS) * len(scenarios)
    completed = 0

    for cfg in CONFIGS:
        pipeline = RAGPipeline(
            domain="senpai",
            top_k=cfg.top_k,
            use_persona_vector=USE_PERSONA_VECTOR,
            chunk_size=cfg.chunk_size,
            reranking=cfg.reranking,
        )
        for param in ALWAYS_CONSTANT_PARAMS:
            actual = pipeline.pipeline_config().get(param)
            expected = {"domain": "senpai", "embedding_model": "all-MiniLM-L6-v2",
                        "tfidf_weight": 0.4, "semantic_weight": 0.6,
                        "use_persona_vector": USE_PERSONA_VECTOR}[param]
            assert actual == expected, (
                f"config {cfg.config_id}: constant param '{param}' drifted "
                f"(expected {expected!r}, got {actual!r})"
            )

        for scenario in scenarios:
            t0 = time.perf_counter()
            result = await evaluate_scenario(
                scenario=scenario, pipeline=pipeline,
                generator=generator, scorer=scorer, dry_run=dry_run,
            )
            latency_ms = (time.perf_counter() - t0) * 1000

            result["config_id"] = cfg.config_id
            result["chunk_size"] = cfg.chunk_size
            result["top_k_config"] = cfg.top_k
            result["reranking"] = cfg.reranking
            result["latency_ms"] = round(latency_ms, 1)

            all_results.append(result)
            completed += 1
            logger.info(
                "[ablation] %d/%d | %s | %s | faithfulness=%s latency=%.0fms",
                completed, total, cfg.config_id, scenario.get("scenario_id"),
                result.get("faithfulness"), latency_ms,
            )

    output = {
        "metadata": {
            "evaluation": "W03_RAG_Ablation",
            "total_results": len(all_results),
            "n_configs": len(CONFIGS),
            "n_scenarios": len(scenarios),
            "domain": "senpai",
            "use_persona_vector": USE_PERSONA_VECTOR,
            "generation_model": GENERATION_MODEL,
            "generation_temperature": GENERATION_TEMPERATURE,
            "generation_seed": GENERATION_SEED,
            "dry_run": dry_run,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "config_design_note": (
                "12 configs = Stage A (chunk_size x top_k full grid, 3x3=9, "
                "reranking=none) + Stage B (reranking=cross-encoder at top_k=3 "
                "across all 3 chunk_sizes, 3 configs). See rag_ablation.py "
                "module docstring for the full rationale."
            ),
        },
        "configs": [vars(c) for c in CONFIGS],
        "results": all_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    errors = [r for r in all_results if r.get("error")]
    logger.info(
        "Done. %d results (%d configs x %d scenarios), %d errors -> %s",
        len(all_results), len(CONFIGS), len(scenarios), len(errors),
        OUTPUT_PATH.relative_to(REPO_ROOT),
    )


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="RAG configuration ablation (Week 3)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    asyncio.run(run(dry_run=args.dry_run))
