"""
Data Layer — W04 Dashboard
============================

Every function here loads from pre-computed files already on disk and returns a plain
pandas DataFrame (or a small dict/list of primitives for summary numbers). No function in
this module calls an LLM provider or judge — there is no live-inference code path.

Verified computations (severity_weighted_score, krippendorff_alpha, per-platform
sub-scores, Pareto-frontier logic, cost-per-quality-point) are imported unchanged from the
Week 2/3 modules that already established and verified them:
  - week02_evaluation/leaderboard_analysis.py
  - week02_evaluation/multiprovider_eval/platform_scores.py
  - week02_evaluation/multiprovider_eval/cost_quality.py
  - week03_synthesis/pareto_chart.py

Page files (app.py, pages/*.py) must only call functions in this module and render the
result — no computation logic belongs in a page file. This is what makes the "no live
inference on launch" self-check mechanically verifiable: grep the page files for imports of
clients.py / rag_scorer.py / step_verifier.py / llm_judge.py.
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from week02_evaluation.leaderboard_analysis import (      # noqa: E402
    load_judged_dataframe, ACCURACY_SEED_COLS, compute_krippendorff_alpha,
)
from week02_evaluation.multiprovider_eval.platform_scores import build_platform_subscores  # noqa: E402
from week02_evaluation.multiprovider_eval.cost_quality import build_cost_quality_table      # noqa: E402
from week03_synthesis.pareto_chart import pareto_frontier                                   # noqa: E402

DATA_DIR = REPO_ROOT / "data"
LEADERBOARD_CSV = DATA_DIR / "leaderboard_summary.csv"
RAG_PERSONA_RESULTS = REPO_ROOT / "week02_evaluation" / "W02_RAG_Eval_results.json"
RAG_ABLATION_RESULTS = REPO_ROOT / "week03_synthesis" / "W03_RAG_Ablation_results.json"
AGENTIC_RESULTS = REPO_ROOT / "week02_evaluation" / "W02_Agentic_Eval_results.json"
PIC20_ANALYSIS_MD = REPO_ROOT / "week03_synthesis" / "W03_PIC20_Analysis.md"
EVAL_LOG_WK02 = REPO_ROOT / "weekly" / "Wk-02-EvalLog.md"

# Aido Rover real-time analysis constants — same values and rationale as
# W03_System_Eval.ipynb section 2 (2000ms: HRI "still feels prompt" convention for a
# hazard-acknowledgment turn; 0.85: reused from pareto_chart.py's existing ideal-zone cutoff).
AIDO_ROVER_LATENCY_THRESHOLD_MS = 2000
AIDO_ROVER_QUALITY_FLOOR = 0.85

# Display names — .capitalize() mangles "deepseek" -> "Deepseek" (should be "DeepSeek").
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "anthropic": "Anthropic", "openai": "OpenAI", "deepseek": "DeepSeek", "groq": "Groq",
}


def display_name(provider: str) -> str:
    return PROVIDER_DISPLAY_NAMES.get(provider, provider.capitalize())


# ---------------------------------------------------------------------------
# Cache helper — Streamlit's own cache decorator if available, else no-op.
# Keeps this module importable (and testable) outside a Streamlit run.
# ---------------------------------------------------------------------------

try:
    import streamlit as st
    _cache = st.cache_data
except ImportError:  # pragma: no cover - only true in non-streamlit test contexts
    def _cache(func):
        return func


def _latest(pattern: str) -> Path:
    matches = sorted(DATA_DIR.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern!r} in {DATA_DIR}")
    return matches[-1]


# ---------------------------------------------------------------------------
# Leaderboard / provider level
# ---------------------------------------------------------------------------

@_cache
def load_leaderboard() -> pd.DataFrame:
    """data/leaderboard_summary.csv, as-is — the already-verified provider leaderboard."""
    return pd.read_csv(LEADERBOARD_CSV)


@_cache
def load_judged_data() -> pd.DataFrame:
    """Combined trackA + trackB judged rows (160 rows: 20 scenarios x 4 providers x 2 tracks)."""
    paths = [_latest("judged_trackA_full_40*.json"), _latest("judged_trackB_full_40*.json")]
    return load_judged_dataframe(paths)


@_cache
def load_platform_subscores() -> pd.DataFrame:
    """Per (platform, provider) severity-weighted sub-scores, platform-local normalizer."""
    return build_platform_subscores(load_judged_data())


@_cache
def load_cost_quality_table() -> pd.DataFrame:
    """
    Cost-per-quality-point, ranked. Reuses build_cost_quality_table() unchanged (same
    formula as W03_System_Eval.ipynb section 3: cost_per_quality_point =
    estimated_cost_usd / severity_weighted_score_norm), then adds price_per_1k_tokens
    backed out from estimated_cost_usd / total_tokens — not a re-guessed constant.
    """
    leaderboard = load_leaderboard()
    cq = build_cost_quality_table(leaderboard)
    price = (leaderboard.set_index("provider")["estimated_cost_usd"]
             / leaderboard.set_index("provider")["total_tokens"] * 1000)
    cq["price_per_1k_tokens"] = cq["provider"].map(price).round(6)
    return cq


@_cache
def load_latency_quality_pareto() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (full_df_with_is_pareto_optimal_flag, frontier_only_df) for the corrected linear-axis
    latency x quality chart (W03_System_Eval.ipynb section 1) — NOT the cost x quality one.
    """
    leaderboard = load_leaderboard()
    lat_qual = leaderboard[["provider", "mean_latency_ms", "severity_weighted_score_norm"]].copy()
    frontier = pareto_frontier(lat_qual, cost_col="mean_latency_ms", quality_col="severity_weighted_score_norm")
    pareto_providers = set(frontier["provider"])
    lat_qual["is_pareto_optimal"] = lat_qual["provider"].isin(pareto_providers)
    return lat_qual, frontier


# ---------------------------------------------------------------------------
# Aido Rover latency-threshold analysis
# ---------------------------------------------------------------------------

@_cache
def load_aido_rover_threshold_table() -> pd.DataFrame:
    """
    Same table as W03_System_Eval.ipynb section 2: per-provider mean latency and
    platform-local severity_weighted_score_norm on the Aido Rover subset (trackA + trackB
    combined, 32 rows), checked against AIDO_ROVER_LATENCY_THRESHOLD_MS and
    AIDO_ROVER_QUALITY_FLOOR.
    """
    df = load_judged_data()
    rover_df = df[df["platform"] == "Aido Rover"]
    latency = rover_df.groupby("resp_provider")["resp_latency_ms"].mean().round(1)

    subscores = load_platform_subscores()
    quality = (subscores[subscores["platform"] == "Aido Rover"]
               .set_index("provider")["severity_weighted_score_norm"])

    table = pd.DataFrame({
        "mean_latency_ms": latency,
        "severity_weighted_score_norm": quality,
    }).round(4)
    table["meets_latency_threshold"] = table["mean_latency_ms"] < AIDO_ROVER_LATENCY_THRESHOLD_MS
    table["meets_quality_floor"] = table["severity_weighted_score_norm"] >= AIDO_ROVER_QUALITY_FLOOR
    table["meets_both"] = table["meets_latency_threshold"] & table["meets_quality_floor"]
    table["latency_over_threshold_x"] = (table["mean_latency_ms"] / AIDO_ROVER_LATENCY_THRESHOLD_MS).round(2)
    return table.sort_values("mean_latency_ms").reset_index().rename(columns={"index": "provider"})


# ---------------------------------------------------------------------------
# RAG — persona-vector ablation (Week 2)
# ---------------------------------------------------------------------------

@_cache
def load_persona_vector_by_platform() -> pd.DataFrame:
    """Per (platform, use_persona_vector) mean faithfulness/answer_relevance/context_coverage."""
    with RAG_PERSONA_RESULTS.open() as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["results"])
    agg = (df.groupby(["platform", "use_persona_vector"])
           [["faithfulness", "answer_relevance", "context_coverage"]]
           .mean().round(3).reset_index())
    return agg


@_cache
def load_persona_vector_aggregate() -> pd.DataFrame:
    """Aggregate ON vs OFF across all platforms combined, with the delta row."""
    with RAG_PERSONA_RESULTS.open() as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["results"])
    summary = (df.groupby("use_persona_vector")
               [["faithfulness", "answer_relevance", "context_coverage"]]
               .mean().round(3))
    summary.index = summary.index.map({True: "persona_vector=ON", False: "persona_vector=OFF"})
    delta = (summary.loc["persona_vector=ON"] - summary.loc["persona_vector=OFF"]).round(3)
    delta.name = "delta (ON - OFF)"
    return pd.concat([summary, delta.to_frame().T])


# ---------------------------------------------------------------------------
# RAG — 12-config ablation (Week 3)
# ---------------------------------------------------------------------------

@_cache
def load_rag_ablation_configs() -> pd.DataFrame:
    """Per-config_id aggregated faithfulness/answer_relevance/context_coverage/latency_ms."""
    with RAG_ABLATION_RESULTS.open() as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["results"])
    agg = df.groupby("config_id").agg(
        chunk_size=("chunk_size", "first"), top_k=("top_k_config", "first"),
        reranking=("reranking", "first"),
        faithfulness=("faithfulness", "mean"), answer_relevance=("answer_relevance", "mean"),
        context_coverage=("context_coverage", "mean"), latency_ms=("latency_ms", "mean"),
    ).round(3).reset_index()
    return agg


@_cache
def load_rag_ablation_pareto() -> pd.DataFrame:
    """Pareto-optimal RAG config(s) — min latency at max faithfulness."""
    agg = load_rag_ablation_configs()
    return pareto_frontier(agg, cost_col="latency_ms", quality_col="faithfulness")


# ---------------------------------------------------------------------------
# Agentic evaluation (Week 2, post-verifier-fix data)
# ---------------------------------------------------------------------------

@_cache
def load_agentic_data() -> pd.DataFrame:
    with AGENTIC_RESULTS.open() as f:
        raw = json.load(f)
    return pd.DataFrame(raw["results"])


@_cache
def load_agentic_completion_by_platform() -> pd.DataFrame:
    """task_completion_rate mean (= rate) per (provider, platform)."""
    df = load_agentic_data()
    return (df.groupby(["provider", "platform"])["task_completion_rate"]
            .agg(completion_rate="mean", n_scenarios="count").reset_index())


@_cache
def load_agentic_step_efficiency_by_completion() -> pd.DataFrame:
    """
    step_efficiency split by (provider, completion status) — NOT a single undifferentiated
    distribution. Per W02_Evaluation_Memo.md: raw step_efficiency conflates "efficient
    success" (high efficiency on a completed run) with "gave up early" (low efficiency on a
    failed run) if shown as one pooled number.
    """
    df = load_agentic_data()
    df = df.copy()
    df["status"] = df["task_completion_rate"].map({1: "completed", 0: "not completed"})
    return (df.groupby(["provider", "status"])["step_efficiency"]
            .agg(mean_step_efficiency="mean", n="count").round(3).reset_index())


@_cache
def load_agentic_error_recovery_sparse() -> dict:
    """
    Actual sparse error-recovery data — NOT a heatmap-ready full matrix. Only
    transcripts that actually contained an error observation have a non-null
    error_recovery_rate; everything else means "no error occurred, metric not
    applicable" (not "0% recovery").
    """
    df = load_agentic_data()
    applicable = df[df["error_recovery_rate"].notna()]
    return {
        "n_total_transcripts": len(df),
        "n_applicable": len(applicable),
        "applicable_rows": applicable[["scenario_id", "provider", "platform", "error_recovery_rate"]],
        "sufficient_for_comparison": len(applicable) >= 5,  # arbitrary-but-stated floor for "a real comparison"
    }


@_cache
def load_verifier_fix_history() -> pd.DataFrame:
    """
    Documented historical fact, not a live recomputation: the step verifier originally
    credited a step as complete whenever the agent's Final Answer *claimed* it was done,
    without requiring a corresponding Action+Observation in the transcript. This was found
    and fixed; the "before" numbers reflect the buggy verifier and are not reproducible by
    re-running current code (which has the fix) — they are cited from
    weekly/Wk-02-EvalLog.md, the durable record of the fix, rather than silently omitted.
    """
    return pd.DataFrame([
        {"provider": "anthropic", "completion_rate_before_fix": 0.70, "completion_rate_after_fix": 0.25,
         "n_scenarios": 20, "source": str(EVAL_LOG_WK02.relative_to(REPO_ROOT))},
        {"provider": "deepseek", "completion_rate_before_fix": 0.30, "completion_rate_after_fix": 0.15,
         "n_scenarios": 20, "source": str(EVAL_LOG_WK02.relative_to(REPO_ROOT))},
    ])


# ---------------------------------------------------------------------------
# PIC 2.0 readiness (Week 3) — parsed from the markdown source, not re-typed by hand
# ---------------------------------------------------------------------------

@_cache
def load_pic20_readiness() -> pd.DataFrame:
    """
    Parses the "## Summary Table" section of W03_PIC20_Analysis.md. This is a per
    PIC-2.0-model-class table (GRPO/STUM/SEOM/AMDC/HTD-IRL/CRL-MRS) — no platform-to-class
    mapping is documented anywhere in this repo's source material (checked: neither the
    landscape brief nor this analysis states one), so this function returns the class-level
    table as written rather than fabricating a platform join.
    """
    text = PIC20_ANALYSIS_MD.read_text(encoding="utf-8")
    section = text.split("## Summary Table", 1)[1]
    rows = []
    for line in section.strip().splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= {"-"}:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == "Class":
            continue
        readiness_num = float(cells[1].split("/")[0])
        rows.append({
            "class": cells[0], "readiness": readiness_num, "readiness_label": cells[1],
            "strongest_evidence": cells[2], "largest_gap": cells[3],
        })
    return pd.DataFrame(rows)


@_cache
def load_deployment_risks() -> list[dict]:
    """
    Top-3 concrete platform-level deployment risks, each computed from real data (not
    hardcoded numbers) and citing its source. These are separate from load_pic20_readiness()
    — that table is per PIC-2.0 class; these are per InGen platform, drawn directly from the
    Week 2/3 evaluation results.
    """
    risks = []

    rover = load_aido_rover_threshold_table()
    leader = load_leaderboard().sort_values("severity_weighted_score_norm", ascending=False).iloc[0]
    leader_row = rover[rover["provider"] == leader["provider"]].iloc[0]
    risks.append({
        "finding": (
            f"{display_name(leader['provider'])} leads the overall quality leaderboard "
            f"({leader['severity_weighted_score_norm']:.3f}) but fails Aido Rover's "
            f"{AIDO_ROVER_LATENCY_THRESHOLD_MS}ms real-time latency threshold by "
            f"{leader_row['latency_over_threshold_x']:.1f}x ({leader_row['mean_latency_ms']:.0f}ms actual)."
        ),
        "source": "week03_synthesis/W03_System_Eval.ipynb (section 2) / W03_PIC20_Analysis.md (GRPO)",
    })

    completion = load_agentic_completion_by_platform()
    humanoid = completion[completion["platform"] == "Aido Humanoid"]
    zero_providers = humanoid[humanoid["completion_rate"] == 0.0]["provider"].tolist()
    risks.append({
        "finding": (
            f"{' and '.join(display_name(p) for p in zero_providers)} complete 0% of Aido Humanoid "
            f"agentic scenarios ({int(humanoid['n_scenarios'].iloc[0])} scenarios each) — the platform "
            f"requiring the most compound multi-step manipulation planning."
        ) if zero_providers else "No provider fully failed Aido Humanoid.",
        "source": "week02_evaluation/W02_Agentic_Eval_results.json / W03_PIC20_Analysis.md (HTD-IRL)",
    })

    judged = load_judged_data()
    violations = judged[judged["judge_majority_failure"] == "safety_boundary_violation"]
    platform_counts = violations["platform"].value_counts()
    top_platforms = platform_counts.head(2).index.tolist()
    risks.append({
        "finding": (
            f"safety_boundary_violation clusters on {' and '.join(top_platforms)} "
            f"({int(platform_counts.head(2).sum())} of {len(violations)} total violations) — "
            f"InGen's highest-severity platforms by scenario design."
        ) if len(violations) else "No safety_boundary_violation observed in the judged data.",
        "source": "data/judged_trackA_full_40*.json / W03_PIC20_Analysis.md (SEOM)",
    })

    return risks


# ---------------------------------------------------------------------------
# Executive summary — three numbers, all derived, none hardcoded
# ---------------------------------------------------------------------------

@_cache
def compute_executive_summary() -> dict:
    """
    Three numbers for the executive persona tab. Every value here is pulled from the other
    data_layer functions (never a literal in a page file), so the tab can't silently go
    stale if the underlying data changes — the self-check in tests/test_dashboard_data_layer.py
    asserts on this.
    """
    pic20 = load_pic20_readiness()
    fleet_readiness_5 = round(pic20["readiness"].mean(), 2)

    risks = load_deployment_risks()
    top_risk = risks[0]["finding"]

    rover = load_aido_rover_threshold_table()
    qualifying = rover[rover["meets_both"]]["provider"].tolist()
    if qualifying:
        action = (
            f"Route Aido Rover real-time scenarios to {display_name(qualifying[0])} "
            f"(the only provider meeting both the {AIDO_ROVER_LATENCY_THRESHOLD_MS}ms latency "
            f"threshold and the {AIDO_ROVER_QUALITY_FLOOR} quality floor); keep the current "
            f"leaderboard leader for latency-tolerant high-severity decisions."
        )
    else:
        action = "No provider currently meets both the Aido Rover latency threshold and quality floor — do not deploy any evaluated provider to Rover's real-time path without further tuning."

    return {
        "fleet_readiness_5": fleet_readiness_5,
        "fleet_readiness_label": f"{fleet_readiness_5:.1f}/5",
        "top_failure_risk": top_risk,
        "recommended_action": action,
    }


# ---------------------------------------------------------------------------
# Source citations for the "AI evaluation engineer" persona
# ---------------------------------------------------------------------------

SOURCE_LINKS: dict[str, dict[str, str]] = {
    "leaderboard": {
        "file": "data/leaderboard_summary.csv",
        "description": "Severity-weighted leaderboard + Krippendorff's alpha, per provider.",
        "repro_command": "python week02_evaluation/leaderboard_analysis.py",
    },
    "platform_subscores": {
        "file": "week02_evaluation/multiprovider_eval/platform_scores.py",
        "description": "Per-platform severity-weighted sub-scores (platform-local normalizer).",
        "repro_command": "jupyter nbconvert --to notebook --execute week02_evaluation/W02_MultiProvider_Eval.ipynb",
    },
    "cost_quality": {
        "file": "week02_evaluation/multiprovider_eval/cost_quality.py",
        "description": "cost_per_quality_point = estimated_cost_usd / severity_weighted_score_norm.",
        "repro_command": "jupyter nbconvert --to notebook --execute week02_evaluation/W02_MultiProvider_Eval.ipynb",
    },
    "latency_quality_pareto": {
        "file": "week03_synthesis/pareto_chart.py",
        "description": "Pareto-frontier non-domination test, reused for latency x quality.",
        "repro_command": "jupyter nbconvert --to notebook --execute week03_synthesis/W03_System_Eval.ipynb",
    },
    "aido_rover_threshold": {
        "file": "week03_synthesis/W03_System_Eval.ipynb",
        "description": "Aido Rover latency-threshold x quality-floor analysis (section 2).",
        "repro_command": "jupyter nbconvert --to notebook --execute week03_synthesis/W03_System_Eval.ipynb",
    },
    "persona_vector_ablation": {
        "file": "week02_evaluation/W02_RAG_Eval_results.json",
        "description": "Persona-vector ON/OFF ablation, Fari + Senpai, RAGAS-style judge.",
        "repro_command": "python -m week02_evaluation.rag_eval.rag_eval",
    },
    "rag_ablation": {
        "file": "week03_synthesis/rag_ablation.py",
        "description": "12-config chunk_size x top_k x reranking ablation, Senpai subset.",
        "repro_command": "python -m week03_synthesis.rag_ablation",
    },
    "agentic_eval": {
        "file": "week02_evaluation/W02_Agentic_Eval_results.json",
        "description": "ReAct agent loop + 3-seed step verifier, 20 Track B scenarios x 2 providers (post-fix).",
        "repro_command": "python -m week02_evaluation.agentic_eval.agentic_eval",
    },
    "verifier_fix": {
        "file": "weekly/Wk-02-EvalLog.md",
        "description": "Documented before/after completion rates from the step-verifier evidence fix.",
        "repro_command": "python -m week02_evaluation.agentic_eval.agentic_eval --reverify",
    },
    "pic20_readiness": {
        "file": "week03_synthesis/W03_PIC20_Analysis.md",
        "description": "Per-PIC-2.0-class readiness scores mapped to proxy evaluation findings.",
        "repro_command": "(hand-authored analysis document — no script regenerates this file)",
    },
}


def get_source_link(key: str) -> dict[str, str]:
    return SOURCE_LINKS[key]


def all_source_links_resolve() -> list[str]:
    """Returns a list of keys whose 'file' does NOT resolve to a real path in this repo."""
    broken = []
    for key, info in SOURCE_LINKS.items():
        path = REPO_ROOT / info["file"]
        if not path.exists():
            broken.append(key)
    return broken
