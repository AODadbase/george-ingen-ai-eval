"""
Week 2 Analysis: Provider Leaderboard + Krippendorff's Alpha
=============================================================

Loads the judged evaluation results from Week 1, then computes:
  1. Krippendorff's Alpha  — inter-seed reliability of task_accuracy scores
  2. Severity-Weighted Aggregate Score per provider
  3. Provider-level summary leaderboard table

Usage
-----
    # Analyse all judged files in data/:
    python week02_evaluation/leaderboard_analysis.py

    # Analyse a specific file:
    python week02_evaluation/leaderboard_analysis.py \
        --input data/judged_trackA_full_40_*.json

Output
------
    data/leaderboard_summary.csv   — provider-level summary table
    Printed leaderboard to stdout
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# Judge seed columns used in llm_judge.py
JUDGE_SEEDS = [42, 100, 2026]
ACCURACY_SEED_COLS = [f"judge_s{s}_task_accuracy" for s in JUDGE_SEEDS]
GROUNDING_SEED_COLS = [f"judge_s{s}_contextual_grounding" for s in JUDGE_SEEDS]

# Token cost estimates (USD per 1K tokens, input+output blended)
# Update these when provider pricing changes.
COST_PER_1K_TOKENS: dict[str, float] = {
    "openai":    0.005,    # gpt-4o blended
    "anthropic": 0.004,    # claude-sonnet-4-6 blended
    "deepseek":  0.0003,   # deepseek-chat blended
    "groq":      0.00008,  # llama-3.1-8b-instant blended
}


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_judged_dataframe(paths: list[Path]) -> pd.DataFrame:
    """
    Load one or more judged result JSON files into a single flat DataFrame.

    Each row is one (scenario × provider) pair with all judge columns
    appended by llm_judge.py.
    """
    frames = []
    for path in paths:
        logger.info("Loading %s", path.name)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        rows = data.get("results", data) if isinstance(data, dict) else data
        frames.append(pd.DataFrame(rows))

    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d rows from %d file(s)", len(df), len(paths))

    # Ensure numeric types for key columns
    numeric_cols = (
        ACCURACY_SEED_COLS + GROUNDING_SEED_COLS
        + ["severity_class", "resp_latency_ms", "resp_prompt_tokens",
           "resp_completion_tokens", "judge_mean_accuracy", "judge_mean_grounding"]
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# 2. Krippendorff's Alpha
# ---------------------------------------------------------------------------

def compute_krippendorff_alpha(
    df: pd.DataFrame,
    seed_cols: list[str],
    level_of_measurement: str = "ordinal",
) -> float:
    """
    Compute Krippendorff's Alpha for inter-seed reliability.

    The krippendorff library expects a reliability data matrix with shape
    (raters, items) where missing values are represented as np.nan.
    The library handles missing values natively — we do NOT dropna() first,
    because that would discard items where only some seeds scored (which is
    exactly the pattern produced by TPM rate-limit misses).

    Parameters
    ----------
    df : DataFrame with judge seed columns
    seed_cols : list of column names (one per seed)
    level_of_measurement : "ordinal" for 1–5 Likert scores

    Returns
    -------
    float — Krippendorff's alpha; NaN if insufficient data
    """
    sub = df[seed_cols]
    # Keep only items where at least 2 of 3 seeds have a score
    has_two = sub.notna().sum(axis=1) >= 2
    sub = sub[has_two]
    if len(sub) < 2:
        logger.warning("Too few scoreable rows (%d) to compute alpha.", len(sub))
        return float("nan")

    # Shape: (n_raters=3, n_items=N) — np.nan encodes missing
    reliability_data = sub.to_numpy(dtype=float).T

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement=level_of_measurement,
            value_domain=[1, 2, 3, 4, 5],  # explicit ordinal domain for accuracy
        )
    return float(alpha)


# ---------------------------------------------------------------------------
# 3. Severity-Weighted Aggregate Score
# ---------------------------------------------------------------------------

def severity_weighted_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute the severity-weighted aggregate accuracy score per row.

    Formula:
        weighted_score = judge_mean_accuracy × severity_class

    Severity-5 failures/successes thus contribute 5× as much as severity-1
    scenarios to the provider aggregate, matching the Week 2 spec.

    Returns a Series aligned to df.index.
    """
    if "judge_mean_accuracy" not in df.columns or "severity_class" not in df.columns:
        raise ValueError("DataFrame must contain 'judge_mean_accuracy' and 'severity_class'.")
    return df["judge_mean_accuracy"] * df["severity_class"]


# ---------------------------------------------------------------------------
# 4. Provider-level leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-provider metrics into a summary leaderboard DataFrame.

    Columns
    -------
    provider                    : provider name
    n_scenarios                 : total (scenario, provider) pairs attempted
    n_scored                    : rows where judge_mean_accuracy is not null
    judge_coverage_pct          : n_scored / n_scenarios × 100
    mean_latency_ms             : mean API latency of the provider
    total_tokens                : sum of prompt + completion tokens
    estimated_cost_usd          : tokens × per-provider price / 1000
    severity_weighted_score     : sum(mean_accuracy × severity_class) over scored rows
    severity_weighted_score_norm: sw_score / max_possible_over_scored_rows
    mean_task_accuracy          : mean of judge_mean_accuracy (scored rows only)
    mean_grounding              : mean of judge_mean_grounding (scored rows only)
    krippendorff_alpha          : inter-seed reliability for this provider
    top_failure_mode            : most common failure mode

    NOTE: severity_weighted_score_norm is normalised against the scored rows only,
    so a provider with fewer scored rows is not penalised for missing data.
    Compare providers on this column, not the raw sw_score sum.
    """
    # Compute per-row weighted score
    df = df.copy()
    df["weighted_score"] = severity_weighted_score(df)

    # Total tokens
    df["total_tokens"] = (
        df.get("resp_prompt_tokens", 0).fillna(0)
        + df.get("resp_completion_tokens", 0).fillna(0)
    )

    records = []
    for provider, grp in df.groupby("resp_provider"):
        n = len(grp)

        # Latency
        latency_col = "resp_latency_ms" if "resp_latency_ms" in grp.columns else "wall_time_ms"
        mean_latency = grp[latency_col].mean() if latency_col in grp.columns else float("nan")

        # Token cost
        total_tokens = grp["total_tokens"].sum()
        cost_per_1k = COST_PER_1K_TOKENS.get(str(provider).lower(), 0.001)
        estimated_cost = total_tokens * cost_per_1k / 1000

        # --- Scored rows (rows where judge_mean_accuracy is not null) ---
        scored = grp[grp["judge_mean_accuracy"].notna()]
        n_scored = len(scored)
        coverage_pct = round(n_scored / n * 100, 1) if n > 0 else 0.0

        # Severity-weighted score — normalised against scored rows only.
        # Max possible per scored row = 5 (max accuracy) × severity_class.
        # This prevents providers with more missing scores from ranking lower
        # simply due to data absence rather than answer quality.
        sw_sum = scored["weighted_score"].sum(skipna=True)
        sw_max = (scored["severity_class"] * 5).sum() if n_scored > 0 else 0
        sw_norm = sw_sum / sw_max if sw_max > 0 else float("nan")

        # Mean accuracy + grounding over scored rows
        mean_acc = scored["judge_mean_accuracy"].mean() if n_scored > 0 else float("nan")
        mean_grd = scored["judge_mean_grounding"].mean() if n_scored > 0 else float("nan")

        # Per-provider Krippendorff's alpha
        alpha = compute_krippendorff_alpha(grp, ACCURACY_SEED_COLS)

        # Top failure mode
        if "judge_majority_failure" in grp.columns:
            mode_counts = grp["judge_majority_failure"].value_counts()
            top_failure = mode_counts.index[0] if len(mode_counts) > 0 else "unknown"
        else:
            top_failure = "unknown"

        records.append({
            "provider":                      provider,
            "n_scenarios":                   n,
            "n_scored":                      n_scored,
            "judge_coverage_pct":            coverage_pct,
            "mean_latency_ms":               round(mean_latency, 1),
            "total_tokens":                  int(total_tokens),
            "estimated_cost_usd":            round(estimated_cost, 4),
            "severity_weighted_score":       round(sw_sum, 2),
            "severity_weighted_score_norm":  round(sw_norm, 4),
            "mean_task_accuracy":            round(mean_acc, 3),
            "mean_grounding":                round(mean_grd, 3),
            "krippendorff_alpha":            round(alpha, 4),
            "top_failure_mode":              top_failure,
        })

    leaderboard = pd.DataFrame(records)
    leaderboard = leaderboard.sort_values("severity_weighted_score_norm", ascending=False)
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard.index += 1  # 1-based rank
    leaderboard.index.name = "rank"
    return leaderboard


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(input_paths: list[Path]) -> pd.DataFrame:
    # --- Load ---
    df = load_judged_dataframe(input_paths)

    # --- Global Krippendorff's alpha ---
    alpha_accuracy = compute_krippendorff_alpha(df, ACCURACY_SEED_COLS, "ordinal")
    alpha_grounding = compute_krippendorff_alpha(df, GROUNDING_SEED_COLS, "ordinal")
    logger.info(
        "Global Krippendorff α — task_accuracy: %.4f | contextual_grounding: %.4f",
        alpha_accuracy, alpha_grounding,
    )

    # --- Leaderboard ---
    leaderboard = build_leaderboard(df)

    # --- Print ---
    print("\n" + "=" * 70)
    print("  InGen AI Model Evaluation — Provider Leaderboard")
    print("=" * 70)
    print(f"  Global Krippendorff α (task_accuracy):    {alpha_accuracy:.4f}")
    print(f"  Global Krippendorff α (grounding):        {alpha_grounding:.4f}")
    print(f"  Total evaluated rows:                     {len(df)}")
    print("=" * 70)
    with pd.option_context("display.max_columns", 20, "display.width", 120,
                            "display.float_format", "{:.3f}".format):
        print(leaderboard.to_string())
    print("=" * 70 + "\n")

    # --- Save ---
    out_path = DATA_DIR / "leaderboard_summary.csv"
    leaderboard.to_csv(out_path)
    logger.info("Leaderboard saved → %s", out_path.relative_to(REPO_ROOT))

    return leaderboard


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Provider leaderboard + Krippendorff's alpha")
    p.add_argument(
        "--input", type=Path, nargs="+", metavar="PATH",
        help="One or more judged result JSON files. Defaults to all judged_*.json in data/.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.input:
        paths = args.input
    else:
        paths = sorted(DATA_DIR.glob("judged_*.json"))
        if not paths:
            raise SystemExit(f"No judged_*.json files found in {DATA_DIR}. "
                             "Run llm_judge.py first.")

    main(paths)
