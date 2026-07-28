"""
Full Failure-Mode Distribution
=================================

leaderboard_analysis.py's leaderboard only reports the single most common
failure mode per provider (top_failure_mode). This module reports the
complete count/percentage breakdown across every category in the judge's
taxonomy, both overall and per platform — a provider whose failures
concentrate on one platform is a more useful finding than an aggregate rate.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from week01_benchmark.eval_harness.llm_judge import FAILURE_MODES  # noqa: E402

# FAILURE_MODES (7 categories) is the static judge taxonomy from llm_judge.py.
# "provider_error" is not in that list — it's assigned at runtime in
# llm_judge.py's evaluate_all() for rows where the original provider call
# failed before judging could happen. Both belong in a complete breakdown.
ALL_CATEGORIES = list(FAILURE_MODES) + ["provider_error"]


def failure_mode_distribution(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Count + percentage breakdown of judge_majority_failure across every
    category in ALL_CATEGORIES (zero-filled for categories a group never
    hit), grouped by the given columns — e.g. ["resp_provider"] for the
    overall breakdown or ["platform", "resp_provider"] for the per-platform one.
    """
    counts = (
        df.groupby(group_cols + ["judge_majority_failure"])
        .size()
        .rename("count")
        .reset_index()
    )

    groups = df[group_cols].drop_duplicates()
    full_index = groups.merge(
        pd.DataFrame({"judge_majority_failure": ALL_CATEGORIES}), how="cross"
    )
    merged = full_index.merge(
        counts, on=group_cols + ["judge_majority_failure"], how="left"
    )
    merged["count"] = merged["count"].fillna(0).astype(int)

    totals = merged.groupby(group_cols)["count"].transform("sum")
    merged["pct"] = (merged["count"] / totals * 100).round(1)
    return merged.sort_values(group_cols + ["judge_majority_failure"]).reset_index(drop=True)
