"""
Cost x Quality Table
======================

Extends the provider leaderboard (severity_weighted_score_norm, already
computed by leaderboard_analysis.build_leaderboard) with a derived
cost_per_quality_point metric, ranked for the "best value" recommendation
in the Week 2 memo.

Pricing assumptions (USD per 1K tokens, blended input+output) are the
exact COST_PER_1K_TOKENS constants already used to produce
leaderboard_summary.csv — see week02_evaluation/leaderboard_analysis.py.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from week02_evaluation.leaderboard_analysis import COST_PER_1K_TOKENS  # noqa: E402,F401


def build_cost_quality_table(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """
    leaderboard: output of leaderboard_analysis.build_leaderboard(df) —
    already has provider, estimated_cost_usd, severity_weighted_score_norm,
    mean_latency_ms.

    Adds cost_per_quality_point = estimated_cost_usd / severity_weighted_score_norm
    and re-ranks by severity_weighted_score_norm descending.
    """
    out = leaderboard.copy()
    out["cost_per_quality_point"] = (
        out["estimated_cost_usd"] / out["severity_weighted_score_norm"]
    ).round(6)
    cols = [
        "provider", "severity_weighted_score_norm", "estimated_cost_usd",
        "mean_latency_ms", "cost_per_quality_point",
    ]
    out = out[cols].sort_values("severity_weighted_score_norm", ascending=False)
    return out.reset_index(drop=True)
