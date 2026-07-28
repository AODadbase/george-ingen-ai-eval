"""
Per-Platform Sub-Scores
=========================

Extends the provider-wide leaderboard (leaderboard_analysis.py) with a
platform x provider breakdown, reusing the exact same weighted-score
formula but normalized within each platform's own severity budget rather
than the provider-wide one — so a platform where a provider quietly
underperforms is visible even when the provider's overall score looks fine.
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from week02_evaluation.leaderboard_analysis import severity_weighted_score  # noqa: E402


def build_platform_subscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (platform, provider), scored rows only.

    severity_weighted_score_norm is normalized against THIS platform's own
    scored-row severity budget (sum(severity_class) * 5) — not the
    provider-wide normalizer used in leaderboard_analysis.build_leaderboard.
    """
    df = df.copy()
    df["weighted_score"] = severity_weighted_score(df)

    records = []
    for (platform, provider), grp in df.groupby(["platform", "resp_provider"]):
        scored = grp[grp["judge_mean_accuracy"].notna()]
        n_scored = len(scored)
        sw_sum = scored["weighted_score"].sum(skipna=True)
        sw_max = (scored["severity_class"] * 5).sum() if n_scored > 0 else 0
        sw_norm = sw_sum / sw_max if sw_max > 0 else float("nan")
        mean_acc = scored["judge_mean_accuracy"].mean() if n_scored > 0 else float("nan")

        records.append({
            "platform": platform,
            "provider": provider,
            "n_scenarios_scored": n_scored,
            "mean_task_accuracy": round(mean_acc, 3) if n_scored else float("nan"),
            "severity_weighted_score": round(sw_sum, 3),
            "severity_weighted_score_norm": round(sw_norm, 4),
        })

    out = pd.DataFrame(records)
    return out.sort_values(["platform", "provider"]).reset_index(drop=True)


def pivot_metric(subscores: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Platform rows x provider columns pivot of the given metric column."""
    return subscores.pivot(index="platform", columns="provider", values=metric)
