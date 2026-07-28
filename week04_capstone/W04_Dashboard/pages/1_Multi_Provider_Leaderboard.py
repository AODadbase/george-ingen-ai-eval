"""
View 1 — Multi-Provider Leaderboard.

No computation logic in this file — every number and chart comes from data_layer.py.
"""

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_layer as dl                              # noqa: E402
from persona import persona_selector, render_source   # noqa: E402

st.set_page_config(page_title="Leaderboard — W04 Dashboard", layout="wide")
persona = persona_selector()

st.title("Multi-Provider Leaderboard")

# --- Severity-weighted scorecard ---
st.header("Severity-weighted scorecard")
st.caption(
    "All 4 providers, reused directly from data/leaderboard_summary.csv — same numbers as "
    "the Week 2 memo, not reformatted."
)
leaderboard = dl.load_leaderboard()
st.dataframe(leaderboard, use_container_width=True, hide_index=True)
if persona == "AI evaluation engineer":
    render_source("leaderboard")

st.divider()

# --- Per-platform drill-down ---
st.header("Per-platform drill-down")
subscores = dl.load_platform_subscores()
platforms = sorted(subscores["platform"].unique())
selected_platform = st.selectbox("Platform", platforms, key="leaderboard_platform")
platform_view = subscores[subscores["platform"] == selected_platform].sort_values(
    "severity_weighted_score_norm", ascending=False
)
st.dataframe(
    platform_view[["provider", "n_scenarios_scored", "mean_task_accuracy",
                    "severity_weighted_score", "severity_weighted_score_norm"]],
    use_container_width=True, hide_index=True,
)
st.caption(
    "severity_weighted_score_norm here is normalized against THIS platform's own severity "
    "budget, not the provider-wide leaderboard normalizer above — the two are not directly comparable."
)
if persona == "AI evaluation engineer":
    render_source("platform_subscores")

if persona == "Product manager":
    st.info(
        f"On **{selected_platform}**, the top provider is "
        f"**{dl.display_name(platform_view.iloc[0]['provider'])}** "
        f"({platform_view.iloc[0]['severity_weighted_score_norm']:.3f})."
    )

st.divider()

# --- Cost x quality table ---
st.header("Cost x quality")
cost_quality = dl.load_cost_quality_table()
st.caption(
    "cost_per_quality_point = estimated_cost_usd / severity_weighted_score_norm "
    "(lower is better value). price_per_1k_tokens is backed out from "
    "estimated_cost_usd / total_tokens — not a re-guessed pricing constant."
)
st.dataframe(cost_quality, use_container_width=True, hide_index=True)
if persona == "AI evaluation engineer":
    render_source("cost_quality")

st.divider()

# --- Latency x quality Pareto frontier ---
st.header("Latency x quality Pareto frontier")
lat_qual, frontier = dl.load_latency_quality_pareto()
st.caption(
    "Linear x-axis (not log) — the cost x quality chart in week03_synthesis/pareto_chart.py "
    "used a log axis, which is why its known x0=0 bug mattered there; that bug has since "
    "been fixed at the source, but this chart avoids the whole axis class by not needing "
    "log compression for a ~22x latency range."
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=frontier["mean_latency_ms"], y=frontier["severity_weighted_score_norm"],
    mode="lines", line=dict(color="rgba(255,215,0,0.7)", width=2, dash="dot"),
    name="Pareto Frontier", hoverinfo="skip",
))
for _, row in lat_qual.iterrows():
    fig.add_trace(go.Scatter(
        x=[row["mean_latency_ms"]], y=[row["severity_weighted_score_norm"]],
        mode="markers+text",
        marker=dict(size=20, line=dict(width=3 if row["is_pareto_optimal"] else 1,
                                        color="gold" if row["is_pareto_optimal"] else "white")),
        text=[dl.display_name(row["provider"])], textposition="top center",
        name=dl.display_name(row["provider"]),
    ))
fig.update_layout(
    xaxis_title="Mean latency (ms, linear scale)",
    yaxis_title="Severity-weighted score (normalized)",
    yaxis_range=[0.70, 1.05], template="plotly_white", height=500,
)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"Pareto-optimal: {', '.join(dl.display_name(p) for p in frontier['provider'])}")
if persona == "AI evaluation engineer":
    render_source("latency_quality_pareto")
