"""
View 4 — System-Level.

No computation logic in this file — every number and chart comes from data_layer.py.
"""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_layer as dl                              # noqa: E402
from persona import persona_selector, render_source   # noqa: E402

st.set_page_config(page_title="System-Level — W04 Dashboard", layout="wide")
persona = persona_selector()

st.title("System-Level View")

# --- Latency distribution per provider ---
st.header("Latency distribution per provider")
judged = dl.load_judged_data()
judged_display = judged.copy()
judged_display["provider"] = judged_display["resp_provider"].map(dl.display_name)
fig = px.box(
    judged_display, x="provider", y="resp_latency_ms", color="provider", points="all",
    title="Response latency distribution (all 160 judged rows, trackA + trackB)",
)
fig.update_layout(template="plotly_white", height=450, showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Aido Rover latency-threshold analysis ---
st.header("Aido Rover real-time latency-threshold analysis")
st.caption(
    f"Threshold: latency < {dl.AIDO_ROVER_LATENCY_THRESHOLD_MS}ms (stated assumption — a "
    "common HRI/voice-assistant bound for a hazard-acknowledgment turn to still read as "
    "\"prompt\" rather than \"stuck\"). Quality floor: "
    f"severity_weighted_score_norm >= {dl.AIDO_ROVER_QUALITY_FLOOR} (reused from "
    "week03_synthesis/pareto_chart.py's existing ideal-zone cutoff, for consistency)."
)
rover = dl.load_aido_rover_threshold_table()
rover_display = rover.copy()
rover_display["provider"] = rover_display["provider"].map(dl.display_name)
st.dataframe(rover_display, use_container_width=True, hide_index=True)

qualifying = rover[rover["meets_both"]]["provider"].tolist()
if qualifying:
    st.success(
        f"Providers meeting BOTH conditions: {', '.join(dl.display_name(p) for p in qualifying)}."
    )
else:
    st.error("No provider meets both conditions.")
if persona == "AI evaluation engineer":
    render_source("aido_rover_threshold")

st.divider()

# --- Cost-per-quality-point bar chart ---
st.header("Cost-per-quality-point")
cq = dl.load_cost_quality_table()
cq_display = cq.copy()
cq_display["provider"] = cq_display["provider"].map(dl.display_name)
fig2 = px.bar(
    cq_display.sort_values("cost_per_quality_point"), x="provider", y="cost_per_quality_point",
    color="provider", text="cost_per_quality_point",
    title="cost_per_quality_point by provider (lower = better value)",
)
fig2.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig2.update_layout(template="plotly_white", height=420, showlegend=False)
st.plotly_chart(fig2, use_container_width=True)
if persona == "AI evaluation engineer":
    render_source("cost_quality")
