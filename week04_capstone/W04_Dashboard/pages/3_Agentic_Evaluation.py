"""
View 3 — Agentic Evaluation.

No computation logic in this file — every number and chart comes from data_layer.py.
"""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_layer as dl                              # noqa: E402
from persona import persona_selector, render_source   # noqa: E402

st.set_page_config(page_title="Agentic Evaluation — W04 Dashboard", layout="wide")
persona = persona_selector()

st.title("Agentic Evaluation")

# --- Verifier fix disclosure, up front ---
st.header("Data note: step-verifier evidence fix")
fix_history = dl.load_verifier_fix_history()
st.error(
    "The numbers on this page are **post-fix**. The step verifier originally credited a "
    "required step as complete whenever the agent's Final Answer *claimed* it was done, "
    "without requiring a corresponding Action+Observation in the transcript. This inflated "
    "completion rates. The bug was found and fixed; both numbers are shown below rather than "
    "hiding the correction — the correction is itself a finding worth surfacing."
)
fix_display = fix_history.copy()
fix_display["provider"] = fix_display["provider"].map(dl.display_name)
st.dataframe(fix_display, use_container_width=True, hide_index=True)
if persona == "AI evaluation engineer":
    render_source("verifier_fix")

st.divider()

# --- Task completion + step efficiency per provider per platform ---
st.header("Task completion rate per provider per platform")
completion = dl.load_agentic_completion_by_platform()
completion_pivot = completion.pivot(index="platform", columns="provider", values="completion_rate")
completion_pivot.columns = [dl.display_name(c) for c in completion_pivot.columns]
st.dataframe(completion_pivot.style.format("{:.2f}"), use_container_width=True)

fig = px.imshow(
    completion_pivot, text_auto=".2f", color_continuous_scale="RdYlGn", zmin=0, zmax=1,
    title="Task completion rate — platform x provider", aspect="auto",
)
fig.update_layout(template="plotly_white", height=400)
st.plotly_chart(fig, use_container_width=True)
if persona == "AI evaluation engineer":
    render_source("agentic_eval")

st.divider()

# --- Step efficiency split by completion status ---
st.header("Step efficiency — split by completion status")
st.caption(
    "Per week02_evaluation/W02_Evaluation_Memo.md: a single pooled step_efficiency number "
    "conflates \"efficient success\" with \"gave up early.\" Split by completion status instead."
)
step_eff = dl.load_agentic_step_efficiency_by_completion()
step_eff_display = step_eff.copy()
step_eff_display["provider"] = step_eff_display["provider"].map(dl.display_name)
fig2 = px.bar(
    step_eff_display, x="provider", y="mean_step_efficiency", color="status",
    barmode="group", text="n",
    title="Mean step_efficiency by provider and completion status (bar label = n runs)",
    color_discrete_map={"completed": "#2E8B57", "not completed": "#C0392B"},
)
fig2.add_hline(y=1.0, line_dash="dot", annotation_text="efficiency = 1.0 (minimum required actions)")
fig2.update_layout(template="plotly_white", height=420)
st.plotly_chart(fig2, use_container_width=True)
st.dataframe(step_eff_display, use_container_width=True, hide_index=True)
st.caption(
    "Completed runs use MORE actions than the minimum required (efficiency > 1); "
    "not-completed runs use FEWER actions than required (efficiency < 1) — consistent with "
    "abandoning early rather than inefficient success."
)

st.divider()

# --- Error recovery — sparse, labeled honestly ---
st.header("Error recovery rate")
recovery = dl.load_agentic_error_recovery_sparse()
st.warning(
    f"Only **{recovery['n_applicable']} of {recovery['n_total_transcripts']}** transcripts "
    "had an applicable error observation to recover from. This is **not enough data for a "
    "real per-provider-per-platform comparison** — no heatmap is shown here because a heatmap "
    "would visually imply a full comparison exists when it does not. The actual sparse data:"
)
recovery_display = recovery["applicable_rows"].copy()
recovery_display["provider"] = recovery_display["provider"].map(dl.display_name)
st.dataframe(recovery_display, use_container_width=True, hide_index=True)
st.caption(
    f"sufficient_for_comparison = {recovery['sufficient_for_comparison']} "
    "(stated floor: at least 5 applicable transcripts before treating this as comparable across providers)."
)

if persona == "Product manager":
    st.info(
        "Do not use error-recovery rate as a selection criterion yet — the current agentic "
        "scenario set produces too few error observations to compare providers on this metric."
    )
