"""
View 2 — RAG Performance.

No computation logic in this file — every number and chart comes from data_layer.py.
"""

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_layer as dl                              # noqa: E402
from persona import persona_selector, render_source   # noqa: E402

st.set_page_config(page_title="RAG Performance — W04 Dashboard", layout="wide")
persona = persona_selector()

st.title("RAG Performance")

# --- Persona-vector ablation ---
st.header("Persona-vector ablation (ON vs OFF)")

aggregate = dl.load_persona_vector_aggregate()
delta_answer_relevance = aggregate.loc["delta (ON - OFF)", "answer_relevance"]
st.warning(
    f"**Verified finding**: enabling the persona-vector layer *reduces* answer_relevance by "
    f"{delta_answer_relevance:+.3f} overall — the opposite of IGuide's design intuition "
    f"that a domain descriptor prefix improves retrieval relevance."
)
st.dataframe(aggregate, use_container_width=True)

st.subheader("Per-platform breakdown")
st.caption(
    "The aggregate delta above hides that this effect is entirely concentrated on Fari — "
    "Senpai shows zero effect. Shown separately here rather than only as an aggregate."
)
by_platform = dl.load_persona_vector_by_platform()
by_platform_display = by_platform.copy()
by_platform_display["use_persona_vector"] = by_platform_display["use_persona_vector"].map(
    {True: "ON", False: "OFF"}
)
st.dataframe(by_platform_display, use_container_width=True, hide_index=True)

fig = px.bar(
    by_platform_display, x="platform", y="answer_relevance", color="use_persona_vector",
    barmode="group", title="answer_relevance by platform x persona-vector condition",
    color_discrete_map={"ON": "#C97B44", "OFF": "#4A90D9"},
)
fig.update_layout(template="plotly_white", height=400, yaxis_range=[0, 1.05])
st.plotly_chart(fig, use_container_width=True)
if persona == "AI evaluation engineer":
    render_source("persona_vector_ablation")

st.divider()

# --- 12-config RAG ablation ---
st.header("RAG configuration ablation (chunk_size x top_k x reranking)")
st.caption(
    "12 configs on the Senpai scenario subset: full chunk_size x top_k grid at "
    "reranking=none (9 configs) + reranking=cross-encoder at top_k=3 across all 3 "
    "chunk_sizes (3 configs). See week03_synthesis/rag_ablation.py for the full design "
    "rationale (the plan's own '12 (3x2x2)' framing doesn't square with "
    "chunk_size x top_k x reranking = 3x3x2 = 18)."
)

configs = dl.load_rag_ablation_configs()
pareto_configs = dl.load_rag_ablation_pareto()
optimal_id = pareto_configs.iloc[0]["config_id"]

fig2 = px.scatter(
    configs, x="latency_ms", y="faithfulness", color="reranking", symbol="chunk_size",
    hover_data=["config_id", "top_k", "answer_relevance", "context_coverage"],
    title="Faithfulness x latency, 12 configurations",
)
optimal_row = configs[configs["config_id"] == optimal_id].iloc[0]
fig2.add_scatter(
    x=[optimal_row["latency_ms"]], y=[optimal_row["faithfulness"]],
    mode="markers", marker=dict(size=22, color="gold", symbol="star"),
    name=f"Pareto-optimal: {optimal_id}",
)
fig2.update_layout(template="plotly_white", height=450)
st.plotly_chart(fig2, use_container_width=True)

st.dataframe(
    configs.sort_values(["reranking", "chunk_size", "top_k"]),
    use_container_width=True, hide_index=True,
)

st.markdown(
    f"**Pareto-optimal config: `{optimal_id}`** — wins on latency "
    f"({optimal_row['latency_ms']:.0f}ms) and faithfulness ({optimal_row['faithfulness']:.2f}), "
    f"**but this is not an unqualified win**: it costs answer_relevance "
    f"(1.00 → {optimal_row['answer_relevance']:.2f}) and context_coverage "
    f"(0.85 → {optimal_row['context_coverage']:.2f}) relative to production's `top_k=3` setting. "
    f"With only one retrieved document, the model has nothing to hallucinate *from* (hence "
    f"perfect faithfulness) but also less material to build a complete answer from. "
    f"The memo and paper draft were careful about this trade-off; this dashboard states it "
    f"the same way rather than simplifying it into \"the best config.\""
)
if persona == "AI evaluation engineer":
    render_source("rag_ablation")

if persona == "Product manager":
    st.info(
        "Recommendation: keep production's `top_k=3` unless latency becomes the binding "
        "constraint for Senpai specifically — the ~20% latency saving from `top_k=1` is real "
        "but trades away answer completeness."
    )
