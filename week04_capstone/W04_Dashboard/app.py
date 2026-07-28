"""
W04 Dashboard — entry page.

Streamlit multipage app: this file is the landing page; pages/1-4 are the four views. Every
page (including this one) loads exclusively from data_layer.py, which loads exclusively from
pre-computed files already on disk — no LLM provider or judge is ever called here.
"""

import streamlit as st

import data_layer as dl
from persona import persona_selector

st.set_page_config(
    page_title="InGen AI Model Evaluation — W04 Dashboard",
    layout="wide",
)

persona = persona_selector()

st.title("InGen AI Model Evaluation — Capstone Dashboard")
st.caption(
    "Every number in this dashboard is pre-computed and loaded from a file already on disk. "
    "No page here calls an LLM provider or judge on launch."
)

st.markdown(
    """
Use the sidebar to navigate the four views:

1. **Multi-Provider Leaderboard** — severity-weighted scorecard, per-platform drill-down, cost x quality, latency x quality Pareto frontier.
2. **RAG Performance** — persona-vector ablation (Week 2) and the 12-config chunk_size x top_k x reranking ablation (Week 3).
3. **Agentic Evaluation** — task completion, step efficiency, and the step-verifier evidence fix.
4. **System-Level** — latency distribution, Aido Rover real-time threshold, cost-per-quality-point.

The **View as** selector in the sidebar (currently: **{persona}**) applies to every page —
it changes how much detail and which framing is shown, not which data is loaded.
""".format(persona=persona)
)

if persona == "Executive":
    st.divider()
    summary = dl.compute_executive_summary()
    # Every heading below is the finding itself, not a topic label ("Summary", "Risks", ...) —
    # required for the executive tab specifically, matching the Week 4 slide-deck self-check.
    st.header(f"Fleet-wide readiness: {summary['fleet_readiness_label']}")
    st.caption(
        "Mean readiness across PIC 2.0's six model classes "
        "(week03_synthesis/W03_PIC20_Analysis.md) — the closest thing this program has to a "
        "single fleet-wide number, since it is already a documented, cited readiness score "
        "rather than an invented composite."
    )
    st.header(summary["top_failure_risk"])
    st.header(summary["recommended_action"])

if persona == "Product manager":
    st.divider()
    st.header("Per-PIC-2.0-class readiness (not per-platform — see caveat below)")
    st.caption(
        "week03_synthesis/W03_PIC20_Analysis.md documents readiness per PIC-2.0 model class "
        "(GRPO/STUM/SEOM/AMDC/HTD-IRL/CRL-MRS), not per InGen platform — no platform-to-class "
        "mapping is documented anywhere in this repo's source material (checked: neither the "
        "landscape brief nor this analysis states one). Shown below as-written rather than "
        "fabricating a join. Platform-level risk findings are shown separately underneath, "
        "drawn directly from the evaluation data instead."
    )
    pic20 = dl.load_pic20_readiness()
    st.dataframe(
        pic20[["class", "readiness_label", "strongest_evidence", "largest_gap"]],
        use_container_width=True, hide_index=True,
    )
    st.subheader("Top-3 platform-level deployment risks")
    for i, risk in enumerate(dl.load_deployment_risks(), 1):
        st.markdown(f"**{i}.** {risk['finding']}")
        st.caption(f"Source: {risk['source']}")
