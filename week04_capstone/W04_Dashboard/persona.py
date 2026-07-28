"""
Shared persona selector — pure UI, no computation.

Used identically (same widget, same `key="persona"`) at the top of every page so the
selection persists across Streamlit's page navigation via st.session_state, per the "one
top-level selector that filters what's shown, not three separate apps" requirement.
"""

import streamlit as st

import data_layer as dl

PERSONAS = ["AI evaluation engineer", "Product manager", "Executive"]
DEFAULT_PERSONA = PERSONAS[0]


def persona_selector() -> str:
    return st.sidebar.radio(
        "View as", PERSONAS, key="persona",
        help="Filters the detail level shown on every page — not a separate app per role.",
    )


def current_persona() -> str:
    """Read the persona without rendering the widget again (e.g. inside a helper)."""
    return st.session_state.get("persona", DEFAULT_PERSONA)


def render_source(key: str) -> None:
    """Engineer-persona-only source citation, resolved against data_layer.SOURCE_LINKS."""
    info = dl.get_source_link(key)
    st.caption(
        f"Source: `{info['file']}` — {info['description']}  \n"
        f"Reproduce: `{info['repro_command']}`"
    )
