"""
Self-check tests for the W04 dashboard — an actual test suite, not a manual note.

Run with:
    python -m pytest tests/test_dashboard_data_layer.py -v

Three checks, matching the dashboard's stated self-check requirements exactly:
  1. No page file imports a live-inference module (clients.py / rag_scorer.py /
     step_verifier.py / llm_judge.py) — mechanically verifies "no model inference on launch".
  2. Every source-citation key referenced by a page file resolves to a real file in this repo.
  3. The executive tab's three numbers are pulled from data_layer.py, not hardcoded literals
     in the page file.
"""

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = REPO_ROOT / "week04_capstone" / "W04_Dashboard"
sys.path.insert(0, str(DASHBOARD_DIR))

import data_layer as dl  # noqa: E402

FORBIDDEN_MODULES = ("clients", "rag_scorer", "step_verifier", "llm_judge")


def _page_files() -> list[Path]:
    files = [DASHBOARD_DIR / "app.py"]
    files += sorted((DASHBOARD_DIR / "pages").glob("*.py"))
    assert files, "No dashboard page files found — check DASHBOARD_DIR"
    return files


def _imported_module_names(source: str) -> set[str]:
    """All module names referenced by import / from-import statements, last path segment."""
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.rsplit(".", 1)[-1])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.rsplit(".", 1)[-1])
    return names


# ---------------------------------------------------------------------------
# Check 1 — no live-inference imports in any page file
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("page_path", _page_files(), ids=lambda p: p.name)
def test_page_has_no_live_inference_imports(page_path: Path):
    source = page_path.read_text(encoding="utf-8")
    imported = _imported_module_names(source)
    violations = imported & set(FORBIDDEN_MODULES)
    assert not violations, (
        f"{page_path.relative_to(REPO_ROOT)} imports live-inference module(s) {violations} — "
        "a page file must only call data_layer.py functions, never a provider/judge client "
        "directly. No live inference path may exist at dashboard launch."
    )


def test_data_layer_itself_has_no_live_inference_imports():
    """data_layer.py loads pre-computed files only — same rule applies to it."""
    source = (DASHBOARD_DIR / "data_layer.py").read_text(encoding="utf-8")
    imported = _imported_module_names(source)
    violations = imported & set(FORBIDDEN_MODULES)
    assert not violations, f"data_layer.py imports live-inference module(s) {violations}"


# ---------------------------------------------------------------------------
# Check 2 — every source-citation key used by a page resolves to a real file
# ---------------------------------------------------------------------------

def test_all_data_layer_source_links_resolve_to_real_files():
    broken = dl.all_source_links_resolve()
    assert not broken, (
        f"SOURCE_LINKS entries with a non-existent 'file' path: {broken} — every number shown "
        "under the AI evaluation engineer persona must cite a file that actually exists."
    )


def test_every_render_source_call_uses_a_real_key():
    """
    Every render_source("<key>") call in a page file must reference a key that actually
    exists in data_layer.SOURCE_LINKS (and therefore, by the check above, a real file).
    """
    import re
    pattern = re.compile(r'render_source\(\s*["\']([^"\']+)["\']\s*\)')
    missing: list[str] = []
    for page_path in _page_files():
        source = page_path.read_text(encoding="utf-8")
        for key in pattern.findall(source):
            if key not in dl.SOURCE_LINKS:
                missing.append(f"{page_path.name}: render_source({key!r}) — no such SOURCE_LINKS key")
    assert not missing, "\n".join(missing)


# ---------------------------------------------------------------------------
# Check 3 — executive tab's three numbers come from data_layer, not literals in app.py
# ---------------------------------------------------------------------------

def test_executive_tab_calls_data_layer_for_its_numbers():
    source = (DASHBOARD_DIR / "app.py").read_text(encoding="utf-8")
    assert "compute_executive_summary" in source, (
        "app.py's executive tab must call dl.compute_executive_summary() — found no reference."
    )


def test_executive_tab_numbers_are_not_hardcoded_literals():
    """
    compute_executive_summary()'s three long, data-specific sentences would only appear as
    literal text in app.py's source if someone had hardcoded them instead of computing them
    dynamically. They should NOT appear as raw source text — only as f-string / dict-key
    references to the `summary` variable.
    """
    source = (DASHBOARD_DIR / "app.py").read_text(encoding="utf-8")
    summary = dl.compute_executive_summary()

    for field in ("fleet_readiness_label", "top_failure_risk", "recommended_action"):
        value = summary[field]
        assert value not in source, (
            f"app.py contains the literal computed value of '{field}' "
            f"({value!r}) as raw source text — it must be referenced dynamically via "
            f"summary[{field!r}], not hardcoded."
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
