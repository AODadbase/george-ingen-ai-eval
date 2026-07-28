"""
Data-Quality Guard
===================

Hard-fails before any leaderboard computation runs if the judged data does
not meet the Week 2 completeness bar.

A prior notebook in this repo had a row-count assertion that was silently
skipped whenever a ``dry_run`` flag was set, letting a 6-row placeholder
pass as if it were the real 40-row result. These checks are unconditional
``assert`` statements with no bypass flag — any violation raises and names
the exact file and rows responsible.
"""

import json
from pathlib import Path

EXPECTED_ROWS_PER_FILE = 80
REQUIRED_EVALUATION_SET = "full_40"
FORBIDDEN_METADATA_KEYS = ("dry_run", "smoke_test")


def load_raw(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        for key in FORBIDDEN_METADATA_KEYS:
            if key in data:
                raise AssertionError(
                    f"{path.name}: found forbidden metadata key '{key}'={data[key]!r} — "
                    "this file may be a smoke-test / dry-run artifact, not the real "
                    "full_40 result."
                )
        rows = data.get("results", data)
    else:
        rows = data
    return rows


def assert_row_count(path: Path, rows: list[dict]) -> None:
    assert len(rows) == EXPECTED_ROWS_PER_FILE, (
        f"{path.name}: expected exactly {EXPECTED_ROWS_PER_FILE} rows "
        f"(20 scenarios x 4 providers), got {len(rows)}."
    )


def assert_no_dry_run_flags(path: Path, rows: list[dict]) -> None:
    offenders = [
        (r.get("scenario_id"), r.get("resp_provider"))
        for r in rows
        if any(k in r for k in FORBIDDEN_METADATA_KEYS)
    ]
    assert not offenders, (
        f"{path.name}: rows carry a dry_run/smoke_test flag — not a real full_40 "
        f"result. Offending (scenario_id, provider) rows: {offenders}"
    )


def assert_evaluation_set(path: Path, rows: list[dict]) -> None:
    values = {r.get("evaluation_set") for r in rows}
    assert values == {REQUIRED_EVALUATION_SET}, (
        f"{path.name}: evaluation_set must be exactly {{'{REQUIRED_EVALUATION_SET}'}} "
        f"for every row, found {values}. Refusing to treat this as the real full "
        "evaluation."
    )


def assert_full_judge_coverage(path: Path, rows: list[dict]) -> None:
    """
    Per-provider judge_coverage_pct must be 100.0 — no row may have a null
    judge_mean_accuracy. Names the exact (scenario_id, provider) pairs that
    are missing rather than reporting an aggregate percentage.
    """
    missing = [
        (r.get("scenario_id"), r.get("resp_provider"))
        for r in rows
        if r.get("judge_mean_accuracy") is None
    ]
    assert not missing, (
        f"{path.name}: {len(missing)} row(s) have a null judge_mean_accuracy "
        "(judge_coverage_pct < 100.0 for at least one provider). "
        f"Missing (scenario_id, provider) pairs: {missing}"
    )


def run_guard(paths: list[Path]) -> dict[str, list[dict]]:
    """
    Run every hard-assert check on every path.

    Returns {path.name: rows} on success. Raises AssertionError on the
    first violation encountered.
    """
    loaded: dict[str, list[dict]] = {}
    for path in paths:
        rows = load_raw(path)
        assert_row_count(path, rows)
        assert_no_dry_run_flags(path, rows)
        assert_evaluation_set(path, rows)
        assert_full_judge_coverage(path, rows)
        loaded[path.name] = rows
    return loaded
