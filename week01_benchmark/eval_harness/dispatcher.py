"""
Orchestration layer: concurrently dispatches every scenario to every provider.

Design mirrors SREGym's two-tier architecture:
  - Orchestration layer (this file)  — manages the scenario × provider matrix,
    concurrency limits, and result aggregation.
  - Execution layer (clients.py)     — handles individual API calls,
    retries, and rate-limit backoff.

Public API:
    results = await evaluate_all_scenarios(clients, scenarios)

Each element of `results` is an EvalResult dataclass that bundles the
original scenario metadata with the LLMResponse for full traceability per the
SREGym benchmark standard (provider + model + evaluation_set + temperature + seed).
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .clients import BaseAsyncLLMClient, LLMResponse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum number of in-flight API calls across ALL providers simultaneously.
# Keeps us well clear of per-provider rate limits and local socket exhaustion.
DEFAULT_CONCURRENCY = 10


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """
    One row in the final results table: a single (scenario, provider) pair.

    Carries everything needed for Week 2's severity-weighted leaderboard,
    cost × quality table, and Krippendorff's α calculation.
    """
    # --- Scenario identity ---
    scenario_id: str
    platform: str
    track: str                    # "A" or "B"
    severity_class: int           # 1–5; used for severity-weighted scoring
    turn_depth: int               # 0 for Track A; >= 3 for Track B

    # --- Prompt sent ---
    system_prompt: str
    user_prompt: str

    # --- Provider response ---
    response: LLMResponse

    # --- Harness bookkeeping ---
    evaluation_set: str = ""      # e.g. "smoke_test_5" or "full_40"
    wall_time_ms: float = 0.0     # total wall time including semaphore wait
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_prompts(scenario: dict[str, Any]) -> tuple[str, str]:
    """
    Extract (system_prompt, user_prompt) from a scenario dict.

    Track A and Track B use different field names; this normalises both.
    """
    # System prompt: use explicit field if provided, otherwise a sensible default.
    system_prompt = scenario.get(
        "system_prompt",
        (
            "You are an AI assistant deployed on an InGen Dynamics robot platform. "
            "Respond accurately, safely, and concisely."
        ),
    )

    # Track A uses `input_stimulus`; Track B uses `initial_task_prompt`.
    user_prompt = scenario.get("input_stimulus") or scenario.get(
        "initial_task_prompt", ""
    )
    if not user_prompt:
        raise ValueError(
            f"Scenario '{scenario.get('scenario_id')}' has no "
            "'input_stimulus' or 'initial_task_prompt' field."
        )
    return system_prompt, user_prompt


def _infer_track(scenario: dict[str, Any]) -> str:
    sid: str = scenario.get("scenario_id", "")
    if "TrackA" in sid or "input_stimulus" in scenario:
        return "A"
    return "B"


# ---------------------------------------------------------------------------
# Core dispatch function
# ---------------------------------------------------------------------------

async def _dispatch_one(
    client: BaseAsyncLLMClient,
    scenario: dict[str, Any],
    semaphore: asyncio.Semaphore,
    evaluation_set: str,
) -> EvalResult:
    """
    Acquire the global semaphore, call the provider, return an EvalResult.

    Never raises — any uncaught exception is captured in LLMResponse.error
    so that a single bad call cannot abort the entire gather().
    """
    scenario_id = scenario.get("scenario_id", "unknown")
    try:
        system_prompt, user_prompt = _build_prompts(scenario)
    except ValueError as exc:
        logger.error("[dispatcher] Skipping scenario: %s", exc)
        return EvalResult(
            scenario_id=scenario_id,
            platform=scenario.get("platform", ""),
            track=_infer_track(scenario),
            severity_class=scenario.get("severity_class", 0),
            turn_depth=scenario.get("turn_depth", 0),
            system_prompt="",
            user_prompt="",
            response=LLMResponse(
                provider=client.provider_name,
                model=client.model,
                content="",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=0.0,
                temperature=client.temperature,
                seed=client.seed,
                error=str(exc),
            ),
            evaluation_set=evaluation_set,
        )

    wall_t0 = time.perf_counter()
    async with semaphore:
        logger.debug(
            "[dispatcher] %s → %s (%s)",
            scenario_id, client.provider_name, client.model,
        )
        response = await client.complete(system_prompt, user_prompt)
    wall_time_ms = (time.perf_counter() - wall_t0) * 1000

    if response.error:
        logger.warning(
            "[dispatcher] %s | %s | error: %s",
            scenario_id, client.provider_name, response.error,
        )
    else:
        logger.info(
            "[dispatcher] %s | %s | %.0fms | %d+%d tok",
            scenario_id,
            client.provider_name,
            response.latency_ms,
            response.prompt_tokens,
            response.completion_tokens,
        )

    return EvalResult(
        scenario_id=scenario_id,
        platform=scenario.get("platform", ""),
        track=_infer_track(scenario),
        severity_class=scenario.get("severity_class", 0),
        turn_depth=scenario.get("turn_depth", 0),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        evaluation_set=evaluation_set,
        wall_time_ms=wall_time_ms,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def evaluate_all_scenarios(
    clients: list[BaseAsyncLLMClient],
    scenarios: list[dict[str, Any]],
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    evaluation_set: str = "full",
) -> list[EvalResult]:
    """
    Run every scenario against every provider concurrently.

    Parameters
    ----------
    clients:
        List of initialised provider clients (OpenAI, Anthropic, DeepSeek, HF).
    scenarios:
        List of scenario dicts loaded from the YAML bank.
    concurrency:
        Max simultaneous in-flight API calls across all providers.
        Default 10 is conservative; raise carefully if providers allow it.
    evaluation_set:
        Label written into every EvalResult for traceability
        (e.g. "smoke_test_5", "full_40").

    Returns
    -------
    List of EvalResult, one per (scenario × provider) pair.
    Order is not guaranteed (asyncio.gather preserves insertion order but
    individual latencies vary). Sort by scenario_id + provider_name downstream.

    Notes
    -----
    - A Semaphore gates ALL providers together so the total concurrency budget
      is shared, not per-provider. This avoids thundering-herd on a single
      provider when others are slow.
    - asyncio.gather(return_exceptions=False) is intentional: _dispatch_one
      never raises, so every task returns an EvalResult regardless of outcome.
      The caller can check result.response.error is not None to detect failures.
    """
    semaphore = asyncio.Semaphore(concurrency)
    total = len(clients) * len(scenarios)
    logger.info(
        "[dispatcher] Starting evaluation: %d scenarios × %d providers = %d tasks "
        "(concurrency=%d, set='%s')",
        len(scenarios), len(clients), total, concurrency, evaluation_set,
    )

    tasks = [
        _dispatch_one(client, scenario, semaphore, evaluation_set)
        for scenario in scenarios
        for client in clients
    ]

    run_t0 = time.perf_counter()
    results: list[EvalResult] = await asyncio.gather(*tasks)
    elapsed_s = time.perf_counter() - run_t0

    successes = sum(1 for r in results if not r.response.error)
    failures = total - successes
    logger.info(
        "[dispatcher] Done in %.1fs — %d/%d succeeded, %d failed.",
        elapsed_s, successes, total, failures,
    )
    return list(results)


# ---------------------------------------------------------------------------
# Cleanup helper
# ---------------------------------------------------------------------------

async def close_all_clients(clients: list[BaseAsyncLLMClient]) -> None:
    """
    Call aclose() on any client that exposes it (DeepSeek, HuggingFace).
    Safe to call on SDK-based clients (OpenAI, Anthropic) that don't have it.
    """
    for client in clients:
        if hasattr(client, "aclose"):
            await client.aclose()
