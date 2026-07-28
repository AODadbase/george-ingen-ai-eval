"""
Agentic Evaluation Runner
==========================

Runs 20 TrackB scenarios × 2 providers (anthropic + deepseek) through the
ReAct agent loop, verifies each step with the 3-seed judge, and computes
the three SREGym metrics:

  task_completion_rate  : 1 if all steps verified + no early-exit, else 0
  step_efficiency       : actual_actions_taken / len(required_steps)
                          (lower is better; 1.0 = optimal; always ≥ 1.0)
  error_recovery_rate   : fraction of error observations after which the
                          agent's next action was corrective (not the same
                          action repeated verbatim; not abandonment)

Outputs
-------
  week02_evaluation/W02_Agentic_Eval_results.json

Usage
-----
    # Full run (all 20 scenarios × 2 providers):
    python -m week02_evaluation.agentic_eval.agentic_eval

    # Dry-run (skips LLM calls, uses placeholder values):
    python -m week02_evaluation.agentic_eval.agentic_eval --dry-run

    # Limit to N scenarios (for smoke testing):
    python -m week02_evaluation.agentic_eval.agentic_eval --limit 2
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from week01_benchmark.eval_harness.clients import AnthropicClient, DeepSeekClient  # noqa: E402
from .agent_loop import ReActAgentLoop, AgentRunResult                             # noqa: E402
from .step_verifier import StepVerifier, VerificationResult                        # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_TEMPERATURE = 0.0
AGENT_SEED = 42
AGENT_MODEL_ANTHROPIC = "claude-sonnet-4-6"
AGENT_MODEL_DEEPSEEK = "deepseek-chat"
ENV_MODEL_ANTHROPIC = "claude-sonnet-4-6"   # env simulator uses same provider
ENV_MODEL_DEEPSEEK = "deepseek-chat"

SCENARIO_YAML = REPO_ROOT / "week01_benchmark" / "trackB_agentic.yaml"
OUTPUT_PATH = REPO_ROOT / "week02_evaluation" / "W02_Agentic_Eval_results.json"

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_task_completion_rate(
    verification: Optional[VerificationResult],
    agent_run: AgentRunResult,
) -> int:
    """1 if all steps verified True + no early-exit; else 0."""
    if verification is None:
        return 0
    if verification.early_exit_triggered or agent_run.early_exit_triggered:
        return 0
    if not verification.step_verdicts:
        return 0
    return int(all(sv.final_verdict is True for sv in verification.step_verdicts))


def compute_step_efficiency(
    agent_run: AgentRunResult,
    n_required_steps: int,
) -> Optional[float]:
    """
    actual_actions_taken / n_required_steps.

    Lower is better; 1.0 = agent completed the task in the minimum number
    of actions. Must always be ≥ 1.0 (warn if < 1.0, which indicates a
    transcript parsing issue rather than a real result).
    """
    if n_required_steps == 0:
        return None
    if agent_run.actual_actions_taken == 0:
        return None
    efficiency = agent_run.actual_actions_taken / n_required_steps
    if efficiency < 1.0:
        logger.warning(
            "[metrics] %s | step_efficiency=%.2f < 1.0 — possible transcript parse bug "
            "(actual_actions=%d, required=%d)",
            agent_run.scenario_id, efficiency,
            agent_run.actual_actions_taken, n_required_steps,
        )
    return round(efficiency, 4)


def compute_error_recovery_rate(transcript_dicts: list[dict]) -> Optional[float]:
    """
    Of actions whose Observation indicated an error, the fraction where
    the agent's *next* action was corrective (different tool or different
    meaningful input — not an exact repeat, not an abandonment).

    Heuristic definition of "corrective":
      - The next action uses a different tool, OR
      - The next action uses the same tool but with a meaningfully different
        input (edit distance > 20% of the longer string)
      - AND the agent did not produce a Final Answer immediately after
        (which would indicate premature abandonment)
    """
    error_indices = []
    actions = []

    for i, entry in enumerate(transcript_dicts):
        if entry.get("role") == "action":
            actions.append((i, entry))
        elif entry.get("role") == "observation" and entry.get("is_error"):
            error_indices.append(i)

    if not error_indices:
        return None  # no errors → metric not applicable

    recoveries = 0
    for err_idx in error_indices:
        # Find the action before this observation
        preceding_action = None
        for t_idx, action in reversed(actions):
            if t_idx < err_idx:
                preceding_action = action
                break
        if preceding_action is None:
            continue

        # Find the action that follows this observation
        next_action = None
        for t_idx, action in actions:
            if t_idx > err_idx:
                next_action = action
                break

        if next_action is None:
            # Check if immediately followed by Final Answer
            for entry in transcript_dicts[err_idx + 1:]:
                if entry.get("role") == "final_answer":
                    break  # abandoned
            continue  # no next action → not recovered

        prev_tool = preceding_action.get("tool", "")
        next_tool = next_action.get("tool", "")
        prev_input = preceding_action.get("tool_input", "")
        next_input = next_action.get("tool_input", "")

        if next_tool != prev_tool:
            recoveries += 1
        elif prev_input and next_input:
            # Approximate similarity: positional character overlap / max length.
            # This is NOT true edit distance — it does not handle transpositions,
            # insertions, or deletions correctly. It is a fast proxy that works
            # well when the same tool is called with an identical or near-identical
            # input (clear repetition) vs. a clearly different input (recovery).
            # Limitation: inputs that are similar in content but differ in word
            # order or length may be mis-classified. Spot-check against the
            # transcript for any scenario where error_recovery_rate seems surprising.
            longer = max(len(prev_input), len(next_input))
            if longer > 0:
                common = sum(
                    a == b for a, b in zip(prev_input[:longer], next_input[:longer])
                )
                similarity = common / longer
                if similarity < 0.8:  # < 80% positional overlap → treat as corrective
                    recoveries += 1

    return round(recoveries / len(error_indices), 4) if error_indices else None


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def load_scenarios(yaml_path: Path, limit: Optional[int] = None) -> list[dict]:
    with yaml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    scenarios = raw if isinstance(raw, list) else raw.get("scenarios", [])
    if limit:
        scenarios = scenarios[:limit]
    logger.info("[runner] Loaded %d TrackB scenarios", len(scenarios))
    return scenarios


# ---------------------------------------------------------------------------
# Build one result dict (SREGym metadata standard)
# ---------------------------------------------------------------------------

def build_result_dict(
    scenario: dict,
    agent_run: AgentRunResult,
    verification: Optional[VerificationResult],
    dry_run: bool,
) -> dict:
    n_required = len(scenario.get("required_steps", []))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if dry_run:
        return {
            "scenario_id": scenario.get("scenario_id"),
            "platform": scenario.get("platform"),
            "provider": agent_run.provider,
            "model": agent_run.model,
            "task_completion_rate": 1,
            "step_efficiency": 1.25,
            "error_recovery_rate": 0.5,
            "per_step_verdicts": [{"step": i, "verdict": True, "seed_verdicts": {42: True, 100: True, 2026: True}}
                                   for i in range(n_required)],
            "early_exit_triggered": False,
            "transcript": [],
            "actual_actions_taken": n_required + 1,
            "temperature": AGENT_TEMPERATURE,
            "seed": AGENT_SEED,
            "timestamp_utc": ts,
            "error": None,
            "dry_run": True,
        }

    tcr = compute_task_completion_rate(verification, agent_run)
    eff = compute_step_efficiency(agent_run, n_required)
    err_rec = compute_error_recovery_rate(agent_run.to_transcript_dicts())

    per_step_verdicts = []
    if verification:
        for sv in verification.step_verdicts:
            per_step_verdicts.append({
                "step_index": sv.step_index,
                "required_step": sv.required_step,
                "success_criterion": sv.success_criterion,
                "final_verdict": sv.final_verdict,
                "seed_verdicts": sv.seed_verdicts,
                "reasoning_per_seed": sv.reasoning_per_seed,
            })

    errors = []
    if agent_run.error:
        errors.append(f"agent: {agent_run.error}")
    if verification and verification.error:
        errors.append(f"verifier: {verification.error}")

    return {
        "scenario_id": scenario.get("scenario_id"),
        "platform": scenario.get("platform"),
        "provider": agent_run.provider,
        "model": agent_run.model,
        "task_completion_rate": tcr,
        "step_efficiency": eff,
        "error_recovery_rate": err_rec,
        "per_step_verdicts": per_step_verdicts,
        "early_exit_triggered": (
            agent_run.early_exit_triggered
            or (verification.early_exit_triggered if verification else False)
        ),
        "transcript": agent_run.to_transcript_dicts(),
        "actual_actions_taken": agent_run.actual_actions_taken,
        "n_required_steps": n_required,
        "temperature": AGENT_TEMPERATURE,
        "seed": AGENT_SEED,
        "timestamp_utc": ts,
        "error": "; ".join(errors) if errors else None,
        "dry_run": False,
    }


# ---------------------------------------------------------------------------
# Re-verification (re-run the judge against saved transcripts, no agent re-run)
# ---------------------------------------------------------------------------

async def reverify(results_path: Path) -> None:
    """
    Re-run step verification against already-saved transcripts, without
    re-running the (expensive, slow) agent loop.

    Use this after fixing a bug in the step_verifier judge prompt — e.g. the
    fix requiring the judge to find Action+Observation evidence for a step
    rather than trusting the agent's self-reported Final Answer narrative.
    A manual audit of TrackB_Fari_02/anthropic found the judge marking all
    4 required steps True off a single check_log call plus a confident but
    unsubstantiated Final Answer — the step verifier was grading the Final
    Answer's prose, not the transcript's actual evidence.

    Only task_completion_rate, per_step_verdicts, and early_exit_triggered
    are recomputed. agent-loop-derived fields (transcript,
    actual_actions_taken, step_efficiency, error_recovery_rate) are
    untouched, since re-verification doesn't change what the agent did.
    """
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY not found in .env")

    with results_path.open("r", encoding="utf-8") as fh:
        output = json.load(fh)

    scenarios = {s["scenario_id"]: s for s in load_scenarios(SCENARIO_YAML)}
    verifier = StepVerifier(openai_api_key=openai_key)

    changed = 0
    for row in output["results"]:
        scenario = scenarios.get(row["scenario_id"])
        if scenario is None:
            logger.warning("[reverify] Unknown scenario_id %s — skipping", row["scenario_id"])
            continue

        old_tcr = row.get("task_completion_rate")
        verification = await verifier.verify(
            scenario=scenario,
            transcript_dicts=row.get("transcript", []),
        )

        agent_run_stub = AgentRunResult(
            scenario_id=row["scenario_id"],
            provider=row["provider"],
            model=row["model"],
            actual_actions_taken=row.get("actual_actions_taken", 0),
            early_exit_triggered=False,  # agent_loop no longer sets this in-loop
        )
        new_tcr = compute_task_completion_rate(verification, agent_run_stub)

        row["task_completion_rate"] = new_tcr
        row["per_step_verdicts"] = [
            {
                "step_index": sv.step_index,
                "required_step": sv.required_step,
                "success_criterion": sv.success_criterion,
                "final_verdict": sv.final_verdict,
                "seed_verdicts": sv.seed_verdicts,
                "reasoning_per_seed": sv.reasoning_per_seed,
            }
            for sv in verification.step_verdicts
        ]
        row["early_exit_triggered"] = verification.early_exit_triggered

        if new_tcr != old_tcr:
            changed += 1
            logger.info("[reverify] %s | %s | TCR %s -> %s",
                        row["scenario_id"], row["provider"], old_tcr, new_tcr)

    output["metadata"]["reverified_at"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output["metadata"]["reverify_note"] = (
        "Step verifier prompt patched to require Action+Observation evidence for a step "
        "rather than trusting the agent's self-reported Final Answer narrative. "
        "task_completion_rate / per_step_verdicts / early_exit_triggered were recomputed "
        "against the original saved transcripts; agent-loop fields (transcript, "
        "actual_actions_taken, step_efficiency, error_recovery_rate) are unchanged."
    )

    with results_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    logger.info("[reverify] Done. %d/%d rows changed task_completion_rate.",
                changed, len(output["results"]))


# ---------------------------------------------------------------------------
# Main async runner
# ---------------------------------------------------------------------------

async def run(dry_run: bool, limit: Optional[int]) -> None:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    openai_key    = os.environ.get("OPENAI_API_KEY", "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    deepseek_key  = os.environ.get("DEEPSEEK_API_KEY", "")

    if not dry_run:
        if not anthropic_key:
            raise SystemExit("ANTHROPIC_API_KEY not found in .env")
        if not deepseek_key:
            raise SystemExit("DEEPSEEK_API_KEY not found in .env")
        if not openai_key:
            raise SystemExit("OPENAI_API_KEY not found in .env (needed for step verifier)")

    scenarios = load_scenarios(SCENARIO_YAML, limit=limit)

    # --- Provider configs: (agent_client, env_client) pairs ---
    providers = []
    if not dry_run:
        anthropic_agent = AnthropicClient(
            api_key=anthropic_key, model=AGENT_MODEL_ANTHROPIC,
            temperature=AGENT_TEMPERATURE, seed=AGENT_SEED,
        )
        anthropic_env = AnthropicClient(
            api_key=anthropic_key, model=ENV_MODEL_ANTHROPIC,
            temperature=0.3, seed=99,  # slightly stochastic env for realism
        )
        deepseek_agent = DeepSeekClient(
            api_key=deepseek_key, model=AGENT_MODEL_DEEPSEEK,
            temperature=AGENT_TEMPERATURE, seed=AGENT_SEED,
        )
        deepseek_env = DeepSeekClient(
            api_key=deepseek_key, model=ENV_MODEL_DEEPSEEK,
            temperature=0.3, seed=99,
        )
        providers = [
            ("anthropic", anthropic_agent, anthropic_env),
            ("deepseek", deepseek_agent, deepseek_env),
        ]
        verifier = StepVerifier(openai_api_key=openai_key)
    else:
        # Dry-run: create placeholder clients (not used)
        from week01_benchmark.eval_harness.clients import OpenAIClient
        dummy = OpenAIClient(api_key="dry_run", model="gpt-4o",
                             temperature=0.0, seed=42)
        providers = [("anthropic", dummy, dummy), ("deepseek", dummy, dummy)]
        verifier = None

    all_results = []
    total = len(scenarios) * len(providers)
    completed = 0

    for provider_name, agent_client, env_client in providers:
        loop = ReActAgentLoop(agent_client=agent_client, env_client=env_client)
        logger.info("=== Provider: %s (%d scenarios) ===", provider_name, len(scenarios))

        for scenario in scenarios:
            scenario_id = scenario.get("scenario_id", "?")
            n_steps = len(scenario.get("required_steps", []))
            logger.info(
                "[runner] %s | %s | turn_depth=%d steps=%d",
                scenario_id, provider_name,
                scenario.get("turn_depth", 3), n_steps,
            )

            if dry_run:
                # Stub result
                from .agent_loop import AgentRunResult
                stub_run = AgentRunResult(
                    scenario_id=scenario_id,
                    provider=provider_name,
                    model="dry_run",
                    actual_actions_taken=n_steps + 1,
                )
                result_dict = build_result_dict(scenario, stub_run, None, dry_run=True)
            else:
                # 1. Run agent loop
                agent_run = await loop.run(scenario)
                logger.info(
                    "[runner] %s | %s | actions=%d early_exit=%s error=%s",
                    scenario_id, provider_name,
                    agent_run.actual_actions_taken,
                    agent_run.early_exit_triggered,
                    agent_run.error,
                )

                # 2. Verify steps
                try:
                    verification = await verifier.verify(
                        scenario=scenario,
                        transcript_dicts=agent_run.to_transcript_dicts(),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("[runner] %s | verification error: %s", scenario_id, exc)
                    verification = None

                result_dict = build_result_dict(scenario, agent_run, verification, dry_run=False)

            all_results.append(result_dict)
            completed += 1
            logger.info("[runner] Progress: %d/%d | TCR=%s eff=%s",
                        completed, total,
                        result_dict.get("task_completion_rate"),
                        result_dict.get("step_efficiency"))

    # --- Close shared aiohttp sessions (DeepSeek clients) ---
    for _, agent_client, env_client in providers:
        for client in (agent_client, env_client):
            aclose = getattr(client, "aclose", None)
            if aclose is not None:
                await aclose()

    # --- Save ---
    output = {
        "metadata": {
            "evaluation": "W02_Agentic_Eval",
            "total_results": len(all_results),
            "providers": ["anthropic", "deepseek"],
            "agent_models": {
                "anthropic": AGENT_MODEL_ANTHROPIC,
                "deepseek": AGENT_MODEL_DEEPSEEK,
            },
            "temperature": AGENT_TEMPERATURE,
            "seed": AGENT_SEED,
            "judge_model": "gpt-4o",
            "judge_seeds": [42, 100, 2026],
            "dry_run": dry_run,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        },
        "results": all_results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    errors = [r for r in all_results if r.get("error")]
    logger.info(
        "Done. %d results, %d errors → %s",
        len(all_results), len(errors),
        OUTPUT_PATH.relative_to(REPO_ROOT),
    )
    if errors:
        for r in errors:
            logger.warning("  %s | %s | %s", r["scenario_id"], r["provider"], r["error"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic evaluation runner")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip all API calls; write placeholder metrics.")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit to first N scenarios (for smoke testing).")
    p.add_argument("--reverify", action="store_true",
                   help="Re-run step verification against the saved transcripts in "
                        "OUTPUT_PATH without re-running the agent loop. Use after a "
                        "step_verifier prompt fix.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.reverify:
        asyncio.run(reverify(OUTPUT_PATH))
    else:
        asyncio.run(run(dry_run=args.dry_run, limit=args.limit))
