"""
Step Verifier — 3-seed Judge
==============================

After the agent loop completes, verifies each required step against
the transcript using the same 3-seed judge pattern as llm_judge.py
(seeds 42, 100, 2026) so Krippendorff's alpha can be computed on
step-verification agreement.

Per step:
  - Ask GPT-4o (3 seeds) whether the step's success criterion was met
    by the transcript → binary verdict per seed
  - Majority vote (≥ 2/3) → final per-step verdict
  - All three raw seed verdicts are stored for α computation downstream

Also checks whether any early_exit_failure_conditions appear in the
transcript, returning a binary early_exit_triggered flag.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import openai

logger = logging.getLogger(__name__)

# Reuse the same judge config as llm_judge.py
JUDGE_MODEL = "gpt-4o"
JUDGE_SEEDS = [42, 100, 2026]
JUDGE_TEMPERATURE = 0.0
JUDGE_TIMEOUT_S = 60.0
JUDGE_MAX_RETRIES = 5
JUDGE_BACKOFF_BASE_S = 8.0

_SEED_SYSTEM_PROMPTS = {
    42: (
        "You are a rigorous AI evaluation judge assessing whether a robot agent "
        "completed a specific required step. Return ONLY a valid JSON object — "
        "no prose, no markdown."
    ),
    100: (
        "You are an expert in physical AI deployment evaluation. Assess strictly "
        "whether the step success criterion was demonstrably met in the transcript. "
        "Return ONLY a valid JSON object — no prose, no markdown."
    ),
    2026: (
        "You are a senior ML evaluation engineer judging agent task completion. "
        "Be conservative: only mark True if the transcript clearly satisfies the "
        "success criterion. Return ONLY a valid JSON object — no prose, no markdown."
    ),
}

_STEP_JUDGE_PROMPT = """You are verifying whether a robot agent's action transcript satisfies a specific step's success criterion.

SCENARIO TASK:
{task}

REQUIRED STEP:
{required_step}

SUCCESS CRITERION FOR THIS STEP:
{success_criterion}

AGENT TRANSCRIPT (Thought/Action/Observation sequence):
{transcript}

QUESTION: Was the success criterion for this step met by the agent's actions in the transcript?

Return ONLY a JSON object with these exact fields:
{{
  "verdict": true or false,
  "reasoning": "<one sentence explaining your decision>"
}}"""

_EARLY_EXIT_CHECK_PROMPT = """You are checking whether an agent triggered any early-exit failure conditions during a task.

TASK:
{task}

EARLY-EXIT FAILURE CONDITIONS (any of these would be an automatic fail):
{early_exit}

AGENT TRANSCRIPT:
{transcript}

Did the agent trigger any of these early-exit failure conditions?

Return ONLY a JSON object:
{{
  "early_exit_triggered": true or false,
  "reasoning": "<one sentence>"
}}"""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class StepVerdict:
    step_index: int
    required_step: str
    success_criterion: str
    seed_verdicts: dict[int, Optional[bool]]   # {seed: True/False/None}
    final_verdict: Optional[bool]               # majority vote
    reasoning_per_seed: dict[int, str] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class VerificationResult:
    scenario_id: str
    step_verdicts: list[StepVerdict] = field(default_factory=list)
    early_exit_triggered: bool = False
    early_exit_reasoning: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Judge call with retry (same pattern as llm_judge.py and rag_scorer.py)
# ---------------------------------------------------------------------------

async def _call_judge(
    client: "openai.AsyncOpenAI",
    system: str,
    prompt: str,
) -> dict:
    last_error = None
    for attempt in range(1, JUDGE_MAX_RETRIES + 1):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=JUDGE_MODEL,
                    temperature=JUDGE_TEMPERATURE,
                    seed=42,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                ),
                timeout=JUDGE_TIMEOUT_S,
            )
            raw = resp.choices[0].message.content or ""
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            return json.loads(cleaned)
        except openai.RateLimitError as exc:
            wait = JUDGE_BACKOFF_BASE_S * (2 ** (attempt - 1))
            logger.warning(
                "[step_verifier] Judge TPM limit (attempt %d/%d) — backing off %.0fs",
                attempt, JUDGE_MAX_RETRIES, wait,
            )
            last_error = exc
            if attempt < JUDGE_MAX_RETRIES:
                await asyncio.sleep(wait)
        except asyncio.TimeoutError as exc:
            last_error = exc
            logger.warning("[step_verifier] Judge timeout attempt %d", attempt)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.error("[step_verifier] Judge error: %s", exc)
            break
    raise RuntimeError(
        f"Judge failed after {JUDGE_MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def _format_transcript(transcript_dicts: list[dict]) -> str:
    """Convert transcript to a compact readable string for the judge prompt."""
    lines = []
    for entry in transcript_dicts:
        role = entry.get("role", "unknown")
        content = entry.get("content", "")
        if role == "thought":
            lines.append(f"Thought: {content}")
        elif role == "action":
            lines.append(f"Action: {entry.get('tool', '')}({entry.get('tool_input', '')})")
        elif role == "observation":
            prefix = "ERROR Observation" if entry.get("is_error") else "Observation"
            lines.append(f"{prefix}: {content}")
        elif role == "final_answer":
            lines.append(f"Final Answer: {content}")
    return "\n".join(lines) or "(empty transcript)"


# ---------------------------------------------------------------------------
# Step Verifier
# ---------------------------------------------------------------------------

class StepVerifier:
    """
    Verifies each required step against an agent transcript using
    GPT-4o with 3 system-prompt seeds.

    Majority vote (≥ 2 of 3 seeds agreeing) → final per-step verdict.
    """

    def __init__(self, openai_api_key: str) -> None:
        self._client = openai.AsyncOpenAI(api_key=openai_api_key)

    async def verify(
        self,
        scenario: dict,
        transcript_dicts: list[dict],
    ) -> VerificationResult:
        """
        Run full verification for one agent run.

        Parameters
        ----------
        scenario : dict from trackB_agentic.yaml
        transcript_dicts : agent_loop.AgentRunResult.to_transcript_dicts()
        """
        scenario_id = scenario.get("scenario_id", "unknown")
        task = scenario.get("initial_task_prompt", "")
        required_steps: list[str] = scenario.get("required_steps", [])
        criteria: list[str] = scenario.get("success_criteria_per_step", [])
        early_exit: str = scenario.get("early_exit_failure_conditions", "")

        transcript_str = _format_transcript(transcript_dicts)
        result = VerificationResult(scenario_id=scenario_id)

        # --- Verify each step (concurrent across seeds, sequential across steps) ---
        for i, (step, criterion) in enumerate(zip(required_steps, criteria)):
            sv = await self._verify_step(
                task=task,
                step_index=i,
                required_step=step,
                criterion=criterion,
                transcript_str=transcript_str,
            )
            result.step_verdicts.append(sv)
            logger.info(
                "[step_verifier] %s | step %d | verdict=%s (seeds: %s)",
                scenario_id, i, sv.final_verdict,
                {s: v for s, v in sv.seed_verdicts.items()},
            )

        # --- Early-exit check ---
        if early_exit and transcript_dicts:
            try:
                data = await _call_judge(
                    self._client,
                    _SEED_SYSTEM_PROMPTS[42],
                    _EARLY_EXIT_CHECK_PROMPT.format(
                        task=task,
                        early_exit=early_exit,
                        transcript=transcript_str,
                    ),
                )
                result.early_exit_triggered = bool(data.get("early_exit_triggered", False))
                result.early_exit_reasoning = data.get("reasoning", "")
            except Exception as exc:  # noqa: BLE001
                result.error = f"Early-exit check failed: {exc}"
                logger.error("[step_verifier] %s | early-exit check error: %s",
                             scenario_id, exc)

        return result

    async def _verify_step(
        self,
        task: str,
        step_index: int,
        required_step: str,
        criterion: str,
        transcript_str: str,
    ) -> StepVerdict:
        """Run one step through all 3 judge seeds concurrently."""
        prompt = _STEP_JUDGE_PROMPT.format(
            task=task,
            required_step=required_step,
            success_criterion=criterion,
            transcript=transcript_str,
        )

        async def judge_with_seed(seed: int) -> tuple[int, Optional[bool], str]:
            try:
                data = await _call_judge(
                    self._client,
                    _SEED_SYSTEM_PROMPTS[seed],
                    # Override seed in the client call via a slightly varied prompt
                    prompt + f"\n<!-- evaluation_seed={seed} -->",
                )
                verdict = bool(data.get("verdict", False))
                reasoning = data.get("reasoning", "")
                return seed, verdict, reasoning
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "[step_verifier] step %d seed=%d error: %s", step_index, seed, exc
                )
                return seed, None, str(exc)

        tasks = [judge_with_seed(s) for s in JUDGE_SEEDS]
        raw_results = await asyncio.gather(*tasks)

        seed_verdicts: dict[int, Optional[bool]] = {}
        reasoning_per_seed: dict[int, str] = {}
        errors = []

        for seed, verdict, reasoning in raw_results:
            seed_verdicts[seed] = verdict
            reasoning_per_seed[seed] = reasoning
            if verdict is None:
                errors.append(f"seed={seed}: {reasoning}")

        # Majority vote
        valid_verdicts = [v for v in seed_verdicts.values() if v is not None]
        if len(valid_verdicts) >= 2:
            final_verdict = sum(valid_verdicts) >= 2  # True if ≥ 2/3 agree True
        elif len(valid_verdicts) == 1:
            final_verdict = valid_verdicts[0]
        else:
            final_verdict = None  # all three seeds failed

        return StepVerdict(
            step_index=step_index,
            required_step=required_step,
            success_criterion=criterion,
            seed_verdicts=seed_verdicts,
            final_verdict=final_verdict,
            reasoning_per_seed=reasoning_per_seed,
            error="; ".join(errors) if errors else None,
        )
