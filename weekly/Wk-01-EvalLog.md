# Week 1 Evaluation Log

**Period**: Week 1 (Physical AI Landscape, Benchmark Design & Harness Engineering)
**Author**: George Wang

## Evaluated

Designed and built the 40-scenario benchmark bank across InGen's five platforms (Fari, Senpai,
Sentinel Prime AI, Aido Rover, Aido Humanoid) — 20 conversational scenarios (Track A) and 20
agentic multi-step scenarios (Track B, all turn_depth ≥ 3). Built the Python evaluation harness
(`eval_harness/`) with a two-tier architecture — orchestration (`dispatcher.py`) and execution
(`clients.py`) — dispatching to all four providers (OpenAI GPT-4o, Anthropic Claude, DeepSeek,
Llama-3.1-8B via Groq) concurrently, plus a 3-seed LLM-judge module (`llm_judge.py`) for
downstream Krippendorff's alpha. Ran the harness as a 5-scenario smoke test across all four
providers to confirm connectivity and baseline latency before committing to a full run.

## Found

Platform-coverage validation on the Track A YAML caught a real gap before it reached the first
full evaluation run: the initial draft had only 16 scenarios and omitted Fari entirely — the
one platform in this bank where failure severity is highest (medication and emergency-response
scenarios). Corrected by authoring four Fari scenarios and re-validating platform coverage
programmatically (4 scenarios × 5 platforms, both tracks) rather than by inspection, since
inspection is exactly how the gap was missed the first time.

## Mechanism

The design choice that most directly reflects my SREGym experience is Track B's
`required_steps` / `success_criteria_per_step` pairing — parallel, equal-length lists giving
one independently verifiable success condition per step, rather than grading on final-state
outcome alone. This is the same principle SREGym used for the Kubernetes-remediation agent
benchmark: a multi-step remediation can reach a healthy final cluster state through the wrong
mechanism (masked fault, unrelated concurrent fix), so grading only the end state overstates
agent competence. Requiring `early_exit_failure_conditions` on every Track B scenario follows
the same logic — it lets the harness distinguish "task incomplete" from "agent took a
plausible-looking wrong action," which SREGym treats as separate failure modes with different
operational implications.

**Next week**: full 4-provider run across all 40 scenarios, RAG pipeline evaluation, and the
first real multi-step agentic execution (Week 1's Track B design supports this; it has not yet
been run).
