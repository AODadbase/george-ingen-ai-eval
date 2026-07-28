# Beyond the Aggregate Score: Platform-Specific and Verification-Sensitive Evaluation for Physical AI Deployment

**George (ZhiNeng) Wang**
InGen AI Model Evaluation Internship — Draft for AI Evaluation Workshop submission

## Abstract

Standard LLM benchmarks report a single aggregate score per model, and standard agentic
benchmarks grade task completion from a model's own account of what it did. We show both
practices fail for physical AI deployment evaluation. Running four LLM providers across a
40-scenario benchmark spanning five InGen Dynamics robot platforms, we find that the
aggregate-leaderboard leader (Anthropic, severity-weighted score 0.986/1.0) is unusable on the
platform with the tightest real-time constraint — 10,963 ms mean latency against a 2,000 ms
operational threshold — while a lower-ranked provider (Groq) is the only one to clear both the
latency and quality bar on that platform. Separately, in building a multi-step agentic
evaluation harness, we find that grading task completion from a model's narrated summary rather
than from verified tool-action evidence overstates completion rates by a factor of roughly
2–3x (70%→25% and 30%→15% after a verification fix): a plausible claim of success is not
evidence of success. We argue physical AI evaluation requires platform-decomposed reporting and
evidence-gated agentic verification as first-class requirements, not optional refinements to a
single-number benchmark.

## 1. Introduction

Physical AI systems — robots that perceive, plan, and act in eldercare, education, security,
and outdoor/humanoid contexts — are increasingly built on the same class of large language
models that power conversational assistants and software agents. The evaluation methodologies
built for those adjacent domains, however, assume properties physical deployment does not have:
a single-turn exchange rather than a multi-step task, a wrong output that costs an incorrect
answer rather than a delayed or unsafe physical action, and — most consequentially for this
paper — an environment where the model's own account of what happened is not independently
checkable, unlike a physical robot's action log.

We state the hypothesis this paper tests directly: **a single aggregate quality score, and a
task-completion metric graded from a model's self-reported summary, both systematically
misrepresent a model's deployment readiness for physical AI platforms**, in ways that only
become visible once results are decomposed by platform and verification is required to be
evidence-based rather than narrative-based. We test this by building a four-provider,
five-platform, two-track (conversational + agentic) evaluation harness for InGen Dynamics'
robot fleet and reporting results both in aggregate and decomposed form.

## 2. Related Work

**AgentBench** (Liu et al., 2024) evaluates 29 models as agents across 8 digital environments
(operating systems, databases, knowledge graphs, digital card games, lateral-thinking puzzles,
household tasks, web shopping, and web browsing), and its cross-environment design is what
surfaces its central finding — that poor long-horizon reasoning, not raw scale, drives failure.
Every one of its 8 environments is purely digital, with no sensor noise, no actuator
uncertainty, and no irreversible consequence for a wrong action. Our work differs by evaluating
against InGen's five physical platforms, where an incorrect action (e.g., pushing through a
physical obstacle instead of rerouting) has a hardware or safety consequence AgentBench's domain
cannot represent.

**RAGAS** (Es et al., 2024) introduces reference-free faithfulness, answer-relevance, and
context-relevance metrics for RAG evaluation, validated on general-domain question answering.
We apply RAGAS-style metrics to two safety-adjacent domains (Fari eldercare, Senpai education)
and find a result RAGAS's original validation setting could not have surfaced: a persona-vector
query augmentation intended to improve retrieval relevance instead *reduces* answer relevance by
0.113 on Fari, entirely because the domain descriptor is broad enough to pull narrow queries
toward generic eldercare topics.

**HELM** (Liang et al., 2023) evaluates 30 models on a standardized grid of 16 core scenarios
and 7 metrics (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency), forcing
every model through an identical grid so trade-offs are visible rather than collapsed into one
score — the design principle our platform-decomposed leaderboard borrows most directly. HELM's
efficiency metric is abstract compute/latency cost, with no mechanism for a hard real-time
threshold; we extend this idea with an explicit pass/fail latency-threshold analysis (Section 5)
that HELM's framework does not support.

**SafeAgentBench** (Yin et al., 2024) treats task success and hazard avoidance as two
independent scoring axes rather than one composite score, and finds that models differing
substantially in task completion show almost no corresponding difference in safety awareness —
directly analogous to our finding (Section 5) that the aggregate-leaderboard leader is not the
safety-violation leader. SafeAgentBench's environment remains a simulated household-task
simulator with no real actuator or sensor hardware; our benchmark is anchored to five named,
InGen-specific physical platforms with platform-specific operational-implication fields per
scenario, though we share SafeAgentBench's limitation of running in a text-only, non-physical
harness rather than against real hardware.

**SREGym** (Clark et al., 2026) is a live benchmark for AI site-reliability-engineering agents
using real fault injection — direct system-layer manipulation rather than scripted or
symptom-level perturbation — producing high-fidelity, non-gameable failure scenarios. This
paper's benchmark design borrows SREGym's methodology directly: a minimum turn depth, an
explicit success criterion at each step, and early-exit conditions on unrecoverable failure
(detailed in `W01_Benchmark_Design.md`). SREGym's faults are entirely digital (service failures,
resource exhaustion, network partitions), so its severity classification does not transfer
directly to a scenario where failure is a physical, irreversible-consequence action — the domain
gap this paper's Track B design is built to close. Note: the author has prior experience
extending SREGym's methodology to a follow-on 80+-scenario Kubernetes-benchmark project, but is
not an author of the original SREGym paper.

**InGen's PIC 2.0 platform paper** (Hisham, 2026) reports strong self-reported figures on
exactly the dimensions this paper independently tests — task completion, safety, calibration —
with no baseline comparison methodology, confidence intervals, or third-party audit for any
headline figure. We treat every PIC 2.0 claim as an assertion to be tested rather than a
baseline, consistent with the standard we apply to our own findings (Section 6).

**What this work adds**: no reviewed methodology combines physical embodiment, platform-specific
(not aggregate) reporting, multi-provider comparison, and evidence-gated (not narrative-graded)
agentic verification in one evaluation framework. This gap is what Sections 4–5 address.

## 3. Methodology

**Benchmark design.** 40 scenarios across InGen's five platforms (Fari, Senpai, Sentinel Prime
AI, Aido Rover, Aido Humanoid), split into Track A (20 conversational, 4 per platform,
severity-classed 1–5 with explicit rubric anchors) and Track B (20 agentic multi-step, 4 per
platform, `turn_depth >= 3`, each with parallel `required_steps` / `success_criteria_per_step`
lists and an `early_exit_failure_conditions` field). Full design rationale, including the
SREGym-derived step-verification design, is in `W01_Benchmark_Design.md`.

**Harness architecture.** A two-tier design mirroring SREGym: an orchestration layer
(`dispatcher.py`) managing the scenario × provider matrix and concurrency, and an execution
layer (`clients.py`) handling per-provider API calls with retry/backoff. Four providers were
evaluated: OpenAI (gpt-4o), Anthropic (claude-sonnet-4-6), DeepSeek (deepseek-chat), and
Llama-3.1-8B via Groq, all at temperature=0.0 with fixed seeds for reproducibility. Every result
records provider, model, evaluation set, temperature, and seed.

**Four-provider evaluation.** All 40 scenarios × 4 providers were LLM-judged (GPT-4o, 3 system-
prompt seeds per response for Krippendorff's alpha) on task accuracy and contextual grounding
(1–5 scale), yielding a severity-weighted aggregate score per provider
(`Σ(judge_mean_accuracy × severity_class)`, normalized against the maximum possible score on
scored rows) and a per-platform decomposition of the same score.

**RAG evaluation.** An IGuide-inspired pipeline (append-only document log, composite TF-IDF +
semantic-similarity ranking, optional persona-vector query augmentation) evaluated on Fari and
Senpai's conversational subset using three RAGAS-style LLM-judged metrics (faithfulness, answer
relevance, context coverage), under a controlled ablation isolating the persona-vector variable
— every other parameter (chunk size, top-k, embedding model, judge configuration) held constant
and verified programmatically before any result was reported.

**Agentic evaluation.** A ReAct-style Thought/Action/Observation loop against the two top-ranked
providers from the four-provider comparison (Anthropic, DeepSeek), with a scripted LLM
environment simulator generating observations from each scenario's success criteria. Three
metrics: task completion rate (all required steps verified, no early-exit trigger), step
efficiency (`actual_actions / n_required_steps`), and error recovery rate. Verification method
is itself a methodological finding of this paper (Section 5.3): the step verifier was initially
grading a model's narrated Final Answer for claimed completion, which credited steps with no
supporting tool action in the transcript; it was patched to require Action+Observation evidence
per step and every result re-verified against the original transcripts.

## 4. Initial Results

### 4.1 Four-provider leaderboard is not one number

| Rank | Provider | Severity-weighted score (norm.) | Krippendorff's α |
|---|---|---|---|
| 1 | Anthropic | 0.986 | 0.496 |
| 2 | DeepSeek | 0.919 | 0.760 |
| 3 | Groq | 0.894 | 0.567 |
| 4 | OpenAI | 0.784 | 0.852 |

The #1-ranked provider's inter-rater reliability (0.496) is below the conventional
0.667 minimum-acceptable threshold; the #4-ranked provider's (0.852) is the most reliable of
all four. Aggregate rank and measurement confidence are not correlated. Per-platform
decomposition shows OpenAI's low aggregate rank is driven almost entirely by one platform (Aido
Humanoid, 3.88/5 mean accuracy vs. 5.0/5 on both Fari and Senpai) rather than uniform weakness.

### 4.2 Latency × quality: the leaderboard leader fails the tightest real-time constraint

On Aido Rover — the platform with the strictest real-time requirement in this bank — Anthropic
reaches perfect quality (severity-weighted score 1.0) but a mean latency of 10,963 ms, against a
2,000 ms operational threshold derived from HRI/voice-assistant interactive-response norms.
Only Groq (391.6 ms, 0.908 quality) clears both the latency threshold and a 0.85 quality floor
on this platform; OpenAI clears latency but not quality; DeepSeek and Anthropic clear quality but
not latency.

### 4.3 RAG: persona-vector augmentation reduces answer relevance

| Condition | Faithfulness | Answer Relevance | Context Coverage |
|---|---|---|---|
| persona-vector OFF | 1.000 | 0.925 | 0.962 |
| persona-vector ON | 1.000 | 0.812 | 0.962 |

Contrary to the design intuition that domain-specific query augmentation improves retrieval
relevance, answer relevance drops 0.113 with the layer enabled — entirely on Fari (0.850 →
0.625; Senpai unaffected). The likely mechanism: Fari's persona descriptor is broad enough
(medication, routines, emergency response, nutrition) to pull narrow-question embeddings toward
generic eldercare topics.

A follow-up 12-configuration ablation (chunk size × top-k × reranking, Senpai subset) found the
faithfulness-optimal, lowest-latency configuration (`top_k=1`) cuts latency 20.6% versus the
production-equivalent default (`top_k=3`) at equal faithfulness (1.0) — but at the cost of
answer relevance (1.00 → 0.75) and context coverage (0.85 → 0.75), a genuine trade-off rather
than a clean win. Cross-encoder reranking never improved faithfulness at any chunk size tested
(1.0 → 0.925 uniformly), on a corpus small enough that the base composite ranking was already
near-ceiling.

### 4.4 Agentic evaluation: verification design changes the result by 2–3x

| Provider | Completion rate (pre-fix) | Completion rate (post-fix) |
|---|---|---|
| Anthropic | 70% (14/20) | 25% (5/20) |
| DeepSeek | 30% (6/20) | 15% (3/20) |

The pre-fix verifier credited a required step whenever the agent's Final Answer *claimed*
completion, regardless of whether a corresponding tool action existed in the transcript. In a
representative case, an agent took a single `check_log` action, then wrote a four-part narrative
claiming full resolution of a 4-step task; three of the four steps had no supporting action and
were nonetheless credited. Both providers fail 100% of Aido Humanoid scenarios (0/8 combined)
post-fix — the platform requiring the most compound, multi-step manipulation planning.

## 5. Discussion and Limitations (to be expanded in Week 4)

This 4-week program's dataset is not large enough to support strong statistical claims on every
dimension reported: the SEOM safety-violation rates (Section 4, PIC 2.0 analysis) rest on 1–2
events per provider out of 40 rows, too few to distinguish a true 5% rate from a 2.5%–7.5% range
with confidence. The Krippendorff's alpha values, similarly, are computed over a 3-seed,
40-scenario design per provider — adequate to detect the qualitative pattern reported
(leaderboard rank and reliability are uncorrelated) but not to bound it precisely.

The results in Section 4.2 and 4.4 both support the paper's stated hypothesis: aggregate
scores hid a platform-specific latency failure that only a decomposed report surfaced, and a
narrative-graded completion metric overstated capability by 2–3x until evidence-gating was
enforced. Section 4.3's RAG finding is a secondary confirmation of the same theme at the
component level — a design intuition (persona augmentation helps) that only a controlled,
decomposed measurement (per-platform delta, not an aggregate RAG score) revealed to be false for
one of two platforms tested.

**What this program's 4-week scope could not test**: no direct evaluation of PIC 2.0's own
modules (operating constraints prohibit InGen internal system access — every result here is a
proxy from four market LLMs); no multi-modal evaluation (relevant to the AMDC model class, which
this program's text-only harness cannot approximate at all — the largest evidence gap identified
in `W03_PIC20_Analysis.md`); no multi-agent coordination evaluation (relevant to CRL-MRS,
explicitly out of scope for a single-agent harness). These are named directly rather than left
implicit, consistent with the standard this paper applies to InGen's own PIC 2.0 platform paper
in Section 2.

**Target venue and submission readiness**: this draft targets an AI evaluation workshop
(e.g., a workshop co-located with a systems or agents-focused venue) rather than a full
conference track — the 4-week dataset supports the methodological contribution (platform-
decomposed reporting, evidence-gated agentic verification) but not the scale of evidence a full
track would expect. The result most needing strengthening before submission is Section 4.4: a
single before/after verification comparison on one harness is suggestive but not yet a general
claim about narrative-grading failure modes in agentic evaluation broadly; replicating the
finding on a second, independently designed agentic benchmark would substantially strengthen the
paper's central claim.
