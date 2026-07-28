# W02 Evaluation Memo

**InGen AI Model Evaluation — Week 2**
**George Wang**

This memo summarizes findings from three Week 2 evaluations: the four-provider severity-weighted
comparison across all 40 benchmark scenarios, the production RAG evaluation on Fari and Senpai,
and the multi-step agentic evaluation on Track B. All figures below are computed from the full,
100%-coverage judged dataset (160 scored rows across four providers) and the full 40-row agentic
run (20 scenarios × 2 providers); none are drawn from partial or dry-run data.

## 1. Four-Provider Comparison

| Rank | Provider | Severity-Weighted Score (norm.) | Mean Task Accuracy | Krippendorff's α |
|---|---|---|---|---|
| 1 | Anthropic (claude-sonnet-4-6) | 0.986 | 4.93 / 5 | 0.496 |
| 2 | DeepSeek (deepseek-chat) | 0.919 | 4.78 / 5 | 0.760 |
| 3 | Groq (llama-3.1-8b-instant) | 0.894 | 4.54 / 5 | 0.567 |
| 4 | OpenAI (gpt-4o) | 0.784 | 4.47 / 5 | 0.852 |

**Most important finding**: the ranking is not uniform across platforms. OpenAI's aggregate
score (4th of 4) is dragged down almost entirely by one platform — Aido Humanoid, where its
mean task accuracy is 3.88/5, versus a perfect 5.0/5 on both Fari and Senpai. A provider
selection based on the aggregate leaderboard alone would miss that OpenAI is competitive to
strong on three of five platforms and specifically weak on compound, multi-step-manipulation
contexts (Aido Humanoid) — this is a platform-specific weakness, not a uniform quality gap, and
should inform per-platform provider routing rather than a single fleet-wide model choice.

A second finding worth flagging rather than hiding: Anthropic and Groq's Krippendorff's alpha
(0.50 and 0.57) sit below the conventional 0.667 minimum-acceptable threshold for inter-rater
reliability, while DeepSeek and OpenAI's (0.76, 0.85) are solid. This means the judge's scoring
of Anthropic and Groq responses is meaningfully less consistent across framings than its scoring
of the other two providers — the top-ranked provider's score is also the least reliably
measured one, which tempers how confidently the #1 ranking should be stated.

## 2. RAG Evaluation — Persona-Vector Ablation

Evaluated on the 8 Fari/Senpai conversational scenarios, run twice (persona-vector layer on and
off) with every other pipeline parameter held identical (verified programmatically before any
result was reported).

| Condition | Faithfulness | Answer Relevance | Context Coverage |
|---|---|---|---|
| persona_vector = OFF | 1.000 | 0.925 | 0.962 |
| persona_vector = ON | 1.000 | 0.812 | 0.962 |
| **Delta (ON − OFF)** | **0.000** | **−0.113** | **0.000** |

**Finding**: the persona-vector augmentation layer does not match IGuide's design intuition.
The intuition — that prepending a domain-specific descriptor to the query improves retrieval
relevance — predicts a positive delta. What we observe is a measurable *drop* in answer
relevance (−0.113), entirely concentrated on Fari (0.850 → 0.625; Senpai is unaffected at
1.000 → 1.000 both conditions), with no compensating gain in faithfulness or context coverage.
The mechanism is plausible: Fari's persona descriptor ("eldercare companion... medication
management, daily routines, emergency response, nutrition") is broad enough to shift the query
embedding toward generic eldercare topics even when the actual question is narrow and specific,
diluting relevance rather than sharpening it. Senpai's persona descriptor is comparatively
narrower relative to its question set, which likely explains why it shows no effect either way.
**Recommendation**: do not enable the persona-vector layer for Fari in its current form: the
data does not support IGuide's premise for this platform, and it costs relevance for no
measured benefit.

## 3. Agentic Evaluation — Task Completion, Step Efficiency, and a Verification Fix

Run on all 20 Track B scenarios against the two top-ranked providers from Section 1 (Anthropic,
DeepSeek), using a ReAct-style loop with a 3-seed LLM step verifier.

| Provider | Task Completion Rate | Mean Step Efficiency (successful runs) | Mean Step Efficiency (failed runs) |
|---|---|---|---|
| Anthropic | 25% (5/20) | 1.80 | 0.78 |
| DeepSeek | 15% (3/20) | 1.58 | 0.86 |

**Note on methodology**: these completion-rate figures were revised down from an initial pass
(Anthropic 70%, DeepSeek 30%) after a verification bug was found and fixed. The original step
verifier credited a required step as complete if the agent's narrated *Final Answer* claimed it
was done, even with no corresponding tool action in the transcript. In one representative case
(`TrackB_Fari_02`, Anthropic), the agent took a single `check_log` action, then wrote a
confident four-part summary claiming full task resolution — the verifier credited all four
required steps as complete despite three of them having no supporting action. This is the exact
failure mode SREGym's evaluation philosophy is designed to catch: a plausible-sounding claim of
completion is not the same as verified completion, and grading the claim instead of the
evidence overstates capability. The verifier was patched to require Action+Observation evidence
per step and every result was re-verified against the original transcripts (agent behavior
itself was not re-run, isolating the fix to the grading logic).

**Reading step efficiency correctly**: step efficiency should not be read as a single
cross-provider ranking independent of completion status. Split by outcome, the pattern is
coherent — successful runs average 1.6–1.8× the minimum required actions (real work was done,
with some inefficiency), while failed runs average *below* 1.0× (agents under-attempted the
task rather than genuinely struggling through it). Comparing raw step-efficiency means across
providers without this split is misleading.

**Failure pattern**: both providers fail 100% of Aido Humanoid scenarios (0/4 each) — the
platform requiring compound, multi-step physical-manipulation planning. Beyond that, the two
providers fail on different platforms (Anthropic struggles on Fari and Aido Humanoid; DeepSeek
struggles on Aido Rover, Sentinel Prime AI, and Aido Humanoid), suggesting the completion-rate
gap (25% vs. 15%) is not explained by one dominant failure mode but by DeepSeek failing more
broadly across platform types.

**Data limitation, stated plainly**: error-recovery rate could not be meaningfully compared
across providers — only 1 of 40 transcripts contained an environment-flagged error observation
at all, so the metric is inapplicable to the other 39 rows. No provider comparison is drawn on
this dimension; a future iteration should design scenarios that reliably trigger a recoverable
mid-task error to make this metric measurable.

## Summary

Anthropic leads on aggregate quality and RAG is not yet ready to ship the persona-vector layer
for Fari; but the most operationally useful finding this week is that both the four-provider
leaderboard and the agentic completion rate look different — and more actionable — once broken
down by platform rather than reported as a single number. Aido Humanoid is the platform every
provider struggles with hardest; that is the platform readiness work should prioritize next.
