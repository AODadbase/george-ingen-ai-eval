# Week 3 Evaluation Log

**Period**: Week 3 (System-Level Evaluation, PIC 2.0 Analysis, Research Paper Draft)
**Author**: George Wang

## Evaluated

Built the latency x quality Pareto frontier across all four providers (the primary required
chart, kept on a linear axis specifically to avoid a log-axis shape bug already present
elsewhere in this repo), the Aido Rover latency-threshold analysis (2000ms threshold, 0.85
quality floor), cost-per-quality-point, and a 12-configuration RAG ablation on chunk size x
top-k x reranking for the Senpai subset. Wrote the 4-page PIC 2.0 model-class analysis mapping
all six foundation model classes (GRPO, STUM, SEOM, AMDC, HTD-IRL, CRL-MRS) to this program's
evaluation evidence, and the 8-page workshop paper draft.

## Found

Anthropic — the aggregate leaderboard leader — fails the Aido Rover latency threshold by a wide
margin (10,963ms mean vs. a 2000ms operational threshold) despite reaching perfect quality
(1.0) on that platform; only Groq clears both the latency and quality bar there. This is the
clearest demonstration yet that a single aggregate score actively hides the finding that
matters for a specific deployment decision, not just simplifies it.

Writing the PIC 2.0 analysis surfaced which model class has the largest evidence gap: **AMDC**.
Its defining function is multi-modal drift compensation, and this entire harness — benchmark,
RAG pipeline, agentic loop — is text-only; no image, sensor, or video input was ever passed to
any provider. This is a different kind of gap than CRL-MRS's (multi-agent coordination, also
untested, but flagged as out-of-scope since Week 1 and therefore expected) — AMDC's gap is more
consequential because nothing in the current harness design is even a partial proxy for it,
unlike the other five classes where a text-only LLM is at least a reasonable stand-in.

## Mechanism

The highest-value experiment to close AMDC's gap is not a bigger version of what this program
already does — it requires actually feeding image or sensor input alongside the text prompt for
at least one platform (Sentinel Prime AI's security scenarios are the most natural fit, given
they already reference visual/camera-adjacent context). Scaling up the existing text-only
harness cannot close this gap no matter how much data is added, because the gap is in kind, not
in degree — the same distinction this week's Rover finding drew between "more data on the same
metric" and "a metric measuring the wrong thing entirely."

**Next week**: Streamlit evaluation dashboard, capstone report, executive deck, and the final
signed evaluation rubric.
