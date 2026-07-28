# Week 2 Evaluation Log

**Period**: Week 2 (4-Provider Evaluation, Production RAG Evaluation, Agentic Task Evaluation)
**Author**: George Wang

## Evaluated

Ran the full 40-scenario benchmark against all four providers (160 judged rows, 100% judge
coverage after fixing a rate-limit gap), producing a severity-weighted leaderboard and
per-provider Krippendorff's alpha. Built and ran the production RAG pipeline (IGuide-inspired
architecture) on the 8 Fari/Senpai conversational scenarios under a controlled persona-vector
ablation. Built and ran the multi-step agentic evaluation — a ReAct loop plus 3-seed step
verifier — on all 20 Track B scenarios against the two top-ranked providers (Anthropic,
DeepSeek).

## Found

The persona-vector ablation does not confirm IGuide's design intuition. IGuide's premise is
that prepending a domain-specific descriptor to the query improves retrieval relevance; the
data shows the opposite for Fari — answer relevance drops by 0.113 (0.925 → 0.812) with the
layer enabled, concentrated entirely on Fari and with no corresponding gain in faithfulness or
context coverage. Senpai is unaffected either way. The most plausible mechanism: Fari's persona
descriptor is broad enough (medication, routines, emergency response, nutrition) to pull the
query embedding toward generic eldercare topics even for narrow questions, diluting rather than
sharpening relevance. This directly informs the deployment recommendation in this week's memo:
do not enable persona-vector augmentation for Fari as currently designed.

## Mechanism

Also found and fixed a verification bug in the agentic evaluation: the step verifier initially
credited a required step as complete whenever the agent's Final Answer *claimed* it was done,
without requiring a corresponding tool action in the transcript. This inflated completion rates
substantially (Anthropic 70% → 25%, DeepSeek 30% → 15% after the fix). The mechanism is a
narrative-only claim standing in for verified evidence — exactly the failure mode SREGym's
evaluation philosophy exists to catch, and the same principle applied to my own harness that I
applied when reviewing InGen's PIC 2.0 platform paper: a claim of success is not evidence of
success until something independently checks it against the actual trace of actions taken.

**Next week**: system-level latency × quality Pareto analysis, RAG configuration ablation
(chunk size × top-k × reranking), PIC 2.0 model-class mapping, and the research paper draft.
