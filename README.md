# george-ingen-ai-eval

AI Model Evaluation harness built for the InGen Dynamics 4-week accelerated AI Model
Evaluation Internship. Evaluates four LLM providers (OpenAI GPT-4o, Anthropic Claude,
DeepSeek, and Llama-3.1-8B via Groq) across a 40-scenario benchmark spanning InGen's five
physical AI platforms (Fari, Senpai, Sentinel Prime AI, Aido Rover, Aido Humanoid), with
production-style RAG evaluation and multi-step agentic evaluation layered on top.

No InGen internal systems, customer data, or proprietary documentation are used anywhere in
this repository. All providers are referenced by their public model IDs.

## Program Overview

| Phase | Week | What it produces |
|---|---|---|
| A | 1 | Landscape brief, 40-scenario benchmark (Track A conversational + Track B agentic), operational evaluation harness |
| B | 2 | 4-provider comparison, production RAG evaluation, multi-step agentic evaluation |
| C | 3 | System-level Pareto analysis, PIC 2.0 model-class mapping, research paper draft |
| D | 4 | Streamlit evaluation dashboard, capstone report, executive deck |

## Repository Structure

```
george-ingen-ai-eval/
├── README.md                     # this file
├── requirements.txt              # pinned dependencies
├── .env.example                  # API key template (see below)
├── week01_benchmark/
│   ├── trackA_conversational.yaml    # 20 conversational scenarios (4 per platform x 5 platforms)
│   ├── trackB_agentic.yaml           # 20 agentic multi-step scenarios (4 per platform x 5 platforms)
│   ├── W01_PhysicalAI_Eval_Landscape.md
│   ├── W01_Benchmark_Design.md
│   └── eval_harness/
│       ├── clients.py             # execution layer — async provider clients w/ retry+backoff
│       ├── dispatcher.py          # orchestration layer — concurrent scenario x provider dispatch
│       ├── llm_judge.py           # 3-seed LLM-as-judge (Krippendorff's alpha input)
│       └── main.py                # CLI entry point
├── week02_evaluation/
│   ├── rag_eval/                  # IGuide-inspired RAG pipeline + RAGAS-style scorer + ablation
│   ├── agentic_eval/              # ReAct agent loop + step verifier
│   ├── multiprovider_eval/        # severity-weighted leaderboard, per-platform sub-scores
│   └── W02_Evaluation_Memo.md
├── week03_synthesis/              # (Week 3)
├── week04_capstone/                # (Week 4)
├── data/                          # all pre-computed result CSVs/JSONs (results_*, judged_*, leaderboard_summary.csv)
└── weekly/
    └── Wk-NN-EvalLog.md           # one 300-word log per week
```

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your own API keys — see below
```

### API keys

This repo uses your own API keys / free-tier allocations. Copy `.env.example` to `.env`
and fill in:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...
GROQ_API_KEY=gsk_...          # used for Llama-3.1-8B-Instant; HF_API_KEY / HF_TOKEN also accepted
```

Never commit `.env`. `.env.example` in this repo contains empty placeholders only.

### Model version registry

| Provider label | Model ID | Used for |
|---|---|---|
| `openai` | `gpt-4o` | Track A/B generation, LLM judge, RAG scorer |
| `anthropic` | `claude-sonnet-4-6` | Track A/B generation, agentic evaluation |
| `deepseek` | `deepseek-chat` | Track A/B generation, agentic evaluation |
| `groq` | `llama-3.1-8b-instant` | Track A/B generation |

Every evaluation result recorded by this harness states provider + model + evaluation_set +
temperature + seed, per the SREGym benchmark standard — no result is reported without full
reproduction metadata.

## One-Command Reproduction Path

```bash
# 1. Smoke test — 5 scenarios (one per platform) x 4 providers, verifies all API connections
python -m week01_benchmark.eval_harness.main --smoke

# 2. Full Week 1/2 benchmark run — all 40 scenarios (20 Track A + 20 Track B) x 4 providers
python -m week01_benchmark.eval_harness.main

# 3. LLM-as-judge scoring (3 seeds per response, for Krippendorff's alpha)
python -m week01_benchmark.eval_harness.llm_judge --all

# 4. RAG evaluation — Fari + Senpai, persona-vector ablation
python -m week02_evaluation.rag_eval.rag_eval

# 5. Agentic evaluation — 20 Track B scenarios x top-2 providers (anthropic, deepseek)
python -m week02_evaluation.agentic_eval.agentic_eval

# 6. Multi-provider leaderboard notebook
jupyter nbconvert --to notebook --execute week02_evaluation/multiprovider_eval/W02_MultiProvider_Eval.ipynb
```

All results are written to `/data/` as timestamped JSON (+ CSV where applicable). Every
script supports `--dry-run` (skip API calls, write clearly-flagged placeholder data for
pipeline testing) — dry-run output is never used as a substitute for a real evaluation
result, and metadata always states `"dry_run": true/false` explicitly.

## Benchmark Summary

- **Track A (conversational)**: 20 scenarios, 4 per platform x 5 platforms (Fari, Senpai,
  Sentinel Prime AI, Aido Rover, Aido Humanoid). Severity classes: 7 x severity-5, 5 x
  severity-4, 3 x severity-3, 5 x severity-2.
- **Track B (agentic)**: 20 scenarios, 4 per platform x 5 platforms. All scenarios have
  turn_depth >= 3 (8 at turn_depth=3, 12 at turn_depth=4), each with an explicit
  `required_steps` / `success_criteria_per_step` pair and `early_exit_failure_conditions`.

See `week01_benchmark/W01_Benchmark_Design.md` for the full design rationale.

## License / Confidentiality

Internal internship deliverable. All content is safe for publication to a personal GitHub
repository — no InGen confidential material is included anywhere in this repo.
