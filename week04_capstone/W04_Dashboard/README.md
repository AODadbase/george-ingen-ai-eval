# W04 Dashboard

Streamlit dashboard for the InGen AI Model Evaluation capstone — built for a live final-readout
demo. Every number shown is pre-computed and loaded from a file already on disk; **no page in
this app calls an LLM provider or judge on launch**.

## One-command launch

```bash
pip install -r week04_capstone/W04_Dashboard/requirements.txt
streamlit run week04_capstone/W04_Dashboard/app.py
```

The app needs the Week 2/3 result files to already exist. If any are missing, regenerate them
first via the repo's normal reproduction path (see the root `README.md` and the per-view
"Source" captions under the **AI evaluation engineer** persona, which state the exact command
for each number shown):

```bash
python week02_evaluation/leaderboard_analysis.py
python -m week02_evaluation.rag_eval.rag_eval
python -m week02_evaluation.agentic_eval.agentic_eval
jupyter nbconvert --to notebook --execute week02_evaluation/W02_MultiProvider_Eval.ipynb
python -m week03_synthesis.rag_ablation
jupyter nbconvert --to notebook --execute week03_synthesis/W03_System_Eval.ipynb
```

## Architecture

Clean data-layer / presentation-layer separation:

```
W04_Dashboard/
├── app.py                 # entry page — persona selector + intro + Executive/PM synthesis
├── data_layer.py           # ALL data loading + reused verified computations. No st.* calls.
├── persona.py               # shared persona-selector widget + engineer source-citation helper
├── pages/
│   ├── 1_Multi_Provider_Leaderboard.py
│   ├── 2_RAG_Performance.py
│   ├── 3_Agentic_Evaluation.py
│   └── 4_System_Level.py
├── requirements.txt
└── README.md               # this file
```

- **`data_layer.py`** is the only place that reads a file or does a calculation. Every function
  returns a plain `pandas.DataFrame` (or a small dict/list of primitives for summary numbers).
  It imports the already-verified Week 2/3 modules unchanged rather than recomputing anything:
  `week02_evaluation/leaderboard_analysis.py`, `week02_evaluation/multiprovider_eval/
  platform_scores.py`, `week02_evaluation/multiprovider_eval/cost_quality.py`,
  `week03_synthesis/pareto_chart.py`.
- **Page files** (`app.py`, `pages/*.py`) only call `data_layer.py` functions and render the
  result via Streamlit/Plotly — no computation logic is embedded in a page file.

This split makes the "does every view load entirely from pre-computed files, no model
inference on launch" self-check mechanically checkable: grep every page file for an import of
`clients.py`, `rag_scorer.py`, `step_verifier.py`, or `llm_judge.py` — none should ever appear.
`tests/test_dashboard_data_layer.py` runs this check (and two others) as an actual test suite:

```bash
python -m pytest tests/test_dashboard_data_layer.py -v
```

## Personas

A single sidebar selector (`persona.py`, synced via `st.session_state`) filters what's shown
across every page — this is **one app with a filter**, not three separate apps:

| Persona | What it adds |
|---|---|
| **AI evaluation engineer** | Full methodology detail, a source-file citation + reproduction command under every metric. |
| **Product manager** | Per-PIC-2.0-class readiness (from `W03_PIC20_Analysis.md`, explicitly labeled as class-level, not platform-level — no platform→class mapping is documented anywhere in this repo) + top-3 platform-level deployment risks, each a specific finding with a real number, not a generic category. |
| **Executive** | A three-number summary on the home page: fleet-wide readiness, top failure risk, recommended next action — every heading on this tab is the finding text itself, never a topic label. |

## Views

1. **Multi-Provider Leaderboard** — severity-weighted scorecard (`data/leaderboard_summary.csv`
   as-is), per-platform drill-down, cost x quality table, and the corrected linear-axis
   latency x quality Pareto frontier.
2. **RAG Performance** — the Week 2 persona-vector ablation (aggregate **and** per-platform,
   since the aggregate alone hides that Senpai was unaffected) and the Week 3 12-config
   chunk_size x top_k x reranking ablation, with the Pareto-optimal config's trade-off stated
   honestly (wins on latency/faithfulness, costs answer_relevance and context_coverage).
3. **Agentic Evaluation** — task completion and step efficiency per provider per platform on the
   **post-verifier-fix** data, with the before/after numbers shown rather than hidden; step
   efficiency split by completion status (not one pooled distribution); error-recovery rate
   shown as the actual sparse data (1 of 40 transcripts) with an explicit "not enough for a real
   comparison" label instead of a heatmap that would imply otherwise.
4. **System-Level** — latency distribution per provider, the Aido Rover real-time
   latency-threshold x quality-floor table, and the cost-per-quality-point bar chart.

## Known fix applied while building this dashboard

`week03_synthesis/pareto_chart.py`'s cost x quality chart had an `add_shape()` call with
`x0=0` on a log-scale x-axis (`log(0)` is undefined, which silently corrupted the axis range).
This had previously been worked around once in `W03_System_Eval.ipynb` by using a linear axis
for a different chart instead of fixing the source. It's fixed here at the source (`x0=1e-5`)
so it no longer needs a workaround anywhere it's reused, including this dashboard.
