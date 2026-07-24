"""
Cost vs. Quality Pareto Frontier — InGen AI Model Evaluation
=============================================================

Generates an interactive Plotly chart comparing the four evaluated LLM
providers on two axes that matter most for deployment decisions:
  - X: Estimated cost per 1K tokens (USD, log scale)
  - Y: Severity-Weighted Quality Score (normalised 0–1)

Pareto-optimal providers are those NOT dominated on BOTH axes simultaneously
(i.e., no other provider is both cheaper AND higher quality). They are
highlighted with a connecting frontier line.

Outputs
-------
  week03_synthesis/pareto_cost_quality.html  — interactive (open in browser)
  week03_synthesis/pareto_cost_quality.png   — static PNG for slides
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
LEADERBOARD_CSV = DATA_DIR / "leaderboard_summary.csv"

# ---------------------------------------------------------------------------
# Provider styling
# ---------------------------------------------------------------------------

PROVIDER_STYLE = {
    "anthropic": {
        "color": "#C97B44",          # Anthropic warm orange
        "symbol": "diamond",
        "label": "Anthropic<br>claude-sonnet-4-6",
    },
    "openai": {
        "color": "#10A37F",          # OpenAI green
        "symbol": "circle",
        "label": "OpenAI<br>gpt-4o",
    },
    "deepseek": {
        "color": "#4A90D9",          # DeepSeek blue
        "symbol": "square",
        "label": "DeepSeek<br>deepseek-chat",
    },
    "groq": {
        "color": "#9B59B6",          # Groq purple
        "symbol": "triangle-up",
        "label": "Groq<br>llama-3.1-8b-instant",
    },
}

# Cost per 1K tokens (USD, blended input+output estimate)
COST_PER_1K = {
    "anthropic": 0.004,
    "openai":    0.005,
    "deepseek":  0.0003,
    "groq":      0.00008,
}

# ---------------------------------------------------------------------------
# Pareto frontier helper
# ---------------------------------------------------------------------------

def pareto_frontier(
    df: pd.DataFrame,
    cost_col: str,
    quality_col: str,
) -> pd.DataFrame:
    """
    Identify Pareto-optimal providers.

    A provider is Pareto-optimal if no other provider has BOTH
    strictly lower cost AND strictly higher (or equal) quality.

    Returns the subset of df that lies on the frontier, sorted by cost
    ascending so the frontier line can be drawn left-to-right.
    """
    pareto_mask = []
    for _, row in df.iterrows():
        dominated = False
        for _, other in df.iterrows():
            if (
                other[cost_col] <= row[cost_col]
                and other[quality_col] >= row[quality_col]
                and (
                    other[cost_col] < row[cost_col]
                    or other[quality_col] > row[quality_col]
                )
            ):
                dominated = True
                break
        pareto_mask.append(not dominated)

    return df[pareto_mask].sort_values(cost_col).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build chart
# ---------------------------------------------------------------------------

def build_pareto_chart(df: pd.DataFrame) -> go.Figure:
    # Compute cost per 1K tokens from total_tokens + estimated_cost_usd
    df = df.copy()
    df["cost_per_1k"] = df.apply(
        lambda r: COST_PER_1K.get(r["provider"], r["estimated_cost_usd"] / max(r["total_tokens"], 1) * 1000),
        axis=1,
    )

    frontier_df = pareto_frontier(df, "cost_per_1k", "severity_weighted_score_norm")
    pareto_providers = set(frontier_df["provider"])

    fig = go.Figure()

    # --- Pareto frontier line ---
    fig.add_trace(
        go.Scatter(
            x=frontier_df["cost_per_1k"],
            y=frontier_df["severity_weighted_score_norm"],
            mode="lines",
            line=dict(color="rgba(255,215,0,0.6)", width=2, dash="dot"),
            name="Pareto Frontier",
            hoverinfo="skip",
            showlegend=True,
        )
    )

    # --- Provider scatter points ---
    for _, row in df.iterrows():
        provider = row["provider"]
        style = PROVIDER_STYLE.get(provider, {"color": "#888", "symbol": "circle", "label": provider})
        on_frontier = provider in pareto_providers

        # Ring around Pareto-optimal providers
        marker_line = dict(
            width=3 if on_frontier else 1,
            color="gold" if on_frontier else "white",
        )

        hover_text = (
            f"<b>{style['label'].replace('<br>', ' ')}</b><br>"
            f"Cost / 1K tokens: ${row['cost_per_1k']:.5f}<br>"
            f"SW Quality Score: {row['severity_weighted_score_norm']:.3f}<br>"
            f"Mean Task Accuracy: {row['mean_task_accuracy']:.2f} / 5<br>"
            f"Mean Grounding: {row['mean_grounding']:.2f} / 5<br>"
            f"Latency (mean): {row['mean_latency_ms']:,.0f} ms<br>"
            f"Krippendorff α: {row['krippendorff_alpha']:.3f}<br>"
            f"{'⭐ Pareto-optimal' if on_frontier else ''}"
        )

        fig.add_trace(
            go.Scatter(
                x=[row["cost_per_1k"]],
                y=[row["severity_weighted_score_norm"]],
                mode="markers+text",
                marker=dict(
                    size=22,
                    color=style["color"],
                    symbol=style["symbol"],
                    line=marker_line,
                    opacity=0.92,
                ),
                text=[style["label"]],
                textposition="top center",
                textfont=dict(size=11, color=style["color"], family="Inter, Arial, sans-serif"),
                name=style["label"].replace("<br>", " · "),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=True,
            )
        )

    # --- Shaded "ideal zone" (low cost, high quality) ---
    fig.add_shape(
        type="rect",
        x0=0, x1=0.001,
        y0=0.85, y1=1.05,
        fillcolor="rgba(0,200,100,0.06)",
        line=dict(width=0),
        layer="below",
    )
    fig.add_annotation(
        x=0.0005, y=1.01,
        text="Ideal Zone",
        showarrow=False,
        font=dict(size=10, color="rgba(0,200,100,0.7)", family="Inter, Arial, sans-serif"),
    )

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text=(
                "Cost vs. Quality Pareto Frontier<br>"
                "<sup>InGen AI Model Evaluation · 4 Providers · 40 Scenarios · "
                "Severity-Weighted Scoring</sup>"
            ),
            font=dict(size=20, family="Inter, Arial, sans-serif", color="#1a1a2e"),
            x=0.05,
            xanchor="left",
        ),
        xaxis=dict(
            title=dict(
                text="Estimated Cost per 1K Tokens (USD, log scale)",
                font=dict(size=13, family="Inter, Arial, sans-serif"),
            ),
            type="log",
            tickformat=".5f",
            tickprefix="$",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(
                text="Severity-Weighted Quality Score (normalised)",
                font=dict(size=13, family="Inter, Arial, sans-serif"),
            ),
            range=[0.70, 1.05],
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            zeroline=False,
            tickformat=".2f",
            tickfont=dict(size=11),
        ),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(200,200,200,0.6)",
            borderwidth=1,
            font=dict(size=11, family="Inter, Arial, sans-serif"),
        ),
        width=900,
        height=580,
        margin=dict(l=70, r=200, t=100, b=70),
        font=dict(family="Inter, Arial, sans-serif"),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="rgba(200,200,200,0.8)",
            font=dict(size=12, family="Inter, Arial, sans-serif"),
        ),
    )

    # Annotation: Pareto-optimal label
    for _, row in frontier_df.iterrows():
        fig.add_annotation(
            x=row["cost_per_1k"],
            y=row["severity_weighted_score_norm"] - 0.025,
            text="★",
            showarrow=False,
            font=dict(size=14, color="gold"),
            xanchor="center",
        )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = pd.read_csv(LEADERBOARD_CSV)

    fig = build_pareto_chart(df)

    # Interactive HTML
    html_path = OUT_DIR / "pareto_cost_quality.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"Interactive chart → {html_path.relative_to(REPO_ROOT)}")

    # Static PNG (requires kaleido)
    try:
        png_path = OUT_DIR / "pareto_cost_quality.png"
        fig.write_image(str(png_path), scale=2)
        print(f"Static PNG        → {png_path.relative_to(REPO_ROOT)}")
    except Exception as exc:
        print(f"PNG export skipped: {exc}")

    # Print frontier summary
    cost_df = df.copy()
    cost_df["cost_per_1k"] = cost_df["provider"].map(
        lambda p: COST_PER_1K.get(p, 0)
    )
    frontier = pareto_frontier(cost_df, "cost_per_1k", "severity_weighted_score_norm")
    print("\nPareto-optimal providers:")
    for _, r in frontier.iterrows():
        print(
            f"  {r['provider']:12s}  "
            f"quality={r['severity_weighted_score_norm']:.3f}  "
            f"cost/1K=${cost_df.loc[cost_df['provider']==r['provider'],'cost_per_1k'].values[0]:.5f}"
        )


if __name__ == "__main__":
    main()
