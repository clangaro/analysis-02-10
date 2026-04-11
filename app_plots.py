"""Plotly visualizations for the Streamlit narrative app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats


GROUP_COLORS = {"CTR": "#4C78A8", "ISF": "#F58518", "Male": "#54A24B", "Female": "#E45756"}


# ----------------------------------------------------------------------
# Circadian phase — polar histogram
# ----------------------------------------------------------------------

def phase_polar(df: pd.DataFrame, phase_col: str = "Phase_post", group_col: str = "Light_new") -> go.Figure:
    fig = go.Figure()
    if phase_col not in df.columns:
        fig.add_annotation(text=f"Missing column: {phase_col}", showarrow=False)
        return fig
    bins = np.linspace(0, 360, 25)
    centres = (bins[:-1] + bins[1:]) / 2
    for grp, sub in df.groupby(group_col, dropna=True):
        vals = sub[phase_col].dropna().values
        if len(vals) == 0:
            continue
        counts, _ = np.histogram(vals, bins=bins)
        fig.add_trace(
            go.Barpolar(
                r=counts,
                theta=centres,
                width=[15] * len(centres),
                name=str(grp),
                marker_color=GROUP_COLORS.get(str(grp)),
                opacity=0.7,
            )
        )
    fig.update_layout(
        title="Circadian acrophase distribution (cosinor phase, deg)",
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                             tickvals=[0, 90, 180, 270], ticktext=["0°", "90°", "180°", "270°"]),
            radialaxis=dict(showticklabels=True, ticks=""),
        ),
        height=420,
        legend_title=group_col,
    )
    return fig


# ----------------------------------------------------------------------
# Sleep bursts by hour-of-day
# ----------------------------------------------------------------------

def burst_hour_plot(profile: pd.DataFrame, dark_phase: tuple[float, float] = (12, 24)) -> go.Figure:
    fig = go.Figure()
    if profile.empty:
        return fig
    fig.add_trace(
        go.Bar(
            x=profile["hour_bin"],
            y=profile["n_bouts"],
            name="# rest bouts",
            marker_color="#4C78A8",
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=profile["hour_bin"],
            y=profile["mean_bout_min"],
            name="Mean bout length (min)",
            mode="lines+markers",
            line=dict(color="#F58518", width=2),
            yaxis="y2",
        )
    )
    fig.add_vrect(
        x0=dark_phase[0], x1=dark_phase[1],
        fillcolor="lightgrey", opacity=0.25, line_width=0,
        annotation_text="dark phase", annotation_position="top left",
    )
    fig.update_layout(
        title="Sleep / rest bursts across the 24-h cycle",
        xaxis_title="Hour of day",
        yaxis=dict(title="# rest bouts"),
        yaxis2=dict(title="Mean bout length (min)", overlaying="y", side="right"),
        height=380,
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def burst_group_compare(profiles: dict[str, pd.DataFrame]) -> go.Figure:
    """profiles: {group_label: per-hour profile DataFrame}."""
    fig = go.Figure()
    for label, prof in profiles.items():
        if prof.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=prof["hour_bin"],
                y=prof["n_bouts"],
                name=label,
                mode="lines+markers",
                line=dict(color=GROUP_COLORS.get(label, None)),
            )
        )
    fig.update_layout(
        title="Rest-bout count per hour, by group",
        xaxis_title="Hour of day",
        yaxis_title="# rest bouts",
        height=360,
    )
    return fig


# ----------------------------------------------------------------------
# Correlation scatter (circadian × cognition)
# ----------------------------------------------------------------------

def correlation_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = "Light_new",
    method: str = "pearson",
) -> go.Figure:
    sub = df[[x_col, y_col, color_col]].dropna()
    if sub.empty:
        return go.Figure()
    fig = px.scatter(
        sub, x=x_col, y=y_col, color=color_col,
        color_discrete_map=GROUP_COLORS, trendline="ols",
        opacity=0.75,
    )
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    r, p = fn(sub[x_col], sub[y_col])
    fig.update_layout(
        title=f"{y_col} vs {x_col}  —  {method} r = {r:.3f}, p = {p:.3g}",
        height=420,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> go.Figure:
    sub = df[cols].dropna()
    if sub.empty or len(sub) < 4:
        return go.Figure()
    corr = sub.corr(method=method)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig.update_layout(title=f"{method.title()} correlation matrix", height=460)
    return fig


# ----------------------------------------------------------------------
# Effect-size forest
# ----------------------------------------------------------------------

def effect_size_forest(effects_df: pd.DataFrame) -> go.Figure:
    df = effects_df.copy().sort_values("Cohen_d")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Cohen_d"],
            y=df["Outcome"],
            mode="markers",
            marker=dict(size=11, color=df["Cohen_d"], colorscale="RdBu", cmin=-1, cmax=1, showscale=True),
            error_x=dict(
                type="data",
                array=1.96 * np.sqrt((df["n_CTR"] + df["n_ISF"]) / (df["n_CTR"] * df["n_ISF"]) +
                                      (df["Cohen_d"] ** 2) / (2 * (df["n_CTR"] + df["n_ISF"]))),
                visible=True,
            ),
            name="Cohen's d (95% CI)",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    for d, lbl in [(0.2, "small"), (0.5, "medium"), (0.8, "large")]:
        fig.add_vline(x=d, line_dash="dot", line_color="lightgrey")
        fig.add_vline(x=-d, line_dash="dot", line_color="lightgrey")
    fig.update_layout(
        title="Effect sizes (CTR vs ISF) — Cohen's d with approximate 95% CI",
        xaxis_title="Cohen's d",
        height=520,
        margin=dict(l=200),
    )
    return fig


# ----------------------------------------------------------------------
# Power
# ----------------------------------------------------------------------

def power_curve_plot(curve_df: pd.DataFrame, target_power: float = 0.8, observed_n: int | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve_df["n_per_group"], y=curve_df["power"],
                             mode="lines+markers", line=dict(color="#4C78A8")))
    fig.add_hline(y=target_power, line_dash="dash", line_color="grey",
                  annotation_text=f"target = {target_power}")
    if observed_n is not None:
        fig.add_vline(x=observed_n, line_dash="dot", line_color="orange",
                      annotation_text=f"current n = {observed_n}")
    fig.update_layout(
        title="Statistical power vs sample size",
        xaxis_title="n per group",
        yaxis_title="Power",
        yaxis=dict(range=[0, 1.05]),
        height=360,
    )
    return fig


def observed_vs_required_power(effects_df: pd.DataFrame) -> go.Figure:
    df = effects_df.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Observed", x=df["Outcome"], y=df["Power_observed"], marker_color="#4C78A8"))
    fig.add_trace(go.Bar(name="If d=0.5", x=df["Outcome"], y=df["Power_d05"], marker_color="#F58518"))
    fig.add_hline(y=0.8, line_dash="dash", line_color="grey")
    fig.update_layout(
        title="Observed power vs power-if-medium-effect",
        barmode="group",
        height=420,
        xaxis_tickangle=-30,
        yaxis=dict(range=[0, 1.05], title="Power"),
    )
    return fig


# ----------------------------------------------------------------------
# Bayes factors
# ----------------------------------------------------------------------

def bayes_factor_plot(bf_df: pd.DataFrame) -> go.Figure:
    df = bf_df.copy()
    df["log_bf01"] = np.log10(df["BF01"].clip(lower=1e-3))
    df = df.sort_values("log_bf01")
    fig = go.Figure(
        go.Bar(
            x=df["log_bf01"],
            y=df["outcome"] + " ~ " + df["predictor"],
            orientation="h",
            marker=dict(color=df["log_bf01"], colorscale="RdBu", cmin=-2, cmax=2, showscale=True),
            text=df["BF01"].round(2),
            textposition="outside",
        )
    )
    for x, lbl in [(np.log10(3), "BF01=3"), (np.log10(10), "BF01=10")]:
        fig.add_vline(x=x, line_dash="dot", line_color="grey", annotation_text=lbl)
    fig.update_layout(
        title="Bayes factors (BF01) — log10 scale; positive = evidence for null",
        xaxis_title="log10(BF01)",
        height=460,
        margin=dict(l=240),
    )
    return fig


# ----------------------------------------------------------------------
# Learning curves
# ----------------------------------------------------------------------

def barnes_learning_curve(barnes_df: pd.DataFrame, group_col: str = "Light_new", y_col: str = "EntryZone_freq_new") -> go.Figure:
    agg = (
        barnes_df.groupby(["Trial", group_col])[y_col]
        .agg(["mean", "sem"])
        .reset_index()
    )
    fig = go.Figure()
    for grp, sub in agg.groupby(group_col):
        fig.add_trace(
            go.Scatter(
                x=sub["Trial"],
                y=sub["mean"],
                error_y=dict(type="data", array=sub["sem"]),
                mode="lines+markers",
                name=str(grp),
                line=dict(color=GROUP_COLORS.get(str(grp))),
            )
        )
    fig.update_layout(
        title=f"Barnes maze learning curve — {y_col} (mean ± SEM)",
        xaxis_title="Trial",
        yaxis_title=y_col,
        height=380,
    )
    return fig


# ----------------------------------------------------------------------
# Diagnostic: Q-Q plot
# ----------------------------------------------------------------------

def qq_plot(values, label: str = "") -> go.Figure:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return go.Figure()
    qs = (np.arange(1, len(arr) + 1) - 0.5) / len(arr)
    theo = stats.norm.ppf(qs)
    obs = np.sort(arr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theo, y=obs, mode="markers", name="data"))
    lo, hi = theo.min(), theo.max()
    slope, intercept = np.polyfit(theo, obs, 1)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[slope * lo + intercept, slope * hi + intercept],
                             mode="lines", name="ref", line=dict(color="grey", dash="dash")))
    fig.update_layout(
        title=f"Q-Q plot{' — ' + label if label else ''}",
        xaxis_title="Theoretical quantiles",
        yaxis_title="Observed quantiles",
        height=320,
    )
    return fig
