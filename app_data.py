"""
Cached data loading and merging for the Streamlit narrative app.

Loads the CSVs that already live at the repo root. The analysis scripts
in the repo execute on import (load CSVs at module level), so we cannot
import them safely; instead we read the CSVs directly here and pass tidy
DataFrames into app_stats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent
IR_DIR = REPO / "Raw IR Monitor Data"


# ----------------------------------------------------------------------
# Raw CSV loaders
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_circadian() -> pd.DataFrame:
    df = pd.read_csv(REPO / "Circadian_raw.csv")
    df = df.rename(columns={"PRE.POST": "PRE_POST"})
    return df


@st.cache_data(show_spinner=False)
def load_circadian_computed() -> pd.DataFrame:
    return pd.read_csv(REPO / "circadian_computed_raw.csv")


@st.cache_data(show_spinner=False)
def load_barnes() -> pd.DataFrame:
    return pd.read_csv(REPO / "Barnes_clean.csv")


@st.cache_data(show_spinner=False)
def load_nor() -> pd.DataFrame:
    df = pd.read_csv(REPO / "UCBAge_Novel_clean.csv")
    df = df.rename(columns={"Animal_ID": "ID"})
    # discrimination index
    n = df["N_obj_nose_duration_s"]
    f = df["F_obj_nose_duration_s"]
    df["DI_duration"] = (n - f) / (n + f)
    return df


@st.cache_data(show_spinner=False)
def load_learning_slopes() -> pd.DataFrame:
    return pd.read_csv(REPO / "learning_slopes_per_mouse.csv")


@st.cache_data(show_spinner=False)
def load_effect_sizes() -> pd.DataFrame:
    return pd.read_csv(REPO / "effect_sizes_and_power.csv")


@st.cache_data(show_spinner=False)
def load_bayes_factors() -> pd.DataFrame:
    return pd.read_csv(REPO / "bayes_factors_circadian_behaviour.csv")


@st.cache_data(show_spinner=False)
def load_clusters() -> pd.DataFrame:
    return pd.read_csv(REPO / "circadian_clusters.csv")


@st.cache_data(show_spinner=False)
def load_interactions() -> pd.DataFrame:
    return pd.read_csv(REPO / "circadian_behaviour_interactions.csv")


@st.cache_data(show_spinner=False)
def load_single_predictors() -> pd.DataFrame:
    return pd.read_csv(REPO / "circadian_behaviour_single.csv")


# ----------------------------------------------------------------------
# Wide circadian (PRE/POST per mouse)
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def circadian_wide() -> pd.DataFrame:
    """
    Pivot Circadian_raw to one row per mouse with _PRE / _POST columns.
    Keeps grouping vars (Sex_new, Age_new, Light_new) from POST row.
    """
    df = load_circadian()
    metrics = ["Amplitude", "Period", "Phase", "MESOR", "RA", "IV", "IS"]
    keep = ["ID", "PRE_POST"] + metrics
    long = df[keep].copy()
    wide = long.pivot_table(index="ID", columns="PRE_POST", values=metrics)
    wide.columns = [f"{m}_{p.lower()}" for m, p in wide.columns]
    wide = wide.reset_index()

    grp = (
        df[["ID", "Sex_new", "Age_new", "Light_new"]]
        .drop_duplicates(subset="ID")
        .reset_index(drop=True)
    )
    wide = wide.merge(grp, on="ID", how="left")

    # delta = POST - PRE
    for m in metrics:
        if f"{m}_post" in wide.columns and f"{m}_pre" in wide.columns:
            wide[f"{m}_delta"] = wide[f"{m}_post"] - wide[f"{m}_pre"]
    return wide


# ----------------------------------------------------------------------
# Cognition wide (one row per mouse)
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def barnes_per_mouse() -> pd.DataFrame:
    """Per-mouse Barnes summary: trial-6 endpoints + learning slope merge."""
    b = load_barnes()
    t6 = b[b["Trial"] == 6].copy()
    out = t6[
        [
            "ID",
            "EntryZone_freq_new",
            "Hole_errors",
            "Goal_Box_latency_new",
            "Entry_latency_new",
            "DistanceMoved_cm",
            "Sex_new",
            "Age_new",
            "Light_new",
        ]
    ].rename(
        columns={
            "EntryZone_freq_new": "barnes_entries_t6",
            "Hole_errors": "barnes_errors_t6",
            "Goal_Box_latency_new": "barnes_goal_latency_t6",
            "Entry_latency_new": "barnes_entry_latency_t6",
            "DistanceMoved_cm": "barnes_distance_t6",
        }
    )
    slopes = load_learning_slopes()[["ID", "learning_slope"]]
    return out.merge(slopes, on="ID", how="left")


@st.cache_data(show_spinner=False)
def nor_per_mouse() -> pd.DataFrame:
    df = load_nor()
    keep = [
        "ID",
        "DI_duration",
        "N_obj_nose_duration_s",
        "F_obj_nose_duration_s",
        "Sex_new",
        "Age_new",
        "Light_new",
    ]
    return df[keep].copy()


# ----------------------------------------------------------------------
# Master merged table — one row per mouse, all variables
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def master_table() -> pd.DataFrame:
    circ = circadian_wide()
    barnes = barnes_per_mouse()
    nor = nor_per_mouse()
    m = circ.merge(
        barnes.drop(columns=["Sex_new", "Age_new", "Light_new"], errors="ignore"),
        on="ID",
        how="left",
    )
    m = m.merge(
        nor.drop(columns=["Sex_new", "Age_new", "Light_new"], errors="ignore"),
        on="ID",
        how="left",
    )
    return m


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    sexes: Optional[list[str]] = None,
    ages: Optional[list[str]] = None,
    lights: Optional[list[str]] = None,
    phase_window: Optional[tuple[float, float]] = None,
    phase_col: str = "Phase_post",
) -> pd.DataFrame:
    out = df.copy()
    if sexes and "Sex_new" in out.columns:
        out = out[out["Sex_new"].isin(sexes)]
    if ages and "Age_new" in out.columns:
        out = out[out["Age_new"].isin(ages)]
    if lights and "Light_new" in out.columns:
        out = out[out["Light_new"].isin(lights)]
    if phase_window is not None and phase_col in out.columns:
        lo, hi = phase_window
        if lo <= hi:
            out = out[(out[phase_col] >= lo) & (out[phase_col] <= hi)]
        else:
            # wraparound (e.g., 350°–20°)
            out = out[(out[phase_col] >= lo) | (out[phase_col] <= hi)]
    return out


# ----------------------------------------------------------------------
# IR file discovery (for sleep-burst computation)
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def list_ir_files() -> pd.DataFrame:
    """Returns a DataFrame {ID, path} for available raw PIR files."""
    if not IR_DIR.exists():
        return pd.DataFrame(columns=["ID", "path"])
    rows = []
    for f in sorted(IR_DIR.glob("#*.csv")):
        stem = f.stem  # "#01 IR"
        digits = "".join(c for c in stem.split()[0] if c.isdigit())
        if digits:
            rows.append({"ID": int(digits), "path": str(f)})
    return pd.DataFrame(rows)
