"""
Sensitivity analysis: Full statistical pipeline on z-score normalised circadian metrics.

Runs the same models as analysis.py but using circadian metrics recomputed
from z-score normalised raw activity data (circadian_computed_normalised.csv).

This addresses the concern that inter-sensor gain differences could bias
circadian metrics — particularly RA (r=0.12 between raw and normalised).

Reports side-by-side: original ClockLab metrics vs normalised metrics.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# ============================================================
# Helpers
# ============================================================

def clean_colnames(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str).str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

def set_cats(df, col, cats):
    if col in df.columns:
        df[col] = df[col].astype("category").cat.set_categories(cats, ordered=True)
    return df

def logit_clip(x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


# ============================================================
# Load data
# ============================================================

# Original ClockLab metrics
circ_orig = clean_colnames(pd.read_csv("Circadian_raw.csv"))
circ_orig["ID"] = pd.to_numeric(circ_orig["ID"], errors="coerce").astype("Int64")

# Normalised metrics (from metrics_script.py)
circ_norm = pd.read_csv("circadian_computed_normalised.csv")
circ_norm = circ_norm[circ_norm["PRE_POST"].isin(["PRE", "POST"])].copy()
circ_norm["ID"] = pd.to_numeric(circ_norm["ID"], errors="coerce").astype("Int64")

# Merge Light/Age/Sex from original (normalised file doesn't have these)
meta = circ_orig[["ID", "Light_new", "Age_new", "Sex_new"]].drop_duplicates("ID")
circ_norm = circ_norm.merge(meta, on="ID", how="left")

# Barnes and NOR
barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))
nor = clean_colnames(pd.read_csv("UCBAge_Novel_clean.csv"))
if "Animal_ID" in nor.columns and "ID" not in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})
for df in (barnes, nor):
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

barnes = set_cats(barnes, "Light_new", ["CTR", "ISF"])
barnes["Age_new"] = barnes["Age_new"].astype("category")
barnes["Sex_new"] = barnes["Sex_new"].astype("category")
barnes["Trial"] = pd.to_numeric(barnes["Trial"], errors="coerce")

nor = set_cats(nor, "Light_new", ["CTR", "ISF"])
nor["Age_new"] = nor["Age_new"].astype("category")
nor["Sex_new"] = nor["Sex_new"].astype("category")


# ============================================================
# Run pipeline on both datasets
# ============================================================

results = []

for label, circ in [("Original (ClockLab)", circ_orig), ("Z-score normalised", circ_norm)]:
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE: {label}")
    print(f"{'=' * 70}")

    circ = circ.copy()
    circ = set_cats(circ, "PRE_POST", ["PRE", "POST"])
    circ = set_cats(circ, "Light_new", ["CTR", "ISF"])
    circ["Age_new"] = circ["Age_new"].astype("category")
    circ["Sex_new"] = circ["Sex_new"].astype("category")

    n_mice = circ["ID"].nunique()
    print(f"  Mice: {n_mice}")

    # --- Circadian LME models ---
    for metric in ["IS", "IV", "RA", "Amplitude"]:
        if metric not in circ.columns:
            continue
        d = circ[["ID", "PRE_POST", "Light_new", "Age_new", "Sex_new", metric]].dropna().copy()
        if d["ID"].nunique() < 5:
            continue
        formula = f"{metric} ~ PRE_POST * Light_new + Age_new + Sex_new"
        try:
            m = smf.mixedlm(formula, d, groups=d["ID"]).fit(method="lbfgs", reml=True)
            interaction = [t for t in m.pvalues.index if "PRE_POST" in t and "Light_new" in t]
            if interaction:
                p = float(m.pvalues[interaction[0]])
                beta = float(m.params[interaction[0]])
            else:
                p, beta = np.nan, np.nan
            results.append({"pipeline": label, "model": f"Circadian LME: {metric}",
                           "term": "PRE_POST x Light", "beta": beta, "p": p, "n": d["ID"].nunique()})
            print(f"  {metric}: interaction beta={beta:.4f}, p={p:.4f}")
        except Exception as e:
            print(f"  {metric}: FAILED ({e})")

    # --- Mouse-level predictors ---
    wide_IS = circ.pivot_table(index="ID", columns="PRE_POST", values="IS", aggfunc="mean")
    IS_pre = wide_IS.get("PRE").rename("IS_pre")
    delta_IS = (wide_IS.get("POST") - wide_IS.get("PRE")).rename("delta_IS")
    mouse_covars = (
        circ.sort_values(["ID", "PRE_POST"])
        .groupby("ID")[["Light_new", "Age_new", "Sex_new"]]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
    )
    circ_mouse = pd.concat([mouse_covars, IS_pre, delta_IS], axis=1).reset_index()

    # --- Barnes Trial 6 ---
    bm = barnes.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")
    bt6 = bm[bm["Trial"] == 6].copy()
    bt6["EntryZone_freq_new"] = pd.to_numeric(bt6["EntryZone_freq_new"], errors="coerce")
    bt6["Hole_errors"] = pd.to_numeric(bt6["Hole_errors"], errors="coerce")
    bt6["total_pokes"] = bt6["EntryZone_freq_new"] + bt6["Hole_errors"]
    bt6 = bt6[bt6["total_pokes"] > 0].copy()
    bt6["probe_accuracy"] = bt6["EntryZone_freq_new"] / bt6["total_pokes"]
    bt6["IS_pre"] = pd.to_numeric(bt6["IS_pre"], errors="coerce")
    bt6["delta_IS"] = pd.to_numeric(bt6["delta_IS"], errors="coerce")
    bt6 = bt6.dropna(subset=["probe_accuracy", "Light_new", "Age_new", "Sex_new",
                              "IS_pre", "delta_IS"]).copy()
    bt6["Light_new"] = bt6["Light_new"].astype(str)
    bt6["Age_new"] = bt6["Age_new"].astype(str)
    bt6["Sex_new"] = bt6["Sex_new"].astype(str)

    if bt6["ID"].nunique() >= 5:
        try:
            fit = smf.ols(
                "probe_accuracy ~ C(Light_new, Treatment('CTR')) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
                data=bt6
            ).fit(cov_type="HC3")
            term = "C(Light_new, Treatment('CTR'))[T.ISF]"
            p = float(fit.pvalues.get(term, np.nan))
            beta = float(fit.params.get(term, np.nan))
            results.append({"pipeline": label, "model": "Barnes T6 probe accuracy (OLS)",
                           "term": "Light[ISF]", "beta": beta, "p": p, "n": bt6["ID"].nunique()})
            print(f"  Barnes T6: Light beta={beta:.4f}, p={p:.4f}")
        except Exception as e:
            print(f"  Barnes T6: FAILED ({e})")

    # --- NOR ---
    nm = nor.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")
    N_dur = pd.to_numeric(nm.get("N_obj_nose_duration_s"), errors="coerce")
    F_dur = pd.to_numeric(nm.get("F_obj_nose_duration_s"), errors="coerce")
    nm = nm[(N_dur.notna()) & (F_dur.notna())].copy()
    N_dur = N_dur.loc[nm.index]; F_dur = F_dur.loc[nm.index]
    nm["DI"] = (N_dur - F_dur) / (N_dur + F_dur + 1e-9)
    nm = nm.dropna(subset=["DI", "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS"]).copy()

    if nm["ID"].nunique() >= 5:
        try:
            fit = smf.ols(
                "DI ~ Light_new + IS_pre + delta_IS + Age_new + Sex_new",
                data=nm
            ).fit(cov_type="HC3")
            term = "Light_new[T.ISF]"
            p = float(fit.pvalues.get(term, np.nan))
            beta = float(fit.params.get(term, np.nan))
            results.append({"pipeline": label, "model": "NOR DI (OLS)",
                           "term": "Light[ISF]", "beta": beta, "p": p, "n": nm["ID"].nunique()})
            print(f"  NOR: Light beta={beta:.4f}, p={p:.4f}")
        except Exception as e:
            print(f"  NOR: FAILED ({e})")


# ============================================================
# Side-by-side comparison
# ============================================================

res_df = pd.DataFrame(results)

print(f"\n{'=' * 70}")
print("  SIDE-BY-SIDE: ORIGINAL vs NORMALISED")
print(f"{'=' * 70}")

orig = res_df[res_df["pipeline"].str.contains("Original")].set_index("model")
norm = res_df[res_df["pipeline"].str.contains("normalised")].set_index("model")

comparison = pd.DataFrame({
    "n_orig": orig["n"],
    "beta_orig": orig["beta"],
    "p_orig": orig["p"],
    "n_norm": norm["n"],
    "beta_norm": norm["beta"],
    "p_norm": norm["p"],
})

comparison["same_direction"] = np.sign(comparison["beta_orig"]) == np.sign(comparison["beta_norm"])
comparison["both_ns"] = (comparison["p_orig"] > 0.05) & (comparison["p_norm"] > 0.05)
comparison["conclusion_changed"] = (
    ((comparison["p_orig"] <= 0.05) & (comparison["p_norm"] > 0.05)) |
    ((comparison["p_orig"] > 0.05) & (comparison["p_norm"] <= 0.05))
)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(comparison.to_string())

print(f"\n{'=' * 70}")
print("  VERDICT")
print(f"{'=' * 70}")
if comparison["conclusion_changed"].any():
    changed = comparison[comparison["conclusion_changed"]]
    print("CONCLUSIONS CHANGED for:")
    print(changed[["beta_orig", "p_orig", "beta_norm", "p_norm"]].to_string())
else:
    print("No conclusions changed. All effects that were non-significant with")
    print("original metrics remain non-significant with normalised metrics.")

res_df.to_csv("normalised_pipeline_results.csv", index=False)
comparison.to_csv("normalised_pipeline_comparison.csv")
print("\nSaved: normalised_pipeline_results.csv, normalised_pipeline_comparison.csv")
