"""
Does Circadian Rhythm Predict Cognition?

Tests whether individual circadian metrics (IS, IV, RA, Amplitude) predict
behavioural outcomes (Barnes probe accuracy, learning slope, NOR DI).

Analyses:
1. Single-predictor models: each circadian metric → each behaviour outcome
   (controlling for Age + Sex)
2. Interaction models: does the combination of two circadian metrics
   predict behaviour better than either alone?
3. PRE vs POST vs delta: which timepoint of circadian measurement
   is most predictive?

All models use robust (HC3) standard errors.
FDR correction is applied within each analysis family.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats
from itertools import combinations


# ============================================================
# Helpers
# ============================================================

def clean_colnames(df):
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True).str.strip("_"))
    return df


# ============================================================
# Load and prepare data
# ============================================================

circ = clean_colnames(pd.read_csv("Circadian_raw.csv"))
barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))
nor = clean_colnames(pd.read_csv("UCBAge_Novel_clean.csv"))
if "Animal_ID" in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})

for df in (circ, barnes, nor):
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

barnes["Trial"] = pd.to_numeric(barnes["Trial"], errors="coerce")
for col in ["EntryZone_freq_new", "Hole_errors"]:
    barnes[col] = pd.to_numeric(barnes[col], errors="coerce")
barnes["total_pokes"] = barnes["EntryZone_freq_new"] + barnes["Hole_errors"]
barnes["probe_accuracy"] = np.where(
    barnes["total_pokes"] > 0,
    barnes["EntryZone_freq_new"] / barnes["total_pokes"], np.nan)

# --- Build mouse-level circadian table (PRE, POST, delta for each metric) ---
circ_metrics = ["IS", "IV", "RA", "Amplitude"]

mouse_circ = pd.DataFrame({"ID": circ["ID"].unique()}).set_index("ID")

for metric in circ_metrics:
    wide = circ.pivot_table(index="ID", columns="PRE_POST", values=metric, aggfunc="mean")
    if "PRE" in wide.columns:
        mouse_circ[f"{metric}_pre"] = wide["PRE"]
    if "POST" in wide.columns:
        mouse_circ[f"{metric}_post"] = wide["POST"]
    if "PRE" in wide.columns and "POST" in wide.columns:
        mouse_circ[f"delta_{metric}"] = wide["POST"] - wide["PRE"]

# Add Age, Sex, Light
covars = (circ.sort_values(["ID", "PRE_POST"])
          .groupby("ID")[["Age_new", "Sex_new", "Light_new"]]
          .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan))
mouse_circ = mouse_circ.join(covars)
mouse_circ = mouse_circ.reset_index()

# --- Build behaviour outcomes ---
# Barnes Trial 6 probe accuracy
bt6 = barnes[barnes["Trial"] == 6][["ID", "probe_accuracy"]].dropna().copy()
bt6 = bt6.rename(columns={"probe_accuracy": "barnes_accuracy"})

# Barnes learning slope (Trials 1-5)
slopes = []
for mid in barnes["ID"].unique():
    dm = barnes[(barnes["ID"] == mid) & (barnes["Trial"] <= 5)].dropna(subset=["probe_accuracy"])
    if len(dm) >= 3:
        slope, _, _, _, _ = stats.linregress(dm["Trial"], dm["probe_accuracy"])
        slopes.append({"ID": mid, "learning_slope": slope})
slope_df = pd.DataFrame(slopes)

# NOR DI
N_dur = pd.to_numeric(nor.get("N_obj_nose_duration_s"), errors="coerce")
F_dur = pd.to_numeric(nor.get("F_obj_nose_duration_s"), errors="coerce")
nor_di = nor[["ID"]].copy()
nor_di["nor_DI"] = ((N_dur - F_dur) / (N_dur + F_dur + 1e-9)).values
nor_di = nor_di.dropna()

# --- Merge everything into one mouse-level table ---
master = mouse_circ.copy()
master = master.merge(bt6, on="ID", how="left")
master = master.merge(slope_df, on="ID", how="left")
master = master.merge(nor_di, on="ID", how="left")

# Ensure types
master["Age_new"] = master["Age_new"].astype(str)
master["Sex_new"] = master["Sex_new"].astype(str)

print("=" * 70)
print("CIRCADIAN METRICS AS PREDICTORS OF BEHAVIOUR")
print("=" * 70)
print(f"Mice with circadian + Barnes accuracy: {master['barnes_accuracy'].notna().sum()}")
print(f"Mice with circadian + learning slope:  {master['learning_slope'].notna().sum()}")
print(f"Mice with circadian + NOR DI:          {master['nor_DI'].notna().sum()}")


# ============================================================
# Circadian predictor columns
# ============================================================

pre_predictors = [f"{m}_pre" for m in circ_metrics]
post_predictors = [f"{m}_post" for m in circ_metrics]
delta_predictors = [f"delta_{m}" for m in circ_metrics]
all_predictors = pre_predictors + post_predictors + delta_predictors

behaviour_outcomes = [
    ("barnes_accuracy", "Barnes T6 probe accuracy"),
    ("learning_slope", "Barnes learning slope (T1-5)"),
    ("nor_DI", "NOR Discrimination Index"),
]


# ============================================================
# ANALYSIS 1: Single circadian predictor → behaviour
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 1: SINGLE CIRCADIAN PREDICTOR → BEHAVIOUR")
print("  Model: behaviour ~ circadian_metric + Age + Sex (robust HC3)")
print("=" * 70)

single_results = []

for outcome, outcome_label in behaviour_outcomes:
    print(f"\n--- {outcome_label} ---")
    print(f"  {'Predictor':<20s} {'beta':>8s} {'SE':>8s} {'p':>8s} {'R²':>6s}")
    print("  " + "-" * 55)

    for pred in all_predictors:
        d = master[["ID", outcome, pred, "Age_new", "Sex_new"]].dropna()
        if len(d) < 15:
            continue

        m = smf.ols(
            f"{outcome} ~ {pred} + C(Age_new) + C(Sex_new)",
            data=d
        ).fit(cov_type="HC3")

        p = float(m.pvalues.get(pred, np.nan))
        beta = float(m.params.get(pred, np.nan))
        se = float(m.bse.get(pred, np.nan))
        r2 = m.rsquared
        sig = " *" if p < 0.05 else ""

        single_results.append({
            "outcome": outcome, "outcome_label": outcome_label,
            "predictor": pred, "beta": beta, "se": se, "p": p, "r2": r2,
        })
        print(f"  {pred:<20s} {beta:>+8.4f} {se:>8.4f} {p:>8.4f} {r2:>6.3f}{sig}")

# FDR correction within each outcome
sr_df = pd.DataFrame(single_results)
for outcome, _ in behaviour_outcomes:
    mask = sr_df["outcome"] == outcome
    if mask.sum() > 0:
        _, fdr, _, _ = multipletests(sr_df.loc[mask, "p"], method="fdr_bh")
        sr_df.loc[mask, "p_fdr"] = fdr

print("\n--- FDR-significant results (single predictors) ---")
sig_single = sr_df[sr_df["p_fdr"] < 0.05]
if len(sig_single) > 0:
    print(sig_single[["outcome_label", "predictor", "beta", "p", "p_fdr", "r2"]].to_string(index=False))
else:
    print("  No single circadian predictor survived FDR correction.")

# Show best predictor per outcome (by raw p)
print("\n--- Best single predictor per outcome (lowest p) ---")
for outcome, label in behaviour_outcomes:
    sub = sr_df[sr_df["outcome"] == outcome].sort_values("p")
    if len(sub) > 0:
        best = sub.iloc[0]
        print(f"  {label}: {best['predictor']} (beta={best['beta']:+.4f}, p={best['p']:.4f}, R²={best['r2']:.3f})")


# ============================================================
# ANALYSIS 2: Interaction of two circadian metrics → behaviour
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 2: CIRCADIAN METRIC INTERACTIONS → BEHAVIOUR")
print("  Model: behaviour ~ metric1 * metric2 + Age + Sex (robust HC3)")
print("  Tests whether the COMBINATION of two metrics predicts better")
print("=" * 70)

# Use PRE metrics for interactions (baseline predictors, no treatment contamination)
interaction_results = []

for outcome, outcome_label in behaviour_outcomes:
    print(f"\n--- {outcome_label} ---")
    print(f"  {'Metric1':<15s} {'Metric2':<15s} {'interaction_beta':>16s} {'inter_p':>8s} {'model_R²':>9s} {'base_R²':>9s} {'ΔR²':>6s}")
    print("  " + "-" * 85)

    for m1, m2 in combinations(pre_predictors, 2):
        d = master[["ID", outcome, m1, m2, "Age_new", "Sex_new"]].dropna()
        if len(d) < 20:
            continue

        # Base model (additive)
        m_base = smf.ols(
            f"{outcome} ~ {m1} + {m2} + C(Age_new) + C(Sex_new)",
            data=d
        ).fit(cov_type="HC3")

        # Interaction model
        m_int = smf.ols(
            f"{outcome} ~ {m1} * {m2} + C(Age_new) + C(Sex_new)",
            data=d
        ).fit(cov_type="HC3")

        # Find interaction term
        int_term = f"{m1}:{m2}"
        if int_term not in m_int.pvalues.index:
            # Try reverse order
            int_term = f"{m2}:{m1}"
        if int_term not in m_int.pvalues.index:
            continue

        p_int = float(m_int.pvalues[int_term])
        beta_int = float(m_int.params[int_term])
        r2_base = m_base.rsquared
        r2_int = m_int.rsquared
        delta_r2 = r2_int - r2_base
        sig = " *" if p_int < 0.05 else ""

        interaction_results.append({
            "outcome": outcome, "outcome_label": outcome_label,
            "metric1": m1, "metric2": m2,
            "interaction_beta": beta_int, "interaction_p": p_int,
            "base_r2": r2_base, "interaction_r2": r2_int, "delta_r2": delta_r2,
        })
        print(f"  {m1:<15s} {m2:<15s} {beta_int:>+16.4f} {p_int:>8.4f} {r2_int:>9.3f} {r2_base:>9.3f} {delta_r2:>+6.3f}{sig}")

    # Also test POST and delta interactions
    for pred_set, pred_label in [(post_predictors, "POST"), (delta_predictors, "delta")]:
        for m1, m2 in combinations(pred_set, 2):
            d = master[["ID", outcome, m1, m2, "Age_new", "Sex_new"]].dropna()
            if len(d) < 20:
                continue
            m_base = smf.ols(f"{outcome} ~ {m1} + {m2} + C(Age_new) + C(Sex_new)", data=d).fit(cov_type="HC3")
            m_int = smf.ols(f"{outcome} ~ {m1} * {m2} + C(Age_new) + C(Sex_new)", data=d).fit(cov_type="HC3")
            int_term = f"{m1}:{m2}"
            if int_term not in m_int.pvalues.index:
                int_term = f"{m2}:{m1}"
            if int_term not in m_int.pvalues.index:
                continue
            p_int = float(m_int.pvalues[int_term])
            beta_int = float(m_int.params[int_term])
            delta_r2 = m_int.rsquared - m_base.rsquared
            sig = " *" if p_int < 0.05 else ""
            interaction_results.append({
                "outcome": outcome, "outcome_label": outcome_label,
                "metric1": m1, "metric2": m2,
                "interaction_beta": beta_int, "interaction_p": p_int,
                "base_r2": m_base.rsquared, "interaction_r2": m_int.rsquared,
                "delta_r2": delta_r2,
            })
            if p_int < 0.05:
                print(f"  {m1:<15s} {m2:<15s} {beta_int:>+16.4f} {p_int:>8.4f} {m_int.rsquared:>9.3f} {m_base.rsquared:>9.3f} {delta_r2:>+6.3f}{sig}")

# FDR correction
ir_df = pd.DataFrame(interaction_results)
if len(ir_df) > 0:
    for outcome, _ in behaviour_outcomes:
        mask = ir_df["outcome"] == outcome
        if mask.sum() > 0:
            _, fdr, _, _ = multipletests(ir_df.loc[mask, "interaction_p"], method="fdr_bh")
            ir_df.loc[mask, "p_fdr"] = fdr

    print("\n--- FDR-significant interactions ---")
    sig_int = ir_df[ir_df["p_fdr"] < 0.05]
    if len(sig_int) > 0:
        print(sig_int[["outcome_label", "metric1", "metric2", "interaction_beta",
                        "interaction_p", "p_fdr", "delta_r2"]].to_string(index=False))
    else:
        print("  No circadian metric interactions survived FDR correction.")

    # Show best interaction per outcome
    print("\n--- Best interaction per outcome (lowest p, uncorrected) ---")
    for outcome, label in behaviour_outcomes:
        sub = ir_df[ir_df["outcome"] == outcome].sort_values("interaction_p")
        if len(sub) > 0:
            best = sub.iloc[0]
            print(f"  {label}: {best['metric1']} x {best['metric2']} "
                  f"(p={best['interaction_p']:.4f}, ΔR²={best['delta_r2']:+.3f})")


# ============================================================
# ANALYSIS 3: Comparison of PRE vs POST vs delta as predictors
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 3: WHICH TIMEPOINT IS MOST PREDICTIVE?")
print("  Comparing PRE, POST, and delta versions of each metric")
print("=" * 70)

for metric in circ_metrics:
    print(f"\n--- {metric} ---")
    for outcome, label in behaviour_outcomes:
        print(f"  {label}:")
        for version, suffix in [("PRE", "_pre"), ("POST", "_post"), ("delta", "delta_")]:
            pred = f"{suffix.lstrip('_')}{metric}" if version == "delta" else f"{metric}{suffix}"
            if pred not in master.columns:
                # Try alternate naming
                pred = f"delta_{metric}" if version == "delta" else pred
            if pred not in master.columns:
                continue
            d = master[["ID", outcome, pred, "Age_new", "Sex_new"]].dropna()
            if len(d) < 15:
                continue
            m = smf.ols(f"{outcome} ~ {pred} + C(Age_new) + C(Sex_new)", data=d).fit(cov_type="HC3")
            p = float(m.pvalues.get(pred, np.nan))
            beta = float(m.params.get(pred, np.nan))
            sig = " *" if p < 0.05 else ""
            print(f"    {version:>6s}: beta={beta:>+.4f}, p={p:.4f}, R²={m.rsquared:.3f}{sig}")


# ============================================================
# ANALYSIS 4: Full multi-predictor model
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 4: FULL MODEL — ALL PRE CIRCADIAN METRICS TOGETHER")
print("  Model: behaviour ~ IS_pre + IV_pre + RA_pre + Amp_pre + Age + Sex")
print("=" * 70)

for outcome, label in behaviour_outcomes:
    preds = [p for p in pre_predictors if p in master.columns]
    d = master[["ID", outcome] + preds + ["Age_new", "Sex_new"]].dropna()
    if len(d) < 20:
        continue

    formula = f"{outcome} ~ {' + '.join(preds)} + C(Age_new) + C(Sex_new)"
    m = smf.ols(formula, data=d).fit(cov_type="HC3")

    print(f"\n--- {label} (n={len(d)}) ---")
    print(f"  R² = {m.rsquared:.3f}, Adj R² = {m.rsquared_adj:.3f}, F = {m.fvalue:.2f}, p = {m.f_pvalue:.4f}")
    for pred in preds:
        sig = " *" if m.pvalues[pred] < 0.05 else ""
        print(f"  {pred:<20s} beta={m.params[pred]:>+.4f}, p={m.pvalues[pred]:.4f}{sig}")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

sr_df.to_csv("circadian_behaviour_single.csv", index=False)
ir_df.to_csv("circadian_behaviour_interactions.csv", index=False)
print("\nSaved: circadian_behaviour_single.csv, circadian_behaviour_interactions.csv")
