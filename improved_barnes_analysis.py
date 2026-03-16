"""
Improved Barnes Maze Analysis for Publication

Three improvements over the original analysis.py:

1. PROBE ACCURACY as primary endpoint
   Uses EntryZone_freq_new / (EntryZone_freq_new + Hole_errors) instead of
   Goal_Box_feq_new / (Goal_Box_feq_new + Hole_errors). Goal_Box_feq_new is
   near-zero for almost all mice at Trial 6 (max=1), creating a floor effect.
   EntryZone_freq_new captures entries to the correct zone (mean ~3.6, range 0-16)
   giving real variance for modelling.

2. LEARNING CURVE across all 6 trials
   Mixed-effects model with Trial as a continuous predictor, testing whether
   40Hz light accelerates learning (Trial x Light interaction).
   This uses ALL data (6 trials x 90 mice = ~530 observations) rather than
   just Trial 6, dramatically increasing statistical power.

3. EFFECT SIZES and POWER ANALYSIS
   Reports Cohen's d for all Light group comparisons and post-hoc power
   for detecting medium effects (d=0.5) given the sample sizes.

Author: Generated for Kriegsfeld Lab analysis pipeline
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats


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

def cohens_d(group1, group2):
    """Compute Cohen's d (pooled SD denominator)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_sd)

def post_hoc_power(n1, n2, d, alpha=0.05):
    """
    Approximate post-hoc power for two-sample t-test.
    Uses the non-central t distribution.
    """
    from scipy.stats import nct, t as t_dist
    df = n1 + n2 - 2
    ncp = d * np.sqrt(n1 * n2 / (n1 + n2))  # non-centrality parameter
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)
    return float(power)


# ============================================================
# Load data
# ============================================================

circ = clean_colnames(pd.read_csv("Circadian_raw.csv"))
barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))
nor = clean_colnames(pd.read_csv("UCBAge_Novel_clean.csv"))

if "Animal_ID" in nor.columns and "ID" not in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})

for df in (circ, barnes, nor):
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

barnes = set_cats(barnes, "Light_new", ["CTR", "ISF"])
barnes["Age_new"] = barnes["Age_new"].astype("category")
barnes["Sex_new"] = barnes["Sex_new"].astype("category")
barnes["Trial"] = pd.to_numeric(barnes["Trial"], errors="coerce")

# Build mouse-level circadian predictors
wide_IS = circ.pivot_table(index="ID", columns="PRE_POST", values="IS", aggfunc="mean")
IS_pre = wide_IS.get("PRE").rename("IS_pre")
delta_IS = (wide_IS.get("POST") - wide_IS.get("PRE")).rename("delta_IS")
mouse_covars = (
    circ.sort_values(["ID", "PRE_POST"])
    .groupby("ID")[["Light_new", "Age_new", "Sex_new"]]
    .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
)
circ_mouse = pd.concat([mouse_covars, IS_pre, delta_IS], axis=1).reset_index()

# Build probe accuracy for all trials
barnes["EntryZone_freq_new"] = pd.to_numeric(barnes["EntryZone_freq_new"], errors="coerce")
barnes["Hole_errors"] = pd.to_numeric(barnes["Hole_errors"], errors="coerce")
barnes["total_pokes"] = barnes["EntryZone_freq_new"] + barnes["Hole_errors"]
barnes["probe_accuracy"] = np.where(
    barnes["total_pokes"] > 0,
    barnes["EntryZone_freq_new"] / barnes["total_pokes"],
    np.nan
)

# Merge circadian predictors
barnes_m = barnes.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")

print("=" * 70)
print("IMPROVED BARNES MAZE ANALYSIS")
print("=" * 70)


# ============================================================
# ANALYSIS 1: Probe Accuracy at Trial 6 (replacing p_correct)
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 1: PROBE ACCURACY AT TRIAL 6")
print("  Endpoint: EntryZone_freq_new / (EntryZone_freq_new + Hole_errors)")
print("  This replaces Goal_Box_feq_new which was at floor (max=1 at T6)")
print("=" * 70)

bt6 = barnes_m[barnes_m["Trial"] == 6].copy()
bt6 = bt6.dropna(subset=["probe_accuracy", "Light_new", "Age_new", "Sex_new"]).copy()
bt6["IS_pre"] = pd.to_numeric(bt6["IS_pre"], errors="coerce")
bt6["delta_IS"] = pd.to_numeric(bt6["delta_IS"], errors="coerce")

print(f"\nTrial 6 mice: {bt6['ID'].nunique()}")
print(f"\nProbe accuracy descriptives:")
print(bt6.groupby("Light_new")["probe_accuracy"].describe().round(4).to_string())

# 1a. Primary model: Binomial GLM with robust SEs
bt6_model = bt6.dropna(subset=["IS_pre", "delta_IS"]).copy()
bt6_model["Light_new"] = bt6_model["Light_new"].astype(str)
bt6_model["Age_new"] = bt6_model["Age_new"].astype(str)
bt6_model["Sex_new"] = bt6_model["Sex_new"].astype(str)

glm_fit = smf.glm(
    "probe_accuracy ~ C(Light_new, Treatment('CTR')) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    data=bt6_model,
    family=sm.families.Binomial(),
    freq_weights=bt6_model["total_pokes"]
).fit(cov_type="HC3")

print("\n--- Binomial GLM (robust HC3) ---")
print(glm_fit.summary())

# 1b. Also fit robust OLS as a sensitivity check (more interpretable)
ols_fit = smf.ols(
    "probe_accuracy ~ C(Light_new, Treatment('CTR')) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    data=bt6_model
).fit(cov_type="HC3")

print("\n--- Robust OLS (sensitivity check) ---")
print(ols_fit.summary())

# 1c. Non-parametric confirmation
ctr = bt6[bt6["Light_new"] == "CTR"]["probe_accuracy"].dropna()
isf = bt6[bt6["Light_new"] == "ISF"]["probe_accuracy"].dropna()
u_stat, u_p = stats.mannwhitneyu(ctr, isf, alternative="two-sided")
print(f"\nMann-Whitney U: U={u_stat:.1f}, p={u_p:.4f}")
print(f"  CTR median={ctr.median():.4f}, ISF median={isf.median():.4f}")


# ============================================================
# ANALYSIS 2: LEARNING CURVE ACROSS ALL 6 TRIALS
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 2: LEARNING CURVE (TRIALS 1-6)")
print("  Mixed-effects model with random intercept + slope per mouse")
print("  Tests whether ISF accelerates learning (Trial x Light interaction)")
print("=" * 70)

lc = barnes_m.dropna(subset=["probe_accuracy", "Light_new", "Age_new", "Sex_new", "Trial"]).copy()
lc["Light_new"] = lc["Light_new"].astype(str)
lc["Age_new"] = lc["Age_new"].astype(str)
lc["Sex_new"] = lc["Sex_new"].astype(str)

print(f"\nTotal observations: {len(lc)} ({lc['ID'].nunique()} mice x up to 6 trials)")

# Mean probe accuracy by Trial x Light
print("\nMean probe accuracy by Trial and Light group:")
pivot = lc.groupby(["Trial", "Light_new"])["probe_accuracy"].agg(["mean", "sem", "count"])
print(pivot.round(4).to_string())

# 2a. Random intercept model (simpler, more stable)
lme_ri = smf.mixedlm(
    "probe_accuracy ~ Trial * C(Light_new, Treatment('CTR')) + C(Age_new) + C(Sex_new)",
    data=lc,
    groups=lc["ID"]
).fit(method="lbfgs", reml=True)

print("\n--- Learning Curve LME (random intercept) ---")
print(lme_ri.summary())

# Extract key terms
for term in lme_ri.pvalues.index:
    if "Trial" in term:
        print(f"  {term}: beta={lme_ri.params[term]:.4f}, p={lme_ri.pvalues[term]:.4f}")

# 2b. Random intercept + slope model (if it converges)
try:
    lme_rs = smf.mixedlm(
        "probe_accuracy ~ Trial * C(Light_new, Treatment('CTR')) + C(Age_new) + C(Sex_new)",
        data=lc,
        groups=lc["ID"],
        re_formula="~Trial"
    ).fit(method="lbfgs", reml=True)

    print("\n--- Learning Curve LME (random intercept + slope) ---")
    print(f"AIC: RI={lme_ri.aic:.1f}, RI+RS={lme_rs.aic:.1f}")
    if lme_rs.aic < lme_ri.aic:
        print("Random slope model preferred (lower AIC).")
        print(lme_rs.summary())
        lme_final = lme_rs
    else:
        print("Random intercept model preferred (lower AIC).")
        lme_final = lme_ri
except Exception as e:
    print(f"\nRandom slope model did not converge: {e}")
    print("Using random intercept model.")
    lme_final = lme_ri

# 2c. Age effect on learning
print("\n--- Age effect on learning ---")
for term in lme_final.pvalues.index:
    if "Age" in term:
        print(f"  {term}: beta={lme_final.params[term]:.4f}, p={lme_final.pvalues[term]:.4f}")

# 2d. Per-trial Light comparison (descriptive)
print("\n--- Per-trial Light group comparison (Mann-Whitney) ---")
trial_tests = []
for t in sorted(lc["Trial"].unique()):
    dt = lc[lc["Trial"] == t]
    c = dt[dt["Light_new"] == "CTR"]["probe_accuracy"].dropna()
    i = dt[dt["Light_new"] == "ISF"]["probe_accuracy"].dropna()
    if len(c) > 2 and len(i) > 2:
        u, p = stats.mannwhitneyu(c, i, alternative="two-sided")
        trial_tests.append({"Trial": t, "CTR_mean": c.mean(), "ISF_mean": i.mean(),
                           "U": u, "p": p})
tt_df = pd.DataFrame(trial_tests)
if len(tt_df) > 0:
    _, tt_df["p_fdr"], _, _ = multipletests(tt_df["p"], method="fdr_bh", alpha=0.05)
    print(tt_df.round(4).to_string(index=False))


# ============================================================
# ANALYSIS 3: EFFECT SIZES AND POWER
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 3: EFFECT SIZES AND POST-HOC POWER")
print("=" * 70)

effect_rows = []

# --- Circadian metrics ---
for metric in ["IS", "IV", "RA", "Amplitude"]:
    for period in ["PRE", "POST"]:
        d = circ[(circ["PRE_POST"] == period)].copy()
        c = d[d["Light_new"] == "CTR"][metric].dropna()
        i = d[d["Light_new"] == "ISF"][metric].dropna()
        if len(c) > 2 and len(i) > 2:
            d_val = cohens_d(i, c)
            pwr = post_hoc_power(len(c), len(i), abs(d_val))
            pwr_medium = post_hoc_power(len(c), len(i), 0.5)
            effect_rows.append({
                "Outcome": f"Circadian {metric} ({period})",
                "n_CTR": len(c), "n_ISF": len(i),
                "mean_CTR": c.mean(), "mean_ISF": i.mean(),
                "Cohen_d": d_val,
                "Power_observed": pwr,
                "Power_d05": pwr_medium,
            })

# --- Circadian delta_IS ---
c = circ_mouse[circ_mouse["Light_new"] == "CTR"]["delta_IS"].dropna() if "delta_IS" in circ_mouse.columns else pd.Series()
i = circ_mouse[circ_mouse["Light_new"] == "ISF"]["delta_IS"].dropna() if "delta_IS" in circ_mouse.columns else pd.Series()
if len(c) > 2 and len(i) > 2:
    d_val = cohens_d(i, c)
    pwr = post_hoc_power(len(c), len(i), abs(d_val))
    pwr_medium = post_hoc_power(len(c), len(i), 0.5)
    effect_rows.append({
        "Outcome": "delta_IS (POST-PRE)",
        "n_CTR": len(c), "n_ISF": len(i),
        "mean_CTR": c.mean(), "mean_ISF": i.mean(),
        "Cohen_d": d_val,
        "Power_observed": pwr,
        "Power_d05": pwr_medium,
    })

# --- Barnes Trial 6 probe accuracy ---
c = bt6[bt6["Light_new"] == "CTR"]["probe_accuracy"].dropna()
i = bt6[bt6["Light_new"] == "ISF"]["probe_accuracy"].dropna()
d_val = cohens_d(i, c)
pwr = post_hoc_power(len(c), len(i), abs(d_val))
pwr_medium = post_hoc_power(len(c), len(i), 0.5)
effect_rows.append({
    "Outcome": "Barnes T6 probe accuracy",
    "n_CTR": len(c), "n_ISF": len(i),
    "mean_CTR": c.mean(), "mean_ISF": i.mean(),
    "Cohen_d": d_val,
    "Power_observed": pwr,
    "Power_d05": pwr_medium,
})

# --- Barnes Trial 6 other endpoints ---
for col_name, col in [("Hole_errors", "Hole_errors"), ("DistanceMoved_cm", "DistanceMoved_cm"),
                       ("Goal_Box_latency_new", "Goal_Box_latency_new")]:
    if col in bt6.columns:
        c = pd.to_numeric(bt6[bt6["Light_new"] == "CTR"][col], errors="coerce").dropna()
        i = pd.to_numeric(bt6[bt6["Light_new"] == "ISF"][col], errors="coerce").dropna()
        if len(c) > 2 and len(i) > 2:
            d_val = cohens_d(i, c)
            pwr = post_hoc_power(len(c), len(i), abs(d_val))
            pwr_medium = post_hoc_power(len(c), len(i), 0.5)
            effect_rows.append({
                "Outcome": f"Barnes T6 {col_name}",
                "n_CTR": len(c), "n_ISF": len(i),
                "mean_CTR": c.mean(), "mean_ISF": i.mean(),
                "Cohen_d": d_val,
                "Power_observed": pwr,
                "Power_d05": pwr_medium,
            })

# --- NOR ---
nor_m = nor.copy()
N_dur = pd.to_numeric(nor_m.get("N_obj_nose_duration_s"), errors="coerce")
F_dur = pd.to_numeric(nor_m.get("F_obj_nose_duration_s"), errors="coerce")
nor_m = nor_m[(N_dur.notna()) & (F_dur.notna())].copy()
N_dur = N_dur.loc[nor_m.index]
F_dur = F_dur.loc[nor_m.index]
nor_m["DI"] = (N_dur - F_dur) / (N_dur + F_dur + 1e-9)

c = nor_m[nor_m["Light_new"] == "CTR"]["DI"].dropna()
i = nor_m[nor_m["Light_new"] == "ISF"]["DI"].dropna()
if len(c) > 2 and len(i) > 2:
    d_val = cohens_d(i, c)
    pwr = post_hoc_power(len(c), len(i), abs(d_val))
    pwr_medium = post_hoc_power(len(c), len(i), 0.5)
    effect_rows.append({
        "Outcome": "NOR DI",
        "n_CTR": len(c), "n_ISF": len(i),
        "mean_CTR": c.mean(), "mean_ISF": i.mean(),
        "Cohen_d": d_val,
        "Power_observed": pwr,
        "Power_d05": pwr_medium,
    })

effect_df = pd.DataFrame(effect_rows)

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("\n--- Effect Sizes and Power (ISF vs CTR) ---")
print(effect_df.to_string(index=False))

# Summary statistics
print(f"\nMedian |Cohen's d| across all outcomes: {effect_df['Cohen_d'].abs().median():.3f}")
print(f"Mean power to detect observed effects: {effect_df['Power_observed'].mean():.3f}")
print(f"Mean power to detect d=0.5 (medium): {effect_df['Power_d05'].mean():.3f}")

# Interpretation
max_d = effect_df.loc[effect_df["Cohen_d"].abs().idxmax()]
print(f"\nLargest effect: {max_d['Outcome']} (d={max_d['Cohen_d']:.3f})")
underpowered = effect_df[effect_df["Power_d05"] < 0.80]
if len(underpowered) > 0:
    print(f"\nOutcomes underpowered to detect d=0.5: {len(underpowered)}/{len(effect_df)}")
else:
    print(f"\nAll outcomes adequately powered (>80%) to detect d=0.5")

# Save
effect_df.to_csv("effect_sizes_and_power.csv", index=False)
print("\nSaved: effect_sizes_and_power.csv")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
