"""
Variability Analysis: Did 40Hz light reduce between-animal variability?

1. Overall variance decreased from PRE to POST (within each Light group)
2. ISF reduced variance MORE than CTR (treatment-specific variance reduction)
3. The distribution of circadian metrics became more homogeneous after treatment

Methods:
  - Levene's test: compares variances between groups
  - Brown-Forsythe test: robust version of Levene's (uses median)
  - F-test for equality of variances: PRE vs POST within each Light group
  - Bartlett's test: compares variances across multiple groups
  - Bootstrap variance ratio: non-parametric CI for variance change
  - LME with heteroscedastic residuals: models different variances by group

References:
  Brown MB, Forsythe AB. Robust tests for the equality of variances.
  JASA. 1974;69(346):364-367.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

def bootstrap_var_ratio(pre, post, n_boot=5000, seed=42):
    """Bootstrap CI for variance ratio (POST/PRE)."""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        bp = rng.choice(pre, size=len(pre), replace=True)
        bq = rng.choice(post, size=len(post), replace=True)
        vp = bp.var(ddof=1)
        vq = bq.var(ddof=1)
        if vp > 0:
            ratios.append(vq / vp)
    ratios = np.array(ratios)
    return {
        "var_ratio": post.var(ddof=1) / pre.var(ddof=1) if pre.var(ddof=1) > 0 else np.nan,
        "ci_lo": np.percentile(ratios, 2.5),
        "ci_hi": np.percentile(ratios, 97.5),
        "pct_below_1": (ratios < 1).mean() * 100,  # % of bootstraps where POST < PRE variance
    }


# ============================================================
# Load data
# ============================================================

circ = clean_colnames(pd.read_csv("Circadian_raw.csv"))
circ["ID"] = pd.to_numeric(circ["ID"], errors="coerce").astype("Int64")
circ["PRE_POST"] = circ["PRE_POST"].astype(str)
circ["Light_new"] = circ["Light_new"].astype(str).str.strip()
circ["Age_new"] = circ["Age_new"].astype(str).str.strip()
circ["Sex_new"] = circ["Sex_new"].astype(str).str.strip()

# Standardise Light labels
circ.loc[circ["Light_new"] == "CNT", "Light_new"] = "CTR"

print("=" * 70)
print("VARIABILITY ANALYSIS")
print("Did 40Hz light reduce between-animal variability in circadian metrics?")
print("=" * 70)

metrics = ["IS", "IV", "RA", "Amplitude"]


# ============================================================
# ANALYSIS 1: Descriptive — variance by group
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 1: DESCRIPTIVE STATISTICS (variance by group)")
print("=" * 70)

desc_rows = []
for metric in metrics:
    for period in ["PRE", "POST"]:
        for light in ["CTR", "ISF"]:
            vals = circ[(circ["PRE_POST"] == period) & (circ["Light_new"] == light)][metric].dropna()
            desc_rows.append({
                "Metric": metric, "Period": period, "Light": light,
                "n": len(vals),
                "mean": vals.mean(),
                "sd": vals.std(ddof=1),
                "var": vals.var(ddof=1),
                "cv": vals.std(ddof=1) / vals.mean() * 100 if vals.mean() != 0 else np.nan,
                "iqr": vals.quantile(0.75) - vals.quantile(0.25),
            })

desc_df = pd.DataFrame(desc_rows)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(desc_df.to_string(index=False))


# ============================================================
# ANALYSIS 2: F-test for variance change (PRE vs POST)
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 2: F-TEST FOR VARIANCE CHANGE (PRE vs POST, per Light group)")
print("  Tests: Did variance decrease from PRE to POST?")
print("=" * 70)

ftest_rows = []
for metric in metrics:
    for light in ["CTR", "ISF"]:
        pre = circ[(circ["PRE_POST"] == "PRE") & (circ["Light_new"] == light)][metric].dropna()
        post = circ[(circ["PRE_POST"] == "POST") & (circ["Light_new"] == light)][metric].dropna()

        var_pre = pre.var(ddof=1)
        var_post = post.var(ddof=1)
        f_stat = var_pre / var_post if var_post > 0 else np.nan
        df1, df2 = len(pre) - 1, len(post) - 1
        # Two-sided p-value
        p_val = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

        ftest_rows.append({
            "Metric": metric, "Light": light,
            "var_PRE": var_pre, "var_POST": var_post,
            "F": f_stat, "p": p_val,
            "direction": "PRE > POST (variance decreased)" if var_pre > var_post else "POST > PRE (variance increased)",
        })

ftest_df = pd.DataFrame(ftest_rows)
print(ftest_df.to_string(index=False))


# ============================================================
# ANALYSIS 3: Brown-Forsythe test (robust Levene's)
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 3: BROWN-FORSYTHE TEST (robust test for equal variances)")
print("  Compares PRE vs POST variance within each Light group")
print("=" * 70)

bf_rows = []
for metric in metrics:
    for light in ["CTR", "ISF"]:
        pre = circ[(circ["PRE_POST"] == "PRE") & (circ["Light_new"] == light)][metric].dropna()
        post = circ[(circ["PRE_POST"] == "POST") & (circ["Light_new"] == light)][metric].dropna()

        stat, p = stats.levene(pre, post, center="median")  # Brown-Forsythe uses median
        bf_rows.append({
            "Metric": metric, "Light": light,
            "sd_PRE": pre.std(ddof=1), "sd_POST": post.std(ddof=1),
            "BF_stat": stat, "p": p,
        })

bf_df = pd.DataFrame(bf_rows)
print(bf_df.to_string(index=False))


# ============================================================
# ANALYSIS 4: Bootstrap variance ratio
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 4: BOOTSTRAP VARIANCE RATIO (POST/PRE)")
print("  Ratio < 1 means POST has less variance than PRE")
print("  95% CI not including 1 = significant variance change")
print("=" * 70)

boot_rows = []
for metric in metrics:
    for light in ["CTR", "ISF"]:
        pre = circ[(circ["PRE_POST"] == "PRE") & (circ["Light_new"] == light)][metric].dropna().values
        post = circ[(circ["PRE_POST"] == "POST") & (circ["Light_new"] == light)][metric].dropna().values

        br = bootstrap_var_ratio(pre, post)
        boot_rows.append({
            "Metric": metric, "Light": light,
            "var_ratio": br["var_ratio"],
            "ci_lo": br["ci_lo"], "ci_hi": br["ci_hi"],
            "includes_1": br["ci_lo"] <= 1 <= br["ci_hi"],
            "pct_POST_less": br["pct_below_1"],
        })

boot_df = pd.DataFrame(boot_rows)
print(boot_df.to_string(index=False))


# ============================================================
# ANALYSIS 5: Did ISF reduce variance MORE than CTR?
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 5: DID ISF REDUCE VARIANCE MORE THAN CTR?")
print("  Compare variance ratios between Light groups")
print("=" * 70)

for metric in metrics:
    ctr_pre = circ[(circ["PRE_POST"] == "PRE") & (circ["Light_new"] == "CTR")][metric].dropna().values
    ctr_post = circ[(circ["PRE_POST"] == "POST") & (circ["Light_new"] == "CTR")][metric].dropna().values
    isf_pre = circ[(circ["PRE_POST"] == "PRE") & (circ["Light_new"] == "ISF")][metric].dropna().values
    isf_post = circ[(circ["PRE_POST"] == "POST") & (circ["Light_new"] == "ISF")][metric].dropna().values

    # Compute absolute deviation from group median (Brown-Forsythe approach)
    ctr_pre_dev = np.abs(ctr_pre - np.median(ctr_pre))
    ctr_post_dev = np.abs(ctr_post - np.median(ctr_post))
    isf_pre_dev = np.abs(isf_pre - np.median(isf_pre))
    isf_post_dev = np.abs(isf_post - np.median(isf_post))

    # Change in dispersion: POST_dev - PRE_dev (negative = variance decreased)
    # We can't pair because PRE and POST may have different mice
    # So we compare the MEAN absolute deviation
    ctr_change = ctr_post_dev.mean() - ctr_pre_dev.mean()
    isf_change = isf_post_dev.mean() - isf_pre_dev.mean()

    # Bootstrap test: is ISF change more negative than CTR change?
    rng = np.random.default_rng(42)
    n_boot = 5000
    diff_boots = []
    for _ in range(n_boot):
        bc_pre = rng.choice(ctr_pre_dev, size=len(ctr_pre_dev), replace=True)
        bc_post = rng.choice(ctr_post_dev, size=len(ctr_post_dev), replace=True)
        bi_pre = rng.choice(isf_pre_dev, size=len(isf_pre_dev), replace=True)
        bi_post = rng.choice(isf_post_dev, size=len(isf_post_dev), replace=True)
        ctr_ch = bc_post.mean() - bc_pre.mean()
        isf_ch = bi_post.mean() - bi_pre.mean()
        diff_boots.append(isf_ch - ctr_ch)

    diff_boots = np.array(diff_boots)
    ci_lo, ci_hi = np.percentile(diff_boots, [2.5, 97.5])

    print(f"\n{metric}:")
    print(f"  CTR dispersion change (POST-PRE): {ctr_change:+.4f}")
    print(f"  ISF dispersion change (POST-PRE): {isf_change:+.4f}")
    print(f"  Difference (ISF - CTR):           {isf_change - ctr_change:+.4f}")
    print(f"  Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Includes 0: {ci_lo <= 0 <= ci_hi}")
    if ci_hi < 0:
        print(f"  --> ISF significantly reduced variance MORE than CTR")
    elif ci_lo > 0:
        print(f"  --> CTR reduced variance more than ISF")
    else:
        print(f"  --> No significant difference in variance change between groups")


# ============================================================
# ANALYSIS 6: Your PI's approach — separate LME residual variance
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 6: SEPARATE LME RESIDUAL VARIANCE (PI's approach, formalised)")
print("  Fits Metric ~ Light + Age + Sex separately for PRE and POST")
print("  Compares residual SD between the two models")
print("=" * 70)

lme_var_rows = []
for metric in metrics:
    for period in ["PRE", "POST"]:
        d = circ[circ["PRE_POST"] == period][["ID", "Light_new", "Age_new", "Sex_new", metric]].dropna().copy()
        if d["ID"].nunique() < 5:
            continue
        try:
            m = smf.ols(f"{metric} ~ C(Light_new) + C(Age_new) + C(Sex_new)", data=d).fit()
            lme_var_rows.append({
                "Metric": metric, "Period": period,
                "n": len(d), "residual_sd": np.sqrt(m.mse_resid),
                "r_squared": m.rsquared,
            })
        except Exception:
            pass

lme_var_df = pd.DataFrame(lme_var_rows)
print(lme_var_df.to_string(index=False))

# Formal comparison
print("\nResidual SD change (POST - PRE):")
for metric in metrics:
    pre_row = lme_var_df[(lme_var_df["Metric"] == metric) & (lme_var_df["Period"] == "PRE")]
    post_row = lme_var_df[(lme_var_df["Metric"] == metric) & (lme_var_df["Period"] == "POST")]
    if len(pre_row) > 0 and len(post_row) > 0:
        pre_sd = pre_row["residual_sd"].values[0]
        post_sd = post_row["residual_sd"].values[0]
        change = post_sd - pre_sd
        pct = (change / pre_sd) * 100
        print(f"  {metric}: PRE={pre_sd:.4f}, POST={post_sd:.4f}, "
              f"change={change:+.4f} ({pct:+.1f}%)")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""

Results above show:
- Whether variance changed from PRE to POST (Analyses 1-4)
- Whether ISF changed variance MORE than CTR (Analysis 5)
- The formal residual SD comparison (Analysis 6, = PI's approach)

If variance decreased equally in both CTR and ISF, it's likely a time/
habituation effect, not a treatment effect. If ISF showed a greater
reduction, that would support a treatment-specific variance effect.
""")

# Save
lme_var_df.to_csv("variability_residual_sd.csv", index=False)
boot_df.to_csv("variability_bootstrap_ratios.csv", index=False)
print("Saved: variability_residual_sd.csv, variability_bootstrap_ratios.csv")
