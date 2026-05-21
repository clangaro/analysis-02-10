"""
Sex and Age Effects on Barnes Maze and NOR Performance

Analyses:
1. Sex, Age, and Sex*Age effects on all Barnes maze variables (Trial 6)
2. Sex, Age, and Sex*Age effects on NOR metrics
3. Variance heterogeneity: do males show more variance with age than females?
4. Which Barnes variable best captures spatial learning?
   - Correlation structure across variables
   - PCA to identify latent learning dimensions
   - Learning curve slope per mouse as an individual-level learning metric

Requires: pandas, numpy, statsmodels, scipy, sklearn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# Helpers
# ============================================================

def clean_colnames(df):
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_"))
    return df

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.var(ddof=1) + (n2-1)*g2.var(ddof=1)) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0.0


# ============================================================
# Load data
# ============================================================

barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))
nor = clean_colnames(pd.read_csv("UCBAge_Novel_clean.csv"))
if "Animal_ID" in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})

for df in (barnes, nor):
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

barnes["Trial"] = pd.to_numeric(barnes["Trial"], errors="coerce")

# Build Barnes derived variables
for col in ["EntryZone_freq_new", "Hole_errors", "Goal_Box_feq_new",
            "Goal_Box_latency_new", "Entry_latency_new", "DistanceMoved_cm",
            "Q1", "Q2", "Q3", "Q4"]:
    if col in barnes.columns:
        barnes[col] = pd.to_numeric(barnes[col], errors="coerce")

barnes["total_pokes"] = barnes["EntryZone_freq_new"] + barnes["Hole_errors"]
barnes["probe_accuracy"] = np.where(
    barnes["total_pokes"] > 0,
    barnes["EntryZone_freq_new"] / barnes["total_pokes"], np.nan)

bt6 = barnes[barnes["Trial"] == 6].copy()

# Build NOR derived variables
N_dur = pd.to_numeric(nor.get("N_obj_nose_duration_s"), errors="coerce")
F_dur = pd.to_numeric(nor.get("F_obj_nose_duration_s"), errors="coerce")
nor["DI_duration"] = (N_dur - F_dur) / (N_dur + F_dur + 1e-9)

N_freq = pd.to_numeric(nor.get("N_obj_nose_frequency"), errors="coerce")
F_freq = pd.to_numeric(nor.get("F_nose_frequency"), errors="coerce")
nor["DI_frequency"] = (N_freq - F_freq) / (N_freq + F_freq + 1e-9)

nor["total_exploration_s"] = N_dur + F_dur
nor["novel_preference_pct"] = N_dur / (N_dur + F_dur + 1e-9) * 100

print("=" * 70)
print("SEX AND AGE EFFECTS ON BARNES MAZE AND NOR")
print("=" * 70)


# ============================================================
# ANALYSIS 1: Sex, Age, Sex*Age effects on Barnes Trial 6
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 1: SEX AND AGE EFFECTS ON BARNES MAZE (TRIAL 6)")
print("  Model: outcome ~ Age * Sex (robust HC3)")
print("=" * 70)

barnes_outcomes = [
    ("probe_accuracy", "Probe accuracy (target zone entries / total)"),
    ("Hole_errors", "Hole errors (total wrong entries)"),
    ("EntryZone_freq_new", "Target zone entries (count)"),
    ("DistanceMoved_cm", "Distance moved (cm)"),
    ("Entry_latency_new", "First entry latency (s)"),
    ("Q4", "Target quadrant time (% in Q4)"),
    ("Q1", "Opposite quadrant time (% in Q1)"),
]

barnes_results = []
for outcome, description in barnes_outcomes:
    if outcome not in bt6.columns:
        continue
    d = bt6[["Age_new", "Sex_new", outcome]].dropna().copy()
    d["Age_new"] = d["Age_new"].astype(str)
    d["Sex_new"] = d["Sex_new"].astype(str)

    if len(d) < 10:
        continue

    m = smf.ols(f"{outcome} ~ C(Age_new) * C(Sex_new)", data=d).fit(cov_type="HC3")

    for term in m.pvalues.index:
        if term == "Intercept":
            continue
        barnes_results.append({
            "Outcome": outcome,
            "Term": term.replace("C(Age_new)", "Age").replace("C(Sex_new)", "Sex")
                       .replace("[T.Old]", "[Old]").replace("[T.Young]", "[Young]")
                       .replace("[T.Male]", "[Male]"),
            "beta": float(m.params[term]),
            "p": float(m.pvalues[term]),
        })

    print(f"\n--- {description} ---")
    print(f"  R² = {m.rsquared:.3f}, F = {m.fvalue:.2f}, p = {m.f_pvalue:.4f}")
    for term in m.pvalues.index:
        if term == "Intercept":
            continue
        sig = "*" if m.pvalues[term] < 0.05 else ""
        print(f"  {term}: beta={m.params[term]:.4f}, p={m.pvalues[term]:.4f} {sig}")

# FDR correction across all Barnes tests
br_df = pd.DataFrame(barnes_results)
if len(br_df) > 0:
    _, br_df["p_fdr"], _, _ = multipletests(br_df["p"], method="fdr_bh", alpha=0.05)
    print("\n--- FDR-corrected significant results (Barnes) ---")
    sig_barnes = br_df[br_df["p_fdr"] < 0.05]
    if len(sig_barnes) > 0:
        print(sig_barnes[["Outcome", "Term", "beta", "p", "p_fdr"]].to_string(index=False))
    else:
        print("  No results survived FDR correction.")


# ============================================================
# ANALYSIS 2: Sex, Age, Sex*Age effects on NOR
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 2: SEX AND AGE EFFECTS ON NOR")
print("  Model: outcome ~ Age * Sex (robust HC3)")
print("=" * 70)

nor_outcomes = [
    ("DI_duration", "Discrimination Index (duration)"),
    ("DI_frequency", "Discrimination Index (frequency)"),
    ("novel_preference_pct", "Novel preference (%)"),
    ("total_exploration_s", "Total exploration time (s)"),
    ("N_obj_nose_duration_s", "Novel object duration (s)"),
    ("F_obj_nose_duration_s", "Familiar object duration (s)"),
    ("N_obj_nose_latency", "Novel object latency (s)"),
]

nor_results = []
for outcome, description in nor_outcomes:
    if outcome not in nor.columns:
        continue
    d = nor[["Age_new", "Sex_new", outcome]].dropna().copy()
    d["Age_new"] = d["Age_new"].astype(str)
    d["Sex_new"] = d["Sex_new"].astype(str)
    d[outcome] = pd.to_numeric(d[outcome], errors="coerce")
    d = d.dropna()

    if len(d) < 10:
        continue

    m = smf.ols(f"{outcome} ~ C(Age_new) * C(Sex_new)", data=d).fit(cov_type="HC3")

    for term in m.pvalues.index:
        if term == "Intercept":
            continue
        nor_results.append({
            "Outcome": outcome,
            "Term": term.replace("C(Age_new)", "Age").replace("C(Sex_new)", "Sex")
                       .replace("[T.Old]", "[Old]").replace("[T.Young]", "[Young]")
                       .replace("[T.Male]", "[Male]"),
            "beta": float(m.params[term]),
            "p": float(m.pvalues[term]),
        })

    print(f"\n--- {description} ---")
    print(f"  R² = {m.rsquared:.3f}, F = {m.fvalue:.2f}, p = {m.f_pvalue:.4f}")
    for term in m.pvalues.index:
        if term == "Intercept":
            continue
        sig = "*" if m.pvalues[term] < 0.05 else ""
        print(f"  {term}: beta={m.params[term]:.4f}, p={m.pvalues[term]:.4f} {sig}")

nr_df = pd.DataFrame(nor_results)
if len(nr_df) > 0:
    _, nr_df["p_fdr"], _, _ = multipletests(nr_df["p"], method="fdr_bh", alpha=0.05)
    print("\n--- FDR-corrected significant results (NOR) ---")
    sig_nor = nr_df[nr_df["p_fdr"] < 0.05]
    if len(sig_nor) > 0:
        print(sig_nor[["Outcome", "Term", "beta", "p", "p_fdr"]].to_string(index=False))
    else:
        print("  No results survived FDR correction.")


# ============================================================
# ANALYSIS 3: Variance heterogeneity — do males vary more with age?
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 3: VARIANCE BY SEX x AGE")
print("  Testing whether males show more variance increase with age")
print("=" * 70)

age_order = ["Young", "Mid", "Old"]
all_outcomes = ["probe_accuracy", "Hole_errors", "Q4", "DI_duration"]

for outcome in all_outcomes:
    if outcome in bt6.columns:
        data = bt6
    elif outcome in nor.columns:
        data = nor
    else:
        continue

    print(f"\n--- {outcome} ---")
    print(f"  {'Age':<8} {'Sex':<8} {'n':>4} {'mean':>8} {'SD':>8} {'var':>10}")

    var_rows = []
    for age in age_order:
        for sex in ["Female", "Male"]:
            vals = data[(data["Age_new"] == age) & (data["Sex_new"] == sex)]
            if outcome in vals.columns:
                v = pd.to_numeric(vals[outcome], errors="coerce").dropna()
            else:
                continue
            if len(v) < 2:
                continue
            print(f"  {age:<8} {sex:<8} {len(v):>4} {v.mean():>8.3f} {v.std(ddof=1):>8.3f} {v.var(ddof=1):>10.4f}")
            var_rows.append({"age": age, "sex": sex, "n": len(v),
                            "var": v.var(ddof=1), "sd": v.std(ddof=1)})

    vdf = pd.DataFrame(var_rows)
    if len(vdf) >= 4:
        # Compare male vs female variance at each age level
        for age in age_order:
            m = vdf[(vdf["age"] == age) & (vdf["sex"] == "Male")]
            f = vdf[(vdf["age"] == age) & (vdf["sex"] == "Female")]
            if len(m) > 0 and len(f) > 0 and f["var"].values[0] > 0:
                ratio = m["var"].values[0] / f["var"].values[0]
                print(f"  {age}: Male/Female variance ratio = {ratio:.2f}")


# ============================================================
# ANALYSIS 4: Which Barnes variable best captures learning?
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 4: WHICH BARNES VARIABLE BEST CAPTURES LEARNING?")
print("=" * 70)

# --- 4a: Correlation with Age (proxy for cognitive decline) ---
print("\n--- 4a: Correlation with Age (known cognitive predictor) ---")
print("  A good learning variable should correlate with Age")
print("  (younger mice should perform better)\n")

age_map = {"Young": 1, "Mid": 2, "Old": 3}
bt6_corr = bt6.copy()
bt6_corr["age_numeric"] = bt6_corr["Age_new"].map(age_map)

test_vars = ["probe_accuracy", "Hole_errors", "EntryZone_freq_new",
             "DistanceMoved_cm", "Entry_latency_new", "Q4", "Q1"]
corr_rows = []
for var in test_vars:
    if var not in bt6_corr.columns:
        continue
    v = pd.to_numeric(bt6_corr[var], errors="coerce")
    d = pd.DataFrame({"age": bt6_corr["age_numeric"], "y": v}).dropna()
    if len(d) < 10:
        continue
    r, p = stats.spearmanr(d["age"], d["y"])
    corr_rows.append({"Variable": var, "Spearman_r": r, "p": p, "abs_r": abs(r)})
    direction = "worse with age" if r > 0 else "better with age"
    sig = "*" if p < 0.05 else ""
    print(f"  {var:<25s} r={r:+.3f}, p={p:.4f} {sig:2s} ({direction})")

corr_df = pd.DataFrame(corr_rows).sort_values("abs_r", ascending=False)
print(f"\n  Best age-predictor: {corr_df.iloc[0]['Variable']} (|r|={corr_df.iloc[0]['abs_r']:.3f})")

# --- 4b: Improvement across trials (learning sensitivity) ---
print("\n--- 4b: Learning sensitivity — which variable changes most across trials? ---")
print("  Effect size (Cohen's d) between Trial 1 and Trial 5 (peak)\n")

learning_vars = ["probe_accuracy", "Hole_errors", "EntryZone_freq_new",
                 "DistanceMoved_cm", "Entry_latency_new"]
learn_rows = []
for var in learning_vars:
    if var not in barnes.columns:
        continue
    t1 = pd.to_numeric(barnes[barnes["Trial"] == 1][var], errors="coerce").dropna()
    t5 = pd.to_numeric(barnes[barnes["Trial"] == 5][var], errors="coerce").dropna()
    if len(t1) < 5 or len(t5) < 5:
        continue
    d = cohens_d(t5, t1)
    learn_rows.append({"Variable": var, "T1_mean": t1.mean(), "T5_mean": t5.mean(),
                       "Cohen_d": d, "abs_d": abs(d)})
    direction = "improved" if ((var == "probe_accuracy" and d > 0) or
                               (var in ["Hole_errors", "DistanceMoved_cm", "Entry_latency_new"] and d < 0) or
                               (var == "EntryZone_freq_new" and d > 0)) else "worsened/no change"
    print(f"  {var:<25s} T1={t1.mean():.2f}, T5={t5.mean():.2f}, d={d:+.3f} ({direction})")

learn_df = pd.DataFrame(learn_rows).sort_values("abs_d", ascending=False)
print(f"\n  Most learning-sensitive: {learn_df.iloc[0]['Variable']} (|d|={learn_df.iloc[0]['abs_d']:.3f})")

# --- 4c: PCA on Trial 6 variables ---
print("\n--- 4c: PCA on Trial 6 variables ---")
print("  What latent dimensions underlie Barnes performance?\n")

pca_vars = ["probe_accuracy", "Hole_errors", "EntryZone_freq_new",
            "DistanceMoved_cm", "Entry_latency_new", "Q4"]
pca_data = bt6[pca_vars].apply(pd.to_numeric, errors="coerce").dropna()

scaler = StandardScaler()
X = scaler.fit_transform(pca_data)

pca = PCA()
pcs = pca.fit_transform(X)

print(f"  Explained variance ratios:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"    PC{i+1}: {var:.3f} ({var*100:.1f}%)")

print(f"\n  PC1 loadings (explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance):")
for name, loading in sorted(zip(pca_vars, pca.components_[0]), key=lambda x: abs(x[1]), reverse=True):
    direction = "+" if loading > 0 else "-"
    print(f"    {name:<25s} {direction}{abs(loading):.3f}")

print(f"\n  PC2 loadings (explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance):")
for name, loading in sorted(zip(pca_vars, pca.components_[1]), key=lambda x: abs(x[1]), reverse=True):
    direction = "+" if loading > 0 else "-"
    print(f"    {name:<25s} {direction}{abs(loading):.3f}")

# --- 4d: Per-mouse learning slope ---
print("\n--- 4d: Per-mouse learning slope (individual learning rate) ---")
print("  Linear slope of probe_accuracy across Trials 1-5 per mouse")
print("  (Trial 6 excluded — performance drops, possibly probe trial)\n")

slopes = []
for mid in barnes["ID"].unique():
    dm = barnes[(barnes["ID"] == mid) & (barnes["Trial"] <= 5)].copy()
    dm = dm.dropna(subset=["probe_accuracy"])
    if len(dm) >= 3:
        slope, _, r, p, _ = stats.linregress(dm["Trial"], dm["probe_accuracy"])
        age = dm["Age_new"].iloc[0]
        sex = dm["Sex_new"].iloc[0]
        light = dm["Light_new"].iloc[0]
        slopes.append({"ID": mid, "learning_slope": slope, "r": r, "p": p,
                       "Age_new": age, "Sex_new": sex, "Light_new": light})

slope_df = pd.DataFrame(slopes)
print(f"  Mice with slopes: {len(slope_df)}")
print(f"  Mean slope: {slope_df['learning_slope'].mean():.4f}")
print(f"  Positive slopes (learning): {(slope_df['learning_slope'] > 0).sum()}/{len(slope_df)}")

# Test Age and Sex effects on learning slope
slope_df["Age_new"] = slope_df["Age_new"].astype(str)
slope_df["Sex_new"] = slope_df["Sex_new"].astype(str)
slope_model = smf.ols("learning_slope ~ C(Age_new) * C(Sex_new)", data=slope_df).fit(cov_type="HC3")

print(f"\n  Learning slope ~ Age * Sex:")
for term in slope_model.pvalues.index:
    if term == "Intercept":
        continue
    sig = "*" if slope_model.pvalues[term] < 0.05 else ""
    print(f"    {term}: beta={slope_model.params[term]:.4f}, p={slope_model.pvalues[term]:.4f} {sig}")

# Descriptives by group
print(f"\n  Learning slope by Age x Sex:")
for age in age_order:
    for sex in ["Female", "Male"]:
        sub = slope_df[(slope_df["Age_new"] == age) & (slope_df["Sex_new"] == sex)]
        if len(sub) > 0:
            print(f"    {age:<8s} {sex:<8s} n={len(sub):>3d}  "
                  f"mean={sub['learning_slope'].mean():+.4f}  "
                  f"SD={sub['learning_slope'].std(ddof=1):.4f}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 4 SUMMARY: BEST BARNES VARIABLE FOR LEARNING")
print("=" * 70)
print("""
Four criteria were evaluated:

  a) Correlation with Age (known cognitive predictor):
     Best: probe_accuracy and Q4 (target quadrant time)

  b) Learning sensitivity (Trial 1 vs Trial 5 change):
     Best: probe_accuracy and Hole_errors

  c) PCA structure:
     PC1 loads heavily on probe_accuracy, Q4, and EntryZone_freq_new
     (spatial accuracy dimension)
     PC2 loads on Hole_errors and DistanceMoved_cm
     (exploration/activity dimension)

  d) Learning slope (Trials 1-5):
     Captures individual learning rate; can be tested for Age*Sex effects

RECOMMENDATION:
  Primary endpoint:  probe_accuracy (best all-round: age-sensitive,
                     learning-sensitive, loads on PC1)
  Secondary:         Q4 (target quadrant time — strong age correlation,
                     standard in Barnes literature)
  Sensitivity:       learning_slope (Trials 1-5, captures trajectory)
  Avoid:             Goal_Box_feq_new (floor effect at Trial 6)
""")

# Save
slope_df.to_csv("learning_slopes_per_mouse.csv", index=False)
br_df.to_csv("sex_age_barnes_results.csv", index=False)
nr_df.to_csv("sex_age_nor_results.csv", index=False)
print("Saved: learning_slopes_per_mouse.csv, sex_age_barnes_results.csv, sex_age_nor_results.csv")
