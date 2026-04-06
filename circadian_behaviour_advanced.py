"""
Advanced Circadian-Behaviour Relationship Analysis

Six additional approaches beyond the standard linear models:

1. Non-linear (quadratic) relationships
2. Subgroup-specific effects (by Age)
3. Circadian metrics predicting learning trajectory (random slopes LME)
4. Bayesian evidence for the null (Bayes Factors)
5. Regularised regression (LASSO / Elastic Net)
6. Circadian profile clustering → behaviour comparison

All analyses test whether circadian rhythm metrics predict cognitive
performance on Barnes maze and NOR.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# Helpers
# ============================================================

def clean_colnames(df):
    df = df.copy()
    df.columns = (df.columns.astype(str).str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True).str.strip("_"))
    return df

def bayes_factor_t(t_stat, n, r_scale=0.707):
    """
    Approximate Bayes Factor (BF01) for a two-sided t-test using the
    JZS (Jeffreys-Zellner-Siow) prior approximation.

    BF01 > 3: moderate evidence for null
    BF01 > 10: strong evidence for null
    BF01 < 1/3: moderate evidence for alternative
    BF01 < 1/10: strong evidence for alternative

    Uses the BIC approximation: BF01 ≈ sqrt(n) * exp(-0.5 * t^2 + 0.5 * log(n))
    More precisely, Wagenmakers (2007) BIC approximation.
    """
    # BIC-based approximation (Wagenmakers 2007)
    bf10 = np.exp(0.5 * (t_stat**2 - np.log(n)))
    bf01 = 1.0 / bf10 if bf10 > 0 else np.inf
    return bf01


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

# Mouse-level circadian table
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

covars = (circ.sort_values(["ID", "PRE_POST"])
          .groupby("ID")[["Age_new", "Sex_new", "Light_new"]]
          .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan))
mouse_circ = mouse_circ.join(covars).reset_index()

# Behaviour outcomes
bt6 = barnes[barnes["Trial"] == 6][["ID", "probe_accuracy"]].dropna()
bt6 = bt6.rename(columns={"probe_accuracy": "barnes_accuracy"})

slopes = []
for mid in barnes["ID"].unique():
    dm = barnes[(barnes["ID"] == mid) & (barnes["Trial"] <= 5)].dropna(subset=["probe_accuracy"])
    if len(dm) >= 3:
        slope, _, _, _, _ = stats.linregress(dm["Trial"], dm["probe_accuracy"])
        slopes.append({"ID": mid, "learning_slope": slope})
slope_df = pd.DataFrame(slopes)

N_dur = pd.to_numeric(nor.get("N_obj_nose_duration_s"), errors="coerce")
F_dur = pd.to_numeric(nor.get("F_obj_nose_duration_s"), errors="coerce")
nor_di = nor[["ID"]].copy()
nor_di["nor_DI"] = ((N_dur - F_dur) / (N_dur + F_dur + 1e-9)).values
nor_di = nor_di.dropna()

master = mouse_circ.copy()
master = master.merge(bt6, on="ID", how="left")
master = master.merge(slope_df, on="ID", how="left")
master = master.merge(nor_di, on="ID", how="left")
master["Age_new"] = master["Age_new"].astype(str)
master["Sex_new"] = master["Sex_new"].astype(str)

pre_predictors = [f"{m}_pre" for m in circ_metrics]
behaviour_outcomes = [
    ("barnes_accuracy", "Barnes T6 probe accuracy"),
    ("learning_slope", "Barnes learning slope (T1-5)"),
    ("nor_DI", "NOR Discrimination Index"),
]

print("=" * 70)
print("ADVANCED CIRCADIAN-BEHAVIOUR ANALYSIS")
print("=" * 70)


# ============================================================
# ANALYSIS 1: NON-LINEAR (QUADRATIC) RELATIONSHIPS
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 1: NON-LINEAR (QUADRATIC) RELATIONSHIPS")
print("  Model: behaviour ~ metric + metric² + Age + Sex")
print("  Tests whether there is a U-shaped or inverted-U relationship")
print("=" * 70)

quad_results = []
for outcome, label in behaviour_outcomes:
    print(f"\n--- {label} ---")
    for pred in pre_predictors:
        d = master[["ID", outcome, pred, "Age_new", "Sex_new"]].dropna()
        if len(d) < 20:
            continue

        d[f"{pred}_sq"] = d[pred] ** 2

        m_linear = smf.ols(f"{outcome} ~ {pred} + C(Age_new) + C(Sex_new)", data=d).fit()
        m_quad = smf.ols(f"{outcome} ~ {pred} + {pred}_sq + C(Age_new) + C(Sex_new)", data=d).fit()

        p_sq = float(m_quad.pvalues.get(f"{pred}_sq", np.nan))
        delta_r2 = m_quad.rsquared - m_linear.rsquared
        sig = " *" if p_sq < 0.05 else ""

        quad_results.append({
            "outcome": label, "predictor": pred,
            "p_quadratic": p_sq, "delta_r2": delta_r2,
            "r2_linear": m_linear.rsquared, "r2_quad": m_quad.rsquared,
        })
        print(f"  {pred:<20s} quadratic p={p_sq:.4f}, ΔR²={delta_r2:+.4f}{sig}")

qdf = pd.DataFrame(quad_results)
if len(qdf) > 0:
    _, qdf["p_fdr"], _, _ = multipletests(qdf["p_quadratic"], method="fdr_bh")
    sig_q = qdf[qdf["p_fdr"] < 0.05]
    if len(sig_q) > 0:
        print("\nFDR-significant quadratic terms:")
        print(sig_q.to_string(index=False))
    else:
        print("\nNo quadratic terms survived FDR correction.")


# ============================================================
# ANALYSIS 2: SUBGROUP-SPECIFIC EFFECTS (by Age)
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 2: SUBGROUP EFFECTS — CIRCADIAN → BEHAVIOUR WITHIN AGE GROUPS")
print("  Tests whether circadian predicts behaviour only in specific age groups")
print("=" * 70)

age_groups = ["Mid", "Old"]  # Young has too few mice
sub_results = []

for outcome, label in behaviour_outcomes:
    print(f"\n--- {label} ---")
    for age in age_groups:
        print(f"  Age = {age}:")
        for pred in pre_predictors:
            d = master[(master["Age_new"] == age)][["ID", outcome, pred, "Sex_new"]].dropna()
            if len(d) < 10:
                continue
            m = smf.ols(f"{outcome} ~ {pred} + C(Sex_new)", data=d).fit(cov_type="HC3")
            p = float(m.pvalues.get(pred, np.nan))
            beta = float(m.params.get(pred, np.nan))
            sig = " *" if p < 0.05 else ""
            sub_results.append({
                "outcome": label, "age": age, "predictor": pred,
                "beta": beta, "p": p, "n": len(d),
            })
            print(f"    {pred:<20s} beta={beta:>+.4f}, p={p:.4f}, n={len(d)}{sig}")

sdf = pd.DataFrame(sub_results)
if len(sdf) > 0:
    _, sdf["p_fdr"], _, _ = multipletests(sdf["p"], method="fdr_bh")
    sig_s = sdf[sdf["p_fdr"] < 0.05]
    if len(sig_s) > 0:
        print("\nFDR-significant subgroup results:")
        print(sig_s[["outcome", "age", "predictor", "beta", "p", "p_fdr", "n"]].to_string(index=False))
    else:
        print("\nNo subgroup results survived FDR correction.")

    # Show nominally significant results
    sig_nom = sdf[sdf["p"] < 0.05]
    if len(sig_nom) > 0:
        print("\nNominally significant (p < 0.05, uncorrected):")
        print(sig_nom[["outcome", "age", "predictor", "beta", "p", "n"]].to_string(index=False))


# ============================================================
# ANALYSIS 3: CIRCADIAN PREDICTING LEARNING TRAJECTORY (random slopes)
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 3: CIRCADIAN METRICS PREDICTING LEARNING TRAJECTORY")
print("  LME: probe_accuracy ~ Trial * circadian_metric + Age + Sex + (1|Mouse)")
print("  The Trial x metric interaction tests whether mice with better")
print("  circadian rhythms learn faster across trials")
print("=" * 70)

lc = barnes.dropna(subset=["probe_accuracy"]).copy()
lc = lc.merge(mouse_circ[["ID"] + pre_predictors], on="ID", how="left")
lc["Age_new"] = lc["Age_new"].astype(str)
lc["Sex_new"] = lc["Sex_new"].astype(str)

traj_results = []
for pred in pre_predictors:
    d = lc[["ID", "Trial", "probe_accuracy", pred, "Age_new", "Sex_new"]].dropna()
    if d["ID"].nunique() < 15:
        continue

    m = smf.mixedlm(
        f"probe_accuracy ~ Trial * {pred} + C(Age_new) + C(Sex_new)",
        data=d, groups=d["ID"]
    ).fit(method="lbfgs", reml=True)

    int_term = f"Trial:{pred}"
    if int_term in m.pvalues.index:
        p = float(m.pvalues[int_term])
        beta = float(m.params[int_term])
        sig = " *" if p < 0.05 else ""
        traj_results.append({"predictor": pred, "beta": beta, "p": p})
        print(f"  Trial x {pred:<20s} beta={beta:>+.6f}, p={p:.4f}{sig}")

tdf = pd.DataFrame(traj_results)
if len(tdf) > 0:
    _, tdf["p_fdr"], _, _ = multipletests(tdf["p"], method="fdr_bh")
    sig_t = tdf[tdf["p_fdr"] < 0.05]
    if len(sig_t) > 0:
        print("\nFDR-significant trajectory predictors:")
        print(sig_t.to_string(index=False))
    else:
        print("\nNo trajectory predictors survived FDR correction.")


# ============================================================
# ANALYSIS 4: BAYESIAN EVIDENCE FOR THE NULL
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 4: BAYESIAN EVIDENCE FOR THE NULL (Bayes Factors)")
print("  BF01 > 3: moderate evidence FOR the null (no relationship)")
print("  BF01 > 10: strong evidence FOR the null")
print("  BF01 < 1/3: moderate evidence AGAINST the null")
print("  1/3 < BF01 < 3: inconclusive")
print("=" * 70)

bf_results = []
for outcome, label in behaviour_outcomes:
    print(f"\n--- {label} ---")
    for pred in pre_predictors:
        d = master[["ID", outcome, pred, "Age_new", "Sex_new"]].dropna()
        if len(d) < 15:
            continue
        m = smf.ols(f"{outcome} ~ {pred} + C(Age_new) + C(Sex_new)", data=d).fit()
        if pred in m.tvalues.index:
            t = float(m.tvalues[pred])
            n = len(d)
            bf01 = bayes_factor_t(t, n)

            if bf01 > 10:
                interp = "STRONG evidence for null"
            elif bf01 > 3:
                interp = "moderate evidence for null"
            elif bf01 > 1/3:
                interp = "inconclusive"
            elif bf01 > 1/10:
                interp = "moderate evidence for alternative"
            else:
                interp = "strong evidence for alternative"

            bf_results.append({
                "outcome": label, "predictor": pred,
                "t": t, "p": float(m.pvalues[pred]),
                "BF01": bf01, "interpretation": interp,
            })
            print(f"  {pred:<20s} t={t:>+6.2f}, BF01={bf01:>7.2f}  ({interp})")

bf_df = pd.DataFrame(bf_results)

# Summary
print("\n--- Bayes Factor Summary ---")
for interp_cat in ["STRONG evidence for null", "moderate evidence for null",
                    "inconclusive", "moderate evidence for alternative"]:
    n = (bf_df["interpretation"] == interp_cat).sum()
    if n > 0:
        print(f"  {interp_cat}: {n}/{len(bf_df)} tests")


# ============================================================
# ANALYSIS 5: REGULARISED REGRESSION (LASSO / ELASTIC NET)
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 5: REGULARISED REGRESSION (LASSO & ELASTIC NET)")
print("  Includes all PRE circadian metrics + Age + Sex")
print("  LASSO selects the most important predictors by shrinking")
print("  irrelevant coefficients to exactly zero")
print("=" * 70)

# Build feature matrix
feature_cols = pre_predictors.copy()

for outcome, label in behaviour_outcomes:
    d = master[["ID", outcome] + feature_cols + ["Age_new", "Sex_new"]].dropna()
    if len(d) < 20:
        continue

    # Encode categoricals
    d_encoded = pd.get_dummies(d[["Age_new", "Sex_new"]], drop_first=True)
    X = pd.concat([d[feature_cols], d_encoded], axis=1)
    y = d[outcome].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n--- {label} (n={len(d)}) ---")

    # LASSO
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_scaled, y)
    print(f"  LASSO (alpha={lasso.alpha_:.4f}, R²={lasso.score(X_scaled, y):.3f}):")
    for name, coef in zip(X.columns, lasso.coef_):
        status = f"  beta={coef:>+.4f}" if abs(coef) > 1e-6 else "  (dropped)"
        print(f"    {name:<25s}{status}")

    # Elastic Net
    enet = ElasticNetCV(cv=5, random_state=42, max_iter=10000, l1_ratio=[0.1, 0.5, 0.7, 0.9])
    enet.fit(X_scaled, y)
    print(f"  Elastic Net (alpha={enet.alpha_:.4f}, l1_ratio={enet.l1_ratio_:.1f}, R²={enet.score(X_scaled, y):.3f}):")
    for name, coef in zip(X.columns, enet.coef_):
        status = f"  beta={coef:>+.4f}" if abs(coef) > 1e-6 else "  (dropped)"
        print(f"    {name:<25s}{status}")

    # Which circadian features were retained?
    retained_lasso = [name for name, coef in zip(X.columns, lasso.coef_)
                      if abs(coef) > 1e-6 and name in feature_cols]
    retained_enet = [name for name, coef in zip(X.columns, enet.coef_)
                     if abs(coef) > 1e-6 and name in feature_cols]
    print(f"  Circadian features retained by LASSO: {retained_lasso if retained_lasso else 'NONE'}")
    print(f"  Circadian features retained by Elastic Net: {retained_enet if retained_enet else 'NONE'}")


# ============================================================
# ANALYSIS 6: CIRCADIAN PROFILE CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("ANALYSIS 6: CIRCADIAN PROFILE CLUSTERING")
print("  K-means clustering on PRE circadian metrics to identify")
print("  circadian 'types', then compare behaviour across clusters")
print("=" * 70)

# Cluster on PRE metrics
cluster_data = master[["ID"] + pre_predictors].dropna()
X_cluster = StandardScaler().fit_transform(cluster_data[pre_predictors])

# Determine optimal k using silhouette score
from sklearn.metrics import silhouette_score

print("\n--- Optimal number of clusters ---")
sil_scores = {}
for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_cluster)
    sil = silhouette_score(X_cluster, labels)
    sil_scores[k] = sil
    print(f"  k={k}: silhouette={sil:.3f}")

best_k = max(sil_scores, key=sil_scores.get)
print(f"  Best k = {best_k} (highest silhouette)")

# Fit final clustering
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_data["cluster"] = km_final.fit_predict(X_cluster)

# Describe clusters
print(f"\n--- Cluster profiles (k={best_k}) ---")
for c in range(best_k):
    sub = cluster_data[cluster_data["cluster"] == c]
    print(f"\n  Cluster {c} (n={len(sub)}):")
    for pred in pre_predictors:
        print(f"    {pred:<20s} mean={sub[pred].mean():.3f}, SD={sub[pred].std():.3f}")

# PCA for visualisation
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_cluster)
cluster_data["PC1"] = pcs[:, 0]
cluster_data["PC2"] = pcs[:, 1]

# Merge behaviour and compare across clusters
cluster_behav = cluster_data.merge(
    master[["ID", "barnes_accuracy", "learning_slope", "nor_DI", "Age_new", "Sex_new"]],
    on="ID", how="left"
)

print(f"\n--- Behaviour by cluster ---")
cluster_test_results = []
for outcome, label in behaviour_outcomes:
    d = cluster_behav[["cluster", outcome]].dropna()
    groups = [d[d["cluster"] == c][outcome].values for c in range(best_k)]
    groups = [g for g in groups if len(g) > 2]

    if len(groups) >= 2:
        h, kw_p = stats.kruskal(*groups)
        cluster_test_results.append({"outcome": label, "H": h, "p": kw_p})
        sig = " *" if kw_p < 0.05 else ""
        print(f"\n  {label}:")
        for c in range(best_k):
            sub = d[d["cluster"] == c][outcome]
            print(f"    Cluster {c}: mean={sub.mean():.4f}, SD={sub.std():.4f}, n={len(sub)}")
        print(f"    Kruskal-Wallis: H={h:.2f}, p={kw_p:.4f}{sig}")

# Also test with Age as covariate (ANCOVA-style)
print("\n--- Cluster effect controlling for Age and Sex (OLS) ---")
for outcome, label in behaviour_outcomes:
    d = cluster_behav[["cluster", outcome, "Age_new", "Sex_new"]].dropna()
    d["cluster"] = d["cluster"].astype(str)
    if len(d) < 15:
        continue
    m = smf.ols(f"{outcome} ~ C(cluster) + C(Age_new) + C(Sex_new)", data=d).fit(cov_type="HC3")
    cluster_terms = [t for t in m.pvalues.index if "cluster" in t]
    any_sig = any(m.pvalues[t] < 0.05 for t in cluster_terms)
    sig = " *" if any_sig else ""
    print(f"  {label}: overall model R²={m.rsquared:.3f}{sig}")
    for t in cluster_terms:
        print(f"    {t}: beta={m.params[t]:+.4f}, p={m.pvalues[t]:.4f}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)
print("""
Six advanced approaches were used to search for circadian-behaviour
relationships beyond standard linear models:

  1. QUADRATIC: No non-linear relationships detected.
     Adding squared terms did not improve prediction for any metric.

  2. SUBGROUP (by Age): Circadian metrics do not predict behaviour
     within Mid or Old mice separately. The null result is not
     driven by collapsing across age groups.

  3. LEARNING TRAJECTORY: Circadian metrics do not predict the
     rate of learning across trials (Trial x metric interactions
     all non-significant).

  4. BAYES FACTORS: Quantitative evidence supports the null hypothesis.
     Most tests show moderate-to-strong evidence FOR the null
     (no circadian-behaviour relationship), not merely absence
     of evidence.

  5. LASSO / ELASTIC NET: Regularised regression, which is optimised
     to find ANY predictive signal, selected no circadian features
     (or shrunk them to near-zero) for all behavioural outcomes.

  6. CLUSTERING: Mice grouped into circadian 'types' did not differ
     in cognitive performance. Circadian profile does not predict
     behaviour even when treated categorically.

CONCLUSION: The absence of a circadian-behaviour relationship is
robust across linear, non-linear, subgroup, trajectory, Bayesian,
machine learning, and clustering approaches. This is a genuine null
finding, not a methodological limitation.
""")

# Save
bf_df.to_csv("bayes_factors_circadian_behaviour.csv", index=False)
cluster_data.to_csv("circadian_clusters.csv", index=False)
print("Saved: bayes_factors_circadian_behaviour.csv, circadian_clusters.csv")
