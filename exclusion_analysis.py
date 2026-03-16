"""
Sensitivity analysis: re-run all primary models excluding mice flagged
for sensor issues (long dropouts, >80% zero activity).

Compares:
  - FULL sample (original analysis)
  - CLEAN sample (sensor-flagged mice excluded)

Reports side-by-side p-values, effect sizes, and direction of effects
so you can assess whether sensor artefacts drive any conclusions.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# ============================================================
# Helpers (same as analysis.py)
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

circ = clean_colnames(pd.read_csv("Circadian_raw.csv"))
barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))
nor = clean_colnames(pd.read_csv("UCBAge_Novel_clean.csv"))

if "Animal_ID" in nor.columns and "ID" not in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})

for df in (circ, barnes, nor):
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

circ = set_cats(circ, "PRE_POST", ["PRE", "POST"])
circ = set_cats(circ, "Light_new", ["CTR", "ISF"])
circ["Age_new"] = circ["Age_new"].astype("category")
circ["Sex_new"] = circ["Sex_new"].astype("category")

barnes = set_cats(barnes, "Light_new", ["CTR", "ISF"])
barnes["Age_new"] = barnes["Age_new"].astype("category")
barnes["Sex_new"] = barnes["Sex_new"].astype("category")
barnes["Trial"] = pd.to_numeric(barnes["Trial"], errors="coerce")

nor = set_cats(nor, "Light_new", ["CTR", "ISF"])
nor["Age_new"] = nor["Age_new"].astype("category")
nor["Sex_new"] = nor["Sex_new"].astype("category")


# ============================================================
# Identify flagged mice from computed sensor analysis
# ============================================================

sensor_df = pd.read_csv("circadian_computed_raw.csv")
# A mouse is flagged if ANY of its periods (PRE or POST) is flagged
flagged_ids = set(sensor_df[sensor_df["sensor_flag"] == True]["ID"].unique())
all_ids = set(circ["ID"].dropna().unique())
clean_ids = all_ids - flagged_ids

print("=" * 70)
print("SENSOR-BASED EXCLUSION ANALYSIS")
print("=" * 70)
print(f"\nTotal mice in circadian data:    {len(all_ids)}")
print(f"Mice flagged for sensor issues:  {len(flagged_ids)}")
print(f"Clean mice remaining:            {len(clean_ids)}")
print(f"\nFlagged IDs: {sorted(flagged_ids)}")


# ============================================================
# Build mouse-level predictors
# ============================================================

def build_mouse_table(circ_df):
    wide_IS = circ_df.pivot_table(index="ID", columns="PRE_POST", values="IS", aggfunc="mean")
    IS_pre = wide_IS.get("PRE").rename("IS_pre")
    delta_IS = (wide_IS.get("POST") - wide_IS.get("PRE")).rename("delta_IS")
    covars = (
        circ_df.sort_values(["ID", "PRE_POST"])
        .groupby("ID")[["Light_new", "Age_new", "Sex_new"]]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
    )
    mouse = pd.concat([covars, IS_pre, delta_IS], axis=1).reset_index()
    mouse = mouse.rename(columns={
        "Light_new": "Light_new_mouse",
        "Age_new": "Age_new_mouse",
        "Sex_new": "Sex_new_mouse",
    })
    return mouse


# ============================================================
# Model functions
# ============================================================

def fit_circadian_lme(circ_df, outcome):
    """Fit circadian mixed-effects model. Returns (model, interaction_p, interaction_beta)."""
    d = circ_df[["ID", "PRE_POST", "Light_new", "Age_new", "Sex_new", outcome]].dropna().copy()
    if d["ID"].nunique() < 5:
        return None, np.nan, np.nan

    formula = f"{outcome} ~ PRE_POST * Light_new + Age_new + Sex_new"
    try:
        m = smf.mixedlm(formula, d, groups=d["ID"]).fit(method="lbfgs", reml=True)
        interaction_term = [t for t in m.pvalues.index if "PRE_POST" in t and "Light_new" in t]
        if interaction_term:
            p = float(m.pvalues[interaction_term[0]])
            beta = float(m.params[interaction_term[0]])
        else:
            p, beta = np.nan, np.nan
        return m, p, beta
    except Exception as e:
        print(f"    LME failed for {outcome}: {e}")
        return None, np.nan, np.nan


def fit_barnes_glm(barnes_df, circ_mouse_df):
    """Fit Barnes Trial 6 binomial GLM. Returns (model, light_p, light_beta, n_mice).
    Falls back to Fisher's exact test if perfect separation is detected."""
    bm = barnes_df.merge(circ_mouse_df[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")
    bt6 = bm[bm["Trial"] == 6].copy()
    bt6["Light_new"] = bt6["Light_new"].astype(str)
    bt6["Age_new"] = bt6["Age_new"].astype(str)
    bt6["Sex_new"] = bt6["Sex_new"].astype(str)
    bt6["IS_pre"] = pd.to_numeric(bt6["IS_pre"], errors="coerce")
    bt6["delta_IS"] = pd.to_numeric(bt6["delta_IS"], errors="coerce")
    bt6["Goal_Box_feq_new"] = pd.to_numeric(bt6["Goal_Box_feq_new"], errors="coerce")
    bt6["Hole_errors"] = pd.to_numeric(bt6["Hole_errors"], errors="coerce")
    bt6 = bt6.dropna(subset=["Light_new", "Age_new", "Sex_new", "IS_pre",
                              "delta_IS", "Goal_Box_feq_new", "Hole_errors"]).copy()
    bt6 = bt6[(bt6["Goal_Box_feq_new"] >= 0) & (bt6["Hole_errors"] >= 0)].copy()
    bt6["total_entries"] = bt6["Goal_Box_feq_new"] + bt6["Hole_errors"]
    bt6 = bt6[bt6["total_entries"] > 0].copy()
    bt6["p_correct"] = bt6["Goal_Box_feq_new"] / bt6["total_entries"]

    n_mice = bt6["ID"].nunique()
    if n_mice < 5:
        return None, np.nan, np.nan, 0

    # Check for perfect separation: if one group is entirely 0 or entirely 1
    group_means = bt6.groupby("Light_new")["p_correct"].mean()
    has_separation = any(group_means == 0.0) or any(group_means == 1.0)
    n_nonzero = (bt6["p_correct"] > 0).sum()

    if has_separation or n_nonzero < 3:
        # Binomial GLM will fail with perfect separation.
        # Fall back to Fisher's exact test on any_correct vs Light
        from scipy.stats import fisher_exact
        bt6["any_correct"] = (bt6["Goal_Box_feq_new"] > 0).astype(int)
        ct = pd.crosstab(bt6["Light_new"], bt6["any_correct"])
        if ct.shape == (2, 2):
            odds, p = fisher_exact(ct.values)
            print(f"    Barnes: perfect separation detected, using Fisher's exact (OR={odds:.3f}, p={p:.4f})")
            return None, p, np.log(odds + 1e-10), n_mice
        else:
            print(f"    Barnes: all mice in same outcome category, cannot test.")
            return None, np.nan, 0.0, n_mice

    try:
        fit = smf.glm(
            "p_correct ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
            data=bt6, family=sm.families.Binomial(),
            freq_weights=bt6["total_entries"]
        ).fit(cov_type="HC3")
        term = "C(Light_new)[T.ISF]"
        p = float(fit.pvalues.get(term, np.nan))
        beta = float(fit.params.get(term, np.nan))
        return fit, p, beta, n_mice
    except Exception as e:
        print(f"    Barnes GLM failed: {e}")
        return None, np.nan, np.nan, 0


def fit_nor_ols(nor_df, circ_mouse_df):
    """Fit NOR DI robust OLS. Returns (model, light_p, light_beta)."""
    nm = nor_df.merge(circ_mouse_df[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")
    nm = nm.dropna(subset=["N_obj_nose_duration_s", "F_obj_nose_duration_s",
                           "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS"]).copy()
    n = pd.to_numeric(nm["N_obj_nose_duration_s"], errors="coerce")
    f = pd.to_numeric(nm["F_obj_nose_duration_s"], errors="coerce")
    nm["DI_duration"] = (n - f) / (n + f + 1e-9)

    if nm["ID"].nunique() < 5:
        return None, np.nan, np.nan, 0

    try:
        fit = smf.ols(
            "DI_duration ~ Light_new + IS_pre + delta_IS + Age_new + Sex_new",
            data=nm
        ).fit(cov_type="HC3")
        term = "Light_new[T.ISF]"
        p = float(fit.pvalues.get(term, np.nan))
        beta = float(fit.params.get(term, np.nan))
        return fit, p, beta, nm["ID"].nunique()
    except Exception as e:
        print(f"    NOR OLS failed: {e}")
        return None, np.nan, np.nan, 0


def bootstrap_mediation(df, y_col, n_boot=3000, seed=0):
    """Cluster bootstrap mediation: Light -> delta_IS -> y."""
    rng = np.random.default_rng(seed)
    ids = df["ID"].dropna().unique()
    if len(ids) < 10:
        return {"indirect": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    a_formula = "delta_IS ~ Light_new_mouse + Age_new_mouse + Sex_new_mouse + IS_pre"
    b_formula = f"{y_col} ~ Light_new_mouse + delta_IS + Age_new_mouse + Sex_new_mouse + IS_pre"

    try:
        a_fit = smf.ols(a_formula, data=df).fit()
        b_fit = smf.ols(b_formula, data=df).fit()
    except Exception:
        return {"indirect": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    lt = [t for t in a_fit.params.index if t.startswith("Light_new_mouse")]
    if len(lt) != 1 or "delta_IS" not in b_fit.params.index:
        return {"indirect": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    a = float(a_fit.params[lt[0]])
    b = float(b_fit.params["delta_IS"])
    indirect = a * b

    boots = []
    for _ in range(n_boot):
        samp_ids = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([df[df["ID"] == i] for i in samp_ids], ignore_index=True)
        try:
            af = smf.ols(a_formula, data=boot).fit()
            bf = smf.ols(b_formula, data=boot).fit()
            lt2 = [t for t in af.params.index if t.startswith("Light_new_mouse")]
            if len(lt2) == 1 and "delta_IS" in bf.params.index:
                boots.append(float(af.params[lt2[0]]) * float(bf.params["delta_IS"]))
        except Exception:
            continue

    boots = np.array(boots)
    if len(boots) < 200:
        return {"indirect": indirect, "ci_lo": np.nan, "ci_hi": np.nan}

    return {
        "indirect": indirect,
        "ci_lo": float(np.quantile(boots, 0.025)),
        "ci_hi": float(np.quantile(boots, 0.975)),
    }


# ============================================================
# Run all models on FULL and CLEAN samples
# ============================================================

results = []

for label, exclude_ids in [("FULL", set()), ("CLEAN", flagged_ids)]:
    print(f"\n{'=' * 70}")
    print(f"  {label} SAMPLE (excluding {len(exclude_ids)} mice)")
    print(f"{'=' * 70}")

    circ_s = circ[~circ["ID"].isin(exclude_ids)].copy()
    barnes_s = barnes[~barnes["ID"].isin(exclude_ids)].copy()
    nor_s = nor[~nor["ID"].isin(exclude_ids)].copy()

    n_circ = circ_s["ID"].nunique()
    n_barnes = barnes_s["ID"].nunique()
    n_nor = nor_s["ID"].nunique()
    print(f"  Circadian mice: {n_circ}, Barnes mice: {n_barnes}, NOR mice: {n_nor}")

    mouse_table = build_mouse_table(circ_s)

    # --- Circadian LME models ---
    for metric in ["IS", "IV", "RA", "Amplitude"]:
        if metric not in circ_s.columns:
            continue
        _, p, beta = fit_circadian_lme(circ_s, metric)
        results.append({
            "sample": label, "model": f"Circadian LME: {metric}",
            "term": "PRE_POST x Light", "beta": beta, "p": p,
            "n_mice": n_circ,
        })
        print(f"  Circadian {metric}: interaction beta={beta:.4f}, p={p:.4f}")

    # --- Barnes Trial 6 ---
    _, p, beta, n_bt6 = fit_barnes_glm(barnes_s, mouse_table)
    results.append({
        "sample": label, "model": "Barnes T6: p_correct (Binomial GLM)",
        "term": "Light[ISF]", "beta": beta, "p": p,
        "n_mice": n_bt6,
    })
    print(f"  Barnes T6 Light: beta={beta:.4f}, p={p:.4f} (n={n_bt6})")

    # --- NOR ---
    _, p, beta, n_nm = fit_nor_ols(nor_s, mouse_table)
    results.append({
        "sample": label, "model": "NOR: DI (robust OLS)",
        "term": "Light[ISF]", "beta": beta, "p": p,
        "n_mice": n_nm,
    })
    print(f"  NOR Light: beta={beta:.4f}, p={p:.4f} (n={n_nm})")

    # --- Mediation: Barnes ---
    bm = barnes_s.merge(mouse_table[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")
    bt6 = bm[bm["Trial"] == 6].copy()
    bt6["Goal_Box_feq_new"] = pd.to_numeric(bt6["Goal_Box_feq_new"], errors="coerce")
    bt6["Hole_errors"] = pd.to_numeric(bt6["Hole_errors"], errors="coerce")
    bt6 = bt6.dropna(subset=["Goal_Box_feq_new", "Hole_errors"]).copy()
    bt6["total_entries"] = bt6["Goal_Box_feq_new"] + bt6["Hole_errors"]
    bt6 = bt6[bt6["total_entries"] > 0].copy()
    bt6["p_correct"] = bt6["Goal_Box_feq_new"] / bt6["total_entries"]
    bt6["logit_p_correct"] = logit_clip(bt6["p_correct"].values)
    bt6 = bt6.merge(
        mouse_table[["ID", "Light_new_mouse", "Age_new_mouse", "Sex_new_mouse"]],
        on="ID", how="left"
    )
    bt6["IS_pre"] = pd.to_numeric(bt6["IS_pre"], errors="coerce")
    bt6["delta_IS"] = pd.to_numeric(bt6["delta_IS"], errors="coerce")
    bt6 = bt6.dropna(subset=["logit_p_correct", "Light_new_mouse", "Age_new_mouse",
                              "Sex_new_mouse", "IS_pre", "delta_IS"])

    med_b = bootstrap_mediation(bt6, "logit_p_correct")
    results.append({
        "sample": label, "model": "Mediation: Light->dIS->Barnes",
        "term": "indirect (a*b)", "beta": med_b["indirect"],
        "p": np.nan,  # CI-based
        "n_mice": bt6["ID"].nunique(),
        "ci_lo": med_b["ci_lo"], "ci_hi": med_b["ci_hi"],
    })
    ci_str = f"[{med_b['ci_lo']:.4f}, {med_b['ci_hi']:.4f}]"
    print(f"  Mediation Barnes: indirect={med_b['indirect']:.4f}, 95% CI {ci_str}")

    # --- Mediation: NOR ---
    nor_s2 = nor_s.copy()
    nor_s2 = nor_s2.dropna(subset=["N_obj_nose_duration_s", "F_obj_nose_duration_s"]).copy()
    n_dur = pd.to_numeric(nor_s2["N_obj_nose_duration_s"], errors="coerce")
    f_dur = pd.to_numeric(nor_s2["F_obj_nose_duration_s"], errors="coerce")
    nor_s2["DI_duration"] = (n_dur - f_dur) / (n_dur + f_dur + 1e-9)
    nor_endp = nor_s2.groupby("ID")["DI_duration"].mean().rename("nor_endp").reset_index()
    med_n_df = mouse_table.merge(nor_endp, on="ID", how="inner")
    med_n_df = med_n_df.dropna(subset=["Light_new_mouse", "Age_new_mouse",
                                        "Sex_new_mouse", "IS_pre", "delta_IS", "nor_endp"])

    med_n = bootstrap_mediation(med_n_df, "nor_endp")
    results.append({
        "sample": label, "model": "Mediation: Light->dIS->NOR",
        "term": "indirect (a*b)", "beta": med_n["indirect"],
        "p": np.nan,
        "n_mice": med_n_df["ID"].nunique(),
        "ci_lo": med_n["ci_lo"], "ci_hi": med_n["ci_hi"],
    })
    ci_str = f"[{med_n['ci_lo']:.4f}, {med_n['ci_hi']:.4f}]"
    print(f"  Mediation NOR: indirect={med_n['indirect']:.4f}, 95% CI {ci_str}")


# ============================================================
# Summary comparison table
# ============================================================

res_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("  SIDE-BY-SIDE COMPARISON: FULL vs CLEAN")
print("=" * 70)

# Pivot for comparison
full = res_df[res_df["sample"] == "FULL"].set_index("model")
clean = res_df[res_df["sample"] == "CLEAN"].set_index("model")

comparison = pd.DataFrame({
    "n_full": full["n_mice"],
    "beta_full": full["beta"],
    "p_full": full["p"],
    "n_clean": clean["n_mice"],
    "beta_clean": clean["beta"],
    "p_clean": clean["p"],
})

# Add direction consistency check
comparison["same_direction"] = np.sign(comparison["beta_full"]) == np.sign(comparison["beta_clean"])
comparison["both_ns"] = (comparison["p_full"] > 0.05) & (comparison["p_clean"] > 0.05)
comparison["conclusion_changed"] = (
    ((comparison["p_full"] <= 0.05) & (comparison["p_clean"] > 0.05)) |
    ((comparison["p_full"] > 0.05) & (comparison["p_clean"] <= 0.05))
)

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(comparison.to_string())

# Mediation CIs
print("\n--- Mediation 95% bootstrap CIs ---")
med_rows = res_df[res_df["model"].str.contains("Mediation")]
for _, row in med_rows.iterrows():
    ci_lo = row.get("ci_lo", np.nan)
    ci_hi = row.get("ci_hi", np.nan)
    includes_zero = (ci_lo <= 0 <= ci_hi) if pd.notna(ci_lo) and pd.notna(ci_hi) else "N/A"
    print(f"  {row['sample']:5s} | {row['model']}: indirect={row['beta']:.4f}, "
          f"CI=[{ci_lo:.4f}, {ci_hi:.4f}], includes_zero={includes_zero}")

# Final verdict
print("\n" + "=" * 70)
print("  VERDICT")
print("=" * 70)
any_changed = comparison["conclusion_changed"].any()
if any_changed:
    changed = comparison[comparison["conclusion_changed"]]
    print("WARNING: Some conclusions changed after excluding sensor-flagged mice:")
    print(changed[["beta_full", "p_full", "beta_clean", "p_clean"]].to_string())
else:
    print("No conclusions changed. All effects that were non-significant in the")
    print("full sample remain non-significant after excluding sensor-flagged mice.")
    print("The sensor issues do NOT drive the primary findings.")

# Save
res_df.to_csv("exclusion_analysis_results.csv", index=False)
comparison.to_csv("exclusion_analysis_comparison.csv")
print("\nSaved: exclusion_analysis_results.csv, exclusion_analysis_comparison.csv")
