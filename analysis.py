"""
Publication-level analysis pipeline (fixed for Patsy column-name issues)

WHAT THIS SCRIPT DOES (high level):
1) Circadian: Mixed-effects model for Light (ISF vs CTR) x PRE/POST on IS (primary),
   with model selection for random slope (PRE/POST | mouse). Secondary rhythm metrics optional with FDR.
2) Behaviour:
   - Barnes: nose-poke COUNT across trials with Poisson mixed model (with optional OLRE for overdispersion),
     using baseline IS (PRE) + delta IS (POST-PRE) as predictors.
     - change nose poke to error rate 
     - use only trial six 
     - entry zone frequency and goal box frequency signal correct choices 
     (from all holes they visited get percentage of visits to entry zone or goal box, then use that as outcome in poisson model)
        - NOR: discrimination index (DI) ~ Light + baseline IS + delta IS + covariates, with robust SEs (HC3).
3) Mediation (bootstrap, cluster by mouse):
   Light -> delta_IS -> (Barnes endpoint OR NOR endpoint).

IMPORTANT FIX:
- Any column names with dots (e.g., PRE.POST) are renamed to safe names (PRE_POST) so Patsy will not error.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM


# =========================
# 0) Helpers
# =========================
def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names Patsy-safe: replace non-word characters with underscores."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)  # anything not [A-Za-z0-9_]
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

def require_columns(df: pd.DataFrame, cols: list[str], df_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}\n"
            f"Available columns include: {df.columns.tolist()[:40]}{' ...' if len(df.columns) > 40 else ''}"
        )

def set_cats(df: pd.DataFrame, col: str, cats: list[str]) -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].astype("category").cat.set_categories(cats, ordered=True)
    return df

def logit_clip(x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


# =========================
# 1) Load + clean column names
# =========================
circ = pd.read_csv("Circadian_raw.csv")
barnes = pd.read_csv("Barnes_clean.csv")
nor = pd.read_csv("UCBAge_Novel_clean.csv")

# Harmonise ID name early if needed
if "Animal_ID" in nor.columns and "ID" not in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})

# Clean column names (THIS FIXES PRE.POST -> PRE_POST etc.)
circ = clean_colnames(circ)
barnes = clean_colnames(barnes)
nor = clean_colnames(nor)

# Coerce IDs
for df in (circ, barnes, nor):
    require_columns(df, ["ID"], df_name="A dataset")
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

# Confirm core columns exist after cleaning
require_columns(circ, ["PRE_POST", "Light_new", "Age_new", "Sex_new"], "Circadian_raw.csv (cleaned)")
require_columns(barnes, ["Trial", "Light_new", "Age_new", "Sex_new"], "Barnes_clean.csv (cleaned)")
require_columns(nor, ["Light_new", "Age_new", "Sex_new"], "UCBAge_Novel_clean.csv (cleaned)")

# Set categorical ordering (CTR reference, ISF treatment)
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


# =========================
# 2) QC: PRE/POST completeness
# =========================
prepost_counts = circ.groupby("ID")["PRE_POST"].nunique()
missing_pair = prepost_counts[prepost_counts < 2].index.tolist()
print(f"\nCircadian mice total: {circ['ID'].nunique()}")
print(f"Mice missing PRE or POST in circadian: {len(missing_pair)}")


# =========================
# 3) Circadian mixed model with random-effects selection
# Primary: IS
# =========================
def fit_circadian_mixedlm(outcome: str, use_logit_if_bounded: bool = True):
    require_columns(circ, ["ID", "PRE_POST", "Light_new", "Age_new", "Sex_new", outcome], f"Circadian for outcome={outcome}")

    d = circ[["ID", "PRE_POST", "Light_new", "Age_new", "Sex_new", outcome]].dropna().copy()

    # Optional transformation if bounded in [0,1]
    y = outcome
    if use_logit_if_bounded:
        y_min, y_max = d[outcome].min(), d[outcome].max()
        if np.isfinite(y_min) and np.isfinite(y_max) and (y_min >= 0) and (y_max <= 1):
            d[f"{outcome}_logit"] = logit_clip(d[outcome])
            y = f"{outcome}_logit"

    formula = f"{y} ~ PRE_POST * Light_new + Age_new + Sex_new"

    # Fit both models under ML for comparison
    m_ri_ml = smf.mixedlm(formula, d, groups=d["ID"]).fit(method="lbfgs", reml=False)
    m_rs_ml = smf.mixedlm(formula, d, groups=d["ID"], re_formula="~PRE_POST").fit(method="lbfgs", reml=False)

    aic_ri, aic_rs = m_ri_ml.aic, m_rs_ml.aic
    chosen = "RI+RS" if aic_rs < aic_ri else "RI"

    # Refit chosen with REML for final reporting
    if chosen == "RI+RS":
        m_final = smf.mixedlm(formula, d, groups=d["ID"], re_formula="~PRE_POST").fit(method="lbfgs", reml=True)
    else:
        m_final = smf.mixedlm(formula, d, groups=d["ID"]).fit(method="lbfgs", reml=True)

    print(f"\n=== Circadian MixedLM: {outcome} (chosen random effects: {chosen}) ===")
    print(m_final.summary())

    return m_final, d, y, chosen

# Primary
require_columns(circ, ["IS"], "Circadian_raw.csv (needs IS for primary analysis)")
circ_IS_model, circ_IS_used, circ_IS_y, circ_IS_re = fit_circadian_mixedlm("IS")

# Secondary (optional) with FDR on the interaction term(s)
secondary_metrics = [m for m in ["IV", "RA", "Amplitude"] if m in circ.columns]
sec_rows = []
sec_models = {}

for out in secondary_metrics:
    m, d, y, chosen = fit_circadian_mixedlm(out)
    sec_models[out] = m
    for term in m.pvalues.index:
        if "PRE_POST[T.POST]:Light_new" in term:
            sec_rows.append({"outcome": out, "term": term, "beta": float(m.params[term]), "p": float(m.pvalues[term])})

if sec_rows:
    sec_df = pd.DataFrame(sec_rows)
    rej, p_fdr, _, _ = multipletests(sec_df["p"].values, method="fdr_bh", alpha=0.05)
    sec_df["p_fdr_bh"] = p_fdr
    sec_df["sig_fdr_0.05"] = rej
    print("\n=== Secondary circadian interaction terms (FDR BH) ===")
    print(sec_df.sort_values(["outcome", "term"]).to_string(index=False))


# =========================
# 4) Build mouse-level baseline IS (PRE) and delta IS (POST-PRE)
# =========================
wide_IS = circ.pivot_table(index="ID", columns="PRE_POST", values="IS", aggfunc="mean")
IS_pre = wide_IS.get("PRE").rename("IS_pre")
delta_IS = (wide_IS.get("POST") - wide_IS.get("PRE")).rename("delta_IS")

mouse_covars = (
    circ.sort_values(["ID", "PRE_POST"])
        .groupby("ID")[["Light_new", "Age_new", "Sex_new"]]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
)

circ_mouse = pd.concat([mouse_covars, IS_pre, delta_IS], axis=1).reset_index()
print(f"\nMouse-level table rows (circ_mouse): {circ_mouse.shape[0]}")

# IMPORTANT: rename circ_mouse covariates to avoid merge suffix collisions
circ_mouse = circ_mouse.rename(columns={
    "Light_new": "Light_new_mouse",
    "Age_new": "Age_new_mouse",
    "Sex_new": "Sex_new_mouse"
})


# =========================
# 5) Barnes (UPDATED): Trial 6 only, correct vs wrong entry proportions
# Correct = Goal_Box_feq_new
# Wrong   = Hole_errors
# =========================

TRIAL_END = 6

require_columns(
    barnes,
    ["ID", "Trial", "Light_new", "Age_new", "Sex_new", "Goal_Box_feq_new", "Hole_errors"],
    "Barnes_clean.csv (cleaned)"
)

# Merge circadian predictors
barnes_m = barnes.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")

# Restrict to Trial 6
barnes_t6 = barnes_m[barnes_m["Trial"] == TRIAL_END].copy()

# Coerce types
barnes_t6["Light_new"] = barnes_t6["Light_new"].astype(str)
barnes_t6["Age_new"] = barnes_t6["Age_new"].astype(str)
barnes_t6["Sex_new"] = barnes_t6["Sex_new"].astype(str)
barnes_t6["IS_pre"] = pd.to_numeric(barnes_t6["IS_pre"], errors="coerce")
barnes_t6["delta_IS"] = pd.to_numeric(barnes_t6["delta_IS"], errors="coerce")

barnes_t6["Goal_Box_feq_new"] = pd.to_numeric(barnes_t6["Goal_Box_feq_new"], errors="coerce")
barnes_t6["Hole_errors"] = pd.to_numeric(barnes_t6["Hole_errors"], errors="coerce")

barnes_t6 = barnes_t6.dropna(subset=[
    "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS", "Goal_Box_feq_new", "Hole_errors"
]).copy()

# Keep non-negative counts
barnes_t6 = barnes_t6[(barnes_t6["Goal_Box_feq_new"] >= 0) & (barnes_t6["Hole_errors"] >= 0)].copy()

# Build totals + proportions
barnes_t6["total_entries"] = barnes_t6["Goal_Box_feq_new"] + barnes_t6["Hole_errors"]
barnes_t6 = barnes_t6[barnes_t6["total_entries"] > 0].copy()

barnes_t6["p_correct"] = barnes_t6["Goal_Box_feq_new"] / barnes_t6["total_entries"]
barnes_t6["p_wrong"] = barnes_t6["Hole_errors"] / barnes_t6["total_entries"]

print(f"\nBarnes Trial {TRIAL_END}: n mice with total_entries>0 = {barnes_t6['ID'].nunique()}")

# ------------------------------------------------------------
# Primary Barnes endpoint at Trial 6: percent correct (binomial GLM)
# Use successes/trials formulation via freq_weights.
# ------------------------------------------------------------
binom_fit = smf.glm(
    formula="p_correct ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    data=barnes_t6,
    family=sm.families.Binomial(),
    freq_weights=barnes_t6["total_entries"]
).fit(cov_type="HC3")

print(f"\n=== Barnes Trial {TRIAL_END} (Primary): Percent correct ~ Light + IS_pre + delta_IS + covariates (Binomial GLM) ===")
print(binom_fit.summary())

# Note: coefficients are log-odds; exp(beta) gives odds ratios.
print("\nOdds ratios (exp(beta)) for key terms:")
for term in ["C(Light_new)[T.ISF]", "IS_pre", "delta_IS"]:
    if term in binom_fit.params.index:
        b = binom_fit.params[term]
        ci = binom_fit.conf_int().loc[term].values
        print(f"  {term}: OR={np.exp(b):.3f} (95% CI {np.exp(ci[0]):.3f}–{np.exp(ci[1]):.3f})")

# ------------------------------------------------------------
# Optional: also report p_wrong descriptively (it is 1 - p_correct)
# ------------------------------------------------------------
print(f"\nTrial {TRIAL_END} mean p_correct = {barnes_t6['p_correct'].mean():.3f}, mean p_wrong = {barnes_t6['p_wrong'].mean():.3f}")


# =========================
# Barnes sensitivity endpoints at Trial 6 (FDR within family)
# - Secondary latency: Goal_Box_latency_new (log transform; robust OLS)
# - Entry_latency_new (log transform; robust OLS)
# - DistanceMoved_cm (robust OLS)
# - EntryZone_freq_new (count; NB GLM)
# =========================

sens_tests = []

# Secondary latency outcomes (continuous): use robust OLS on log scale
for lat_col in ["Goal_Box_latency_new", "Entry_latency_new"]:
    if lat_col in barnes.columns:
        d = barnes_m[barnes_m["Trial"] == TRIAL_END].copy()
        d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
        d = d.dropna(subset=[lat_col, "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS"]).copy()
        d = d[d[lat_col] > 0].copy()
        d["log_lat"] = np.log(d[lat_col].astype(float))

        fit = smf.ols(
            "log_lat ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
            data=d
        ).fit(cov_type="HC3")

        sens_tests.append({"outcome": lat_col, "model": "OLS(log)", "p_light": float(fit.pvalues.get("C(Light_new)[T.ISF]", np.nan))})

# Distance moved (continuous): robust OLS
if "DistanceMoved_cm" in barnes.columns:
    d = barnes_m[barnes_m["Trial"] == TRIAL_END].copy()
    d["DistanceMoved_cm"] = pd.to_numeric(d["DistanceMoved_cm"], errors="coerce")
    d = d.dropna(subset=["DistanceMoved_cm", "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS"]).copy()

    fit = smf.ols(
        "DistanceMoved_cm ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
        data=d
    ).fit(cov_type="HC3")

    sens_tests.append({"outcome": "DistanceMoved_cm", "model": "OLS", "p_light": float(fit.pvalues.get("C(Light_new)[T.ISF]", np.nan))})

# EntryZone_freq_new (count): NB GLM
if "EntryZone_freq_new" in barnes.columns:
    d = barnes_m[barnes_m["Trial"] == TRIAL_END].copy()
    d["EntryZone_freq_new"] = pd.to_numeric(d["EntryZone_freq_new"], errors="coerce")
    d = d.dropna(subset=["EntryZone_freq_new", "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS"]).copy()
    d = d[d["EntryZone_freq_new"] >= 0].copy()
    d["EntryZone_freq_new"] = d["EntryZone_freq_new"].astype(int)

    fit = smf.glm(
        "EntryZone_freq_new ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
        data=d,
        family=sm.families.NegativeBinomial()
    ).fit(cov_type="HC3")

    sens_tests.append({"outcome": "EntryZone_freq_new", "model": "NB GLM", "p_light": float(fit.pvalues.get("C(Light_new)[T.ISF]", np.nan))})

if sens_tests:
    sens_df = pd.DataFrame(sens_tests).dropna()
    rej, p_fdr, _, _ = multipletests(sens_df["p_light"].values, method="fdr_bh", alpha=0.05)
    sens_df["p_fdr_bh"] = p_fdr
    sens_df["sig_fdr_0.05"] = rej
    print(f"\n=== Barnes Trial {TRIAL_END} sensitivity outcomes (FDR within family) ===")
    print(sens_df.sort_values("p_light").to_string(index=False))
else:
    print(f"\nNo Barnes sensitivity outcomes available for Trial {TRIAL_END}.")

# =========================
# 6) NOR: discrimination index (DI) and robust linear model
# =========================
require_columns(nor, ["ID", "Light_new", "Age_new", "Sex_new"], "UCBAge_Novel_clean.csv (cleaned)")
require_columns(nor, ["N_obj_nose_duration_s", "F_obj_nose_duration_s"], "UCBAge_Novel_clean.csv (needs durations for DI)")

nor_m = nor.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")
nor_m = nor_m.dropna(subset=[
    "N_obj_nose_duration_s", "F_obj_nose_duration_s",
    "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS"
]).copy()

n = pd.to_numeric(nor_m["N_obj_nose_duration_s"], errors="coerce")
f = pd.to_numeric(nor_m["F_obj_nose_duration_s"], errors="coerce")
nor_m["DI_duration"] = (n - f) / (n + f + 1e-9)

nor_fit = smf.ols(
    "DI_duration ~ Light_new + IS_pre + delta_IS + Age_new + Sex_new",
    data=nor_m
).fit(cov_type="HC3")

print("\n=== NOR OLS with robust (HC3) SEs ===")
print(nor_fit.summary())


# =========================
# 7) Mediation (bootstrap, cluster by mouse)
# Light -> delta_IS -> Behaviour endpoints
# =========================
def barnes_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    max_trial = df["Trial"].max()
    last_trials = df[df["Trial"].isin([max_trial - 1, max_trial])].copy()
    endp = last_trials.groupby("ID")[NOSEPOKE].mean().rename("barnes_endp").reset_index()
    return endp

barnes_endp = barnes_endpoint(barnes_m)

med_df = (
    circ_mouse.merge(barnes_endp, on="ID", how="left")
             .merge(nor_m.groupby("ID")["DI_duration"].mean().rename("nor_endp").reset_index(), on="ID", how="left")
)

med_barnes = med_df.dropna(subset=["Light_new_mouse", "Age_new_mouse", "Sex_new_mouse", "IS_pre", "delta_IS", "barnes_endp"]).copy()
med_nor = med_df.dropna(subset=["Light_new_mouse", "Age_new_mouse", "Sex_new_mouse", "IS_pre", "delta_IS", "nor_endp"]).copy()

def bootstrap_mediation(df: pd.DataFrame, y_col: str, n_boot: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = df["ID"].dropna().unique()

    # Use mouse-level covariates consistently (from circ_mouse)
    a_formula = "delta_IS ~ Light_new_mouse + Age_new_mouse + Sex_new_mouse + IS_pre"
    b_formula = f"{y_col} ~ Light_new_mouse + delta_IS + Age_new_mouse + Sex_new_mouse + IS_pre"

    a_fit = smf.ols(a_formula, data=df).fit()
    b_fit = smf.ols(b_formula, data=df).fit()

    light_terms = [t for t in a_fit.params.index if t.startswith("Light_new_mouse")]
    if len(light_terms) != 1:
        raise RuntimeError(f"Unexpected Light coding in a-model terms: {light_terms}")

    a = float(a_fit.params[light_terms[0]])
    b = float(b_fit.params["delta_IS"])
    indirect = a * b

    boots = []
    for _ in range(n_boot):
        samp_ids = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([df[df["ID"] == i] for i in samp_ids], axis=0, ignore_index=True)

        af = smf.ols(a_formula, data=boot).fit()
        bf = smf.ols(b_formula, data=boot).fit()

        lt = [t for t in af.params.index if t.startswith("Light_new_mouse")]
        if (len(lt) != 1) or ("delta_IS" not in bf.params.index):
            continue
        boots.append(float(af.params[lt[0]]) * float(bf.params["delta_IS"]))

    boots = np.array(boots, dtype=float)
    if len(boots) < 200:
        ci_lo, ci_hi = (np.nan, np.nan)
    else:
        ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975])

    return {
        "endpoint": y_col,
        "indirect_a_times_b": indirect,
        "boot_n": int(len(boots)),
        "ci_2_5pct": float(ci_lo),
        "ci_97_5pct": float(ci_hi),
        "a_model_p_light": float(a_fit.pvalues[light_terms[0]]),
        "b_model_p_deltaIS": float(b_fit.pvalues["delta_IS"]),
        "cprime_p_light": float(b_fit.pvalues[light_terms[0]]),
    }

print("\n=== Mediation: Barnes endpoint ===")
if len(med_barnes) >= 10:
    print(bootstrap_mediation(med_barnes, "barnes_endp"))
else:
    print("Not enough mice with Barnes endpoint for stable bootstrap mediation.")

print("\n=== Mediation: NOR endpoint ===")
if len(med_nor) >= 10:
    print(bootstrap_mediation(med_nor, "nor_endp"))
else:
    print("Not enough mice with NOR endpoint for stable bootstrap mediation.")

print("\nDONE.")

