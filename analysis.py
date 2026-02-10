import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# For Poisson GLMM (Bayesian MAP estimation)
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM


# =========================
# 0) Load
# =========================
circ = pd.read_csv("Circadian_raw.csv")
barnes = pd.read_csv("Barnes_clean.csv")
nor = pd.read_csv("UCBAge_Novel_clean.csv")

# Harmonise ID column name
nor = nor.rename(columns={"Animal_ID": "ID"})

# Coerce IDs
for df in (circ, barnes, nor):
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

# Ensure categorical coding, with CTR as reference and ISF as treatment
def set_cats(df, col, cats):
    if col in df.columns:
        df[col] = df[col].astype("category").cat.set_categories(cats, ordered=True)
    return df

# circadian
circ = set_cats(circ, "PRE.POST", ["PRE", "POST"])
circ = set_cats(circ, "Light_new", ["CTR", "ISF"])
circ["Age_new"] = circ["Age_new"].astype("category")
circ["Sex_new"] = circ["Sex_new"].astype("category")

# barnes
barnes = set_cats(barnes, "Light_new", ["CTR", "ISF"])
barnes["Age_new"] = barnes["Age_new"].astype("category")
barnes["Sex_new"] = barnes["Sex_new"].astype("category")
barnes["Trial"] = pd.to_numeric(barnes["Trial"], errors="coerce")

# NOR
nor = set_cats(nor, "Light_new", ["CTR", "ISF"])
nor["Age_new"] = nor["Age_new"].astype("category")
nor["Sex_new"] = nor["Sex_new"].astype("category")


# =========================
# 1) QC: who has PRE and POST
# =========================
prepost_counts = circ.groupby("ID")["PRE.POST"].nunique()
missing_pair = prepost_counts[prepost_counts < 2].index.tolist()
print(f"\nCircadian mice total: {circ['ID'].nunique()}")
print(f"Mice missing PRE or POST in circadian: {len(missing_pair)}")


# =========================
# 2) Circadian mixed model
# Primary: IS (best single summary for “circadian regularity”)
# Model: IS ~ PRE.POST * Light_new + Age_new + Sex_new + (PRE.POST | ID)
# =========================
def fit_circadian(outcome="IS"):
    d = circ[["ID", "PRE.POST", "Light_new", "Age_new", "Sex_new", outcome]].dropna().copy()

    # Scale continuous outcome (optional) improves optimiser stability; does NOT change inference on fixed effects
    d[f"z_{outcome}"] = (d[outcome] - d[outcome].mean()) / d[outcome].std(ddof=0)

    formula = f"z_{outcome} ~ PRE.POST * Light_new + Age_new + Sex_new"

    # random intercept + random slope for PRE.POST
    md = smf.mixedlm(formula, d, groups=d["ID"], re_formula="~PRE.POST")
    m = md.fit(method="lbfgs", reml=True)

    print(f"\n=== Circadian MixedLM: {outcome} ===")
    print(m.summary())

    return m, d

circ_model_IS, circ_used = fit_circadian("IS")

# Optional secondary circadian checks with FDR
secondary = [x for x in ["IV", "RA", "Amplitude"] if x in circ.columns]
sec_results = []
sec_models = {}

for out in secondary:
    m, d = fit_circadian(out)
    sec_models[out] = (m, d)
    # store the treatment-modulated change: PRE.POST[T.POST]:Light_new[T.ISF]
    for term in m.pvalues.index:
        if "PRE.POST[T.POST]:Light_new" in term:
            sec_results.append({"outcome": out, "term": term, "p": float(m.pvalues[term])})

if sec_results:
    sec_df = pd.DataFrame(sec_results)
    rej, p_fdr, _, _ = multipletests(sec_df["p"].values, method="fdr_bh", alpha=0.05)
    sec_df["p_fdr_bh"] = p_fdr
    sec_df["sig_fdr_0.05"] = rej
    print("\n=== Secondary circadian interaction terms (FDR controlled) ===")
    print(sec_df.sort_values(["outcome", "term"]).to_string(index=False))


# =========================
# 3) Compute per-mouse ΔIS = POST − PRE (predictor for behaviour)
# =========================
wide_IS = circ.pivot_table(index="ID", columns="PRE.POST", values="IS", aggfunc="mean")
circ_delta = (wide_IS.get("POST") - wide_IS.get("PRE")).rename("delta_IS").reset_index()

# Add stable mouse-level covariates
mouse_covars = (
    circ.sort_values(["ID", "PRE.POST"])
        .groupby("ID")[["Light_new", "Age_new", "Sex_new"]]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
        .reset_index()
)

circ_mouse = mouse_covars.merge(circ_delta, on="ID", how="left")


# =========================
# 4) Barnes: nose-poke frequency (count)
# You asked for “nose poke frequency”.
# In your file the closest columns are:
#   - EntryZone_freq_new  (count)
#   - Goal.Box_feq_new    (count)
# Choose ONE as primary; here I use EntryZone_freq_new unless you change it.
# =========================
BARNES_NOSEPOKE = "EntryZone_freq_new"  # change to "Goal.Box_feq_new" if that is your intended measure

barnes_m = barnes.merge(circ_mouse, on="ID", how="left")
barnes_m = barnes_m.dropna(subset=["ID", "Trial", "Light_new", "Age_new", "Sex_new", "delta_IS", BARNES_NOSEPOKE]).copy()

# --- 4A) Preferred: Poisson GLMM with random intercept per mouse
# Fixed: Trial + Light + delta_IS + Age + Sex (+ interaction delta_IS:Light if desired)
# Random: intercept per ID
def fit_barnes_poisson_glmm(df):
    d = df.copy()
    # Ensure integer non-negative counts
    d[BARNES_NOSEPOKE] = pd.to_numeric(d[BARNES_NOSEPOKE], errors="coerce").astype(int)
    d = d[d[BARNES_NOSEPOKE] >= 0].copy()

    # Build fixed effects design via formula
    # Note: PoissonBayesMixedGLM uses a patsy formula with `0 +` in vc formulas.
    formula = f"{BARNES_NOSEPOKE} ~ Trial + Light_new + delta_IS + Age_new + Sex_new"

    # Random intercept per mouse via variance components
    vc = {"ID_re": "0 + C(ID)"}

    model = PoissonBayesMixedGLM.from_formula(formula, vc_formulas=vc, data=d)
    # fit_map gives a MAP estimate; stable and fast for typical sample sizes
    res = model.fit_map()

    print(f"\n=== Barnes Poisson GLMM (MAP): {BARNES_NOSEPOKE} ===")
    print(res.summary())
    return res, d

# --- 4B) Fallback: Linear MixedLM on log1p(count), with (Trial | ID) random effects
def fit_barnes_log_mixedlm(df):
    d = df.copy()
    d[BARNES_NOSEPOKE] = pd.to_numeric(d[BARNES_NOSEPOKE], errors="coerce")
    d = d.dropna(subset=[BARNES_NOSEPOKE]).copy()
    d["log1p_count"] = np.log1p(d[BARNES_NOSEPOKE].astype(float))

    formula = "log1p_count ~ Trial + Light_new + delta_IS + Age_new + Sex_new"
    md = smf.mixedlm(formula, d, groups=d["ID"], re_formula="~Trial")
    m = md.fit(method="lbfgs", reml=True)

    print(f"\n=== Barnes fallback MixedLM on log1p(count): {BARNES_NOSEPOKE} ===")
    print(m.summary())
    return m, d

try:
    barnes_glmm, barnes_used = fit_barnes_poisson_glmm(barnes_m)
except Exception as e:
    print(f"\n[NOTE] Poisson GLMM failed ({e}). Using fallback MixedLM on log1p(count).")
    barnes_lmm, barnes_used = fit_barnes_log_mixedlm(barnes_m)


# =========================
# 5) NOR: circadian ΔIS predicting object recognition
# Compute duration-based discrimination index (DI)
# DI_duration = (Novel - Familiar) / (Novel + Familiar)
# =========================
nor_m = nor.merge(circ_mouse, on="ID", how="left")

needed = ["N_obj_nose_duration_s", "F_obj_nose_duration_s", "delta_IS", "Light_new", "Age_new", "Sex_new"]
nor_m = nor_m.dropna(subset=[c for c in needed if c in nor_m.columns]).copy()

n = pd.to_numeric(nor_m["N_obj_nose_duration_s"], errors="coerce")
f = pd.to_numeric(nor_m["F_obj_nose_duration_s"], errors="coerce")
nor_m["DI_duration"] = (n - f) / (n + f + 1e-9)

# Optional frequency-based DI as a sensitivity analysis
if "N_obj_nose_frequency" in nor_m.columns and "F_nose_frequency" in nor_m.columns:
    nf = pd.to_numeric(nor_m["N_obj_nose_frequency"], errors="coerce")
    ff = pd.to_numeric(nor_m["F_nose_frequency"], errors="coerce")
    nor_m["DI_frequency"] = (nf - ff) / (nf + ff + 1e-9)

print("\n=== NOR OLS: DI_duration ~ delta_IS + Light + Age + Sex ===")
nor_fit = smf.ols("DI_duration ~ delta_IS + Light_new + Age_new + Sex_new", data=nor_m).fit()
print(nor_fit.summary())

if "DI_frequency" in nor_m.columns:
    print("\n=== NOR OLS (sensitivity): DI_frequency ~ delta_IS + Light + Age + Sex ===")
    nor_fit2 = smf.ols("DI_frequency ~ delta_IS + Light_new + Age_new + Sex_new", data=nor_m).fit()
    print(nor_fit2.summary())

print("\nDONE.")

