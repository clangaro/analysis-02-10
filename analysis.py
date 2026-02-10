import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM

# -----------------------------
# 0) Load
# -----------------------------
circ = pd.read_csv("Circadian_raw.csv")
barnes = pd.read_csv("Barnes_clean.csv")
nor = pd.read_csv("UCBAge_Novel_clean.csv").rename(columns={"Animal_ID": "ID"})

for df in (circ, barnes, nor):
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

def set_cats(df, col, cats):
    if col in df.columns:
        df[col] = df[col].astype("category").cat.set_categories(cats, ordered=True)
    return df

# Confirmed: ISF is 40 Hz light group; CTR is control
circ = set_cats(circ, "PRE.POST", ["PRE", "POST"])
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

# -----------------------------
# 1) Circadian: model selection for random slope (PRE.POST | ID)
# Primary: IS
# -----------------------------
def logit_clip(x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))

def fit_circadian_mixedlm(outcome, use_logit_if_bounded=True):
    d = circ[["ID", "PRE.POST", "Light_new", "Age_new", "Sex_new", outcome]].dropna().copy()

    y = outcome
    if use_logit_if_bounded:
        # IS and RA are typically bounded in [0,1] in these datasets
        if d[outcome].min() >= 0 and d[outcome].max() <= 1:
            d[f"{outcome}_logit"] = logit_clip(d[outcome])
            y = f"{outcome}_logit"

    # Fixed effects: main hypothesis is the interaction POST×ISF
    formula = f"{y} ~ PRE.POST * Light_new + Age_new + Sex_new"

    # Random intercept only (ML) for comparison
    m_ri_ml = smf.mixedlm(formula, d, groups=d["ID"]).fit(method="lbfgs", reml=False)

    # Random intercept + random slope for PRE.POST (ML) for comparison
    m_rs_ml = smf.mixedlm(formula, d, groups=d["ID"], re_formula="~PRE.POST").fit(method="lbfgs", reml=False)

    # Compare by AIC (practical) and LR test (approximate; conservative in mixed models)
    aic_ri, aic_rs = m_ri_ml.aic, m_rs_ml.aic
    lr_stat = 2 * (m_rs_ml.llf - m_ri_ml.llf)
    df_lr = (len(m_rs_ml.params) - len(m_ri_ml.params))
    # Chi-square approximation
    from scipy.stats import chi2
    p_lr = 1 - chi2.cdf(lr_stat, df_lr) if df_lr > 0 else np.nan

    chosen = "RI+RS" if aic_rs < aic_ri else "RI"
    print(f"\n[Circadian {outcome}] AIC RI={aic_ri:.2f} | RI+RS={aic_rs:.2f} -> choose {chosen} (LR p≈{p_lr:.4g})")

    # Refit final chosen model with REML (recommended for final parameter estimates)
    if chosen == "RI+RS":
        m_final = smf.mixedlm(formula, d, groups=d["ID"], re_formula="~PRE.POST").fit(method="lbfgs", reml=True)
    else:
        m_final = smf.mixedlm(formula, d, groups=d["ID"]).fit(method="lbfgs", reml=True)

    print(m_final.summary())
    return m_final, d, y

# Primary circadian model
circ_IS_model, circ_IS_used, circ_IS_y = fit_circadian_mixedlm("IS")

# Secondary circadian outcomes with FDR on the *interaction* term(s)
secondary = [m for m in ["IV", "RA", "Amplitude"] if m in circ.columns]
sec_rows = []
sec_models = {}
for out in secondary:
    m, d, y = fit_circadian_mixedlm(out)
    sec_models[out] = m
    for term in m.pvalues.index:
        if "PRE.POST[T.POST]:Light_new" in term:
            sec_rows.append({"outcome": out, "term": term, "p": float(m.pvalues[term]), "beta": float(m.params[term])})

if sec_rows:
    sec_df = pd.DataFrame(sec_rows)
    rej, p_fdr, _, _ = multipletests(sec_df["p"].values, method="fdr_bh", alpha=0.05)
    sec_df["p_fdr_bh"] = p_fdr
    sec_df["sig_fdr_0.05"] = rej
    print("\nSecondary circadian interaction terms (FDR BH):")
    print(sec_df.sort_values(["outcome", "term"]).to_string(index=False))

# -----------------------------
# 2) Compute ΔIS and IS_PRE for behaviour + mediation
# -----------------------------
wide_IS = circ.pivot_table(index="ID", columns="PRE.POST", values="IS", aggfunc="mean")
delta_IS = (wide_IS.get("POST") - wide_IS.get("PRE")).rename("delta_IS")
IS_pre = wide_IS.get("PRE").rename("IS_pre")

mouse_covars = (
    circ.sort_values(["ID", "PRE.POST"])
        .groupby("ID")[["Light_new", "Age_new", "Sex_new"]]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
)

circ_mouse = pd.concat([mouse_covars, IS_pre, delta_IS], axis=1).reset_index()

# -----------------------------
# 3) Barnes (nose-poke count): Poisson mixed model with overdispersion check
# Primary nose poke: EntryZone_freq_new
# -----------------------------
NOSEPOKE = "EntryZone_freq_new"

barnes_m = barnes.merge(circ_mouse, on="ID", how="left")
barnes_m = barnes_m.dropna(subset=["ID", "Trial", "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS", NOSEPOKE]).copy()
barnes_m[NOSEPOKE] = pd.to_numeric(barnes_m[NOSEPOKE], errors="coerce").astype(int)
barnes_m = barnes_m[barnes_m[NOSEPOKE] >= 0].copy()

# Create an observation ID for OLRE (overdispersion)
barnes_m["obs_id"] = np.arange(len(barnes_m)).astype(int)

def fit_barnes_poisson(df, use_olre=False):
    # Fixed effects: learning trend (Trial), treatment, baseline rhythm, change in rhythm, covariates
    formula = f"{NOSEPOKE} ~ Trial + Light_new + IS_pre + delta_IS + Age_new + Sex_new"

    vc = {"mouse_re": "0 + C(ID)"}
    if use_olre:
        vc["olre"] = "0 + C(obs_id)"

    model = PoissonBayesMixedGLM.from_formula(formula, vc_formulas=vc, data=df)
    res = model.fit_map()
    return res

# Fit without OLRE first
barnes_res = fit_barnes_poisson(barnes_m, use_olre=False)
print("\n=== Barnes Poisson mixed model (no OLRE) ===")
print(barnes_res.summary())

# Overdispersion diagnostic (approximate):
# Compare observed variance to mean at the raw level as a quick screen
mean_y = barnes_m[NOSEPOKE].mean()
var_y = barnes_m[NOSEPOKE].var(ddof=1)
disp_ratio = var_y / mean_y if mean_y > 0 else np.nan
print(f"\n[Barnes] crude overdispersion screen: Var/Mean = {disp_ratio:.3f}")

# If Var/Mean substantially > 1, refit with OLRE
if np.isfinite(disp_ratio) and disp_ratio > 1.5:
    barnes_res_olre = fit_barnes_poisson(barnes_m, use_olre=True)
    print("\n=== Barnes Poisson mixed model WITH OLRE (recommended under overdispersion) ===")
    print(barnes_res_olre.summary())
else:
    barnes_res_olre = None

# -----------------------------
# 4) NOR: Discrimination Index and robust regression
# -----------------------------
nor_m = nor.merge(circ_mouse, on="ID", how="left")
nor_m = nor_m.dropna(subset=["N_obj_nose_duration_s", "F_obj_nose_duration_s", "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS"]).copy()

n = pd.to_numeric(nor_m["N_obj_nose_duration_s"], errors="coerce")
f = pd.to_numeric(nor_m["F_obj_nose_duration_s"], errors="coerce")
nor_m["DI_duration"] = (n - f) / (n + f + 1e-9)

# Model includes baseline and change to reduce regression-to-the-mean artefacts
nor_fit = smf.ols("DI_duration ~ Light_new + IS_pre + delta_IS + Age_new + Sex_new", data=nor_m).fit(cov_type="HC3")
print("\n=== NOR OLS with robust (HC3) SEs ===")
print(nor_fit.summary())

# -----------------------------
# 5) Mediation (bootstrap): Light -> delta_IS -> Behaviour
# Use per-mouse endpoints for behaviour.
# Barnes endpoint: mean nosepokes in final trials (default: max Trial - 1 and max Trial)
# -----------------------------
def barnes_endpoint(df):
    # Define "final performance" as mean of last two trials (publication-friendly and simple)
    max_trial = df["Trial"].max()
    last_trials = df[df["Trial"].isin([max_trial - 1, max_trial])].copy()
    endp = last_trials.groupby("ID")[NOSEPOKE].mean().rename("barnes_endp").reset_index()
    return endp

barnes_endp = barnes_endpoint(barnes_m)
med_df = circ_mouse.merge(barnes_endp, on="ID", how="left").merge(
    nor_m.groupby("ID")["DI_duration"].mean().rename("nor_endp").reset_index(), on="ID", how="left"
)

# Drop mice missing endpoints (you can run mediation separately per endpoint)
med_barnes = med_df.dropna(subset=["Light_new", "Age_new", "Sex_new", "delta_IS", "barnes_endp"]).copy()
med_nor = med_df.dropna(subset=["Light_new", "Age_new", "Sex_new", "delta_IS", "nor_endp"]).copy()

def bootstrap_mediation(df, y_col, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    ids = df["ID"].dropna().unique()

    # Path a: delta_IS ~ Light + covars
    # Path b/c': y ~ Light + delta_IS + covars
    a_formula = "delta_IS ~ Light_new + Age_new + Sex_new + IS_pre"
    b_formula = f"{y_col} ~ Light_new + delta_IS + Age_new + Sex_new + IS_pre"

    # Fit on full sample for point estimate
    a_fit = smf.ols(a_formula, data=df).fit()
    b_fit = smf.ols(b_formula, data=df).fit()
    # Extract Light effect in a-model (ISF vs CTR) and delta_IS effect in b-model
    a_term = [t for t in a_fit.params.index if "Light_new" in t]
    if len(a_term) != 1:
        raise RuntimeError("Unexpected Light_new coding; check categories.")
    a = float(a_fit.params[a_term[0]])
    b = float(b_fit.params["delta_IS"])
    indirect = a * b

    # Bootstrap by resampling mice (cluster bootstrap)
    boots = []
    for _ in range(n_boot):
        samp_ids = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([df[df["ID"] == i] for i in samp_ids], axis=0, ignore_index=True)

        af = smf.ols(a_formula, data=boot).fit()
        bf = smf.ols(b_formula, data=boot).fit()

        a_term_b = [t for t in af.params.index if "Light_new" in t]
        if len(a_term_b) != 1 or "delta_IS" not in bf.params.index:
            continue
        boots.append(float(af.params[a_term_b[0]]) * float(bf.params["delta_IS"]))

    boots = np.array(boots, dtype=float)
    ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975]) if len(boots) > 100 else (np.nan, np.nan)

    return {
        "y": y_col,
        "indirect_a_times_b": indirect,
        "boot_n": int(len(boots)),
        "ci_2.5%": float(ci_lo),
        "ci_97.5%": float(ci_hi),
        "a_model_p": float(a_fit.pvalues[a_term[0]]),
        "b_model_p": float(b_fit.pvalues["delta_IS"]),
        "cprime_light_p": float(b_fit.pvalues[a_term[0]])
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
