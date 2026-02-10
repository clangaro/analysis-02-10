"""
Publication-level analysis pipeline (fixed for Patsy column-name issues)

WHAT THIS SCRIPT DOES (high level):
1) Circadian: Mixed-effects model for Light (ISF vs CTR) x PRE/POST on IS (primary),
   with model selection for random slope (PRE/POST | mouse). Secondary rhythm metrics optional with FDR.
2) Behaviour:
   - Barnes: nose-poke COUNT across trials with Poisson mixed model (with optional OLRE for overdispersion),
     using baseline IS (PRE) + delta IS (POST-PRE) as predictors.
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


# =========================
# 5) Barnes: nose-poke frequency (count) as Poisson mixed model
# =========================
# Choose your nose-poke column (after cleaning).
# Common options in your file:
# - EntryZone_freq_new
# - Goal_Box_feq_new  (this is the cleaned version of Goal.Box_feq_new)
NOSEPOKE = "EntryZone_freq_new"  # change to "Goal_Box_feq_new" if that is your intended measure

require_columns(barnes, ["ID", "Trial", "Light_new", "Age_new", "Sex_new", NOSEPOKE], "Barnes_clean.csv (cleaned)")

barnes_m = barnes.merge(circ_mouse, on="ID", how="left")
barnes_m = barnes_m.dropna(subset=["ID", "Trial", "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS", NOSEPOKE]).copy()

barnes_m[NOSEPOKE] = pd.to_numeric(barnes_m[NOSEPOKE], errors="coerce").astype(int)
barnes_m = barnes_m[barnes_m[NOSEPOKE] >= 0].copy()
barnes_m["obs_id"] = np.arange(len(barnes_m)).astype(int)

def fit_barnes_poisson(df: pd.DataFrame, use_olre: bool = False):
    # Fixed effects: Trial + Light + baseline IS + delta IS + covariates
    formula = f"{NOSEPOKE} ~ Trial + Light_new + IS_pre + delta_IS + Age_new + Sex_new"

    vc = {"mouse_re": "0 + C(ID)"}
    if use_olre:
        vc["olre"] = "0 + C(obs_id)"

    model = PoissonBayesMixedGLM.from_formula(formula, vc_formulas=vc, data=df)
    res = model.fit_map()
    return res

# Fit without OLRE
barnes_res = fit_barnes_poisson(barnes_m, use_olre=False)
print(f"\n=== Barnes Poisson mixed model (no OLRE): {NOSEPOKE} ===")
print(barnes_res.summary())

# Crude overdispersion screen (Var/Mean)
mean_y = barnes_m[NOSEPOKE].mean()
var_y = barnes_m[NOSEPOKE].var(ddof=1)
disp_ratio = var_y / mean_y if mean_y > 0 else np.nan
print(f"\n[Barnes] crude overdispersion screen Var/Mean = {disp_ratio:.3f}")

barnes_res_olre = None
if np.isfinite(disp_ratio) and disp_ratio > 1.5:
    barnes_res_olre = fit_barnes_poisson(barnes_m, use_olre=True)
    print(f"\n=== Barnes Poisson mixed model WITH OLRE (recommended under overdispersion): {NOSEPOKE} ===")
    print(barnes_res_olre.summary())


# =========================
# 6) NOR: discrimination index (DI) and robust linear model
# =========================
# After cleaning, these typically remain:
# N_obj_nose_duration_s, F_obj_nose_duration_s
require_columns(nor, ["ID", "Light_new", "Age_new", "Sex_new"], "UCBAge_Novel_clean.csv (cleaned)")
require_columns(nor, ["N_obj_nose_duration_s", "F_obj_nose_duration_s"], "UCBAge_Novel_clean.csv (needs durations for DI)")

nor_m = nor.merge(circ_mouse, on="ID", how="left")
nor_m = nor_m.dropna(subset=["N_obj_nose_duration_s", "F_obj_nose_duration_s", "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS"]).copy()

n = pd.to_numeric(nor_m["N_obj_nose_duration_s"], errors="coerce")
f = pd.to_numeric(nor_m["F_obj_nose_duration_s"], errors="coerce")
nor_m["DI_duration"] = (n - f) / (n + f + 1e-9)

# Robust SEs are used to reduce sensitivity to heteroskedasticity/outliers in small samples
nor_fit = smf.ols("DI_duration ~ Light_new + IS_pre + delta_IS + Age_new + Sex_new", data=nor_m).fit(cov_type="HC3")
print("\n=== NOR OLS with robust (HC3) SEs ===")
print(nor_fit.summary())


# =========================
# 7) Mediation (bootstrap, cluster by mouse)
# Light -> delta_IS -> Behaviour endpoints
# =========================
def barnes_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    """Define a per-mouse Barnes endpoint: mean nosepokes in the last two trials."""
    max_trial = df["Trial"].max()
    last_trials = df[df["Trial"].isin([max_trial - 1, max_trial])].copy()
    endp = last_trials.groupby("ID")[NOSEPOKE].mean().rename("barnes_endp").reset_index()
    return endp

barnes_endp = barnes_endpoint(barnes_m)

med_df = (
    circ_mouse.merge(barnes_endp, on="ID", how="left")
             .merge(nor_m.groupby("ID")["DI_duration"].mean().rename("nor_endp").reset_index(), on="ID", how="left")
)

med_barnes = med_df.dropna(subset=["Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS", "barnes_endp"]).copy()
med_nor = med_df.dropna(subset=["Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS", "nor_endp"]).copy()

def bootstrap_mediation(df: pd.DataFrame, y_col: str, n_boot: int = 5000, seed: int = 0):
    """
    Cluster bootstrap by mouse ID.
    a-model: delta_IS ~ Light + covariates + baseline
    b-model: y ~ Light + delta_IS + covariates + baseline
    indirect = a * b
    """
    rng = np.random.default_rng(seed)
    ids = df["ID"].dropna().unique()

    a_formula = "delta_IS ~ Light_new + Age_new + Sex_new + IS_pre"
    b_formula = f"{y_col} ~ Light_new + delta_IS + Age_new + Sex_new + IS_pre"

    a_fit = smf.ols(a_formula, data=df).fit()
    b_fit = smf.ols(b_formula, data=df).fit()

    # Identify the Light indicator term (ISF vs CTR)
    light_terms = [t for t in a_fit.params.index if t.startswith("Light_new")]
    if len(light_terms) != 1:
        raise RuntimeError(f"Unexpected Light_new coding in a-model terms: {light_terms}")

    a = float(a_fit.params[light_terms[0]])
    b = float(b_fit.params["delta_IS"])
    indirect = a * b

    boots = []
    for _ in range(n_boot):
        samp_ids = rng.choice(ids, size=len(ids), replace=True)
        boot = pd.concat([df[df["ID"] == i] for i in samp_ids], axis=0, ignore_index=True)

        af = smf.ols(a_formula, data=boot).fit()
        bf = smf.ols(b_formula, data=boot).fit()

        lt = [t for t in af.params.index if t.startswith("Light_new")]
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
