import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM


# =========================
# Helpers
# =========================
def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

def require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}\n"
            f"Available columns (first 60): {df.columns.tolist()[:60]}"
        )

def logit_clip(p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# =========================
# Load + clean
# =========================
circ = clean_colnames(pd.read_csv("Circadian_raw.csv"))
barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))
nor = clean_colnames(pd.read_csv("UCBAge_Novel_clean.csv"))

if "Animal_ID" in nor.columns and "ID" not in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})

for df in (circ, barnes, nor):
    require_columns(df, ["ID"], "Dataset")
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

# Required circadian columns after cleaning
require_columns(circ, ["PRE_POST", "Light_new", "Age_new", "Sex_new", "IS"], "Circadian_raw.csv")

# Ensure categorical predictors exist (we will convert to strings later where needed)
circ["PRE_POST"] = circ["PRE_POST"].astype(str)
circ["Light_new"] = circ["Light_new"].astype(str)
circ["Age_new"] = circ["Age_new"].astype(str)
circ["Sex_new"] = circ["Sex_new"].astype(str)

barnes["Light_new"] = barnes["Light_new"].astype(str)
barnes["Age_new"] = barnes["Age_new"].astype(str)
barnes["Sex_new"] = barnes["Sex_new"].astype(str)
barnes["Trial"] = pd.to_numeric(barnes["Trial"], errors="coerce")

nor["Light_new"] = nor["Light_new"].astype(str)
nor["Age_new"] = nor["Age_new"].astype(str)
nor["Sex_new"] = nor["Sex_new"].astype(str)


# =========================
# Circadian: compute IS_pre and delta_IS
# =========================
wide_IS = circ.pivot_table(index="ID", columns="PRE_POST", values="IS", aggfunc="mean")
IS_pre = wide_IS.get("PRE").rename("IS_pre")
delta_IS = (wide_IS.get("POST") - wide_IS.get("PRE")).rename("delta_IS")

# mouse-level covariates from circadian table
mouse_covars = (
    circ.sort_values(["ID", "PRE_POST"])
        .groupby("ID")[["Light_new", "Age_new", "Sex_new"]]
        .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan)
)

circ_mouse = pd.concat([mouse_covars, IS_pre, delta_IS], axis=1).reset_index()
circ_mouse = circ_mouse.rename(columns={"Light_new": "Light_new_mouse",
                                        "Age_new": "Age_new_mouse",
                                        "Sex_new": "Sex_new_mouse"})

print(f"\nMouse-level rows: {circ_mouse.shape[0]}")


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# -------------------------
# Helpers (keep your existing clean_colnames/require_columns/logit_clip)
# -------------------------
def logit_clip(p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ============================================================
# 1) BARNES: Negative Binomial GEE (overdispersion-robust)
# ============================================================
NOSEPOKE = "EntryZone_freq_new"  # or "Goal_Box_feq_new"

# barnes_m must already be merged with circ_mouse and cleaned, as in your script
# Ensure these columns exist in barnes_m:
#   ID, Trial, Light_new, Age_new, Sex_new, IS_pre, delta_IS, NOSEPOKE

# Coerce types
barnes_m = barnes_m.copy()
barnes_m["ID_str"] = barnes_m["ID"].astype(str)
barnes_m["Trial"] = pd.to_numeric(barnes_m["Trial"], errors="coerce")
barnes_m[NOSEPOKE] = pd.to_numeric(barnes_m[NOSEPOKE], errors="coerce")
barnes_m = barnes_m.dropna(subset=["ID_str", "Trial", "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS", NOSEPOKE]).copy()

# Count must be non-negative integer for NB count modelling
barnes_m = barnes_m[barnes_m[NOSEPOKE] >= 0].copy()
barnes_m[NOSEPOKE] = barnes_m[NOSEPOKE].astype(int)

# Quick overdispersion check
mean_y = barnes_m[NOSEPOKE].mean()
var_y = barnes_m[NOSEPOKE].var(ddof=1)
print(f"\n[Barnes] Var/Mean = {var_y / mean_y:.3f} (>>1 supports NB over Poisson)")

# Negative Binomial GEE with robust SE; Exchangeable within-mouse correlation
gee_nb = smf.gee(
    formula=f"{NOSEPOKE} ~ Trial + C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    groups="ID_str",
    data=barnes_m,
    family=sm.families.NegativeBinomial(),
    cov_struct=sm.cov_struct.Exchangeable()
).fit()

print("\n=== Barnes Negative Binomial GEE (clustered by mouse; robust SEs) ===")
print(gee_nb.summary())

# Interpretability: convert coefficients to rate ratios (RR = exp(beta))
print("\n[Barnes] Rate ratios (RR = exp(beta)) and 95% CI:")
params = gee_nb.params
conf = gee_nb.conf_int()
for term in params.index:
    rr = float(np.exp(params[term]))
    lo = float(np.exp(conf.loc[term, 0]))
    hi = float(np.exp(conf.loc[term, 1]))
    print(f"  {term:35s} RR={rr:.3f}  (95% CI {lo:.3f}–{hi:.3f})")


# ============================================================
# 2) NOR: robust regression models
# ============================================================
# nor_m must already be merged with circ_mouse and cleaned, as in your script
# Required: N_obj_nose_duration_s, F_obj_nose_duration_s, plus predictors

nor_m = nor_m.copy()

N_dur = pd.to_numeric(nor_m["N_obj_nose_duration_s"], errors="coerce")
F_dur = pd.to_numeric(nor_m["F_obj_nose_duration_s"], errors="coerce")
nor_m = nor_m[(N_dur.notna()) & (F_dur.notna())].copy()
N_dur = N_dur.loc[nor_m.index]
F_dur = F_dur.loc[nor_m.index]

# (A) Preference proportion
nor_m["p_novel_dur"] = (N_dur / (N_dur + F_dur + 1e-9)).clip(1e-6, 1 - 1e-6)
nor_m["logit_p_novel_dur"] = logit_clip(nor_m["p_novel_dur"].values)

# (B) Discrimination index
nor_m["DI_dur"] = (N_dur - F_dur) / (N_dur + F_dur + 1e-9)

# Robust OLS (HC3) for both outcomes
nor_fit_pref = smf.ols(
    "logit_p_novel_dur ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    data=nor_m
).fit(cov_type="HC3")

print("\n=== NOR Model 1: logit(preference proportion), robust (HC3) SEs ===")
print(nor_fit_pref.summary())

nor_fit_di = smf.ols(
    "DI_dur ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    data=nor_m
).fit(cov_type="HC3")

print("\n=== NOR Model 2: discrimination index, robust (HC3) SEs ===")
print(nor_fit_di.summary())

print("\nDONE.")
