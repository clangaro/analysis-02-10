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


# ============================================================
# 1) BARNES: Overdispersion-robust Poisson mixed model with OLRE
# ============================================================
# Choose your nosepoke column
NOSEPOKE = "EntryZone_freq_new"  # or "Goal_Box_feq_new" if that is the correct one

require_columns(barnes, ["Trial", "Light_new", "Age_new", "Sex_new", NOSEPOKE], "Barnes_clean.csv")

barnes_m = barnes.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")
barnes_m = barnes_m.dropna(subset=["ID", "Trial", "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS", NOSEPOKE]).copy()

# Coerce types for Patsy
barnes_m["ID_str"] = barnes_m["ID"].astype(str)
barnes_m["Trial"] = pd.to_numeric(barnes_m["Trial"], errors="coerce")
barnes_m[NOSEPOKE] = pd.to_numeric(barnes_m[NOSEPOKE], errors="coerce").astype(int)
barnes_m = barnes_m[(barnes_m[NOSEPOKE] >= 0) & barnes_m["Trial"].notna()].copy()

# Overdispersion quick screen
mean_y = barnes_m[NOSEPOKE].mean()
var_y = barnes_m[NOSEPOKE].var(ddof=1)
disp_ratio = var_y / mean_y if mean_y > 0 else np.nan
print(f"\n[Barnes] Var/Mean = {disp_ratio:.3f} (>>1 indicates overdispersion)")

# Observation-level random effect
barnes_m["obs_id"] = np.arange(len(barnes_m)).astype(int)

def fit_barnes_poisson_olre(df: pd.DataFrame):
    """
    Poisson mixed model with:
      - mouse random intercept (accounts for repeated measures)
      - OLRE random intercept (accounts for overdispersion)
    """
    # C() forces categorical interpretation
    formula = f"{NOSEPOKE} ~ Trial + C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)"

    vc = {
        "mouse_re": "0 + C(ID_str)",
        "olre":     "0 + C(obs_id)"
    }

    model = PoissonBayesMixedGLM.from_formula(formula, vc_formulas=vc, data=df)
    res = model.fit_map()
    return res

barnes_res_olre = fit_barnes_poisson_olre(barnes_m)
print(f"\n=== Barnes Poisson mixed model WITH OLRE (overdispersion-robust): {NOSEPOKE} ===")
print(barnes_res_olre.summary())

# Optional: exponentiate fixed-effect posterior means to interpret as rate ratios
# Note: Summary shows posterior means for coefficients on log scale.
print("\n[Barnes] Approximate rate ratios (exp(coef)) for fixed effects:")
for name, val in zip(barnes_res_olre.model.exog_names, barnes_res_olre.params[:len(barnes_res_olre.model.exog_names)]):
    print(f"  {name:35s} RR≈ {np.exp(val):.3f}")


# ==========================================
# 2) NOR: add behavioural models
# ==========================================
# Needed columns for duration-based measures
require_columns(nor, ["N_obj_nose_duration_s", "F_obj_nose_duration_s"], "UCBAge_Novel_clean.csv")

nor_m = nor.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")
nor_m = nor_m.dropna(subset=["Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS",
                             "N_obj_nose_duration_s", "F_obj_nose_duration_s"]).copy()

N_dur = pd.to_numeric(nor_m["N_obj_nose_duration_s"], errors="coerce")
F_dur = pd.to_numeric(nor_m["F_obj_nose_duration_s"], errors="coerce")
nor_m = nor_m[(N_dur.notna()) & (F_dur.notna())].copy()
N_dur = N_dur.loc[nor_m.index]
F_dur = F_dur.loc[nor_m.index]

# (A) Preference proportion: p = Novel / (Novel + Familiar)
# This is in [0,1] and often more stable than DI.
nor_m["p_novel_dur"] = (N_dur / (N_dur + F_dur + 1e-9)).clip(1e-6, 1 - 1e-6)
nor_m["logit_p_novel_dur"] = logit_clip(nor_m["p_novel_dur"].values)

# (B) Discrimination index: DI = (Novel - Familiar) / (Novel + Familiar)
nor_m["DI_dur"] = (N_dur - F_dur) / (N_dur + F_dur + 1e-9)

print("\n=== NOR model 1: logit(preference proportion) with robust SEs ===")
nor_fit_pref = smf.ols(
    "logit_p_novel_dur ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    data=nor_m
).fit(cov_type="HC3")
print(nor_fit_pref.summary())

print("\n=== NOR model 2: discrimination index (DI) with robust SEs ===")
nor_fit_di = smf.ols(
    "DI_dur ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    data=nor_m
).fit(cov_type="HC3")
print(nor_fit_di.summary())

print("\nDONE.")
