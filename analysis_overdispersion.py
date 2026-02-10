import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# =========================
# Helper functions
# =========================
def clean_colnames(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df

def logit_clip(p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# =========================
# Load data
# =========================
circ = clean_colnames(pd.read_csv("Circadian_raw.csv"))
barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))
nor = clean_colnames(pd.read_csv("UCBAge_Novel_clean.csv"))

if "Animal_ID" in nor.columns and "ID" not in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})

for df in (circ, barnes, nor):
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")


# =========================
# Build mouse-level circadian predictors
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

print(f"\nMouse-level rows: {circ_mouse.shape[0]}")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# =========================
# 1) Build Δ metrics for multiple circadian variables
# =========================
circ_metrics = ["IS", "IV", "RA", "Amplitude"]

# Keep only metrics present in dataset
circ_metrics = [m for m in circ_metrics if m in circ.columns]

wide = circ.pivot_table(index="ID", columns="PRE_POST", values=circ_metrics, aggfunc="mean")

# Compute change scores
delta_dict = {}
for m in circ_metrics:
    if (m, "PRE") in wide.columns and (m, "POST") in wide.columns:
        delta_dict[f"delta_{m}"] = wide[(m, "POST")] - wide[(m, "PRE")]

delta_df = pd.DataFrame(delta_dict)
delta_df.index.name = "ID"
delta_df = delta_df.reset_index()

print("\nΔ metrics used for PCA:", delta_df.columns.tolist())

# Merge mouse-level covariates
pca_df = circ_mouse.merge(delta_df, on="ID", how="inner")

# Drop rows with missing Δ values
pca_vars = [c for c in pca_df.columns if c.startswith("delta_")]
pca_df = pca_df.dropna(subset=pca_vars).copy()

print("Rows included in PCA:", pca_df.shape[0])

# =========================
# 2) Standardise Δ metrics
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(pca_df[pca_vars])

# =========================
# 3) Run PCA
# =========================
pca = PCA()
pcs = pca.fit_transform(X)

# Add PC1 to dataframe
pca_df["PC1"] = pcs[:, 0]

print("\nExplained variance ratio:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.3f}")

print("\nPCA loadings (PC1):")
for name, loading in zip(pca_vars, pca.components_[0]):
    print(f"{name:20s}: {loading:.3f}")

# =========================
# 4) Test Light → PC1
# =========================
pca_df["Light_new"] = pca_df["Light_new_mouse"].astype(str)
pca_df["Age_new"] = pca_df["Age_new_mouse"].astype(str)
pca_df["Sex_new"] = pca_df["Sex_new_mouse"].astype(str)

model_pc1 = smf.ols(
    "PC1 ~ C(Light_new) + C(Age_new) + C(Sex_new)",
    data=pca_df
).fit(cov_type="HC3")

print("\n=== Light → Circadian Composite (PC1) ===")
print(model_pc1.summary())

# =========================
# 5) PC1 → Barnes
# =========================
barnes_pca = barnes.merge(pca_df[["ID", "PC1"]], on="ID", how="inner")

barnes_pca["ID_str"] = barnes_pca["ID"].astype(str)
barnes_pca["Trial"] = pd.to_numeric(barnes_pca["Trial"], errors="coerce")
barnes_pca[NOSEPOKE] = pd.to_numeric(barnes_pca[NOSEPOKE], errors="coerce")

barnes_pca = barnes_pca.dropna(subset=["PC1", "Trial", NOSEPOKE]).copy()
barnes_pca = barnes_pca[barnes_pca[NOSEPOKE] >= 0]

gee_nb_pca = smf.gee(
    formula=f"{NOSEPOKE} ~ Trial + PC1 + C(Light_new) + C(Age_new) + C(Sex_new)",
    groups="ID_str",
    data=barnes_pca,
    family=sm.families.NegativeBinomial(),
    cov_struct=sm.cov_struct.Exchangeable()
).fit()

print("\n=== PC1 → Barnes (NB GEE) ===")
print(gee_nb_pca.summary())

# =========================
# 6) PC1 → NOR
# =========================
nor_pca = nor.merge(pca_df[["ID", "PC1"]], on="ID", how="inner")

N_dur = pd.to_numeric(nor_pca["N_obj_nose_duration_s"], errors="coerce")
F_dur = pd.to_numeric(nor_pca["F_obj_nose_duration_s"], errors="coerce")

nor_pca = nor_pca[(N_dur.notna()) & (F_dur.notna())].copy()
N_dur = N_dur.loc[nor_pca.index]
F_dur = F_dur.loc[nor_pca.index]

nor_pca["DI"] = (N_dur - F_dur) / (N_dur + F_dur + 1e-9)

model_nor_pc1 = smf.ols(
    "DI ~ PC1 + C(Light_new) + C(Age_new) + C(Sex_new)",
    data=nor_pca
).fit(cov_type="HC3")

print("\n=== PC1 → NOR ===")
print(model_nor_pc1.summary())

print("\nDONE.")

# ============================================================
# 1) BARNES: Negative Binomial GEE (overdispersion-robust)
# ============================================================
NOSEPOKE = "EntryZone_freq_new"  # change if needed

# Merge circadian predictors into barnes
barnes_m = barnes.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")

# Clean types
barnes_m["ID_str"] = barnes_m["ID"].astype(str)
barnes_m["Trial"] = pd.to_numeric(barnes_m["Trial"], errors="coerce")
barnes_m[NOSEPOKE] = pd.to_numeric(barnes_m[NOSEPOKE], errors="coerce")

barnes_m = barnes_m.dropna(subset=[
    "ID_str", "Trial", "Light_new", "Age_new", "Sex_new",
    "IS_pre", "delta_IS", NOSEPOKE
]).copy()

barnes_m = barnes_m[barnes_m[NOSEPOKE] >= 0].copy()
barnes_m[NOSEPOKE] = barnes_m[NOSEPOKE].astype(int)

# Overdispersion check
mean_y = barnes_m[NOSEPOKE].mean()
var_y = barnes_m[NOSEPOKE].var(ddof=1)
print(f"\n[Barnes] Var/Mean = {var_y / mean_y:.3f}")

# Negative Binomial GEE
gee_nb = smf.gee(
    formula=f"{NOSEPOKE} ~ Trial + C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    groups="ID_str",
    data=barnes_m,
    family=sm.families.NegativeBinomial(),
    cov_struct=sm.cov_struct.Exchangeable()
).fit()

print("\n=== Barnes Negative Binomial GEE (clustered by mouse; robust SEs) ===")
print(gee_nb.summary())

print("\n[Barnes] Rate ratios (exp(beta)):")
params = gee_nb.params
conf = gee_nb.conf_int()
for term in params.index:
    rr = float(np.exp(params[term]))
    lo = float(np.exp(conf.loc[term, 0]))
    hi = float(np.exp(conf.loc[term, 1]))
    print(f"{term:35s} RR={rr:.3f}  (95% CI {lo:.3f}–{hi:.3f})")


# ============================================================
# 2) NOR
# ============================================================
nor_m = nor.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")

N_dur = pd.to_numeric(nor_m["N_obj_nose_duration_s"], errors="coerce")
F_dur = pd.to_numeric(nor_m["F_obj_nose_duration_s"], errors="coerce")

nor_m = nor_m[(N_dur.notna()) & (F_dur.notna())].copy()
N_dur = N_dur.loc[nor_m.index]
F_dur = F_dur.loc[nor_m.index]

nor_m["p_novel"] = (N_dur / (N_dur + F_dur + 1e-9)).clip(1e-6, 1 - 1e-6)
nor_m["logit_p_novel"] = logit_clip(nor_m["p_novel"].values)
nor_m["DI"] = (N_dur - F_dur) / (N_dur + F_dur + 1e-9)

nor_fit_pref = smf.ols(
    "logit_p_novel ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    data=nor_m
).fit(cov_type="HC3")

print("\n=== NOR Model 1: logit preference (robust SE) ===")
print(nor_fit_pref.summary())

nor_fit_di = smf.ols(
    "DI ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    data=nor_m
).fit(cov_type="HC3")

print("\n=== NOR Model 2: discrimination index (robust SE) ===")
print(nor_fit_di.summary())

print("\nDONE.")

