"""
POST-only circadian predictors of behaviour (light excluded)
Primary Barnes: Hole_errors (count; NB-GEE)
Secondary Barnes: Goal_Box_latency_new (log-latency; Gaussian GEE with robust SE)
Exploratory Barnes composite: per-mouse composite from last 2 trials (z-score, higher=better)
NOR: unchanged optional section at bottom (DI + preference) if you want to keep it.

Files required (same folder):
- Circadian_raw.csv
- Barnes_clean.csv
- UCBAge_Novel_clean.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

def require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} missing columns: {missing}\n"
            f"Available columns (first 80): {df.columns.tolist()[:80]}"
        )

def logit_clip(p, eps=1e-6):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else np.nan)


# =========================
# Load + clean
# =========================
circ = clean_colnames(pd.read_csv("Circadian_raw.csv"))
barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))
nor = clean_colnames(pd.read_csv("UCBAge_Novel_clean.csv"))

if "Animal_ID" in nor.columns and "ID" not in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})

for df in (circ, barnes, nor):
    require_cols(df, ["ID"], "Dataset")
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

require_cols(circ, ["PRE_POST"], "Circadian_raw.csv")
require_cols(barnes, ["Trial", "Age_new", "Sex_new", "Hole_errors", "Goal_Box_latency_new"], "Barnes_clean.csv")


# =========================
# Build POST circadian table (one row per mouse)
# =========================
circ_post = circ[circ["PRE_POST"] == "POST"].copy()

# Aggregate to 1 row per mouse at POST (mean for numeric metrics; first for categorical)
numeric_cols = [c for c in circ_post.columns if circ_post[c].dtype.kind in "if"]
agg = {c: "mean" for c in numeric_cols}
for c in ["Age_new", "Sex_new", "Light_new"]:
    if c in circ_post.columns:
        agg[c] = lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan

circ_post = circ_post.groupby("ID", as_index=False).agg(agg)

print(f"\nPOST circadian mice (after aggregation): {circ_post['ID'].nunique()}")


# =========================
# Circadian predictors: PCA composite on POST metrics
# =========================
circ_metrics = [m for m in ["IS", "IV", "RA", "Amplitude"] if m in circ_post.columns]
if len(circ_metrics) < 2:
    raise ValueError(f"Not enough circadian metrics found for PCA. Found: {circ_metrics}")

pca_df = circ_post[["ID"] + circ_metrics].dropna().copy()
print(f"Rows included in POST circadian PCA: {pca_df.shape[0]}")

scaler = StandardScaler()
X = scaler.fit_transform(pca_df[circ_metrics])

pca = PCA()
pcs = pca.fit_transform(X)

pca_df["PC1_post"] = pcs[:, 0]

print("\nExplained variance ratio (POST circadian PCA):")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {var:.3f}")

print("\nPC1 loadings (POST circadian PCA):")
for name, loading in zip(circ_metrics, pca.components_[0]):
    print(f"{name:12s}: {loading:.3f}")


# =========================
# Merge POST circadian predictors into Barnes
# =========================
barnes_m = barnes.merge(pca_df[["ID", "PC1_post"]], on="ID", how="inner") \
                 .merge(circ_post[["ID"] + circ_metrics], on="ID", how="inner")

barnes_m["ID_str"] = barnes_m["ID"].astype(str)
barnes_m["Trial"] = pd.to_numeric(barnes_m["Trial"], errors="coerce")

# Ensure outcomes are numeric
barnes_m["Hole_errors"] = pd.to_numeric(barnes_m["Hole_errors"], errors="coerce")
barnes_m["Goal_Box_latency_new"] = pd.to_numeric(barnes_m["Goal_Box_latency_new"], errors="coerce")
barnes_m["Age_new"] = barnes_m["Age_new"].astype(str)
barnes_m["Sex_new"] = barnes_m["Sex_new"].astype(str)

barnes_m = barnes_m.dropna(subset=["ID_str", "Trial", "Age_new", "Sex_new", "PC1_post", "Hole_errors", "Goal_Box_latency_new"]).copy()

# Count outcome must be non-negative integer
barnes_m = barnes_m[barnes_m["Hole_errors"] >= 0].copy()
barnes_m["Hole_errors"] = barnes_m["Hole_errors"].astype(int)

# Latency must be positive for log transform
barnes_m = barnes_m[barnes_m["Goal_Box_latency_new"] > 0].copy()
barnes_m["log_goal_latency"] = np.log(barnes_m["Goal_Box_latency_new"].astype(float))


# ============================================================
# PRIMARY: Barnes Hole_errors ~ POST circadian (PC1_post)
# NB-GEE, clustered by mouse
# ============================================================
mean_y = barnes_m["Hole_errors"].mean()
var_y = barnes_m["Hole_errors"].var(ddof=1)
print(f"\n[Barnes PRIMARY: Hole_errors] Var/Mean = {var_y/mean_y:.3f} (>>1 supports NB)")

gee_hole_pc1 = smf.gee(
    formula="Hole_errors ~ Trial + PC1_post + C(Age_new) + C(Sex_new)",
    groups="ID_str",
    data=barnes_m,
    family=sm.families.NegativeBinomial(),
    cov_struct=sm.cov_struct.Exchangeable()
).fit()

print("\n=== PRIMARY: Barnes Hole_errors ~ PC1_post (NB GEE; light excluded) ===")
print(gee_hole_pc1.summary())

# Rate ratio for PC1_post
b = gee_hole_pc1.params["PC1_post"]
ci = gee_hole_pc1.conf_int().loc["PC1_post"].values
print("\n[PRIMARY] PC1_post effect on Hole_errors as rate ratio (RR = exp(beta)):")
print(f"  RR = {np.exp(b):.3f} (95% CI {np.exp(ci[0]):.3f}–{np.exp(ci[1]):.3f})")


# ============================================================
# SECONDARY: Barnes Goal_Box_latency_new ~ POST circadian (PC1_post)
# log-latency Gaussian GEE, clustered by mouse
# ============================================================
gee_lat_pc1 = smf.gee(
    formula="log_goal_latency ~ Trial + PC1_post + C(Age_new) + C(Sex_new)",
    groups="ID_str",
    data=barnes_m,
    family=sm.families.Gaussian(),
    cov_struct=sm.cov_struct.Exchangeable()
).fit()

print("\n=== SECONDARY: Barnes log(Goal_Box_latency_new) ~ PC1_post (Gaussian GEE; light excluded) ===")
print(gee_lat_pc1.summary())

# Interpret PC1_post coefficient as multiplicative change in latency: exp(beta)
bL = gee_lat_pc1.params["PC1_post"]
ciL = gee_lat_pc1.conf_int().loc["PC1_post"].values
print("\n[SECONDARY] PC1_post effect on Goal_Box_latency as multiplicative factor (exp(beta)):")
print(f"  exp(beta) = {np.exp(bL):.3f} (95% CI {np.exp(ciL[0]):.3f}–{np.exp(ciL[1]):.3f})")
print("  Values < 1 imply shorter latency (better), > 1 imply longer latency (worse).")


# ============================================================
# EXPLORATORY: Composite cognitive score (per mouse, last 2 trials)
# Composite = mean(z(-Hole_errors_last2), z(-log_latency_last2))
# Higher = better
# Then: Composite ~ PC1_post + Age + Sex (robust OLS)
# ============================================================
# Define last two trials per mouse
max_trial_by_mouse = barnes_m.groupby("ID")["Trial"].max().rename("max_trial")
barnes_tmp = barnes_m.merge(max_trial_by_mouse, on="ID", how="left")
barnes_last2 = barnes_tmp[barnes_tmp["Trial"].isin(barnes_tmp["max_trial"])]
# also include (max_trial - 1) when present
barnes_last2 = pd.concat([
    barnes_last2,
    barnes_tmp[barnes_tmp["Trial"] == (barnes_tmp["max_trial"] - 1)]
]).drop_duplicates()

# Aggregate last2 endpoints per mouse
endp = barnes_last2.groupby("ID").agg(
    hole_last2=("Hole_errors", "mean"),
    loglat_last2=("log_goal_latency", "mean"),
    Age_new=("Age_new", lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan),
    Sex_new=("Sex_new", lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan),
).reset_index()

endp = endp.merge(pca_df[["ID", "PC1_post"]], on="ID", how="inner").dropna()

# Build composite: invert so higher is better
endp["z_hole_better"] = zscore(-endp["hole_last2"])
endp["z_lat_better"] = zscore(-endp["loglat_last2"])
endp["composite_better"] = endp[["z_hole_better", "z_lat_better"]].mean(axis=1)

comp_fit = smf.ols(
    "composite_better ~ PC1_post + C(Age_new) + C(Sex_new)",
    data=endp
).fit(cov_type="HC3")

print("\n=== EXPLORATORY: Composite cognitive score (last 2 trials) ~ PC1_post (robust OLS; light excluded) ===")
print(comp_fit.summary())


# ============================================================
# Sensitivity: Replace PC1_post with each POST circadian metric (FDR)
# For PRIMARY (Hole_errors) and COMPOSITE
# ============================================================
print("\n=== Sensitivity (PRIMARY): Hole_errors ~ each POST circadian metric, FDR ===")
primary_rows = []
for m in circ_metrics:
    dfm = barnes.merge(circ_post[["ID", m]], on="ID", how="inner").copy()
    dfm["ID_str"] = dfm["ID"].astype(str)
    dfm["Trial"] = pd.to_numeric(dfm["Trial"], errors="coerce")
    dfm["Hole_errors"] = pd.to_numeric(dfm["Hole_errors"], errors="coerce")
    dfm[m] = pd.to_numeric(dfm[m], errors="coerce")
    dfm["Age_new"] = dfm["Age_new"].astype(str)
    dfm["Sex_new"] = dfm["Sex_new"].astype(str)

    dfm = dfm.dropna(subset=["ID_str", "Trial", "Hole_errors", m, "Age_new", "Sex_new"]).copy()
    dfm = dfm[dfm["Hole_errors"] >= 0].copy()
    dfm["Hole_errors"] = dfm["Hole_errors"].astype(int)

    fit = smf.gee(
        formula=f"Hole_errors ~ Trial + {m} + C(Age_new) + C(Sex_new)",
        groups="ID_str",
        data=dfm,
        family=sm.families.NegativeBinomial(),
        cov_struct=sm.cov_struct.Exchangeable()
    ).fit()

    beta = float(fit.params[m])
    p = float(fit.pvalues[m])
    ci = fit.conf_int().loc[m].values
    primary_rows.append({
        "metric": m,
        "beta": beta,
        "p": p,
        "RR": float(np.exp(beta)),
        "RR_CI_low": float(np.exp(ci[0])),
        "RR_CI_high": float(np.exp(ci[1])),
        "n_obs": int(dfm.shape[0]),
        "n_mice": int(dfm["ID"].nunique())
    })

primary_df = pd.DataFrame(primary_rows)
rej, p_fdr, _, _ = multipletests(primary_df["p"].values, method="fdr_bh", alpha=0.05)
primary_df["p_fdr_bh"] = p_fdr
primary_df["sig_fdr_0.05"] = rej
print(primary_df.sort_values("p").to_string(index=False))


print("\n=== Sensitivity (EXPLORATORY COMPOSITE): composite_better ~ each POST circadian metric, FDR ===")
comp_rows = []
for m in circ_metrics:
    dfm = endp.merge(circ_post[["ID", m]], on="ID", how="inner").copy()
    dfm[m] = pd.to_numeric(dfm[m], errors="coerce")
    dfm = dfm.dropna(subset=[m, "Age_new", "Sex_new", "composite_better"]).copy()

    fit = smf.ols(
        f"composite_better ~ {m} + C(Age_new) + C(Sex_new)",
        data=dfm
    ).fit(cov_type="HC3")

    comp_rows.append({
        "metric": m,
        "beta": float(fit.params[m]),
        "p": float(fit.pvalues[m]),
        "n_mice": int(dfm.shape[0])
    })

comp_df = pd.DataFrame(comp_rows)
rej, p_fdr, _, _ = multipletests(comp_df["p"].values, method="fdr_bh", alpha=0.05)
comp_df["p_fdr_bh"] = p_fdr
comp_df["sig_fdr_0.05"] = rej
print(comp_df.sort_values("p").to_string(index=False))


# ============================================================
# Optional: NOR using POST circadian PC1_post (light excluded)
# ============================================================
# If you want to keep NOR in the same script, uncomment below.

# require_cols(nor, ["N_obj_nose_duration_s", "F_obj_nose_duration_s", "Age_new", "Sex_new"], "NOR")
# nor_m = nor.merge(pca_df[["ID", "PC1_post"]], on="ID", how="inner").copy()
# N = pd.to_numeric(nor_m["N_obj_nose_duration_s"], errors="coerce")
# F = pd.to_numeric(nor_m["F_obj_nose_duration_s"], errors="coerce")
# nor_m = nor_m[(N.notna()) & (F.notna())].copy()
# N = N.loc[nor_m.index]
# F = F.loc[nor_m.index]
# nor_m["DI"] = (N - F) / (N + F + 1e-9)
# nor_m["p_novel"] = (N / (N + F + 1e-9)).clip(1e-6, 1 - 1e-6)
# nor_m["logit_p_novel"] = logit_clip(nor_m["p_novel"].values)
#
# nor_di = smf.ols("DI ~ PC1_post + C(Age_new) + C(Sex_new)", data=nor_m).fit(cov_type="HC3")
# print("\n=== NOR: DI ~ PC1_post (robust OLS; light excluded) ===")
# print(nor_di.summary())
#
# nor_pref = smf.ols("logit_p_novel ~ PC1_post + C(Age_new) + C(Sex_new)", data=nor_m).fit(cov_type="HC3")
# print("\n=== NOR: logit preference ~ PC1_post (robust OLS; light excluded) ===")
# print(nor_pref.summary())

print("\nDONE.")
