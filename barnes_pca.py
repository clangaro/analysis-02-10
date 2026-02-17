"""
POST circadian rhythms predicting end-of-training Barnes performance (Trial 6)

Goal:
- Compute a stable POST circadian composite (PC1_post) using all mice with POST circadian data.
- Test whether PC1_post predicts Barnes performance specifically at the end of training (Trial 6):
    Primary: Hole_errors (count) -> Negative Binomial GLM with robust (HC3) SE
    Secondary: Goal.Box_feq_new (log latency) -> OLS with robust (HC3) SE
- Light is excluded (you can add it back later if needed).

Files required (same folder):
- Circadian_raw.csv
- Barnes_clean.csv
- UCBAge_Novel_clean.csv (loaded but not used here; kept for compatibility)

Run:
python post_circadian_predicts_barnes_trial6.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

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


# =========================
# Load + clean
# =========================
circ = clean_colnames(pd.read_csv("Circadian_raw.csv"))
barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))
nor = clean_colnames(pd.read_csv("UCBAge_Novel_clean.csv"))  # not used, but loaded for consistency

if "Animal_ID" in nor.columns and "ID" not in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})

for df in (circ, barnes, nor):
    require_cols(df, ["ID"], "Dataset")
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

require_cols(circ, ["PRE_POST"], "Circadian_raw.csv")
require_cols(barnes, ["Trial", "Age_new", "Sex_new", "Hole_errors", "Goal_Box_feq_new"], "Barnes_clean.csv")

# Ensure Trial numeric
barnes["Trial"] = pd.to_numeric(barnes["Trial"], errors="coerce")


# =========================
# Build POST circadian table (one row per mouse)
# =========================
circ_post = circ[circ["PRE_POST"] == "POST"].copy()

# Aggregate to 1 row per mouse at POST (mean for numeric metrics)
numeric_cols = [c for c in circ_post.columns if circ_post[c].dtype.kind in "if"]
agg = {c: "mean" for c in numeric_cols}
circ_post = circ_post.groupby("ID", as_index=False).agg(agg)

print(f"\nPOST circadian mice (after aggregation): {circ_post['ID'].nunique()}")

# Circadian metrics for PCA
circ_metrics = [m for m in ["IS", "IV", "RA", "Amplitude"] if m in circ_post.columns]
if len(circ_metrics) < 2:
    raise ValueError(f"Not enough circadian metrics found for PCA. Found: {circ_metrics}")

# PCA dataset
pca_df = circ_post[["ID"] + circ_metrics].dropna().copy()
print(f"Rows included in POST circadian PCA: {pca_df.shape[0]}")

# Standardise + PCA
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
# Barnes end-of-training dataset: Trial 6 only
# =========================
TRIAL_END = 6

barnes_end = barnes.merge(pca_df[["ID", "PC1_post"]], on="ID", how="inner").copy()
barnes_end = barnes_end[barnes_end["Trial"] == TRIAL_END].copy()

print(f"\nBarnes mice with Trial {TRIAL_END} and PC1_post: {barnes_end['ID'].nunique()}")
print(f"Barnes Trial {TRIAL_END} rows: {barnes_end.shape[0]}")

barnes_end["Age_new"] = barnes_end["Age_new"].astype(str)
barnes_end["Sex_new"] = barnes_end["Sex_new"].astype(str)


# =========================
# PRIMARY: Hole_errors ~ PC1_post + Age + Sex
# Negative Binomial GLM with robust (HC3) SE
# =========================
barnes_end["Hole_errors"] = pd.to_numeric(barnes_end["Hole_errors"], errors="coerce")
df_primary = barnes_end.dropna(subset=["Hole_errors", "PC1_post", "Age_new", "Sex_new"]).copy()
df_primary = df_primary[df_primary["Hole_errors"] >= 0].copy()
df_primary["Hole_errors"] = df_primary["Hole_errors"].astype(int)

mean_y = df_primary["Hole_errors"].mean()
var_y = df_primary["Hole_errors"].var(ddof=1)
print(f"\n[Barnes Trial {TRIAL_END}] Hole_errors Var/Mean = {var_y/mean_y:.3f}")

nb_glm = smf.glm(
    formula="Hole_errors ~ PC1_post + C(Age_new) + C(Sex_new)",
    data=df_primary,
    family=sm.families.NegativeBinomial()
).fit(cov_type="HC3")

print(f"\n=== PRIMARY (Trial {TRIAL_END}): Hole_errors ~ PC1_post (NB GLM; robust SE) ===")
print(nb_glm.summary())

b = nb_glm.params["PC1_post"]
ci = nb_glm.conf_int().loc["PC1_post"].values
print("\n[PRIMARY] PC1_post rate ratio RR = exp(beta):")
print(f"  RR = {np.exp(b):.3f} (95% CI {np.exp(ci[0]):.3f}–{np.exp(ci[1]):.3f})")


# =========================
# SECONDARY: log(Goal_Box_latency_new) ~ PC1_post + Age + Sex
# Robust OLS (HC3)
# =========================
barnes_end["Goal_Box_feq_new"] = pd.to_numeric(barnes_end["Goal_Box_feq_new"], errors="coerce")
df_lat = barnes_end.dropna(subset=["Goal_Box_feq_new", "PC1_post", "Age_new", "Sex_new"]).copy()
df_lat = df_lat[df_lat["Goal_Box_feq_new"] > 0].copy()
df_lat["log_goal_latency"] = np.log(df_lat["Goal_Box_feq_new"].astype(float))

lat_fit = smf.ols(
    "log_goal_latency ~ PC1_post + C(Age_new) + C(Sex_new)",
    data=df_lat
).fit(cov_type="HC3")

print(f"\n=== SECONDARY (Trial {TRIAL_END}): log(Goal_Box_feq_new) ~ PC1_post (robust OLS) ===")
print(lat_fit.summary())

bL = lat_fit.params["PC1_post"]
ciL = lat_fit.conf_int().loc["PC1_post"].values
print("\n[SECONDARY] Multiplicative effect on latency exp(beta):")
print(f"  exp(beta) = {np.exp(bL):.3f} (95% CI {np.exp(ciL[0]):.3f}–{np.exp(ciL[1]):.3f})")
print("  exp(beta) < 1 implies shorter latency (better); > 1 implies longer latency (worse).")

print("\nDONE.")
