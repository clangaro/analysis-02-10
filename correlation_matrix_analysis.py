"""
Correlation matrix across circadian and Barnes Trial 6 variables.

Includes:
- IS_pre
- delta_IS
- POST circadian metrics (IS, IV, RA, Amplitude)
- Trial 6:
    - Hole_errors
    - Goal_Box_feq_new
    - percent correct (p_correct)
    - total_entries
    - Goal_Box_latency_new
    - DistanceMoved_cm (if present)

Outputs:
- Pearson correlation matrix
- Spearman correlation matrix
"""

import pandas as pd
import numpy as np

# =========================
# Helper
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

# =========================
# Load data
# =========================
circ = clean_colnames(pd.read_csv("Circadian_raw.csv"))
barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))

# Ensure ID numeric
for df in (circ, barnes):
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

# =========================
# Build circadian predictors
# =========================
# PRE and POST wide table for IS
wide_IS = circ.pivot_table(index="ID", columns="PRE_POST", values="IS", aggfunc="mean")

IS_pre = wide_IS.get("PRE")
delta_IS = wide_IS.get("POST") - wide_IS.get("PRE")

circ_post = circ[circ["PRE_POST"] == "POST"].copy()

# Aggregate POST metrics per mouse
post_metrics = (
    circ_post.groupby("ID")[["IS", "IV", "RA", "Amplitude"]]
    .mean()
)

circ_mouse = pd.DataFrame({
    "IS_pre": IS_pre,
    "delta_IS": delta_IS
}).join(post_metrics)

# =========================
# Barnes Trial 6 endpoints
# =========================
TRIAL_END = 6
barnes_t6 = barnes[barnes["Trial"] == TRIAL_END].copy()

barnes_t6["Hole_errors"] = pd.to_numeric(barnes_t6["Hole_errors"], errors="coerce")
barnes_t6["Goal_Box_feq_new"] = pd.to_numeric(barnes_t6["Goal_Box_feq_new"], errors="coerce")
barnes_t6["Goal_Box_latency_new"] = pd.to_numeric(barnes_t6["Goal_Box_latency_new"], errors="coerce")

if "DistanceMoved_cm" in barnes_t6.columns:
    barnes_t6["DistanceMoved_cm"] = pd.to_numeric(barnes_t6["DistanceMoved_cm"], errors="coerce")

# Build proportions
barnes_t6["total_entries"] = barnes_t6["Hole_errors"] + barnes_t6["Goal_Box_feq_new"]
barnes_t6 = barnes_t6[barnes_t6["total_entries"] > 0].copy()
barnes_t6["p_correct"] = barnes_t6["Goal_Box_feq_new"] / barnes_t6["total_entries"]

# Keep relevant columns
barnes_vars = [
    "Hole_errors",
    "Goal_Box_feq_new",
    "p_correct",
    "total_entries",
    "Goal_Box_latency_new"
]

if "DistanceMoved_cm" in barnes_t6.columns:
    barnes_vars.append("DistanceMoved_cm")

barnes_mouse = barnes_t6.set_index("ID")[barnes_vars]

# =========================
# Merge all into one mouse-level dataframe
# =========================
full_df = circ_mouse.join(barnes_mouse, how="inner")

# Drop rows with missing values
full_df = full_df.dropna()

print("\nNumber of mice in correlation analysis:", full_df.shape[0])

# =========================
# Pearson correlation
# =========================
pearson_corr = full_df.corr(method="pearson")

print("\n=== Pearson Correlation Matrix ===")
print(pearson_corr.round(3))

# =========================
# Spearman correlation
# =========================
spearman_corr = full_df.corr(method="spearman")

print("\n=== Spearman Correlation Matrix ===")
print(spearman_corr.round(3))
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white", context="talk")

# =========================
# Pearson heatmap
# =========================
plt.figure(figsize=(10, 8))
pearson_corr = full_df.corr(method="pearson")

sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.title("Pearson Correlation Matrix")
plt.tight_layout()
plt.show()

# =========================
# Spearman heatmap
# =========================
plt.figure(figsize=(10, 8))
spearman_corr = full_df.corr(method="spearman")

sns.heatmap(
    spearman_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8}
)

plt.title("Spearman Correlation Matrix")
plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="white", context="talk")

# Compute Spearman correlation
corr = full_df.corr(method="spearman")

# Mask upper triangle for cleaner visualization
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(14, 12))

heatmap = sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 12}
)

plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(rotation=0, fontsize=12)

plt.title("Spearman Correlation Matrix", fontsize=18)
plt.tight_layout()

# Save high-resolution version
plt.savefig("spearman_correlation_matrix.png", dpi=300)
plt.show()
