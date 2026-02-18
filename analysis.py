"""
Publication-level analysis pipeline (Patsy-safe) — FIXED + UPDATED for Trial 6 accuracy

Key updates:
- Barnes endpoint is now Trial 6 "percent correct" using Goal_Box_feq_new (correct) and Hole_errors (wrong):
    p_correct = Goal_Box_feq_new / (Goal_Box_feq_new + Hole_errors)
  Model: Binomial GLM with freq_weights = total_entries, robust (HC3) SEs

- Barnes sensitivity endpoints at Trial 6 with FDR within family:
    Goal_Box_latency_new (log OLS), Entry_latency_new (log OLS),
    DistanceMoved_cm (OLS), EntryZone_freq_new (NB GLM)

- NOR: unchanged (DI_duration robust OLS)

- Mediation (optional, simplified, per-mouse endpoints):
    Light -> delta_IS -> Barnes logit(p_correct)   (OLS; cluster bootstrap by mouse)
    Light -> delta_IS -> NOR DI_duration           (OLS; cluster bootstrap by mouse)

IMPORTANT FIXES:
- Added `import statsmodels.api as sm` (fixes NameError: sm not defined)
- Removed references to NOSEPOKE and PoissonBayesMixedGLM from the Barnes section
  (those caused downstream NameErrors / mismatched endpoints).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# =========================
# 0) Helpers
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

def require_columns(df: pd.DataFrame, cols: list[str], df_name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}\n"
            f"Available columns include: {df.columns.tolist()[:60]}{' ...' if len(df.columns) > 60 else ''}"
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
# 1) Load + clean
# =========================
circ = clean_colnames(pd.read_csv("Circadian_raw.csv"))
barnes = clean_colnames(pd.read_csv("Barnes_clean.csv"))
nor = clean_colnames(pd.read_csv("UCBAge_Novel_clean.csv"))

if "Animal_ID" in nor.columns and "ID" not in nor.columns:
    nor = nor.rename(columns={"Animal_ID": "ID"})

for df in (circ, barnes, nor):
    require_columns(df, ["ID"], "Dataset")
    df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")

require_columns(circ, ["PRE_POST", "Light_new", "Age_new", "Sex_new"], "Circadian_raw.csv (cleaned)")
require_columns(barnes, ["ID", "Trial", "Light_new", "Age_new", "Sex_new", "Goal_Box_feq_new", "Hole_errors"], "Barnes_clean.csv (cleaned)")
require_columns(nor, ["ID", "Light_new", "Age_new", "Sex_new"], "UCBAge_Novel_clean.csv (cleaned)")

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
# 3) Circadian mixed models (as before)
# =========================
def fit_circadian_mixedlm(outcome: str, use_logit_if_bounded: bool = True):
    require_columns(circ, ["ID", "PRE_POST", "Light_new", "Age_new", "Sex_new", outcome], f"Circadian for outcome={outcome}")

    d = circ[["ID", "PRE_POST", "Light_new", "Age_new", "Sex_new", outcome]].dropna().copy()

    y = outcome
    if use_logit_if_bounded:
        y_min, y_max = d[outcome].min(), d[outcome].max()
        if np.isfinite(y_min) and np.isfinite(y_max) and (y_min >= 0) and (y_max <= 1):
            d[f"{outcome}_logit"] = logit_clip(d[outcome])
            y = f"{outcome}_logit"

    formula = f"{y} ~ PRE_POST * Light_new + Age_new + Sex_new"

    m_ri_ml = smf.mixedlm(formula, d, groups=d["ID"]).fit(method="lbfgs", reml=False)
    m_rs_ml = smf.mixedlm(formula, d, groups=d["ID"], re_formula="~PRE_POST").fit(method="lbfgs", reml=False)

    chosen = "RI+RS" if m_rs_ml.aic < m_ri_ml.aic else "RI"

    if chosen == "RI+RS":
        m_final = smf.mixedlm(formula, d, groups=d["ID"], re_formula="~PRE_POST").fit(method="lbfgs", reml=True)
    else:
        m_final = smf.mixedlm(formula, d, groups=d["ID"]).fit(method="lbfgs", reml=True)

    print(f"\n=== Circadian MixedLM: {outcome} (chosen random effects: {chosen}) ===")
    print(m_final.summary())
    return m_final

# Primary circadian
require_columns(circ, ["IS"], "Circadian_raw.csv needs IS")
_ = fit_circadian_mixedlm("IS")

# Secondary circadian interactions with FDR
secondary_metrics = [m for m in ["IV", "RA", "Amplitude"] if m in circ.columns]
sec_rows = []
for out in secondary_metrics:
    m = fit_circadian_mixedlm(out)
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
# 4) Mouse-level predictors: IS_pre and delta_IS
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

circ_mouse = circ_mouse.rename(columns={
    "Light_new": "Light_new_mouse",
    "Age_new": "Age_new_mouse",
    "Sex_new": "Sex_new_mouse"
})


# =========================
# 5) Barnes Trial 6: Percent correct (binomial GLM) + sensitivity family
# =========================
TRIAL_END = 6

barnes_m = barnes.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")
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

barnes_t6 = barnes_t6[(barnes_t6["Goal_Box_feq_new"] >= 0) & (barnes_t6["Hole_errors"] >= 0)].copy()
barnes_t6["total_entries"] = barnes_t6["Goal_Box_feq_new"] + barnes_t6["Hole_errors"]
barnes_t6 = barnes_t6[barnes_t6["total_entries"] > 0].copy()

barnes_t6["p_correct"] = barnes_t6["Goal_Box_feq_new"] / barnes_t6["total_entries"]
print(f"\nBarnes Trial {TRIAL_END}: n mice with total_entries>0 = {barnes_t6['ID'].nunique()}")

# Primary Trial 6 binomial GLM (robust)
binom_fit = smf.glm(
    formula="p_correct ~ C(Light_new) + IS_pre + delta_IS + C(Age_new) + C(Sex_new)",
    data=barnes_t6,
    family=sm.families.Binomial(),
    freq_weights=barnes_t6["total_entries"]
).fit(cov_type="HC3")

print(f"\n=== Barnes Trial {TRIAL_END} PRIMARY: Percent correct (Binomial GLM; robust HC3) ===")
print(binom_fit.summary())

print("\nOdds ratios (exp(beta)) for key terms:")
for term in ["C(Light_new)[T.ISF]", "IS_pre", "delta_IS"]:
    if term in binom_fit.params.index:
        b = binom_fit.params[term]
        ci = binom_fit.conf_int().loc[term].values
        print(f"  {term}: OR={np.exp(b):.3f} (95% CI {np.exp(ci[0]):.3f}–{np.exp(ci[1]):.3f})")

# Sensitivity family at Trial 6 (FDR on Light only)
sens_tests = []

# Latencies (continuous): log OLS
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

# Distance moved (continuous): OLS
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
# 6) NOR (unchanged): DI_duration robust OLS
# =========================
require_columns(nor, ["N_obj_nose_duration_s", "F_obj_nose_duration_s"], "NOR durations needed")

nor_m = nor.merge(circ_mouse[["ID", "IS_pre", "delta_IS"]], on="ID", how="left")
nor_m = nor_m.dropna(subset=["N_obj_nose_duration_s", "F_obj_nose_duration_s",
                             "Light_new", "Age_new", "Sex_new", "IS_pre", "delta_IS"]).copy()

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
# 7) Mediation (optional): per-mouse endpoints (simplified; robust)
# =========================
# Barnes endpoint: logit(p_correct) per mouse at Trial 6
med_barnes = barnes_t6.copy()
med_barnes["logit_p_correct"] = logit_clip(med_barnes["p_correct"].values)

# Merge mouse-level covars for mediation
# Barnes endpoint: logit(p_correct) per mouse at Trial 6
med_barnes = barnes_t6.copy()
med_barnes["logit_p_correct"] = logit_clip(med_barnes["p_correct"].values)

# Merge ONLY mouse-level group covariates (avoid duplicating IS_pre/delta_IS)
med_barnes = med_barnes.merge(
    circ_mouse[["ID", "Light_new_mouse", "Age_new_mouse", "Sex_new_mouse"]],
    on="ID",
    how="left"
)

# Now drop missing values (IS_pre and delta_IS already exist in med_barnes)
med_barnes = med_barnes.dropna(subset=[
    "logit_p_correct", "Light_new_mouse", "Age_new_mouse", "Sex_new_mouse", "IS_pre", "delta_IS"
]).copy()


# NOR endpoint per mouse
nor_endp = nor_m.groupby("ID")["DI_duration"].mean().rename("nor_endp").reset_index()
med_nor = circ_mouse.merge(nor_endp, on="ID", how="inner").dropna(subset=["Light_new_mouse", "Age_new_mouse", "Sex_new_mouse", "IS_pre", "delta_IS", "nor_endp"])

def bootstrap_mediation(df: pd.DataFrame, y_col: str, n_boot: int = 3000, seed: int = 0):
    """
    Cluster bootstrap by mouse.
    a-model: delta_IS ~ Light + covariates + IS_pre
    b-model: y ~ Light + delta_IS + covariates + IS_pre
    indirect = a*b
    """
    rng = np.random.default_rng(seed)
    ids = df["ID"].dropna().unique()

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

print("\n=== Mediation (optional): Barnes logit(p_correct) ===")
if med_barnes["ID"].nunique() >= 10:
    print(bootstrap_mediation(med_barnes, "logit_p_correct"))
else:
    print("Not enough mice for stable mediation on Barnes logit(p_correct).")

print("\n=== Mediation (optional): NOR DI ===")
if med_nor["ID"].nunique() >= 10:
    print(bootstrap_mediation(med_nor, "nor_endp"))
else:
    print("Not enough mice for stable mediation on NOR.")

print("\nDONE.")

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", context="talk")

# -------------------------
# Settings
# -------------------------
OUTDIR = "."  # current folder (same as script / VS Code workspace)
DPI = 300
SHOW_PLOTS = True  # set False if you only want files saved

def save_show(fig, filename):
    path = os.path.join(OUTDIR, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)
    print(f"Saved: {path}")

# -------------------------
# 1) Circadian PRE vs POST by Light group (IS, IV, RA, Amplitude)
# -------------------------
def plot_circadian_prepost(metric):
    if metric not in circ.columns:
        return
    d = circ[["ID", "PRE_POST", "Light_new", metric]].dropna().copy()
    d["PRE_POST"] = d["PRE_POST"].astype(str)
    d["Light_new"] = d["Light_new"].astype(str)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    sns.pointplot(
        data=d,
        x="PRE_POST",
        y=metric,
        hue="Light_new",
        errorbar="se",
        dodge=True,
        ax=ax
    )

    ax.set_title(f"{metric}: PRE vs POST by Light Group")
    ax.set_xlabel("Timepoint")
    ax.set_ylabel(metric)
    ax.legend(title="Light Group", loc="best")

    save_show(fig, f"circadian_{metric}_prepost_by_light.png")

for m in ["IS", "IV", "RA", "Amplitude"]:
    plot_circadian_prepost(m)

# -------------------------
# 2) Barnes Trial 6 percent correct by Light group
# -------------------------
if "barnes_t6" in globals() and "p_correct" in barnes_t6.columns:
    d = barnes_t6.copy()
    d["Light_new"] = d["Light_new"].astype(str)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    sns.stripplot(data=d, x="Light_new", y="p_correct", jitter=0.2, alpha=0.7, ax=ax)
    sns.pointplot(data=d, x="Light_new", y="p_correct", errorbar="se", color="black", ax=ax)

    ax.set_title("Barnes Trial 6: Percent Correct by Light Group")
    ax.set_xlabel("Light Group")
    ax.set_ylabel("Percent Correct (Goal / (Goal + Errors))")
    ax.set_ylim(0, 1)

    save_show(fig, "barnes_trial6_percent_correct_by_light.png")

# -------------------------
# 3) Barnes Trial 6 percent correct vs delta_IS (scatter + overall trend)
# -------------------------
if "barnes_t6" in globals() and all(c in barnes_t6.columns for c in ["delta_IS", "p_correct", "Light_new"]):
    d = barnes_t6.dropna(subset=["delta_IS", "p_correct"]).copy()
    d["Light_new"] = d["Light_new"].astype(str)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    sns.scatterplot(data=d, x="delta_IS", y="p_correct", hue="Light_new", alpha=0.75, ax=ax)
    sns.regplot(data=d, x="delta_IS", y="p_correct", scatter=False, ax=ax)

    ax.set_title("Barnes Trial 6: Percent Correct vs ΔIS")
    ax.set_xlabel("ΔIS (POST − PRE)")
    ax.set_ylabel("Percent Correct")
    ax.set_ylim(0, 1)
    ax.legend(title="Light Group", loc="best")

    save_show(fig, "barnes_trial6_percent_correct_vs_deltaIS.png")

# -------------------------
# 4) NOR DI by Light group
# -------------------------
if "nor_m" in globals() and "DI_duration" in nor_m.columns and "Light_new" in nor_m.columns:
    d = nor_m.dropna(subset=["DI_duration", "Light_new"]).copy()
    d["Light_new"] = d["Light_new"].astype(str)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    sns.stripplot(data=d, x="Light_new", y="DI_duration", jitter=0.2, alpha=0.7, ax=ax)
    sns.pointplot(data=d, x="Light_new", y="DI_duration", errorbar="se", color="black", ax=ax)

    ax.set_title("Novel Object Recognition: DI by Light Group")
    ax.set_xlabel("Light Group")
    ax.set_ylabel("Discrimination Index (DI)")

    save_show(fig, "nor_DI_by_light.png")

print("\nAll plots saved.")

