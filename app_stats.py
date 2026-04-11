"""
Reusable statistical helpers for the Streamlit narrative app.

Where possible, functions are imported from existing repo modules
(metrics_script.py is import-safe). Helpers that originate in the
top-level analysis scripts (analysis.py, circadian_predicts_behaviour.py,
variability_analysis.py, improved_barnes_analysis.py, sex_age_effects.py)
are *re-implemented here* because those scripts execute on import.

Sleep-burst note: the repo has no precomputed sleep-burst metric. This
module derives one from the raw PIR activity series in metrics_script.py
style: a "rest bout" is a maximal run of epochs with activity <= threshold
that exceeds a minimum duration. Bursts-by-phase count and aggregate those
bouts within hour-of-day bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower

import metrics_script as ms


# ----------------------------------------------------------------------
# Column hygiene (ported from analysis.py / circadian_predicts_behaviour.py)
# ----------------------------------------------------------------------

def clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"\.+", "_", regex=True)
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def logit_clip(p: pd.Series, eps: float = 1e-3) -> pd.Series:
    p = p.clip(eps, 1 - eps)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------
# Effect sizes (ported from sex_age_effects.py / improved_barnes_analysis.py)
# ----------------------------------------------------------------------

def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if s == 0:
        return np.nan
    return float((a.mean() - b.mean()) / s)


def hedges_g(a: Sequence[float], b: Sequence[float]) -> float:
    d = cohens_d(a, b)
    if not np.isfinite(d):
        return np.nan
    n = len(a) + len(b)
    j = 1 - 3 / (4 * n - 9)
    return float(d * j)


def bootstrap_mean_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Returns (mean_diff, lo, hi) bootstrap CI for mean(a) - mean(b)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan, np.nan)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def bootstrap_corr_ci(
    x: Sequence[float],
    y: Sequence[float],
    method: str = "pearson",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4:
        return (np.nan, np.nan, np.nan)
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    r0 = float(fn(x, y)[0])
    rs = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        rs[i] = fn(x[idx], y[idx])[0]
    lo, hi = np.nanquantile(rs, [alpha / 2, 1 - alpha / 2])
    return r0, float(lo), float(hi)


# ----------------------------------------------------------------------
# Bayes factor (JZS, t-based) — ported from circadian_predicts_behaviour.py
# ----------------------------------------------------------------------

def bayes_factor_t(t: float, n: int, r: float = 0.707) -> float:
    """JZS Bayes Factor (BF01) for a two-sided t-test. >1 favours null."""
    if not np.isfinite(t) or n < 3:
        return np.nan
    df = n - 2
    from scipy.integrate import quad

    def integrand(g):
        return (1 + n * g * r ** 2) ** (-0.5) * (
            1 + t ** 2 / ((1 + n * g * r ** 2) * df)
        ) ** (-(df + 1) / 2) * (2 * np.pi) ** (-0.5) * g ** (-1.5) * np.exp(-1 / (2 * g))

    try:
        bf10, _ = quad(integrand, 0, np.inf, limit=200)
        h0 = (1 + t ** 2 / df) ** (-(df + 1) / 2)
        bf10 = bf10 / h0
        return float(1 / bf10) if bf10 > 0 else np.nan
    except Exception:
        return np.nan


def interpret_bf01(bf01: float) -> str:
    if not np.isfinite(bf01):
        return "n/a"
    bf10 = 1 / bf01
    val = max(bf01, bf10)
    direction = "null" if bf01 >= 1 else "alternative"
    if val < 1:
        label = "anecdotal"
    elif val < 3:
        label = "anecdotal"
    elif val < 10:
        label = "moderate"
    elif val < 30:
        label = "strong"
    elif val < 100:
        label = "very strong"
    else:
        label = "extreme"
    return f"{label} evidence for {direction}"


# ----------------------------------------------------------------------
# Diagnostic checks
# ----------------------------------------------------------------------

def normality_check(x: Sequence[float]) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"n": len(x), "shapiro_W": np.nan, "shapiro_p": np.nan, "skew": np.nan, "kurtosis": np.nan}
    W, p = stats.shapiro(x) if len(x) <= 5000 else (np.nan, np.nan)
    return {
        "n": int(len(x)),
        "shapiro_W": float(W) if np.isfinite(W) else np.nan,
        "shapiro_p": float(p) if np.isfinite(p) else np.nan,
        "skew": float(stats.skew(x)),
        "kurtosis": float(stats.kurtosis(x)),
    }


def variance_homogeneity(a: Sequence[float], b: Sequence[float]) -> dict:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"levene_W": np.nan, "levene_p": np.nan, "brown_W": np.nan, "brown_p": np.nan}
    lW, lp = stats.levene(a, b, center="mean")
    bW, bp = stats.levene(a, b, center="median")
    return {"levene_W": float(lW), "levene_p": float(lp), "brown_W": float(bW), "brown_p": float(bp)}


def fdr_adjust(pvals: Iterable[float], method: str = "fdr_bh") -> np.ndarray:
    pvals = np.asarray(list(pvals), dtype=float)
    mask = np.isfinite(pvals)
    out = np.full_like(pvals, np.nan)
    if mask.sum() == 0:
        return out
    _, p_adj, _, _ = multipletests(pvals[mask], method=method)
    out[mask] = p_adj
    return out


# ----------------------------------------------------------------------
# Power analysis
# ----------------------------------------------------------------------

def post_hoc_power(d: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    if not np.isfinite(d) or n1 < 2 or n2 < 2:
        return np.nan
    ratio = n2 / n1
    return float(TTestIndPower().power(effect_size=abs(d), nobs1=n1, ratio=ratio, alpha=alpha))


def required_n(d: float, power: float = 0.8, alpha: float = 0.05) -> float:
    if not np.isfinite(d) or d == 0:
        return np.nan
    try:
        return float(TTestIndPower().solve_power(effect_size=abs(d), power=power, alpha=alpha, ratio=1.0))
    except Exception:
        return np.nan


def power_curve(d: float, n_range: Sequence[int], alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for n in n_range:
        rows.append({"n_per_group": int(n), "power": post_hoc_power(d, int(n), int(n), alpha)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Group comparison wrapper (returns a tidy results row)
# ----------------------------------------------------------------------

def compare_two_groups(a: Sequence[float], b: Sequence[float], label_a: str, label_b: str, alpha: float = 0.05) -> dict:
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return {"label_a": label_a, "label_b": label_b, "n_a": len(a), "n_b": len(b)}
    t, p_t = stats.ttest_ind(a, b, equal_var=False)
    U, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    md, lo, hi = bootstrap_mean_diff_ci(a, b)
    norm_a = normality_check(a)
    norm_b = normality_check(b)
    var = variance_homogeneity(a, b)
    bf01 = bayes_factor_t(t, len(a) + len(b))
    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_a": int(len(a)),
        "n_b": int(len(b)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "sd_a": float(a.std(ddof=1)),
        "sd_b": float(b.std(ddof=1)),
        "mean_diff": md,
        "ci_lo": lo,
        "ci_hi": hi,
        "welch_t": float(t),
        "welch_p": float(p_t),
        "mwu_U": float(U),
        "mwu_p": float(p_u),
        "cohen_d": d,
        "hedges_g": g,
        "power_observed": post_hoc_power(d, len(a), len(b), alpha),
        "bf01": bf01,
        "bf_interpret": interpret_bf01(bf01),
        "shapiro_p_a": norm_a["shapiro_p"],
        "shapiro_p_b": norm_b["shapiro_p"],
        "levene_p": var["levene_p"],
        "brown_forsythe_p": var["brown_p"],
    }


# ----------------------------------------------------------------------
# Circular statistics for circadian phase (degrees, 0–360)
# ----------------------------------------------------------------------

def circular_mean_deg(phases_deg: Sequence[float]) -> float:
    p = np.asarray(phases_deg, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return np.nan
    rad = np.deg2rad(p)
    mean_rad = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
    return float(np.rad2deg(mean_rad) % 360)


def mean_resultant_length(phases_deg: Sequence[float]) -> float:
    p = np.asarray(phases_deg, dtype=float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return np.nan
    rad = np.deg2rad(p)
    return float(np.sqrt(np.sin(rad).mean() ** 2 + np.cos(rad).mean() ** 2))


def rayleigh_test(phases_deg: Sequence[float]) -> dict:
    """Rayleigh test for non-uniformity of circular data."""
    p = np.asarray(phases_deg, dtype=float)
    p = p[np.isfinite(p)]
    n = len(p)
    if n < 4:
        return {"n": n, "R": np.nan, "p": np.nan, "mean_deg": np.nan}
    R_bar = mean_resultant_length(p)
    R = n * R_bar
    z = R ** 2 / n
    p_val = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - R ** 2)) - (1 + 2 * n))
    return {"n": int(n), "R": float(R_bar), "p": float(p_val), "mean_deg": circular_mean_deg(p), "z": float(z)}


# ----------------------------------------------------------------------
# Sleep-burst computation from raw PIR activity
# ----------------------------------------------------------------------

@dataclass
class RestBout:
    start_idx: int
    end_idx: int  # inclusive
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_min: float
    hour_of_day: float  # midpoint hour, 0–24


def compute_rest_bouts(
    activity: pd.DataFrame,
    epoch_minutes: int,
    activity_threshold: float = 0.0,
    min_bout_minutes: int = 30,
) -> list[RestBout]:
    """
    Identify maximal runs of consecutive epochs where activity <= threshold,
    longer than min_bout_minutes.

    `activity` must be a DataFrame with columns ['timestamp', 'activity']
    on a regular epoch grid (use metrics_script.regularise_to_epoch first).
    """
    if activity.empty:
        return []
    is_rest = (activity["activity"].values <= activity_threshold).astype(int)
    bouts: list[RestBout] = []
    n = len(is_rest)
    i = 0
    while i < n:
        if is_rest[i] == 0:
            i += 1
            continue
        j = i
        while j < n and is_rest[j] == 1:
            j += 1
        # bout is [i, j-1]
        length_epochs = j - i
        duration_min = length_epochs * epoch_minutes
        if duration_min >= min_bout_minutes:
            t0 = activity["timestamp"].iloc[i]
            t1 = activity["timestamp"].iloc[j - 1]
            mid_hour = ((t0 + (t1 - t0) / 2).hour + (t0 + (t1 - t0) / 2).minute / 60.0)
            bouts.append(
                RestBout(
                    start_idx=i,
                    end_idx=j - 1,
                    start_time=t0,
                    end_time=t1,
                    duration_min=float(duration_min),
                    hour_of_day=float(mid_hour),
                )
            )
        i = j
    return bouts


def burst_summary(bouts: list[RestBout]) -> dict:
    if not bouts:
        return {"n_bouts": 0, "total_rest_min": 0.0, "mean_bout_min": np.nan, "median_bout_min": np.nan, "max_bout_min": np.nan}
    durs = np.array([b.duration_min for b in bouts])
    return {
        "n_bouts": int(len(bouts)),
        "total_rest_min": float(durs.sum()),
        "mean_bout_min": float(durs.mean()),
        "median_bout_min": float(np.median(durs)),
        "max_bout_min": float(durs.max()),
    }


def bursts_by_phase(bouts: list[RestBout], n_bins: int = 24) -> pd.DataFrame:
    """Aggregate rest bouts into hour-of-day bins (default 24 = hourly)."""
    if not bouts:
        return pd.DataFrame({"hour_bin": np.arange(n_bins), "n_bouts": 0, "mean_bout_min": np.nan})
    bin_edges = np.linspace(0, 24, n_bins + 1)
    hours = np.array([b.hour_of_day for b in bouts])
    durs = np.array([b.duration_min for b in bouts])
    idx = np.clip(np.digitize(hours, bin_edges) - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        mask = idx == k
        rows.append(
            {
                "hour_bin": float((bin_edges[k] + bin_edges[k + 1]) / 2),
                "n_bouts": int(mask.sum()),
                "mean_bout_min": float(durs[mask].mean()) if mask.any() else np.nan,
                "total_rest_min": float(durs[mask].sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_mouse_burst_profile(
    ir_csv_path: str,
    epoch_minutes: int = 60,
    activity_threshold: float = 0.0,
    min_bout_minutes: int = 30,
) -> tuple[pd.DataFrame, dict, list[RestBout]]:
    """
    End-to-end: read one mouse's PIR file, regularise, compute rest bouts,
    return (per-hour profile, summary, raw bout list).

    Reuses metrics_script.read_mouse_data + regularise_to_epoch — no duplication.
    """
    raw = ms.read_mouse_data(ir_csv_path)
    reg, ep = ms.regularise_to_epoch(raw, epoch_minutes=epoch_minutes)
    bouts = compute_rest_bouts(reg, ep, activity_threshold=activity_threshold, min_bout_minutes=min_bout_minutes)
    profile = bursts_by_phase(bouts, n_bins=24)
    summary = burst_summary(bouts)
    return profile, summary, bouts
