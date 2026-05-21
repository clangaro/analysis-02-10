"""
Compute circadian metrics from raw PIR time-series (ClockLab IR monitor exports).

Validated against ClockLab output using:
- 60-minute epoch bins (ClockLab default)
- Cross-day IV formula (standard Van Someren et al. 1999)
- Per-minute reporting for Mean/MESOR/Amplitude
- Per-mouse PRE/POST windows based on treatment start dates

Metrics:
- Non-parametric: IS, IV, RA  (Van Someren et al. 1999 NPCRA)
- Cosinor (period fixed at 24 h): MESOR, Amplitude, Phase (deg), F, p
- Also: Mean, Variance, Period (=24)

Data layout handled:
- Flat files:  "#NN IR.csv"  (single CSV per mouse)
- Directory files: numbered folders (e.g. "59/") containing multiple PIR*.CSV
  segments that are concatenated chronologically.

Requires: pandas, numpy, statsmodels, openpyxl
    pip install pandas numpy statsmodels openpyxl
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

logger = logging.getLogger(__name__)

# ClockLab default: 60-minute bins for NPCRA, report values per-minute
CLOCKLAB_EPOCH_MINUTES = 60


# ============================================================
# I/O helpers
# ============================================================

def read_pir_csv(path: str) -> pd.DataFrame:
    """
    Reads a ClockLab IR monitor CSV.

    Handles both formats:
      - "#NN IR.csv" flat files
      - "PIR000_*.CSV" segment files

    Returns DataFrame with columns: 'timestamp', 'activity'.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.rstrip(",")

    ts_col = None
    for c in df.columns:
        if "hh:mm:ss" in c or "YYYY" in c or "timestamp" in c.lower():
            ts_col = c
            break
    if ts_col is None:
        raise ValueError(f"No timestamp column in {path}. Columns: {list(df.columns)}")

    act_col = None
    for c in df.columns:
        if "PIRCountChange" in c:
            act_col = c
            break
    if act_col is None:
        raise ValueError(f"No PIRCountChange column in {path}. Columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    out["activity"] = pd.to_numeric(df[act_col], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["activity"] = out["activity"].fillna(0.0)
    return out


def read_mouse_data(path: str) -> pd.DataFrame:
    """
    Reads activity data for one mouse.
    If path is a file, reads it directly.
    If path is a directory, concatenates all PIR/IR CSV files chronologically.
    """
    p = Path(path)

    if p.is_file():
        return read_pir_csv(str(p))

    if p.is_dir():
        csvs = sorted(p.glob("PIR*.CSV")) + sorted(p.glob("*IR*.csv"))
        csvs += sorted(p.glob("*.CSV"))
        seen = set()
        unique_csvs = []
        for f in csvs:
            if f not in seen and f.stat().st_size > 0:
                seen.add(f)
                unique_csvs.append(f)

        if not unique_csvs:
            raise FileNotFoundError(f"No non-empty CSV files in {path}")

        frames = []
        for csv_path in unique_csvs:
            try:
                frames.append(read_pir_csv(str(csv_path)))
            except (ValueError, pd.errors.EmptyDataError) as e:
                logger.warning("Skipping %s: %s", csv_path.name, e)

        if not frames:
            raise ValueError(f"All CSV files in {path} failed to parse.")

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
        return combined

    raise FileNotFoundError(f"Path does not exist: {path}")


def infer_epoch_minutes(ts: pd.Series) -> int:
    """Infers epoch length in minutes using mode of timestamp diffs."""
    diffs = ts.sort_values().diff().dropna()
    if diffs.empty:
        raise ValueError("Not enough timestamps to infer epoch length.")
    rounded = diffs.dt.total_seconds().round(0)
    mode_seconds = rounded.mode()
    if len(mode_seconds) == 0 or mode_seconds.iloc[0] <= 0:
        med = diffs.dt.total_seconds().median()
        if not np.isfinite(med) or med <= 0:
            raise ValueError("Invalid timestamp spacing.")
        return int(max(1, round(med / 60.0)))
    return int(max(1, round(mode_seconds.iloc[0] / 60.0)))


def regularise_to_epoch(
    df: pd.DataFrame,
    epoch_minutes: Optional[int] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Resamples activity to a fixed epoch grid, summing within each bin.
    Missing epochs filled with 0.
    """
    if epoch_minutes is None:
        epoch_minutes = infer_epoch_minutes(df["timestamp"])
    if start is None:
        start = df["timestamp"].min()
    if end is None:
        end = df["timestamp"].max()
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    d = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if d.empty:
        raise ValueError("No data in the requested [start, end] window.")

    d = d.set_index("timestamp")
    rule = f"{epoch_minutes}min"
    rs = d["activity"].resample(rule).sum().to_frame("activity")
    rs = rs.asfreq(rule, fill_value=0.0)
    rs = rs.reset_index().rename(columns={"timestamp": "timestamp"})
    return rs, epoch_minutes


# ============================================================
# Sensor quality checks
# ============================================================

def flag_sensor_issues(
    df_reg: pd.DataFrame,
    epoch_minutes: int,
    zero_run_threshold_hours: float = 6.0,
) -> Dict[str, object]:
    """
    Detects sensor dropouts and saturation.

    Returns dict with:
      longest_zero_run_hours, n_zero_runs_over_threshold,
      saturation_value, n_saturated_epochs, pct_zero, flag
    """
    x = df_reg["activity"].to_numpy()
    is_zero = x == 0
    pct_zero = float(is_zero.mean() * 100)

    zero_runs = []
    run_len = 0
    for val in is_zero:
        if val:
            run_len += 1
        else:
            if run_len > 0:
                zero_runs.append(run_len)
            run_len = 0
    if run_len > 0:
        zero_runs.append(run_len)

    threshold_epochs = int(zero_run_threshold_hours * 60 / epoch_minutes)
    longest_zero = max(zero_runs) if zero_runs else 0
    longest_zero_hours = longest_zero * epoch_minutes / 60.0
    n_over_threshold = sum(1 for r in zero_runs if r >= threshold_epochs)

    if len(x) > 0 and x.max() > 0:
        sat_val = float(np.percentile(x[x > 0], 99.9))
        n_saturated = int(np.sum(x >= sat_val))
    else:
        sat_val = 0.0
        n_saturated = 0

    flag = (longest_zero_hours >= zero_run_threshold_hours) or (pct_zero > 80)

    return {
        "longest_zero_run_hours": round(longest_zero_hours, 2),
        "n_zero_runs_over_threshold": n_over_threshold,
        "saturation_value": round(sat_val, 2),
        "n_saturated_epochs": n_saturated,
        "pct_zero": round(pct_zero, 2),
        "flag": flag,
    }


# ============================================================
# NPCRA metrics (IS, IV, RA) — matches ClockLab
# ============================================================

@dataclass
class NPCRAResult:
    IS: float
    IV: float
    RA: float
    M10: float
    L5: float


def _split_into_days(df: pd.DataFrame, epoch_minutes: int) -> Tuple[np.ndarray, int, int]:
    """Reshape regularised series into [n_days, p]. Only complete days."""
    x = df["activity"].to_numpy(dtype=float)
    p = int(round((24 * 60) / epoch_minutes))
    if p <= 0:
        raise ValueError("Invalid epoch_minutes.")
    n_complete = len(x) // p
    if n_complete < 2:
        raise ValueError(
            f"Need >= 2 complete days. Got {len(x)} samples = "
            f"{len(x) * epoch_minutes / 60:.1f} hours."
        )
    x = x[: n_complete * p]
    return x.reshape(n_complete, p), n_complete, p


def compute_is_iv_ra(df_reg: pd.DataFrame, epoch_minutes: int) -> NPCRAResult:
    """
    Non-parametric circadian rhythm analysis (Van Someren et al. 1999).

    Uses the CROSS-DAY IV formula (standard, matches ClockLab):
        IV = (N * sum(diff(x)^2)) / ((N-1) * sum((x-mean)^2))
    applied across the full flattened time series.

    M10/L5 computed on the mean 24h profile (standard NPCRA).
    """
    X, n, p = _split_into_days(df_reg, epoch_minutes)
    x = X.reshape(-1)
    N = x.size
    mu = x.mean()
    denom = np.sum((x - mu) ** 2)

    if denom <= 0:
        return NPCRAResult(IS=0.0, IV=0.0, RA=0.0, M10=0.0, L5=0.0)

    # IS
    mean_profile = X.mean(axis=0)
    IS = float((n * np.sum((mean_profile - mu) ** 2)) / denom)

    # IV (cross-day, standard formula — matches ClockLab)
    diffs = np.diff(x)
    IV = float((N * np.sum(diffs ** 2)) / ((N - 1) * denom))

    # RA from mean profile
    win10 = int(round((10 * 60) / epoch_minutes))
    win5 = int(round((5 * 60) / epoch_minutes))
    if win10 < 1 or win5 < 1:
        raise ValueError("Epoch too large for M10/L5.")

    def _rolling_circular(profile: np.ndarray, w: int) -> np.ndarray:
        y = np.concatenate([profile, profile[: w - 1]])
        c = np.cumsum(np.insert(y, 0, 0.0))
        return ((c[w:] - c[:-w]) / w)[: profile.size]

    rm10 = _rolling_circular(mean_profile, win10)
    rm5 = _rolling_circular(mean_profile, win5)
    M10 = float(np.max(rm10))
    L5 = float(np.min(rm5))
    RA = float((M10 - L5) / (M10 + L5)) if (M10 + L5) > 0 else 0.0

    return NPCRAResult(IS=IS, IV=IV, RA=RA, M10=M10, L5=L5)


# ============================================================
# Cosinor (24h)
# ============================================================

@dataclass
class CosinorResult:
    MESOR: float
    Amplitude: float
    Phase_deg: float
    Period: float
    F: float
    p: float


def compute_cosinor_24h(df_reg: pd.DataFrame) -> CosinorResult:
    """
    Fits y = MESOR + a*cos(wt) + b*sin(wt) with w = 2pi/24.
    Phase referenced to midnight for cross-condition comparability.
    """
    y = df_reg["activity"].to_numpy(dtype=float)
    timestamps = df_reg["timestamp"]
    midnight = timestamps.dt.normalize()
    t_hours = (timestamps - midnight).dt.total_seconds().to_numpy(dtype=float) / 3600.0

    omega = 2.0 * np.pi / 24.0
    cos_t = np.cos(omega * t_hours)
    sin_t = np.sin(omega * t_hours)

    X_full = np.column_stack([np.ones_like(y), cos_t, sin_t])
    X_red = np.ones((len(y), 1))

    model_full = sm.OLS(y, X_full).fit()
    model_red = sm.OLS(y, X_red).fit()

    an = anova_lm(model_red, model_full)
    F = float(an["F"].iloc[1])
    p = float(an["Pr(>F)"].iloc[1])

    MESOR = float(model_full.params[0])
    a = float(model_full.params[1])
    b = float(model_full.params[2])
    Amplitude = float(np.sqrt(a * a + b * b))
    Phase_deg = float(np.degrees(np.arctan2(-b, a)) % 360.0)

    return CosinorResult(MESOR=MESOR, Amplitude=Amplitude, Phase_deg=Phase_deg,
                         Period=24.0, F=F, p=p)


# ============================================================
# Full metric computation
# ============================================================

@dataclass
class CircadianMetrics:
    Mean: float
    Variance: float
    MESOR: float
    Amplitude: float
    Phase: float
    Period: float
    F: float
    p: float
    IS: float
    IV: float
    RA: float


def compute_all_metrics(
    df_raw: pd.DataFrame,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    epoch_minutes: int = CLOCKLAB_EPOCH_MINUTES,
) -> CircadianMetrics:
    """
    Computes circadian metrics matching ClockLab conventions:
    - 60-min bins for NPCRA (IS, IV, RA)
    - Per-minute values for Mean, MESOR, Amplitude, Variance
    """
    df_reg, ep = regularise_to_epoch(df_raw, epoch_minutes=epoch_minutes,
                                      start=start, end=end)

    # NPCRA on raw binned data (60-min counts)
    npcra = compute_is_iv_ra(df_reg, epoch_minutes=ep)

    # Cosinor on per-minute rate
    df_permin = df_reg.copy()
    df_permin["activity"] = df_permin["activity"] / ep
    cos = compute_cosinor_24h(df_permin)

    # Report Mean/Variance as per-minute
    activity_permin = df_reg["activity"] / ep
    Mean = float(activity_permin.mean())
    Variance = float(activity_permin.var(ddof=1))

    return CircadianMetrics(
        Mean=Mean, Variance=Variance,
        MESOR=cos.MESOR, Amplitude=cos.Amplitude, Phase=cos.Phase_deg,
        Period=cos.Period, F=cos.F, p=cos.p,
        IS=npcra.IS, IV=npcra.IV, RA=npcra.RA,
    )


# ============================================================
# Normalisation for sensor sensitivity
# ============================================================

def normalise_activity(
    df_raw: pd.DataFrame,
    method: str = "zscore",
) -> pd.DataFrame:
    """
    Normalises raw activity to remove inter-sensor gain differences.

    Methods:
      'zscore'  : (x - mean) / std  — preserves temporal pattern, removes scale.
                  Standard in circadian literature (e.g. Winnebeck et al. 2018).
      'robust'  : (x - median) / IQR — more resistant to outliers/saturation.
      'minmax'  : (x - min) / (max - min) — scales to [0, 1].
      'rank'    : rank-based (percentile) transform — completely non-parametric,
                  eliminates all gain/offset differences.

    After normalisation, all mice have comparable signal scales, so IS/IV/RA
    reflect temporal structure rather than sensor sensitivity.

    Returns a copy with normalised 'activity' column.
    """
    out = df_raw.copy()
    x = out["activity"].to_numpy(dtype=float)

    if method == "zscore":
        mu, sd = x.mean(), x.std()
        if sd > 0:
            out["activity"] = (x - mu) / sd
        else:
            out["activity"] = 0.0

    elif method == "robust":
        med = np.median(x)
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            out["activity"] = (x - med) / iqr
        else:
            out["activity"] = 0.0

    elif method == "minmax":
        xmin, xmax = x.min(), x.max()
        rng = xmax - xmin
        if rng > 0:
            out["activity"] = (x - xmin) / rng
        else:
            out["activity"] = 0.0

    elif method == "rank":
        from scipy.stats import rankdata
        ranks = rankdata(x, method="average")
        out["activity"] = ranks / len(ranks)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore', 'robust', 'minmax', or 'rank'.")

    return out


# ============================================================
# Batch processing with per-mouse PRE/POST windows
# ============================================================

def _extract_mouse_id(path: str) -> int | str:
    """Extract numeric mouse ID from filename or directory name."""
    base = os.path.basename(path.rstrip("/"))
    m = re.search(r"#(\d+)", base)
    if m:
        return int(m.group(1))
    if base.isdigit():
        return int(base)
    m = re.search(r"PIR(\d+)", base, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return base


def load_mouse_metadata(xlsx_path: str) -> pd.DataFrame:
    """
    Loads per-mouse metadata from UCB_Age_Overview.xlsx 'Mice' sheet.

    Returns DataFrame with columns:
      ID, Cohort, Sex, Age_Group, Light_Group, treatment_start, actogram_problems
    """
    mice = pd.read_excel(xlsx_path, sheet_name="Mice", header=0)
    mice.columns = [str(c).strip() for c in mice.columns]

    meta = pd.DataFrame()
    meta["ID"] = pd.to_numeric(mice["Animal ID"], errors="coerce")
    meta["Cohort"] = mice["Cohort"]
    meta["Sex"] = mice["Sex"]
    meta["Age_Group"] = mice["Age Group"]
    meta["Light_Group"] = mice["Light Group"]
    meta["treatment_start"] = pd.to_datetime(mice["Start of Experiment"], errors="coerce")
    meta["actogram_problems"] = mice["Problems with Actograms"].fillna("").astype(str).str.strip()

    meta = meta.dropna(subset=["ID", "treatment_start"])
    meta["ID"] = meta["ID"].astype(int)
    return meta.reset_index(drop=True)


def discover_mouse_sources(folder: str) -> Dict[int | str, str]:
    """Discovers all mouse data files/directories in a folder."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    sources: List[Tuple[int | str, str]] = []
    for f in sorted(folder_path.glob("*.csv")):
        if f.stat().st_size > 0:
            sources.append((_extract_mouse_id(str(f)), str(f)))
    for f in sorted(folder_path.glob("*.CSV")):
        if f.stat().st_size > 0:
            sources.append((_extract_mouse_id(str(f)), str(f)))
    for d in sorted(folder_path.iterdir()):
        if d.is_dir():
            sources.append((_extract_mouse_id(str(d)), str(d)))

    seen: Dict[int | str, str] = {}
    for mid, src in sources:
        if mid not in seen or Path(src).is_dir():
            seen[mid] = src
    return seen


def compute_batch(
    folder: str,
    metadata: pd.DataFrame,
    epoch_minutes: int = CLOCKLAB_EPOCH_MINUTES,
    normalise: Optional[str] = None,
    check_sensors: bool = True,
) -> pd.DataFrame:
    """
    Batch-compute circadian metrics with per-mouse PRE/POST windows.

    PRE  = recording start -> treatment_start
    POST = treatment_start -> recording end

    Parameters
    ----------
    folder : path to Raw IR Monitor Data
    metadata : from load_mouse_metadata()
    epoch_minutes : bin size (default 60 to match ClockLab)
    normalise : None, 'zscore', 'robust', 'minmax', or 'rank'.
                If set, normalises raw activity before computing metrics.
    check_sensors : include sensor quality flags
    """
    sources = discover_mouse_sources(folder)
    id_to_treatment = dict(zip(metadata["ID"], metadata["treatment_start"]))

    rows: List[dict] = []
    n_total = len(sources)

    for i, (mouse_id, source) in enumerate(sorted(sources.items()), 1):
        logger.info("[%d/%d] Processing mouse %s", i, n_total, mouse_id)

        try:
            df_raw = read_mouse_data(source)
        except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
            logger.error("FAILED to read mouse %s: %s", mouse_id, e)
            continue

        if normalise:
            df_raw = normalise_activity(df_raw, method=normalise)

        treatment = id_to_treatment.get(mouse_id)
        if treatment is None:
            logger.warning("No treatment date for mouse %s, computing ALL only", mouse_id)
            windows = {"ALL": (None, None)}
        else:
            windows = {
                "PRE": (None, treatment),
                "POST": (treatment, None),
            }

        for label, (st, en) in windows.items():
            try:
                met = compute_all_metrics(df_raw, start=st, end=en, epoch_minutes=epoch_minutes)
                row = dict(ID=mouse_id, PRE_POST=label, **asdict(met),
                           source=os.path.basename(source))

                if check_sensors:
                    df_reg, ep = regularise_to_epoch(df_raw, epoch_minutes=epoch_minutes,
                                                      start=st, end=en)
                    sensor = flag_sensor_issues(df_reg, ep)
                    row["sensor_flag"] = sensor["flag"]
                    row["pct_zero"] = sensor["pct_zero"]
                    row["longest_zero_run_hours"] = sensor["longest_zero_run_hours"]

                rows.append(row)
            except Exception as e:
                logger.error("FAILED mouse %s (%s): %s", mouse_id, label, e)

    if not rows:
        raise RuntimeError("All mice failed. Check logs.")

    result = pd.DataFrame(rows)
    logger.info("Done. %d rows from %d mice. Normalisation: %s",
                len(result), n_total, normalise or "none")
    return result


# ============================================================
# Validation against ClockLab
# ============================================================

def compare_to_clocklab(clocklab_csv: str, computed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge computed metrics with ClockLab output and compute absolute differences.
    """
    clock = pd.read_csv(clocklab_csv)
    clock = clock.rename(columns={"PRE.POST": "PRE_POST"})
    clock["ID"] = pd.to_numeric(clock["ID"], errors="coerce")

    comp = computed_df.copy()
    comp["ID"] = pd.to_numeric(comp["ID"], errors="coerce")
    if "Variance" in comp.columns:
        comp = comp.rename(columns={"Variance": "X..Variance"})

    merged = clock.merge(comp, on=["ID", "PRE_POST"], how="inner",
                         suffixes=("_clocklab", "_computed"))

    for c in ["Amplitude", "Phase", "Mean", "X..Variance", "MESOR", "F", "p", "RA", "IV", "IS"]:
        cl, co = f"{c}_clocklab", f"{c}_computed"
        if cl in merged.columns and co in merged.columns:
            merged[f"absdiff_{c}"] = (merged[cl] - merged[co]).abs()

    return merged


# ============================================================
# Main: validation + normalised sensitivity analysis
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    DATA_FOLDER = "/Users/carolinalangaro/Desktop/analysis 02:10/Raw IR Monitor Data"
    CLOCKLAB_CSV = "/Users/carolinalangaro/Desktop/analysis 02:10/Circadian_raw.csv"
    METADATA_XLSX = "/Users/carolinalangaro/Downloads/UCB_Age_Overview.xlsx"

    # --- Load metadata ---
    meta = load_mouse_metadata(METADATA_XLSX)
    print(f"Loaded metadata for {len(meta)} mice.\n")

    # =============================================
    # STEP 1: Validate against ClockLab (no normalisation)
    # =============================================
    print("=" * 60)
    print("STEP 1: Computing raw metrics (60-min bins, per-mouse PRE/POST)")
    print("=" * 60)
    computed_raw = compute_batch(DATA_FOLDER, meta, normalise=None)

    diffs = compare_to_clocklab(CLOCKLAB_CSV, computed_raw)
    absdiff_cols = [c for c in diffs.columns if c.startswith("absdiff_")]
    if absdiff_cols and len(diffs) > 0:
        print(f"\nMatched {len(diffs)} rows with ClockLab.")
        print("\nAbsolute differences (computed vs ClockLab):")
        print(diffs[absdiff_cols].describe().round(4).to_string())
    else:
        print(f"\nWARNING: {len(diffs)} rows matched ClockLab. Check PRE_POST labels.")

    # Show sensor flags
    if "sensor_flag" in computed_raw.columns:
        flagged = computed_raw[computed_raw["sensor_flag"]]
        print(f"\n{len(flagged)} rows flagged for sensor issues.")
        if len(flagged) > 0:
            print(flagged[["ID", "PRE_POST", "pct_zero", "longest_zero_run_hours"]].head(20).to_string())

    # =============================================
    # STEP 2: Normalised sensitivity analysis
    # =============================================
    print("\n" + "=" * 60)
    print("STEP 2: Normalised sensitivity analysis (z-score)")
    print("=" * 60)
    computed_norm = compute_batch(DATA_FOLDER, meta, normalise="zscore")

    # Compare raw vs normalised IS/IV/RA
    raw_wide = computed_raw.pivot(index="ID", columns="PRE_POST", values=["IS", "IV", "RA"])
    norm_wide = computed_norm.pivot(index="ID", columns="PRE_POST", values=["IS", "IV", "RA"])

    print("\nRaw metrics summary (PRE):")
    pre_raw = computed_raw[computed_raw["PRE_POST"] == "PRE"]
    print(pre_raw[["IS", "IV", "RA"]].describe().round(4).to_string())

    print("\nNormalised metrics summary (PRE):")
    pre_norm = computed_norm[computed_norm["PRE_POST"] == "PRE"]
    print(pre_norm[["IS", "IV", "RA"]].describe().round(4).to_string())

    # Compute delta_IS for both
    for label, df in [("Raw", computed_raw), ("Normalised", computed_norm)]:
        piv = df.pivot(index="ID", columns="PRE_POST", values="IS")
        if "PRE" in piv.columns and "POST" in piv.columns:
            delta = piv["POST"] - piv["PRE"]
            print(f"\n{label} delta_IS (POST - PRE):")
            print(f"  mean={delta.mean():.4f}, std={delta.std():.4f}, "
                  f"min={delta.min():.4f}, max={delta.max():.4f}")

    # Save outputs
    out_dir = "/Users/carolinalangaro/Desktop/analysis 02:10"
    computed_raw.to_csv(os.path.join(out_dir, "circadian_computed_raw.csv"), index=False)
    computed_norm.to_csv(os.path.join(out_dir, "circadian_computed_normalised.csv"), index=False)
    print(f"\nSaved: circadian_computed_raw.csv, circadian_computed_normalised.csv")
