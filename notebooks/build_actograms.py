"""Build double-plotted actograms for selected animals.

Handles three raw-data layouts under `Raw IR Monitor Data/`:
  1. `#NN IR.csv` (single top-level file, IDs ~1-40)
  2. `NN/#NN.CSV` (single merged file inside folder)
  3. `NN/PIR000_*.CSV` (chunks inside folder; concatenate non-empty chunks)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path("/Users/carolinalangaro/Desktop/analysis_02_10")
RAW  = REPO / "Raw IR Monitor Data"
OUT  = REPO / "notebooks" / "actograms"
OUT.mkdir(parents=True, exist_ok=True)

META = pd.read_csv(REPO / "Circadian_raw.csv").drop_duplicates("ID")[
    ["ID", "Sex_new", "Age_new", "Light_new"]].set_index("ID")

def _read_one(path):
    df = pd.read_csv(path, on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    ts = pd.to_datetime(df["MM:DD:YYYY hh:mm:ss"], errors="coerce", format="mixed")
    out = pd.DataFrame({"ts": ts, "act": pd.to_numeric(df["PIRCountChange"], errors="coerce")})
    return out.dropna(subset=["ts"])

def load_animal(animal_id):
    f = RAW / f"#{animal_id:02d} IR.csv"
    if f.exists():
        return _read_one(f)
    folder = RAW / str(animal_id)
    if folder.is_dir():
        merged = folder / f"#{animal_id}.CSV"
        if merged.exists() and merged.stat().st_size > 0:
            return _read_one(merged)
        chunks = []
        for p in sorted(folder.glob("PIR000_*.CSV")):
            if p.stat().st_size > 0:
                try: chunks.append(_read_one(p))
                except Exception: pass
        if chunks:
            return pd.concat(chunks, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    return None

def make_actogram(animal_id, save_path, bin_min=10, max_days=60):
    df = load_animal(animal_id)
    if df is None or df.empty:
        return f"ID {animal_id}: NO DATA"

    df = df.sort_values("ts").drop_duplicates("ts")
    df = df[df["ts"].dt.year >= 2020]  # drop un-synced device timestamps
    if df.empty:
        return f"ID {animal_id}: no real-clock timestamps"
    # Use only the most recent contiguous window (cap the actogram height)
    cutoff = df["ts"].max() - pd.Timedelta(days=max_days)
    df = df[df["ts"] >= cutoff]
    df = df.set_index("ts").resample(f"{bin_min}min")["act"].sum().reset_index()

    df["date"] = df["ts"].dt.date
    df["min_of_day"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["bin_of_day"] = (df["min_of_day"] // bin_min).astype(int)

    bins_per_day = 24 * 60 // bin_min
    pivot = (df.pivot_table(index="date", columns="bin_of_day", values="act",
                            aggfunc="sum", fill_value=0)
               .reindex(columns=range(bins_per_day), fill_value=0))

    if len(pivot) < 2:
        return f"ID {animal_id}: only {len(pivot)} day(s) of data"

    # Double-plot: each row shows day i (0-24h) + day i+1 (24-48h)
    double = np.full((len(pivot) - 1, bins_per_day * 2), 0.0)
    arr = pivot.values
    for i in range(len(pivot) - 1):
        double[i, :bins_per_day] = arr[i]
        double[i, bins_per_day:] = arr[i + 1]

    # Robust per-animal scaling so mid activity is visible without saturation
    cap = np.percentile(double[double > 0], 95) if (double > 0).any() else 1.0
    double_clip = np.clip(double, 0, cap)

    meta = META.loc[animal_id] if animal_id in META.index else None
    sex  = meta["Sex_new"] if meta is not None else "?"
    age  = meta["Age_new"] if meta is not None else "?"
    light = meta["Light_new"] if meta is not None else "?"
    n_days = len(pivot)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * n_days)))
    ax.imshow(double_clip, cmap="Greys", aspect="auto",
              extent=[0, 48, len(double), 0],  # x: 0-48 hours
              interpolation="nearest")
    ax.set_xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    ax.set_xticklabels(["00", "06", "12", "18", "00", "06", "12", "18", "00"])
    ax.set_xlabel("Clock time (hours, double-plotted)")
    ax.set_ylabel("Day of recording")
    ax.set_yticks(np.arange(0.5, len(double), max(1, len(double)//12)))
    ax.set_yticklabels([str(i+1) for i in range(0, len(double), max(1, len(double)//12))])
    ax.axvline(24, color="#d62828", lw=0.8, ls="--", alpha=0.7)
    ax.set_title(f"Actogram — ID #{animal_id}  ({age} {sex}, {light})  |  "
                 f"{bin_min}-min bins, {n_days} days, {df['ts'].min().date()} -> {df['ts'].max().date()}",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"ID {animal_id} OK: {n_days} days  -> {save_path.name}"


REQUESTED = {
    "old_male":    [1],
    "mid_male":    [27],
    "old_female":  [63, 64, 67, 72, 73, 74, 75, 76, 77, 78, 79, 83, 84, 95, 96],
    "young_male":  [92, 93, 94, 99, 100],
}

for category, ids in REQUESTED.items():
    for aid in ids:
        out_path = OUT / f"actogram_{category}_{aid:03d}.png"
        msg = make_actogram(aid, out_path)
        print(msg)

print(f"\nAll PNGs in: {OUT}")
