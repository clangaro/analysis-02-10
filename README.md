# analysis_02_10

Analysis pipeline and Streamlit dashboard for the circadian-behaviour study.

The public-facing dashboard is hosted on Streamlit Community Cloud and
auto-deploys from `main`. The repo also contains the standalone analysis
scripts, notebooks, and intermediate / final result tables used to produce
the figures.

For a per-file description (what each script computes, what each CSV
contains, what each notebook produces) see [`reports/project_guide.txt`](reports/project_guide.txt).

## Layout

```
.
├── app.py                  Streamlit entry point (`streamlit run app.py`)
├── app_data.py             Cached CSV loaders + filtering helpers
├── app_plots.py            Plotly visualisations used by app.py
├── app_stats.py            Bootstrap CIs, Bayes factors, effect sizes, FDR, power
├── requirements.txt        Pinned deps for Streamlit Cloud
│
├── src/
│   ├── preprocessing/      Raw-data preprocessing (NPCRA/cosinor metrics, normalisation)
│   ├── analysis/           Statistical analyses (Barnes, NOR, circadian × behaviour, …)
│   └── sensitivity/        Exclusion / sensitivity analyses
│
├── data/
│   ├── raw/                Cleaned input CSVs (Barnes, Circadian, UCBAge, CellCounting)
│   └── processed/          Derived CSVs produced by preprocessing
│
├── results/
│   ├── tables/             Result CSVs (effect sizes, Bayes factors, sensitivity, …)
│   └── figures/            Generated PNGs from the analysis scripts and notebooks
│
├── notebooks/              Jupyter notebooks for figures and exploratory work
├── reports/                Methods/explanation/writeup text files
├── poster_analysis/        Poster artefacts (notebook, PDF, dedicated figures)
└── Raw IR Monitor Data/    Raw infrared monitor data (gitignored; input to metrics_script.py)
```

## Running the dashboard locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running the analysis scripts

Scripts read inputs from `data/raw/` and `data/processed/` and write outputs
to `results/tables/` (or back into `data/processed/` for derived datasets).
Run them from the repo root:

```bash
python src/preprocessing/metrics_script.py
python src/analysis/improved_barnes_analysis.py
python src/sensitivity/exclusion_analysis.py
# …etc.
```

Each script resolves paths via a `REPO` constant
(`Path(__file__).resolve().parents[2]`), so it works regardless of the
shell's working directory.

## Notebooks

Notebooks in `notebooks/` resolve `REPO` via `Path.cwd()` (climbing one
level if launched from `notebooks/`), then read from `data/raw/`,
`data/processed/`, and `results/tables/`.
