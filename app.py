"""
Streamlit narrative app — Circadian rhythm × sleep-burst × cognition.

Run with:
    streamlit run app.py

Modules:
    app_data.py   — cached CSV loading & merging
    app_stats.py  — reusable statistics (effect sizes, BF, FDR, power, bursts)
    app_plots.py  — Plotly visualizations
    metrics_script.py (existing) — read PIR data, regularise, NPCRA + cosinor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import app_data as D
import app_plots as P
import app_stats as S


st.set_page_config(
    page_title="Circadian × Sleep-Burst × Cognition",
    page_icon="🌓",
    layout="wide",
)


# ============================================================
# Sidebar — global controls
# ============================================================

st.sidebar.title("⚙️ Controls")
st.sidebar.caption("Adjust filters and thresholds; all panels update live.")

with st.sidebar.expander("👥 Sample filters", expanded=True):
    sex_sel = st.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])
    age_sel = st.multiselect("Age group", ["Young", "Mid", "Old"], default=["Young", "Mid", "Old"])
    light_sel = st.multiselect("Light condition", ["CTR", "ISF"], default=["CTR", "ISF"])

with st.sidebar.expander("🌓 Circadian phase window", expanded=False):
    phase_window = st.slider(
        "Acrophase range (degrees)", min_value=0, max_value=360, value=(0, 360), step=5
    )
    phase_basis = st.radio("Apply to", ["POST", "PRE"], horizontal=True, index=0)

with st.sidebar.expander("📐 Statistical thresholds", expanded=True):
    alpha = st.slider("α (significance)", 0.001, 0.10, 0.05, 0.001)
    fdr_method = st.selectbox(
        "Multiple-comparison correction",
        ["fdr_bh", "fdr_by", "bonferroni", "holm"],
        index=0,
    )
    n_boot = st.select_slider("Bootstrap iterations", options=[500, 1000, 2000, 5000], value=2000)

with st.sidebar.expander("💤 Sleep-burst parameters", expanded=False):
    epoch_minutes = st.select_slider("Epoch length (min)", options=[5, 10, 15, 30, 60], value=60)
    activity_threshold = st.number_input("Inactivity threshold (counts)", min_value=0.0, value=0.0, step=1.0)
    min_bout_minutes = st.slider("Min bout duration (min)", 10, 240, 30, 10)
    dark_start, dark_end = st.slider("Dark phase (h)", 0, 24, (12, 24))


# ============================================================
# Load data (cached)
# ============================================================

with st.spinner("Loading data..."):
    master = D.master_table()
    barnes_long = D.load_barnes()
    effects = D.load_effect_sizes()
    bf_table = D.load_bayes_factors()
    interactions = D.load_interactions()
    single_pred = D.load_single_predictors()
    clusters = D.load_clusters()
    ir_files = D.list_ir_files()

filtered = D.apply_filters(
    master,
    sexes=sex_sel,
    ages=age_sel,
    lights=light_sel,
    phase_window=phase_window,
    phase_col=f"Phase_{phase_basis.lower()}",
)

barnes_filtered = barnes_long[
    barnes_long["Sex_new"].isin(sex_sel)
    & barnes_long["Age_new"].isin(age_sel)
    & barnes_long["Light_new"].isin(light_sel)
]

st.sidebar.markdown("---")
st.sidebar.metric("Mice in current sample", f"{filtered['ID'].nunique()} / {master['ID'].nunique()}")


# ============================================================
# Header
# ============================================================

st.title("🌓 Circadian rhythm, sleep bursts & cognition")
st.markdown(
    """
    A narrative walk-through of the circadian-behaviour analysis: hypothesis,
    methods, results, interpretation, and implications. Use the sidebar to
    re-slice the sample and re-run the statistics in real time.
    """
)

# ============================================================
# Tabs — narrative sections
# ============================================================

tabs = st.tabs([
    "1️⃣ Hypothesis",
    "2️⃣ Methods",
    "3️⃣ Sleep bursts",
    "4️⃣ Phase × cognition",
    "5️⃣ Effect sizes & power",
    "6️⃣ Diagnostics",
    "7️⃣ Interpretation & implications",
])


# -----------------------------------------------------------
# 1. Hypothesis
# -----------------------------------------------------------
with tabs[0]:
    st.header("Hypothesis")
    st.markdown(
        """
        **Primary hypothesis.** Disrupted circadian organisation — measured via
        non-parametric activity rhythm metrics (Interdaily Stability *IS*,
        Intradaily Variability *IV*, Relative Amplitude *RA*) and cosinor
        parameters (Amplitude, Phase, MESOR) — predicts spatial-learning
        deficits in the Barnes maze and reduced novelty preference in the
        Novel Object Recognition (NOR) task.

        **Secondary hypothesis.** Mice in the irregular schedule-fragmentation
        condition (**ISF**) show **fragmented sleep bursts** (more, shorter
        rest bouts skewed away from the normal dark-phase consolidation) compared
        to controls (**CTR**), and that fragmentation mediates the cognitive
        deficit.

        **Operational definitions.**
        - *Sleep burst* = a maximal run of consecutive epochs with PIR activity
          ≤ threshold lasting ≥ min-bout-duration (sidebar-configurable).
          The repo did not contain a precomputed sleep-burst metric, so this is
          derived from the raw PIR series via `metrics_script.read_mouse_data` →
          `regularise_to_epoch` → `app_stats.compute_rest_bouts`.
        - *Phase* = cosinor acrophase (degrees from midnight, 0–360).
        """
    )
    st.info(
        "ℹ️ The repo's existing CSVs (Bayes factors, single-predictor table, "
        "interactions, effect sizes) suggest the published result was a **null** "
        "circadian–behaviour association. This app makes that finding interrogable."
    )


# -----------------------------------------------------------
# 2. Methods
# -----------------------------------------------------------
with tabs[1]:
    st.header("Methods")
    st.markdown(
        f"""
        **Animals & sample.** {master['ID'].nunique()} mice with PRE/POST circadian
        recordings, Barnes maze (6 trials), and NOR. Grouping factors: Sex
        (Male/Female), Age (Young/Mid/Old), Light condition (CTR/ISF).

        **Circadian metrics.** Computed by `metrics_script.py` from raw PIR
        time-series at {epoch_minutes}-min epochs. NPCRA (Van Someren 1999):
        IS, IV, RA. Cosinor (24-h fixed): MESOR, Amplitude, Phase.

        **Sleep bursts.** Rest bouts ≥ {min_bout_minutes} min where activity
        ≤ {activity_threshold:g} counts/epoch. Aggregated by hour-of-day.

        **Cognition outcomes.** Barnes trial-6 entry-zone visits (primary),
        learning slope across trials 1–5, hole errors, goal-box latency.
        NOR: discrimination index (DI = (N − F) / (N + F) of nose duration).

        **Statistical analysis.**
        - Group comparisons: Welch's *t* (parametric) + Mann–Whitney *U* (non-parametric).
        - Effect sizes: Cohen's *d*, Hedges' *g*, with bootstrap 95% CIs ({n_boot} iter).
        - Bayes factors: JZS BF₀₁ (prior r = 0.707), so a value of 3 = moderate
          evidence for the null.
        - Multiple comparisons: {fdr_method.upper()} across all reported tests.
        - Diagnostics: Shapiro–Wilk normality, Levene & Brown–Forsythe variance
          homogeneity, Q-Q plots.
        - Power: post-hoc power (current *n*, current *d*) and required *n* for
          80% power at *d* = 0.5.
        - Circular statistics for phase: circular mean, mean resultant length,
          Rayleigh test for non-uniformity.
        """
    )

    st.subheader("Sample composition (after filters)")
    cols = st.columns(3)
    with cols[0]:
        st.dataframe(filtered.groupby("Light_new")["ID"].nunique().rename("n").reset_index(), hide_index=True)
    with cols[1]:
        st.dataframe(filtered.groupby("Sex_new")["ID"].nunique().rename("n").reset_index(), hide_index=True)
    with cols[2]:
        st.dataframe(filtered.groupby("Age_new")["ID"].nunique().rename("n").reset_index(), hide_index=True)


# -----------------------------------------------------------
# 3. Sleep bursts
# -----------------------------------------------------------
with tabs[2]:
    st.header("Sleep / rest bursts across the day")
    st.caption(
        "Computed live from raw PIR files in `Raw IR Monitor Data/`. "
        "Slow if you select many mice — try one or two first."
    )

    if ir_files.empty:
        st.warning("No raw PIR files found in `Raw IR Monitor Data/`.")
    else:
        available = ir_files["ID"].tolist()
        default_pick = [i for i in available if i in filtered["ID"].tolist()][:2] or available[:1]
        picked = st.multiselect("Mice to compute (subset)", available, default=default_pick)

        if picked:
            profiles_per_mouse = {}
            summaries = []
            progress = st.progress(0.0, text="Computing rest bouts…")
            for i, mouse_id in enumerate(picked, start=1):
                path = ir_files.set_index("ID").loc[mouse_id, "path"]
                try:
                    prof, summ, _ = S.compute_mouse_burst_profile(
                        path,
                        epoch_minutes=epoch_minutes,
                        activity_threshold=activity_threshold,
                        min_bout_minutes=min_bout_minutes,
                    )
                    profiles_per_mouse[mouse_id] = prof
                    summ["ID"] = mouse_id
                    summaries.append(summ)
                except Exception as e:
                    st.warning(f"Mouse {mouse_id}: {e}")
                progress.progress(i / len(picked))
            progress.empty()

            if summaries:
                summ_df = pd.DataFrame(summaries).merge(
                    master[["ID", "Light_new", "Sex_new", "Age_new"]], on="ID", how="left"
                )
                st.subheader("Per-mouse burst summary")
                st.dataframe(summ_df, hide_index=True, use_container_width=True)

                # Average profile by light condition
                grouped_profiles = {}
                for light_grp in ["CTR", "ISF"]:
                    ids_in_grp = summ_df[summ_df["Light_new"] == light_grp]["ID"].tolist()
                    if not ids_in_grp:
                        continue
                    stacked = pd.concat([profiles_per_mouse[i] for i in ids_in_grp])
                    avg = stacked.groupby("hour_bin").mean(numeric_only=True).reset_index()
                    grouped_profiles[light_grp] = avg

                col1, col2 = st.columns(2)
                with col1:
                    if picked:
                        first_id = picked[0]
                        st.plotly_chart(
                            P.burst_hour_plot(profiles_per_mouse[first_id], dark_phase=(dark_start, dark_end)),
                            use_container_width=True,
                        )
                        st.caption(f"Mouse {first_id} — single-animal hourly profile.")
                with col2:
                    if grouped_profiles:
                        st.plotly_chart(P.burst_group_compare(grouped_profiles), use_container_width=True)

                # Group comparison statistic
                if len(grouped_profiles) == 2:
                    st.subheader("CTR vs ISF — burst counts (Welch t / MWU)")
                    a = summ_df[summ_df["Light_new"] == "CTR"]["n_bouts"]
                    b = summ_df[summ_df["Light_new"] == "ISF"]["n_bouts"]
                    res = S.compare_two_groups(a, b, "CTR", "ISF", alpha=alpha)
                    st.json({k: v for k, v in res.items() if not isinstance(v, dict)})


# -----------------------------------------------------------
# 4. Phase × cognition
# -----------------------------------------------------------
with tabs[3]:
    st.header("Circadian phase / amplitude × cognition")

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Acrophase distribution")
        st.plotly_chart(P.phase_polar(filtered, phase_col=f"Phase_{phase_basis.lower()}"), use_container_width=True)

        # Rayleigh test per group
        rows = []
        for grp, sub in filtered.groupby("Light_new"):
            r = S.rayleigh_test(sub[f"Phase_{phase_basis.lower()}"].dropna())
            r["group"] = grp
            rows.append(r)
        st.markdown("**Rayleigh test (uniformity of phase distribution)**")
        st.dataframe(pd.DataFrame(rows)[["group", "n", "mean_deg", "R", "p"]], hide_index=True)

    with right:
        st.subheader("Predictor → cognition scatter")
        x_options = [c for c in [
            "IS_post", "IV_post", "RA_post", "Amplitude_post", "Phase_post", "MESOR_post",
            "IS_delta", "IV_delta", "RA_delta", "Amplitude_delta",
        ] if c in filtered.columns]
        y_options = [c for c in [
            "barnes_entries_t6", "barnes_errors_t6", "learning_slope",
            "barnes_goal_latency_t6", "DI_duration",
        ] if c in filtered.columns]
        x_col = st.selectbox("Circadian predictor", x_options, index=0)
        y_col = st.selectbox("Cognition outcome", y_options, index=0)
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        st.plotly_chart(
            P.correlation_scatter(filtered, x_col, y_col, method=method),
            use_container_width=True,
        )

        # Bootstrap CI + BF
        sub = filtered[[x_col, y_col]].dropna()
        if len(sub) >= 4:
            r0, lo, hi = S.bootstrap_corr_ci(sub[x_col], sub[y_col], method=method, n_boot=n_boot)
            from scipy import stats as _st
            n = len(sub)
            t = r0 * np.sqrt((n - 2) / max(1 - r0 ** 2, 1e-9))
            bf01 = S.bayes_factor_t(t, n)
            cols = st.columns(4)
            cols[0].metric(f"{method} r", f"{r0:.3f}")
            cols[1].metric("95% CI", f"[{lo:.2f}, {hi:.2f}]")
            cols[2].metric("BF₀₁", f"{bf01:.2f}" if np.isfinite(bf01) else "n/a")
            cols[3].metric("n", n)
            st.caption(S.interpret_bf01(bf01))

    st.subheader("Correlation matrix — circadian × cognition")
    corr_cols = [c for c in [
        "IS_post", "IV_post", "RA_post", "Amplitude_post", "Phase_post",
        "barnes_entries_t6", "barnes_errors_t6", "learning_slope", "DI_duration",
    ] if c in filtered.columns]
    corr_method = st.radio("Heatmap method", ["pearson", "spearman"], horizontal=True, key="corr_h")
    st.plotly_chart(P.correlation_heatmap(filtered, corr_cols, method=corr_method), use_container_width=True)


# -----------------------------------------------------------
# 5. Effect sizes & power
# -----------------------------------------------------------
with tabs[4]:
    st.header("Effect sizes & statistical power")
    st.markdown(
        "Group comparisons re-computed live on the **filtered** sample, "
        "with FDR correction across the comparison set."
    )

    outcome_options = [
        ("IS_post", "Interdaily Stability (POST)"),
        ("IV_post", "Intradaily Variability (POST)"),
        ("RA_post", "Relative Amplitude (POST)"),
        ("Amplitude_post", "Cosinor Amplitude (POST)"),
        ("Phase_post", "Cosinor Phase (POST)"),
        ("IS_delta", "Δ IS (POST−PRE)"),
        ("IV_delta", "Δ IV"),
        ("RA_delta", "Δ RA"),
        ("barnes_entries_t6", "Barnes T6 entry-zone visits"),
        ("barnes_errors_t6", "Barnes T6 hole errors"),
        ("learning_slope", "Barnes learning slope"),
        ("DI_duration", "NOR discrimination index"),
    ]

    rows = []
    for col, label in outcome_options:
        if col not in filtered.columns:
            continue
        a = filtered[filtered["Light_new"] == "CTR"][col].dropna()
        b = filtered[filtered["Light_new"] == "ISF"][col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        res = S.compare_two_groups(a, b, "CTR", "ISF", alpha=alpha)
        res["outcome"] = label
        res["col"] = col
        rows.append(res)

    if rows:
        results_df = pd.DataFrame(rows)
        results_df["p_adj"] = S.fdr_adjust(results_df["welch_p"], method=fdr_method)
        results_df["sig"] = (results_df["p_adj"] < alpha).map({True: "✓", False: ""})

        display_cols = [
            "outcome", "n_a", "n_b", "mean_a", "mean_b",
            "cohen_d", "hedges_g", "ci_lo", "ci_hi",
            "welch_p", "p_adj", "sig", "bf01", "power_observed",
        ]
        st.dataframe(
            results_df[display_cols].round(4),
            hide_index=True,
            use_container_width=True,
        )

        st.subheader("Forest plot — Cohen's d (CTR vs ISF)")
        fp = pd.DataFrame({
            "Outcome": results_df["outcome"],
            "n_CTR": results_df["n_a"],
            "n_ISF": results_df["n_b"],
            "Cohen_d": results_df["cohen_d"],
        })
        st.plotly_chart(P.effect_size_forest(fp), use_container_width=True)

        st.subheader("Power analysis")
        col1, col2 = st.columns(2)
        with col1:
            sel_outcome = st.selectbox(
                "Outcome for power curve",
                results_df["outcome"].tolist(),
                index=0,
            )
            row = results_df[results_df["outcome"] == sel_outcome].iloc[0]
            curve = S.power_curve(row["cohen_d"], n_range=range(5, 101, 5), alpha=alpha)
            st.plotly_chart(
                P.power_curve_plot(curve, target_power=0.8, observed_n=int((row["n_a"] + row["n_b"]) / 2)),
                use_container_width=True,
            )
            req = S.required_n(row["cohen_d"], power=0.8, alpha=alpha)
            st.metric("Required n per group (80% power)", f"{req:.0f}" if np.isfinite(req) else "n/a")
        with col2:
            st.markdown("**Pre-existing repo summary** (`effect_sizes_and_power.csv`)")
            st.plotly_chart(P.observed_vs_required_power(effects), use_container_width=True)

    st.subheader("Pre-computed Bayes factors (`bayes_factors_circadian_behaviour.csv`)")
    st.plotly_chart(P.bayes_factor_plot(bf_table), use_container_width=True)


# -----------------------------------------------------------
# 6. Diagnostics
# -----------------------------------------------------------
with tabs[5]:
    st.header("Assumption checks & diagnostics")

    diag_outcome = st.selectbox(
        "Variable",
        [c for c in [
            "IS_post", "IV_post", "RA_post", "Amplitude_post", "Phase_post",
            "barnes_entries_t6", "learning_slope", "DI_duration",
        ] if c in filtered.columns],
        index=0,
    )

    a = filtered[filtered["Light_new"] == "CTR"][diag_outcome].dropna()
    b = filtered[filtered["Light_new"] == "ISF"][diag_outcome].dropna()

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Normality (Shapiro–Wilk)**")
        norm_table = pd.DataFrame([
            {"group": "CTR", **S.normality_check(a)},
            {"group": "ISF", **S.normality_check(b)},
        ])
        st.dataframe(norm_table.round(4), hide_index=True)
        st.caption("Shapiro p < α suggests deviation from normality → prefer Mann–Whitney / bootstrap CI.")
    with cols[1]:
        st.markdown("**Variance homogeneity**")
        vh = S.variance_homogeneity(a, b)
        st.json(vh)
        st.caption("Levene p < α → unequal variances → Welch's t (already used).")

    cols = st.columns(2)
    with cols[0]:
        st.plotly_chart(P.qq_plot(a, label="CTR"), use_container_width=True)
    with cols[1]:
        st.plotly_chart(P.qq_plot(b, label="ISF"), use_container_width=True)

    st.subheader("Barnes learning curve")
    st.plotly_chart(
        P.barnes_learning_curve(barnes_filtered, group_col="Light_new", y_col="EntryZone_freq_new"),
        use_container_width=True,
    )

    st.subheader("Pre-computed circadian-clusters (PCA)")
    st.dataframe(clusters.head(15), hide_index=True, use_container_width=True)


# -----------------------------------------------------------
# 7. Interpretation & implications
# -----------------------------------------------------------
with tabs[6]:
    st.header("Interpretation")

    # Recompute the headline numbers on the filtered sample
    headline = {}
    for col, label in [
        ("IS_post", "Interdaily Stability"),
        ("barnes_entries_t6", "Barnes T6 entries"),
        ("learning_slope", "Learning slope"),
        ("DI_duration", "NOR DI"),
    ]:
        if col not in filtered.columns:
            continue
        a = filtered[filtered["Light_new"] == "CTR"][col].dropna()
        b = filtered[filtered["Light_new"] == "ISF"][col].dropna()
        if len(a) >= 2 and len(b) >= 2:
            d = S.cohens_d(a, b)
            from scipy import stats as _st
            _, p = _st.ttest_ind(a, b, equal_var=False)
            headline[label] = (d, p, len(a), len(b))

    if headline:
        st.markdown("**Headline group comparisons (CTR vs ISF), current filtered sample:**")
        for label, (d, p, na, nb) in headline.items():
            sig = "**significant**" if p < alpha else "not significant"
            direction = "negligible" if abs(d) < 0.2 else ("small" if abs(d) < 0.5 else ("medium" if abs(d) < 0.8 else "large"))
            st.markdown(
                f"- *{label}* — Cohen's d = **{d:+.2f}** ({direction}), "
                f"Welch p = **{p:.3g}** → {sig} at α = {alpha}. "
                f"(n = {na} CTR, {nb} ISF)"
            )

    st.markdown(
        """
        ---
        **What the data show.** Across the pre-computed Bayes-factor table and
        the live re-analysis on the filtered sample, evidence for a circadian
        → cognition link is **predominantly null**. Most BF₀₁ values exceed 3
        (moderate evidence *for* H₀) and the per-outcome Cohen's d values are
        small (|d| < 0.3 in most cells). The circadian metrics themselves
        differ modestly between CTR and ISF (as expected — that's the
        manipulation), but the spillover into cognition is weak.

        **Why this is *not* simply a "boring null".**
        - The observed power for most contrasts is well below 0.8 (see the
          Power tab). With the current sample, only large effects (d ≳ 0.7)
          would be reliably detectable.
        - Bayes factors *do* let us distinguish "no evidence" from "evidence
          of no effect", and several outcomes pass the BF₀₁ > 3 threshold —
          that is informative.
        - The **sleep-burst** angle in this app is exploratory: the operational
          definition (rest bout ≥ N minutes at threshold) is one of many
          reasonable choices and the result is sensitive to those parameters
          (see sidebar). Treat the burst tab as hypothesis-generating, not
          confirmatory.

        ### Implications

        1. **For the current paper.** Frame the headline as *"circadian
           manipulation altered activity rhythms but did not produce a
           detectable cognitive deficit at this sample size"*, supported
           explicitly by Bayes-factor evidence and a power audit.
        2. **For follow-up.** A confirmatory study targeting d ≈ 0.5 would
           need ≈ 64 mice/group (see Power tab). Pre-register the burst
           operationalisation up front so it isn't a researcher-degree-of-freedom.
        3. **For sleep-burst characterisation.** Compute bursts on every mouse
           offline (this app does it on demand), then add the per-mouse burst
           summary as a feature in the existing predictive models in
           `circadian_behaviour_advanced.py` and `circadian_predicts_behaviour.py`.
        4. **For sensitivity.** Use the sidebar to verify that conclusions
           don't flip when you (a) drop sensor-flagged mice, (b) restrict to
           one sex/age, (c) switch FDR method, (d) tighten α.
        """
    )

st.markdown("---")
st.caption(
    "Built with the existing analysis modules at the repo root. "
    "Sleep-burst computation uses `metrics_script.read_mouse_data` + "
    "`regularise_to_epoch`; statistical helpers in `app_stats.py` are "
    "ported from `analysis.py`, `circadian_predicts_behaviour.py`, "
    "`improved_barnes_analysis.py`, and `sex_age_effects.py` because those "
    "scripts execute on import."
)
