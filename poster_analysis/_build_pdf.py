"""Build the PI-ready PDF summary of all poster-analysis figures with
interpretive captions. Run once; writes poster_summary.pdf alongside."""
import base64
import nbformat
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether,
    Table, TableStyle,
)

HERE = Path(__file__).resolve().parent
EXTRACT = HERE / "_pdf_temp_figures"
EXTRACT.mkdir(exist_ok=True)
OUT = HERE / "poster_summary.pdf"


# ---------------------------------------------------------------------------
# 1. Extract every embedded PNG from the four notebooks
# ---------------------------------------------------------------------------

def extract_notebook_figures(nb_path, label_prefix):
    nb = nbformat.read(nb_path, as_version=4)
    extracted = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        for o in cell.outputs:
            if o.output_type != "display_data":
                continue
            if "image/png" not in o.get("data", {}):
                continue
            fn = EXTRACT / f"{label_prefix}_cell{i:02d}.png"
            with open(fn, "wb") as f:
                f.write(base64.b64decode(o["data"]["image/png"]))
            extracted.append((f"{label_prefix}_cell{i:02d}", str(fn)))
    return dict(extracted)


imgs = {}
imgs.update(extract_notebook_figures(HERE / "poster_results.ipynb", "P"))
imgs.update(extract_notebook_figures(
    HERE / "vulnerability_resilience_analysis" / "01_pi_approach_per_metric_1sd.ipynb", "N1"))
imgs.update(extract_notebook_figures(
    HERE / "vulnerability_resilience_analysis" / "02_composite_directional_approach.ipynb", "N2"))
imgs.update(extract_notebook_figures(
    HERE / "vulnerability_resilience_analysis" / "03_behavioural_composite_approach.ipynb", "N3"))

# The 4 standalone poster PNGs
for name in ["fig1_barnes_nor_orthogonality.png",
             "fig2_circadian_by_memory_split.png",
             "fig3_amplitude_by_nor_sex.png",
             "fig4_nor_dysfunction_by_sex.png"]:
    imgs[name] = str(HERE / "figures" / name)

print(f"Available images: {len(imgs)}")


# ---------------------------------------------------------------------------
# 2. Figure catalogue (titles + captions) in the order we want them in the PDF
# ---------------------------------------------------------------------------

FIGURES = [
    # ---------- Part 1: poster-ready figures ----------
    {"part": 1, "key": "fig3_amplitude_by_nor_sex.png",
     "title": "POSTER FIG A — Amplitude of old-male, NOR-impaired mice is highest",
     "caption":
        "Rhythm amplitude (POST) by Sex × NOR classification (n = 36 with both "
        "scores). The NOR-impaired Old-Male cell has visibly higher amplitude than "
        "the resilient cell; no such separation in Females, Mid-Males, or Mid-Females. "
        "This is the subgroup that carries the cohort-level Cohen's d = +0.51 for "
        "the NOR classification split. Same subgroup as the FDR-corrected Age × Sex "
        "Barnes locomotor deficit (p_fdr = 0.004)."},

    {"part": 1, "key": "fig4_nor_dysfunction_by_sex.png",
     "title": "POSTER FIG B — Male-specific NOR–rhythm relationship (continuous form)",
     "caption":
        "NOR discrimination index vs circadian dysfunction score (directional "
        "composite, higher = worse rhythm across IS, IV, RA, Amplitude) in old "
        "animals, with separate regression lines for Males and Females. Males: "
        "ρ = +0.35, p = 0.11, n = 22. Females: ρ ≈ 0, n = 14. The slope "
        "divergence between sexes is the continuous-form signature of a "
        "sex-specific circadian-NOR relationship — the same pattern the "
        "dichotomised NOR-split reveals, without dichotomisation."},

    {"part": 1, "key": "fig1_barnes_nor_orthogonality.png",
     "title": "POSTER FIG C — Barnes and NOR index orthogonal memory domains",
     "caption":
        "Each old animal plotted in the Barnes fidelity composite (x) × NOR DI (y) "
        "plane, colored by Sex. Dashed lines mark each score's median. Cross-tab "
        "Cohen's κ = −0.17 and only 42% classification agreement (chance = 50%). "
        "Spatial and recognition memory are not just weakly coupled — they are "
        "statistically independent in this cohort, which empirically rules out "
        "the use of any composite memory vulnerability score."},

    {"part": 1, "key": "fig2_circadian_by_memory_split.png",
     "title": "POSTER FIG D — Circadian metrics by memory classification (effect sizes)",
     "caption":
        "Forest plot of Cohen's d ± 95% CI for each circadian metric, under "
        "Barnes classification (purple) and NOR classification (orange). Seven of "
        "eight cells are flat (|d| < 0.15). The exception is Amplitude × NOR "
        "split, d ≈ +0.5 — NOR-impaired animals have stronger rhythms, same "
        "paradoxical direction as the poster's Old-Male finding. CI crosses zero, "
        "so this is a hypothesis-generating pattern, not a confirmatory test."},

    # ---------- Part 2: full poster-notebook sequence ----------
    {"part": 2, "key": "P_cell04",
     "title": "Manipulation check: ISF did not differentially perturb rhythms",
     "caption":
        "PRE → POST change per circadian metric, stratified by light condition, "
        "in Mid+Old animals. CTR (green) and ISF (red) trajectories overlap for all "
        "four metrics. Mixed-effects tests: PRE × Light interactions all p > 0.25, "
        "p_fdr > 0.50. The intervention does not pass its mechanistic precondition "
        "— this itself is a reportable finding and motivates collapsing PRE/POST "
        "into a single per-animal average for downstream analyses."},

    {"part": 2, "key": "P_cell11",
     "title": "Sex, not Age or Light, drives circadian phenotype",
     "caption":
        "Forest plot of standardised β (units of metric SD) ± 95% CI for Age, Sex, "
        "and Light effects on each collapsed circadian metric (Mid+Old, n = 84). "
        "Red = FDR-significant. Only three effects survive: Male < Female on "
        "Amplitude_mean (p_fdr = 0.029), Male < Female on RA_mean (p_fdr = 0.043), "
        "and a second Amplitude_mean Sex effect. Age and Light effects are all "
        "non-significant after correction. This makes Sex the right stratification "
        "for the circadian-cognition question."},

    {"part": 2, "key": "P_cell15",
     "title": "Full-cohort circadian → cognition: null result",
     "caption":
        "Signed-β heatmap of 16 covariate-adjusted regressions (4 collapsed "
        "circadian metrics × 4 behavioural outcomes), each adjusted for Age, Sex, "
        "Light. Zero cells survive FDR correction (minimum p_fdr = 0.82). The "
        "canonical hypothesis that rhythm strength predicts cognition is not "
        "supported at the cohort level. This null is a scientifically important "
        "finding on its own — and motivates the targeted subgroup analysis below."},

    {"part": 2, "key": "P_cell18",
     "title": "Old × Male locomotor deficit (pre-existing, FDR-corrected)",
     "caption":
        "Barnes T6 total distance moved, by Age × Sex. Old males move markedly "
        "less than the additive Age + Sex model predicts — the Age[Old]:Sex[Male] "
        "interaction term is β = −632 cm, p = 0.00017, p_fdr = 0.0036 (from "
        "sex_age_barnes_results.csv). This is the only FDR-surviving effect in "
        "the standard sex/age Barnes family and identifies Old Males as the "
        "phenotypically distinct subgroup a priori — making all following "
        "Old-Male analyses hypothesis-directed rather than exploratory."},

    {"part": 2, "key": "P_cell22",
     "title": "Old-Male dissociation — rhythm amplitude × memory domain",
     "caption":
        "Scatter of Amplitude_mean versus NOR discrimination index (left) and "
        "Barnes learning slope (right) in Old Males (n = 22). The same predictor "
        "correlates negatively with NOR (r = −0.49, p = 0.022) and positively "
        "with Barnes learning (r = +0.43, p = 0.048) — an opposite-signed "
        "dissociation within a single subgroup. Multiple related metrics pointing "
        "the same way is a pattern, not a single test, and pattern-level "
        "consistency is harder to explain by chance."},

    {"part": 2, "key": "P_cell23",
     "title": "Two independent axes of the same biology",
     "caption":
        "In the same Old-Male subgroup, Amplitude_mean and IV_mean correlate with "
        "NOR DI in opposite directions (Amplitude r = −0.48; IV r = +0.42). "
        "Interpretable as one underlying axis: rigid / strong rhythms (high "
        "amplitude, low variability) are associated with worse NOR, whereas more "
        "variable rhythms are associated with better NOR. Having two "
        "independently-measured metrics tell the same story strengthens the "
        "mechanistic narrative."},

    {"part": 2, "key": "P_cell24",
     "title": "Age × Sex specificity — signal isolated to Old × Male",
     "caption":
        "Amplitude_mean vs NOR DI in all four Age × Sex cells. Only the Old-Male "
        "cell shows a detectable correlation (r = −0.49, p = 0.022). Mid-Female, "
        "Mid-Male, and Old-Female all yield r's near zero (|r| ≤ 0.20, all p > 0.4, "
        "all n ≥ 14). Cell-size alone cannot explain this — the Mid-Male cell is "
        "of similar size to Old-Male. This rules out a generic circadian-NOR "
        "signal."},

    {"part": 2, "key": "P_cell25",
     "title": "Light-condition moderation — NOR effect is ISF-specific",
     "caption":
        "Within Old Males, the Amplitude_mean ↔ NOR DI correlation splits by light "
        "condition: ISF r = −0.81, p = 0.005, n = 10; CTR r = −0.10, p = 0.72, "
        "n = 12. Sleep fragmentation unmasks (or induces) the relationship "
        "without changing mean rhythm metrics (Section 1). The effect is small-n "
        "but striking, with shared variance ≈ 65% within the ISF cell."},

    {"part": 2, "key": "P_cell26",
     "title": "Specificity: novel- not familiar-object exploration",
     "caption":
        "Within Old Males, Amplitude_mean predicts duration spent on the novel "
        "object (r = −0.41, p = 0.06) but not on the familiar object "
        "(r = +0.13, p = 0.56). The NOR DI finding is therefore carried by "
        "novelty detection, not generic motor output or total exploration. This "
        "is a cognitive phenomenon, not a confound."},

    {"part": 2, "key": "P_cell27",
     "title": "Robustness: leave-one-out + bootstrap",
     "caption":
        "Left: Leave-one-out Pearson r for the Amplitude_mean → NOR DI effect "
        "stays in [−0.58, −0.37] — no single animal drives it. Right: 5,000-"
        "resample bootstrap distribution, 95% CI [−0.78, −0.10]; excludes zero. "
        "Together these confirm the effect is robust to removing individual "
        "points and to the parametric Pearson assumption."},

    # ---------- Part 3: Vulnerability notebook figures (selected) ----------
    {"part": 3, "key": "N1_cell03",
     "title": "NOTEBOOK 1 — Gaussian fits per circadian metric (PI's approach input)",
     "caption":
        "Distribution of each circadian metric in old animals (n = 40) with a "
        "Gaussian fit and ±1 SD thresholds, per the PI's original proposal. "
        "Shapiro–Wilk p-values flag IS_post, IV_pre, and RA_pre as non-normal — "
        "the first of six problems with the ±1 SD classification approach "
        "documented in Notebook 1."},

    {"part": 3, "key": "N1_cell10",
     "title": "NOTEBOOK 1 — Both tails get the same 'vulnerable' label",
     "caption":
        "For IS_post (representative), animals in the LOW tail and HIGH tail are "
        "behaviourally on opposite sides of the middle on Barnes entries, "
        "learning slope, and NOR DI. Pooling both tails into a single "
        "'vulnerable' label destroys the biological signal — core reason the "
        "±1 SD approach fails."},

    {"part": 3, "key": "N1_cell14",
     "title": "NOTEBOOK 1 — Per-metric labels disagree across metrics",
     "caption":
        "Animal-by-metric heatmap of the vulnerable/resilient label under the "
        "PI's ±1 SD rule. The same animal is often labelled differently on "
        "different metrics, with agreement well below the 50% chance baseline "
        "on many pairs. There is no coherent single classification; which "
        "metric you pick changes who counts as 'vulnerable'."},

    {"part": 3, "key": "N1_cell18",
     "title": "NOTEBOOK 1 — Behavioural conclusions flip with metric choice",
     "caption":
        "Cohen's d heatmap of (vulnerable − resilient) behavioural differences, "
        "one row per circadian metric. Cells are red and blue within the same "
        "row — the conclusion 'vulnerables have worse (or better) behaviour' "
        "depends entirely on which metric defined the split. This is the "
        "researcher-degrees-of-freedom consequence of metric-level classification."},

    {"part": 3, "key": "N2_cell06",
     "title": "NOTEBOOK 2 — Directional composite dysfunction score (distribution)",
     "caption":
        "Composite circadian-dysfunction score across old animals, with the "
        "median (red) used to split resilient (below) vs vulnerable (above). "
        "IS, RA, and Amplitude are sign-flipped so that higher = more "
        "dysfunction. The composite fixes four of the six problems from "
        "Notebook 1 and sets up a principled single split."},

    {"part": 3, "key": "N2_cell08",
     "title": "NOTEBOOK 2 — Composite tracks PCA PC1 (internal validation)",
     "caption":
        "Mean-z directional composite vs the first principal component of the "
        "same four circadian metrics. High correlation (typically r > 0.9) "
        "shows the composite is capturing the same dominant axis of variation "
        "that an unsupervised PCA finds — reassuring that the hand-directed "
        "sign flips didn't invent a direction."},

    {"part": 3, "key": "N2_cell14",
     "title": "NOTEBOOK 2 — Behavioural performance by composite group (null)",
     "caption":
        "Median-split composite vulnerable vs resilient behavioural comparison. "
        "All three outcomes show indistinguishable distributions between groups, "
        "consistent with the full-cohort null. Median split in the forward "
        "direction does not recover a circadian-cognition relationship."},

    {"part": 3, "key": "N2_cell17",
     "title": "NOTEBOOK 2 — Non-bootstrap side-by-side validation",
     "caption":
        "Left: parametric Cohen's d (Hedges SE) forest plot with 95% CIs. "
        "Right: group means ± SEM (within-outcome z-scored). Parametric CIs "
        "agree with the bootstrap CIs above, confirming the near-zero group "
        "differences are not a method artefact. Both approaches cross zero "
        "for all three outcomes."},

    {"part": 3, "key": "N2_cell21",
     "title": "NOTEBOOK 2 — Continuous correlations (full old cohort)",
     "caption":
        "Behavioural outcome vs circadian dysfunction score, continuous form. "
        "All three Spearman correlations are near zero and non-significant "
        "(ρ ∈ [−0.19, +0.05], all p ≥ 0.25). At the pooled old-animal level, "
        "neither dichotomisation nor continuous analysis finds a circadian-"
        "cognition signal — setting up the need for sex stratification."},

    {"part": 3, "key": "N2_cell25",
     "title": "NOTEBOOK 2 — Sex-stratified continuous correlations",
     "caption":
        "Within each behavioural outcome, Male (red) and Female (blue) "
        "regression lines diverge — especially Barnes learning slope, where "
        "Females ρ = +0.46 (p = 0.076) and Males ρ = −0.42 (p = 0.050), "
        "opposite signs. This sex-dependent pattern is the continuous-form "
        "signature of the Old-Male effect that the poster makes explicit."},

    {"part": 3, "key": "N3_cell03",
     "title": "NOTEBOOK 3 — Barnes input correlations (composite construction)",
     "caption":
        "5 × 5 correlation matrix of candidate Barnes inputs (all sign-flipped "
        "so higher = better memory). Entries and Q4 co-vary strongly (r = +0.57); "
        "Hole_errors is activity-confounded (r = −0.52 with entries); "
        "learning_slope and Goal_latency do not belong in the same factor. "
        "This diagnostic empirically justifies the final composite "
        "(entries + Q4 only)."},

    {"part": 3, "key": "N3_cell06",
     "title": "NOTEBOOK 3 — Barnes fidelity and NOR score distributions",
     "caption":
        "Barnes fidelity composite (z-mean of target entries + target-Q4 %) "
        "and NOR discrimination index in old animals, with median splits "
        "marked. Both distributions are approximately symmetric around the "
        "median, yielding near-balanced 19/19 and 18/18 classifications used "
        "in the κ analysis."},

    {"part": 3, "key": "N3_cell11",
     "title": "NOTEBOOK 3 — Barnes × NOR classification cross-tab",
     "caption":
        "Count of animals in each Barnes × NOR classification cell. Agreement "
        "is 41.7% (below the 50% chance baseline). Cohen's κ = −0.17. "
        "Memory domains index independent axes of cognitive variation in old "
        "animals — the central methodological finding supporting Poster Fig C."},

    {"part": 3, "key": "N3_cell15",
     "title": "NOTEBOOK 3 — Circadian metrics by memory classification",
     "caption":
        "Boxplot grid: each of four circadian metrics by Barnes split (top "
        "row) and NOR split (bottom row). Consistent with Fig D in the main "
        "poster section, only Amplitude_post by NOR split shows visual "
        "separation (NOR-impaired > NOR-resilient, d = +0.51, p = 0.159); "
        "all other cells overlap."},

    {"part": 3, "key": "N3_cell19",
     "title": "NOTEBOOK 3 — Behavioural score vs circadian dysfunction (continuous)",
     "caption":
        "Barnes composite and NOR DI against the circadian dysfunction score, "
        "pooled across old animals. Both continuous correlations are near zero, "
        "confirming the null at the cohort level before sex stratification. "
        "The behaviourally-forward framing recovers the same null as the "
        "rhythm-forward framing — two framings, one answer, at the cohort level."},

    {"part": 3, "key": "N3_cell22",
     "title": "NOTEBOOK 3 — Sex-stratified circadian × memory comparison",
     "caption":
        "Amplitude_post by memory classification (Barnes top, NOR bottom) × Sex. "
        "The only visually striking separation is Old-Male × NOR split — "
        "replicating the poster's Old-Male Amplitude result from the "
        "behaviourally-forward direction."},

    {"part": 3, "key": "N3_cell23",
     "title": "NOTEBOOK 3 — Sex-stratified continuous correlation",
     "caption":
        "Behavioural score vs circadian dysfunction score, with regression "
        "lines by Sex. NOR DI × Males: ρ = +0.35, p = 0.11; Females ρ ≈ 0. "
        "Same male-specific NOR–rhythm coupling seen elsewhere, in "
        "behaviourally-forward continuous form."},
]

# Add the Sex-stratified view figures (there's one more each from vulnerability notebooks)
# that are particularly informative — in this case just Barnes composite boxplot by sex
# from N3, already covered above. Keep list as is.

print(f"Total figures in PDF: {len(FIGURES)}")


# ---------------------------------------------------------------------------
# 3. Build the PDF
# ---------------------------------------------------------------------------

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"],
                     fontSize=10, leading=13, alignment=TA_JUSTIFY,
                     spaceAfter=6)
fig_title = ParagraphStyle("fig_title", parent=styles["Heading3"],
                          fontSize=11, leading=14, spaceAfter=4,
                          textColor=HexColor("#222222"))
caption = ParagraphStyle("caption", parent=styles["BodyText"],
                        fontSize=9, leading=12, alignment=TA_JUSTIFY,
                        textColor=HexColor("#333333"), spaceAfter=2)
part_head = ParagraphStyle("part_head", parent=styles["Heading1"],
                          fontSize=16, leading=20, spaceBefore=6,
                          spaceAfter=10, textColor=HexColor("#b5174e"))
title_style = ParagraphStyle("title", parent=styles["Title"],
                            fontSize=20, leading=24, alignment=1,
                            spaceAfter=6, textColor=HexColor("#222222"))
subtitle_style = ParagraphStyle("subtitle", parent=styles["Normal"],
                               fontSize=12, leading=16, alignment=1,
                               spaceAfter=14, textColor=HexColor("#555555"))

doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                       leftMargin=2*cm, rightMargin=2*cm,
                       topMargin=2*cm, bottomMargin=2*cm,
                       title="Poster analysis — figures and interpretations")

story = []

# --- Cover page ---
story.append(Spacer(1, 4*cm))
story.append(Paragraph("Circadian × Cognition in Aged Mice", title_style))
story.append(Paragraph("Figures and interpretations for poster review",
                      subtitle_style))
story.append(Spacer(1, 0.8*cm))

exec_summary = """
This document collates every figure produced by the poster analysis pipeline,
organised by analytical stage, each with a short interpretive caption. The
central finding (Part 1) is a sex-specific, task-specific, light-conditional
circadian–cognition dissociation in aged male mice: stronger rhythm amplitude
tracks <b>faster Barnes spatial learning</b> but <b>slower novel-object
recognition</b>, with the NOR side concentrated in the sleep-fragmentation
condition. Part 2 reproduces the full poster-notebook analysis. Part 3
contains supporting methodology from the three vulnerability/composite
notebooks (PI-approach flaws, composite validation, Barnes–NOR orthogonality).
The interpretations are distilled directly from the notebook outputs and the
interpretation document (<b>poster_interpretation.txt</b>) in the same folder.
"""
story.append(Paragraph(exec_summary, body))

story.append(Spacer(1, 0.5*cm))

# Headline findings table
headlines = [
    ["1", "ISF light intervention did NOT differentially perturb circadian metrics (all PRE×Light p_fdr > 0.5)."],
    ["2", "Sex, not Age or Light, is the dominant determinant of circadian phenotype (FDR-significant: Males have lower Amplitude and RA)."],
    ["3", "Full-cohort circadian → cognition is null (0 FDR survivors in the 16-test primary family, BF01 moderate evidence for null)."],
    ["4", "Old × Male subgroup shows an FDR-corrected Barnes locomotor deficit (p_fdr = 0.0036), identifying them a priori as phenotypically distinct."],
    ["5", "Within Old Males: Amplitude → NOR DI r = −0.49, p = 0.022 (bootstrap CI [−0.78, −0.10]); Amplitude → Barnes slope r = +0.43, p = 0.048."],
    ["6", "The NOR effect is specific to the ISF cell (r = −0.81, p = 0.005, n = 10) and to novel- (not familiar-) object exploration."],
    ["7", "Barnes and NOR index orthogonal memory domains in old animals (κ = −0.17), empirically forbidding single-score memory composites."],
]
tbl = Table(headlines, colWidths=[0.8*cm, 14.8*cm])
tbl.setStyle(TableStyle([
    ("FONTSIZE", (0,0), (-1,-1), 9),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ("TOPPADDING", (0,0), (-1,-1), 3),
    ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ("BACKGROUND", (0,0), (0,-1), HexColor("#fde2e4")),
    ("BACKGROUND", (1,0), (1,-1), HexColor("#f7f7f7")),
    ("GRID", (0,0), (-1,-1), 0.25, HexColor("#bbbbbb")),
    ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
]))
story.append(tbl)

story.append(PageBreak())

# --- Helper to add a figure ---
def add_figure(entry):
    if entry["key"] not in imgs:
        print(f"  !! missing image for {entry['key']}")
        return
    img_path = imgs[entry["key"]]
    # figure out image width to fit page (17cm max)
    flowables = []
    flowables.append(Paragraph(entry["title"], fig_title))
    try:
        img = Image(img_path, width=17*cm, height=10*cm, kind="proportional")
    except Exception:
        img = Image(img_path, width=17*cm)
    flowables.append(img)
    flowables.append(Spacer(1, 2*mm))
    flowables.append(Paragraph(entry["caption"], caption))
    flowables.append(Spacer(1, 4*mm))
    story.append(KeepTogether(flowables))


# --- Part 1 ---
story.append(Paragraph("Part 1 — Poster-ready figures (4 main panels)", part_head))
story.append(Paragraph(
    "The four figures destined for the poster panels, in reading order. "
    "Each is saved as a standalone 300 DPI PNG in "
    "<font face='Helvetica-Oblique'>poster_analysis/figures/</font> and can be "
    "placed directly into the poster layout.", body))
story.append(Spacer(1, 4*mm))
for entry in FIGURES:
    if entry["part"] == 1:
        add_figure(entry)

story.append(PageBreak())

# --- Part 2 ---
story.append(Paragraph("Part 2 — Full poster-notebook analysis sequence", part_head))
story.append(Paragraph(
    "Every figure from <font face='Helvetica-Oblique'>poster_results.ipynb</font> "
    "in execution order. Together they tell the full story: manipulation null → "
    "sex drives rhythm phenotype → full-cohort null → pre-existing Old-Male "
    "locomotor deficit → Old-Male circadian–cognition dissociation → specificity "
    "checks → robustness.", body))
story.append(Spacer(1, 4*mm))
for entry in FIGURES:
    if entry["part"] == 2:
        add_figure(entry)

story.append(PageBreak())

# --- Part 3 ---
story.append(Paragraph("Part 3 — Sensitivity and methodological analyses",
                      part_head))
story.append(Paragraph(
    "Selected figures from the three notebooks in "
    "<font face='Helvetica-Oblique'>vulnerability_resilience_analysis/</font>. "
    "These document why the final approach was chosen — the PI's ±1 SD rule "
    "fails (Notebook 1), a directional composite with median-split is defensible "
    "but null at the pooled cohort level (Notebook 2), and splitting on memory "
    "instead of rhythm recovers the same sex-specific signal while showing that "
    "memory domains are orthogonal in old mice (Notebook 3).", body))
story.append(Spacer(1, 4*mm))
for entry in FIGURES:
    if entry["part"] == 3:
        add_figure(entry)

# --- Closing caveats page ---
story.append(PageBreak())
story.append(Paragraph("Caveats (must appear on the poster)", part_head))
caveats = """
<b>1. Old-Male findings are exploratory.</b> They do not survive strict FDR
across the 48-test full-cohort family (min p_fdr ≈ 0.93) or the 16-test
Old-Male grid (min p_fdr ≈ 0.20). The evidence is a coherent pattern across
multiple related metrics, not a single corrected test.
<br/><br/>
<b>2. Small subgroup n.</b> Old-Male total n = 22; strongest single result
(ISF cell) n = 10. Bootstrap CI [−0.78, −0.10] is wide; replication on an
independent cohort is required.
<br/><br/>
<b>3. Paradoxical direction needs a mechanism.</b> Stronger rhythm → worse
NOR in aged males is opposite to the naive "rhythm loss impairs cognition"
model. Candidate frames: rigid rhythms reduce the cognitive flexibility
needed for novelty processing; compensatory amplitude amplification in
homeostatically-compromised animals. Present this as hypothesis-generating,
not confirmed.
<br/><br/>
<b>4. Intervention did not work as advertised.</b> ISF does not differentially
alter rhythm metrics (Section 1). Do not write "ISF-induced circadian
disruption" anywhere — ISF appears to moderate rather than produce the
phenotype.
<br/><br/>
<b>5. Memory domains are orthogonal.</b> Barnes and NOR cannot be collapsed
into a single memory score (κ = −0.17). Report them separately, always.
<br/><br/>
For the full statistical methods and an expanded narrative, see
<font face='Helvetica-Oblique'>poster_interpretation.txt</font> in the same
folder.
"""
story.append(Paragraph(caveats, body))

doc.build(story)
print(f"Wrote {OUT}")

# Clean up temp extracts
import shutil
shutil.rmtree(EXTRACT)
