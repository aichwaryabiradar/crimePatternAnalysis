// Builds the Word report from outputs/results.json, outputs/figures and report/screens.
// Usage (from project root): node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow,
  TableCell, WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, Footer, PageNumber,
} = require("docx");

const R = JSON.parse(fs.readFileSync("outputs/results.json", "utf8"));
const REPO = process.env.REPO_URL || "https://github.com/<your-username>/crime-hotspot";
const frame = (o) => o.index.map((ix, i) => Object.assign({ _index: ix },
  Object.fromEntries(o.columns.map((c, j) => [c, o.data[i][j]]))));
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const f = (x, d = 3) => Number(x).toFixed(d);
const pv = (p) => (p < 1e-4 ? p.toExponential(1) : Number(p).toFixed(4));
const num = (x) => Number(x).toLocaleString("en-IN");

const FONT = "Calibri";
function runs(text) {
  return String(text).split(/(\*\*[^*]+\*\*)/).filter(Boolean).map((s) => s.startsWith("**")
    ? new TextRun({ text: s.slice(2, -2), bold: true, font: FONT, size: 22 })
    : new TextRun({ text: s, font: FONT, size: 22 }));
}
const P = (t) => new Paragraph({ spacing: { after: 120, line: 300 }, alignment: AlignmentType.JUSTIFIED, children: runs(t) });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bul", level: 0 }, spacing: { after: 60, line: 280 }, children: runs(t) });
const numbered = (t, ref) => new Paragraph({ numbering: { reference: ref, level: 0 }, spacing: { after: 60, line: 280 }, children: runs(t) });
const formula = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
  children: [new TextRun({ text: t, font: "Cambria Math", size: 22, italics: true })] });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });
const box = (t) => new Paragraph({ spacing: { before: 120, after: 160 },
  shading: { fill: "EAF1F8", type: ShadingType.CLEAR, color: "auto" },
  border: { left: { style: BorderStyle.SINGLE, size: 24, color: "1F4E79", space: 8 } }, children: runs(t) });

let figNo = 0, tabNo = 0;
function figure(file, caption, widthIn = 6.0, maxH = 8.4) {
  const buf = fs.readFileSync(file);
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  let wp = widthIn * 96, hp = Math.round(wp * h / w);
  if (hp > maxH * 96) { hp = maxH * 96; wp = Math.round(hp * w / h); }
  figNo++;
  return [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: buf, transformation: { width: wp, height: hp },
      altText: { title: caption, description: caption, name: path.basename(file) } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [
    new TextRun({ text: `Figure ${figNo}: ${caption}`, italics: true, size: 19, font: FONT, color: "52514E" })] })];
}
const fig = (n, c, w) => figure(`outputs/figures/${n}`, c, w);
const bd = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
function table(header, rows, widths, caption) {
  const cell = (t, i, head) => new TableCell({ borders: { top: bd, bottom: bd, left: bd, right: bd },
    width: { size: widths[i], type: WidthType.DXA },
    shading: head ? { fill: "1F4E79", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: String(t), bold: head, color: head ? "FFFFFF" : "000000", size: 18, font: FONT })] })] });
  tabNo++;
  return [new Paragraph({ spacing: { before: 120, after: 60 }, keepNext: true,
    children: [new TextRun({ text: `Table ${tabNo}: ${caption}`, bold: true, size: 19, font: FONT })] }),
  new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) }))] }),
  new Paragraph({ spacing: { after: 100 }, children: [] })];
}

// ---------- numbers ----------
const D = R.data, PT = R.patterns, S = R.sampling, A = R.area, prof = A.profile;
const catF = frame(PT.category_freq), slotF = frame(PT.slot_freq), areaF = frame(PT.area_freq);
const cond = frame(PT["P(category|time_slot)"]), byDay = frame(PT.daily_by_weekday), bySlot = frame(PT.cells_by_slot);
const corr = PT.correlation.data.map((r) => ({ feature: r[0], r: r[1], p: r[2] }));
const bay = PT.bayes, daily = PT.daily;
const M = frame(R.comparison.table).map((r) => r); const pairs = frame(R.comparison.pairs);
const byName = Object.fromEntries(M.map((r) => [r.Model, r]));
const LR = byName["Logistic Regression"], DT = byName["Decision Tree"], RF = byName["Random Forest"];
const pRFDT = pairs.find((r) => r.Comparison === "Random Forest vs Decision Tree");
const pRFLR = pairs.find((r) => r.Comparison === "Random Forest vs Logistic Regression");
const pDTLR = pairs.find((r) => r.Comparison === "Decision Tree vs Logistic Regression");
const base = 1 - S.high_risk_test;
const topP = frame(A.top_periods);

const C = [];
// ================= TITLE =================
C.push(
  new Paragraph({ spacing: { before: 1700 }, alignment: AlignmentType.CENTER, children: [
    new TextRun({ text: "Statistical and Machine Learning-Based", bold: true, size: 40, font: FONT, color: "1F4E79" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [
    new TextRun({ text: "Crime Pattern and Hotspot Analysis", bold: true, size: 40, font: FONT, color: "1F4E79" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [
    new TextRun({ text: "Lab Assignment: Applying Probability and Statistics to Evaluate a Machine Learning Model", italics: true, size: 24, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1100 }, children: [
    new TextRun({ text: "Subject: Applied Probability and Statistics", size: 24, font: FONT })] }),
  ...[["Submitted by", "Aishwarya S Biradar"], ["Programme", "M.E. Big Data Analytics"],
    ["Institution", "Manipal School of Information Sciences, MAHE, Manipal"], ["Code repository", REPO],
    ["Tools", "Python, pandas, SciPy, scikit-learn, Streamlit"]].map(([k, v]) => new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [
      new TextRun({ text: k + ": ", bold: true, size: 22, font: FONT }), new TextRun({ text: v, size: 22, font: FONT })] })),
  pageBreak(),
);

// ================= SUMMARY =================
C.push(
  H1("Summary"),
  P(`This project analyses ${num(D.rows_clean)} Chicago Police Department crime incidents (${D.date_min.slice(0, 4)}–${D.date_max.slice(0, 4)}). Crime patterns are described with **frequency analysis, conditional probability, mean and variance, and correlation**. Three models are then trained to predict **crime hotspots**, that is, whether a place (one of Chicago's 77 community areas) at a given weekday and time slot is high risk: **Logistic Regression** (baseline), **Decision Tree** and **Random Forest**.`),
  P(`All three models were evaluated on the same held-out test set of ${num(S.n_test)} place-time cells, using **sampling, confidence intervals and hypothesis tests** to answer the lab question: does a model really perform better, or could the observed difference be due to variation in the test sample?`),
  box(`**Answer.** Decision Tree (${pct(DT.Accuracy)}) and Random Forest (${pct(RF.Accuracy)}) are clearly better than Logistic Regression (${pct(LR.Accuracy)}): McNemar p < 10⁻¹⁷, and the 95% CIs of the gains (about +12 to +20 points) are far from zero. Random Forest's ${pct(pRFDT["Accuracy difference"])} edge over the Decision Tree is **not** significant (p = ${f(pRFDT["McNemar p"], 2)}, 95% CI ${pct(pRFDT["CI low"])} to ${pct(pRFDT["CI high"])}). That difference could be due to test-sample variation.`),
  P("A Streamlit web application lets a user choose a place and see its crime-pattern statistics, the high-risk periods predicted by all three models, and the riskiest places at any day and time."),
  new Paragraph({ spacing: { before: 200 }, children: [new TextRun({ text: "Contents", bold: true, size: 26, font: FONT, color: "1F4E79" })] }),
  ...["1. Problem Statement and Dataset", "2. Sampling and Experimental Methodology",
    "3. Observations, Decision and Limitations", "4. Where Probability and Statistics Are Used",
    "5. Web Application", "6. Code Structure and How to Run", "References"].map((t) =>
    new Paragraph({ spacing: { after: 50 }, indent: { left: 360 }, children: [new TextRun({ text: t, size: 22, font: FONT })] })),
  pageBreak(),
);

// ================= 1 =================
C.push(
  H1("1. Problem Statement and Dataset"),
  H2("1.1 Problem statement"),
  P("Crime patterns vary with time, place and social factors, which makes crime analysis and prediction difficult. The objective is a data-driven framework that analyses crime data (type, time, location, frequency) to identify patterns and predict **high-risk periods and locations**."),
  P("The statistical question of the lab is: when one model scores higher than another on a test set, **is the improvement real, or could it be due to which cases happened to fall in the test sample?** A test set is a random sample, so each accuracy is only an estimate with sampling error."),
  H2("1.2 Dataset"),
  P(`City of Chicago Data Portal, "Crimes – 2001 to Present" (Chicago Police Department): a random sample of 25,000 incidents from ${D.date_min} to ${D.date_max}, as distributed in the open-source rcrimeanalysis package. Each row is one reported incident with its date and time, crime type, location type, community area, coordinates, and arrest and domestic flags.`),
  ...table(["Property", "Value"], [
    ["Incidents (raw → cleaned)", `${num(D.rows_raw)} → ${num(D.rows_clean)}`],
    ["Days covered", num(D.n_days)], ["Community areas (places)", D.n_areas],
    ["Crime types → categories", `${D.n_types} → 5 (property, violent, fraud, drugs/weapons, other)`],
  ], [3500, 5526], "Dataset overview"),
  H2("1.3 Prediction target: high-risk place-time cells"),
  P("Incidents are counted in every **cell = (community area × weekday × time slot)**, giving 77 × 7 × 4 = 2,156 cells. The time slots are night (0–6 h), morning (6–12), afternoon (12–18) and evening (18–24)."),
  P(`A cell is **high risk** if its incident count is at least ${f(S.threshold, 0)}, the 75th percentile of training-cell counts (the top 25%). Model features are the area's location (latitude and longitude of its centre), weekday, time slot and a weekend flag.`),
  pageBreak(),
);

// ================= 2 =================
C.push(
  H1("2. Sampling and Experimental Methodology"),
  H2("2.1 Crime-pattern statistics"),
  bullet("**Frequency analysis:** relative frequency as an estimate of probability, P(A) = n(A) / N."),
  bullet("**Conditional probability:** P(category | time slot) = n(category ∩ slot) / n(slot); one Bayes' theorem example."),
  bullet("**Mean and variance:** of daily incident counts, and of incidents per cell by time slot."),
  bullet("**Correlation:** Pearson r between each cell feature and the incident count, with the test H₀: ρ = 0."),
  H2("2.2 Sampling"),
  P(`The 2,156 cells are split at random into **70% training (${num(S.n_train)} cells)** and **30% held-out test (${num(S.n_test)} cells)** with a fixed seed (42). The high-risk threshold is computed from training cells only, so no test information leaks into training. The high-risk share is ${pct(S.high_risk_train)} in training and ${pct(S.high_risk_test)} in test, so the two samples are similar. All three models are trained on the same training cells and evaluated on the **same test cells**, which makes the comparison paired.`),
  H2("2.3 Models"),
  ...table(["Model", "Role", "How it works"], [
    ["Logistic Regression", "Baseline", "Linear model: log-odds of high risk = b₀ + b·x; features standardised"],
    ["Decision Tree", "Improved model 1", "If/else splits on the features (max depth 6, min 5 cells per leaf)"],
    ["Random Forest", "Improved model 2", "300 trees, each on a bootstrap sample; the vote share is P(high risk)"],
  ], [2300, 1800, 4926], "The three models"),
  H2("2.4 Statistical evaluation"),
  P("**(a) Estimation: accuracy with a 95% confidence interval.** Each test prediction is a Bernoulli trial (right or wrong), so the number correct K ~ Binomial(n, p) and accuracy p̂ = K/n. The Wilson score interval is used:"),
  formula("CI = [ p̂ + z²/2n ± z·√( p̂(1−p̂)/n + z²/4n² ) ] / (1 + z²/n),   z = 1.96"),
  P("Precision, recall, F1 and ROC-AUC are also reported, because high-risk cells are the minority class."),
  P("**(b) Hypothesis test: McNemar's test.** For each pair of models, H₀: both have the same accuracy; H₁: they differ. Because both models predict the same cells, only the cells where exactly one model is right matter: b = A right and B wrong, c = A wrong and B right. Under H₀, c ~ Binomial(b + c, ½); the exact binomial p-value is used. With three pairwise tests, each is judged at α/3 = 0.0167 (Bonferroni correction)."),
  P("**(c) Estimation: bootstrap 95% CI of the accuracy difference.** The test cells are resampled with replacement 2,000 times, keeping each cell's predictions from both models together, and the accuracy difference is recomputed each time. The 2.5th and 97.5th percentiles give the 95% CI. If the interval contains 0, the difference could be due to test-sample variation."),
  pageBreak(),
);

// ================= 3 =================
C.push(
  H1("3. Observations, Decision and Limitations"),
  H2("3.1 Crime patterns"),
  P(`**Frequency.** PROPERTY (${pct(catF[0].probability)}) and VIOLENT (${pct(catF[1].probability)}) crimes dominate. Incidents peak in the afternoon (${pct(slotF[2].probability)}) and evening (${pct(slotF[3].probability)}), against ${pct(slotF[0].probability)} at night. The busiest areas are ${areaF.slice(0, 3).map((r) => `${r._index} (${pct(r.probability)})`).join(", ")}.`),
  ...fig("01_frequency.png", "Relative frequency of crime categories and time slots", 6.2),
  ...table(["Time slot", "PROPERTY", "VIOLENT", "FRAUD", "DRUGS_WEAPONS", "OTHER"],
    cond.map((r) => [r._index, f(r.PROPERTY, 3), f(r.VIOLENT, 3), f(r.FRAUD, 3), f(r.DRUGS_WEAPONS, 3), f(r.OTHER, 3)]),
    [2026, 1400, 1400, 1400, 1400, 1400], "Conditional probability P(category | time slot)"),
  P(`**Conditional probability.** The crime mix depends on the time of day: P(VIOLENT | night) = ${f(cond[0].VIOLENT)} against P(VIOLENT | morning) = ${f(cond[1].VIOLENT)}, and fraud is rare in the evening (${f(cond[3].FRAUD)}). **Bayes' theorem:** P(VIOLENT | DOMESTIC) = P(DOMESTIC | VIOLENT)·P(VIOLENT)/P(DOMESTIC) = ${f(bay["P(DOMESTIC|VIOLENT)"])} × ${f(bay["P(VIOLENT)"])} / ${f(bay["P(DOMESTIC)"])} = ${f(bay["P(VIOLENT|DOMESTIC) Bayes"])}, matching the direct count.`),
  P(`**Mean and variance.** The city records a mean of ${f(daily.mean, 2)} incidents per day in the sample (variance ${f(daily.variance, 2)}, SD ${f(daily.std, 2)}), with similar weekday means (${f(Math.min(...byDay.map((r) => r.mean)), 1)}–${f(Math.max(...byDay.map((r) => r.mean)), 1)}). At the cell level, both the mean **and the variance** rise strongly from night to afternoon:`),
  ...table(["Time slot", "Cells", "Mean incidents", "Variance", "SD"], bySlot.map((r) => [r._index, r.n, f(r.mean, 2), f(r.variance, 1), f(r.std, 2)]),
    [2600, 1300, 1700, 1700, 1726], "Mean and variance of incidents per place-time cell"),
  P("The large variance means a few places carry far more incidents than others: these are the hotspots."),
  ...fig("03_mean_variance.png", "Mean ± standard deviation of incidents per cell by time slot", 5.4),
  P(`**Correlation.** ${corr.map((c) => `${c.feature}: r = ${f(c.r)} (p = ${pv(c.p)})`).join("; ")}. The time slot has the strongest linear relationship with the count. Latitude and longitude are only weakly correlated even though location matters a lot, because risk varies with location in a **non-linear** way (hotspots are scattered). A linear model therefore cannot capture location well, and the model results confirm this.`),
  ...fig("04_correlation.png", "Pearson correlation matrix of cell features and incident count", 4.2),

  H2("3.2 Model comparison on the held-out test set"),
  ...table(["Model", "Accuracy", "95% CI", "Precision", "Recall", "F1", "ROC-AUC"], M.map((r) => [r.Model, pct(r.Accuracy),
    `${pct(r["CI low"])} – ${pct(r["CI high"])}`, f(r.Precision), f(r.Recall), f(r.F1), f(r["ROC-AUC"])]),
    [2200, 1100, 1700, 1050, 1000, 900, 1076], `Performance on the same ${num(S.n_test)} test cells`),
  ...fig("05_accuracy_ci.png", "Test accuracy with 95% confidence intervals (dotted line = always predicting 'low risk')", 5.6),
  P(`Logistic Regression (${pct(LR.Accuracy)}) does **no better than always predicting "low risk"** (${pct(base)}): its recall is only ${f(LR.Recall)}, so it almost never flags a hotspot. Both tree models find about ${pct(DT.Recall, 0)} of hotspots with high precision. The confidence intervals of the tree models do not overlap with the baseline's, but they overlap heavily with each other.`),
  ...fig("06_confusion.png", "Confusion matrices on the test set", 6.2),
  ...table(["Comparison", "Accuracy diff.", "Bootstrap 95% CI", "b", "c", "McNemar p", "Significant at α/3?"],
    pairs.map((r) => [r.Comparison, (r["Accuracy difference"] >= 0 ? "+" : "") + pct(r["Accuracy difference"]),
      `${pct(r["CI low"])} to ${pct(r["CI high"])}`, r["b (A right, B wrong)"], r["c (A wrong, B right)"], pv(r["McNemar p"]),
      r["Significant (α/3)"] ? "Yes" : "No"]), [2726, 1150, 1700, 600, 600, 1100, 1150], "Pairwise hypothesis tests (paired, same test cells)"),
  ...fig("07_bootstrap.png", "Bootstrap distributions of the accuracy difference with the 95% CI (shaded) and the no-difference line", 6.4),
  P(`**Interpretation.** For the Decision Tree vs Logistic Regression, the models disagree on ${pDTLR["b (A right, B wrong)"] + pDTLR["c (A wrong, B right)"]} test cells, and the tree is right on ${pDTLR["c (A wrong, B right)"]} of them. Under H₀ such a lopsided split is essentially impossible (p = ${pv(pDTLR["McNemar p"])}), and the bootstrap CI (${pct(pDTLR["CI low"])} to ${pct(pDTLR["CI high"])}) is far from 0. The same holds for Random Forest vs Logistic Regression (p = ${pv(pRFLR["McNemar p"])}). For Random Forest vs Decision Tree, the split is only ${pRFDT["c (A wrong, B right)"]} vs ${pRFDT["b (A right, B wrong)"]} (p = ${f(pRFDT["McNemar p"], 3)}), and the CI includes 0.`),

  H2("3.3 Hotspot prediction for a chosen place"),
  P(`For a chosen place, here **${A.name}**, the application shows its statistics and every model's predicted risk for all 28 weekday × time-slot periods. ${A.name} recorded ${num(prof.n)} incidents (${pct(prof.share)} of the city; rank #${prof.rank} of ${prof.n_areas}), a mean of ${f(prof.daily.mean, 2)} incidents per day (variance ${f(prof.daily.variance, 2)}), and an arrest probability of ${pct(prof.arrest_rate)} against ${pct(prof.city_arrest_rate)} citywide. Agreement of each model with the observed high-risk periods: ${Object.entries(A.agreement).map(([k, v]) => `${k} ${pct(v, 0)}`).join(", ")}. The riskiest predicted periods (Random Forest) are ${topP.slice(0, 3).map((r) => `${r.Day} ${r["Time slot"]} (${f(r["P(high risk)"], 2)})`).join(", ")}.`),
  ...fig("08_area_example.png", `${A.name}: observed high-risk periods and the predicted P(high risk) of each model`, 6.4),

  H2("3.4 Decision"),
  box(`**Is the improvement real?** Yes for both tree models over the baseline: Decision Tree +${pct(pDTLR["Accuracy difference"])} and Random Forest +${pct(pRFLR["Accuracy difference"])}, with McNemar p < 10⁻¹⁷ and bootstrap CIs well above 0. These gains cannot be explained by test-sample variation; location has a non-linear effect that only the trees capture. **Between the two tree models**, the +${pct(pRFDT["Accuracy difference"])} difference is not significant (p = ${f(pRFDT["McNemar p"], 2)}; CI ${pct(pRFDT["CI low"])} to ${pct(pRFDT["CI high"])}), so it **could be due to test-sample variation**. **Recommendation:** use a tree model. Random Forest gives the highest accuracy and ROC-AUC and smoother risk probabilities, but the simpler Decision Tree is statistically just as accurate on this test set.`),
  H2("3.5 Limitations"),
  bullet("**Sample of 25,000 incidents.** Cell counts are small, so the high-risk label is noisy; the full Chicago data would give more reliable hotspots."),
  bullet("**Small test set (647 cells).** A real 1-point difference between two good models cannot be detected with this many cells."),
  bullet("**Random split, not over time.** Training and test cells come from the same years; a forward-in-time split would better test forecasting."),
  bullet("**Independence assumption.** McNemar and the bootstrap treat cells as independent, but neighbouring areas and adjacent time slots are related."),
  bullet("**Recorded crime ≠ actual crime.** Police data reflect reporting and patrol patterns."),
  pageBreak(),
);

// ================= 4 =================
C.push(
  H1("4. Where Probability and Statistics Are Used"),
  ...table(["Stage", "Concept", "Where in this project"], [
    ["Pattern analysis", "Frequency, relative frequency as probability", "P(category), P(time slot), busiest areas (3.1)"],
    ["Pattern analysis", "Conditional probability, Bayes' theorem", "P(category | time slot); P(VIOLENT | DOMESTIC); P(category | place) in the app"],
    ["Pattern analysis", "Mean, variance, standard deviation", "Daily counts; incidents per cell by time slot; per place in the app"],
    ["Pattern analysis", "Pearson correlation and test of H₀: ρ = 0", "Which features relate to the incident count"],
    ["Labelling", "Percentile (75th quantile)", "Definition of a high-risk cell"],
    ["Sampling", "Random sampling, train/test split", "70/30 split of cells; same test set for all models"],
    ["Modelling", "Probability output P(high risk)", "Logistic (sigmoid), tree leaf frequencies, forest vote share; risk maps"],
    ["Modelling", "Bootstrap sampling", "Each Random Forest tree is trained on a bootstrap sample"],
    ["Estimation", "Binomial model of accuracy; Wilson 95% CI", "Accuracy of each model (3.2)"],
    ["Hypothesis testing", "McNemar's exact test (Binomial), α, p-value", "Pairwise model comparison (3.2)"],
    ["Hypothesis testing", "Bonferroni correction α/3", "Three comparisons at once"],
    ["Estimation", "Bootstrap 95% CI of a difference", "Could the accuracy difference be zero? (3.2)"],
  ], [1900, 3200, 3926], "Probability and statistics used in the project"),
  pageBreak(),
);

// ================= 5 =================
const scr = (n, c) => figure(`report/screens/${n}`, c, 6.3, 8.6);
C.push(
  H1("5. Web Application"),
  P("The Streamlit application (streamlit run app.py) has three pages, selected in the sidebar."),
  bullet("**Hotspot predictor:** the user chooses a place (any of the 77 community areas; type to search) and sees its crime-pattern analysis (incidents, rank, mean and variance per day, arrest probability, P(category | place) vs city, observed counts by weekday and time slot). It then shows the predicted P(high risk) for every period from all three models, the riskiest periods, and a check for a chosen day and time slot. A map and table rank the riskiest places at that time."),
  bullet("**Crime patterns:** frequency analysis, conditional probability, mean and variance, correlation."),
  bullet("**Model comparison:** sampling, accuracy with 95% CIs, confusion matrices, McNemar tests, bootstrap CIs and the conclusion."),
  ...scr("app_predictor_1.png", "Hotspot predictor: crime pattern of the chosen place and the three models' predicted high-risk periods"),
  ...scr("app_predictor_2.png", "Hotspot predictor: prediction for a chosen time and the riskiest places at that time"),
  ...scr("app_patterns.png", "Crime patterns page"),
  ...scr("app_comparison.png", "Model comparison page"),
);

// ================= 6 =================
C.push(
  H1("6. Code Structure and How to Run"),
  ...table(["File", "Purpose"], [
    ["app.py", "Web application: loads data, trains the models once (cached), page menu"],
    ["main.py", "Runs the whole analysis without the website; saves tables, figures, results.json"],
    ["src/config.py", "Settings, community-area names, crime categories"],
    ["src/data.py", "Load and clean incidents; build the place-time cells"],
    ["src/statistics.py", "Frequency, conditional probability, mean/variance, correlation"],
    ["src/models.py", "Train/test split, high-risk label, the three models"],
    ["src/evaluation.py", "Accuracy + Wilson CI, McNemar test, bootstrap CI"],
    ["src/hotspot.py", "Place profile, risk grid for a place, riskiest places"],
    ["src/figures.py", "Figures for this report"],
    ["views/predictor.py, patterns.py, comparison.py", "The three web pages"],
    ["views/charts.py", "Chart helpers"],
    ["tests/test_statistics.py", "Unit tests of the statistical functions"],
  ], [3600, 5426], "Project files"),
  bullet("python -m venv .venv, then .venv\\Scripts\\activate"),
  bullet("pip install -r requirements.txt"),
  bullet("streamlit run app.py   (web application)"),
  bullet("python main.py   (outputs folder: tables and figures)"),
  bullet("python -m pytest -q   (tests)"),
  P(`Repository: ${REPO}`),
  H1("References"),
  numbered("City of Chicago. Crimes – 2001 to Present. Chicago Data Portal. https://data.cityofchicago.org", "ref"),
  numbered("Spaulding, J. S. rcrimeanalysis: An Implementation of Crime Analysis Methods (R package).", "ref"),
  numbered("McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2), 153–157.", "ref"),
  numbered("Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895–1923.", "ref"),
  numbered("Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall.", "ref"),
  numbered("Montgomery, D. C., & Runger, G. C. (2014). Applied Statistics and Probability for Engineers (6th ed.). Wiley.", "ref"),
);

const nCfg = (ref, fmt) => ({ reference: ref, levels: [{ level: 0, format: fmt,
  text: fmt === LevelFormat.BULLET ? "•" : (ref === "ref" ? "[%1]" : "%1."), alignment: AlignmentType.LEFT,
  style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] });
const doc = new Document({
  creator: "Aishwarya S Biradar", title: "Crime Pattern and Hotspot Analysis",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: FONT, color: "1F4E79" }, paragraph: { spacing: { before: 300, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: "2E75B6" }, paragraph: { spacing: { before: 220, after: 100 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [nCfg("bul", LevelFormat.BULLET), nCfg("ref", LevelFormat.DECIMAL)] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "7F7F7F" })] })] }) },
    children: C,
  }],
});
Packer.toBuffer(doc).then((b) => { fs.writeFileSync("report/Crime_Hotspot_Report.docx", b); console.log("wrote report"); });
