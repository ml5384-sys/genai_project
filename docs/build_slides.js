// Builds docs/JobFit_Presentation.pptx   (node docs/build_slides.js)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const R = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, "eval/results", f), "utf8"));
const rank = R("ranking.json"), para = R("paraphrase.json"), cf = R("counterfactual.json"), st = R("stability.json");
const M = rank.methods;
const mini = para.models["sentence-transformers/all-MiniLM-L6-v2"]["with descriptor"];
const f3 = (x) => x.toFixed(3), pct = (x) => `${Math.round(x * 100)}%`;
const FIG = (f) => path.join(ROOT, "docs/figures", f);

const THEME = {
  name: "JobFit", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1B1F2A", lt1: "FFFFFF", dk2: "1F2A44", lt2: "EEF1F6",
    accent1: "1B998B", accent2: "E8A33D", accent3: "2F5D8A", accent4: "C8553D", accent5: "8A94A6", accent6: "5B6475",
    hlink: "2F5D8A", folHlink: "5B6475" },
};
const HEX = THEME.colors;
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "JobFit — Final Presentation";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "DARK", background: { color: HEX.dk2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 8.8, h: 1.2, fontSize: 44, bold: true, align: "left",
    color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", margin: 0 }, text: "" } }],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: HEX.lt1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 26, bold: true, align: "left",
    color: C.text2, fontFace: THEME.headFontFace, valign: "middle", margin: 0 }, text: "" } }],
  slideNumber: { x: 9.2, y: 5.25, w: 0.5, h: 0.25, fontSize: 10, color: HEX.accent5, align: "right" },
});

const T = (s, text, o) => s.addText(text, { isTextBox: true, fontSize: 14, color: C.text1, margin: 0, valign: "top", ...o });
const card = (s, x, y, w, h, fill = HEX.lt2) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
const dot = (s, x, y, n, color = HEX.accent1) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color }, line: { color } });
  T(s, String(n), { x, y, w: 0.42, h: 0.42, fontSize: 15, bold: true, color: C.background1, align: "center", valign: "middle" });
};
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.accent6, valAxisLabelColor: HEX.accent6, dataLabelColor: HEX.dk1, catAxisLabelFontSize: 11,
  valAxisLabelFontSize: 10, dataLabelFontSize: 11 };

// 1 ── Title
pres.addSection({ title: "Intro" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Intro" });
s.addText("JobFit", { placeholder: "title" });
T(s, "Which entry-level jobs should I apply to?", { x: 0.6, y: 2.85, w: 8.8, h: 0.5, fontSize: 22, color: HEX.accent2, fontFace: "Cambria", italic: true });
T(s, "Transparent, repeatable requirement-coverage ranking of 8,000 LinkedIn postings", { x: 0.6, y: 3.4, w: 8.8, h: 0.4, fontSize: 16, color: "C9D2E3" });
T(s, "[Group members]  ·  Generative AI final project  ·  November 2026", { x: 0.6, y: 4.7, w: 8.8, h: 0.35, fontSize: 12, color: "9AA6BD" });
s.addNotes("Hi, we're presenting JobFit. The one-line pitch: you upload your resume, and JobFit ranks thousands of open entry-level postings by how well your resume covers what each posting actually asks for — with a score you can trust and evidence you can check.");

// 2 ── Problem
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
s.addText("Mass-applying wastes time; chatbots don't fix it", { placeholder: "title" });
T(s, "Students send one generic resume everywhere and get no feedback. The common shortcut — paste resume + JD into ChatGPT — has three gaps:",
  { x: 0.5, y: 1.15, w: 9, h: 0.6, fontSize: 15, color: HEX.accent6 });
const gaps = [["One at a time", "A chat handles one posting; there are thousands of open roles"],
  ["Not repeatable", "Ask twice, get two different scores"],
  ["Opaque", "No evidence for what the number is based on"]];
gaps.forEach(([h, b], i) => {
  const x = 0.5 + i * 3.05;
  card(s, x, 2.0, 2.85, 2.6);
  dot(s, x + 0.25, 2.25, i + 1, HEX.accent4);
  T(s, h, { x: x + 0.25, y: 2.85, w: 2.4, h: 0.45, fontSize: 19, bold: true, color: C.text2, fontFace: "Cambria" });
  T(s, b, { x: x + 0.25, y: 3.35, w: 2.4, h: 1.1, fontSize: 14, color: C.text1 });
});
T(s, "JobFit answers one question well: given my resume, which postings are a realistic fit?", { x: 0.5, y: 4.8, w: 9, h: 0.35, fontSize: 14, bold: true, color: HEX.accent1 });
s.addNotes("The entry-level market is competitive, so most of us mass-apply. Pasting a resume and a JD into ChatGPT is the common shortcut, but it has three structural problems: it works one posting at a time, its scores change between runs, and it doesn't show its evidence. JobFit is designed around exactly these three gaps. Important: our score measures requirement coverage, not the probability of getting an interview.");

// 3 ── Data
pres.addSection({ title: "Data & method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data & method" });
s.addText("Data: 8,000 entry-level postings, 586 resumes", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "Postings", labels: ["LinkedIn 2023–24", "Entry / Assoc. / Intern", "FIN·ACCT·IT·CNSL·ANLS", "Sampled database"],
  values: [123849, 47983, 12065, 8000] }], {
  x: 0.4, y: 1.2, w: 5.4, h: 3.6, barDir: "bar", catAxisOrientation: "maxMin", chartColors: [HEX.accent5, HEX.accent5, HEX.accent5, HEX.accent1],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0", valAxisHidden: true, valGridLine: { style: "none" },
  catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 45, showTitle: true, title: "Job database funnel",
  titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: HEX.dk2, ...chartText,
});
card(s, 6.1, 1.2, 3.4, 3.6);
T(s, [
  { text: "Jobs  ", options: { bold: true, color: HEX.dk2 } }, { text: "Kaggle LinkedIn Job Postings. Kept all non-IT postings, filled with IT (seed 42): 5,079 entry · 2,596 associate · 325 intern.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "Resumes  ", options: { bold: true, color: HEX.dk2 } }, { text: "Kaggle Resume Dataset, 5 categories: Accountant, Finance, Banking, IT, Consultant — evaluation only, nothing trained on them.", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 8 } },
  { text: "Pre-built index ships in the repo (19 MB)", options: { italic: true, color: HEX.accent6 } },
], { x: 6.35, y: 1.45, w: 2.95, h: 3.2, fontSize: 13 });
s.addNotes("Jobs come from the Kaggle LinkedIn postings dataset. We filtered 124k postings to 48k entry, associate and internship roles, then to Finance, Accounting, IT, Consulting and Analyst functions, and sampled 8,000, keeping every non-IT posting so IT doesn't dominate. Resumes come from the Kaggle resume dataset — 586 in our five categories — used only for evaluation.");

// 4 ── Architecture
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data & method" });
s.addText("Pipeline: extract once, match per request", { placeholder: "title" });
const steps = [
  ["Structured extraction", "LLM with a fixed JSON schema (T=0, cached) — or offline rule-based taxonomy of 115 skills"],
  ["Semantic matching", "Exact skill hit = 1; else calibrated sentence-embedding similarity to resume bullets"],
  ["Transparent score", "Explicit formula over required / preferred coverage, experience, education, role"],
  ["Rank + explain", "Vectorised over all 8,000 postings; explanation built only from the match output"],
];
steps.forEach(([h, b], i) => {
  const x = 0.5 + i * 2.3;
  card(s, x, 1.3, 2.1, 2.75, i % 2 ? HEX.lt2 : "E4F3F1");
  dot(s, x + 0.2, 1.5, i + 1);
  T(s, h, { x: x + 0.2, y: 2.05, w: 1.75, h: 0.6, fontSize: 16, bold: true, color: C.text2, fontFace: "Cambria" });
  T(s, b, { x: x + 0.2, y: 2.7, w: 1.75, h: 1.3, fontSize: 12, color: C.text1 });
  if (i < 3) T(s, "›", { x: x + 2.1, y: 2.45, w: 0.2, h: 0.4, fontSize: 22, bold: true, color: HEX.accent5, align: "center", valign: "middle" });
});
T(s, [{ text: "Served by FastAPI ", options: { bold: true } }, { text: "(/rank, /rank/upload, /match, /extract, /jobs) in a Docker image that runs fully offline; Gemini key optional." }],
  { x: 0.5, y: 4.3, w: 9, h: 0.5, fontSize: 14, color: HEX.accent6 });
s.addNotes("Four stages. One: structured extraction — every JD becomes a fixed schema of requirements, each marked required or preferred, plus minimum years and degree. We use an LLM with a JSON schema at temperature zero and cache it, so each JD is extracted once; there's also a rule-based extractor so everything runs offline. Two: each requirement is linked to the closest resume evidence. Three: an explicit formula turns that into a score. Four: we rank all postings and explain the top ones.");

// 5 ── Matching detail
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Data & method" });
s.addText("Matching sees past wording; the score is a formula", { placeholder: "title" });
card(s, 0.5, 1.2, 4.3, 3.65, "E4F3F1");
T(s, "Requirement", { x: 0.75, y: 1.4, w: 3.8, h: 0.3, fontSize: 12, color: HEX.accent6 });
T(s, "Power BI", { x: 0.75, y: 1.7, w: 3.8, h: 0.4, fontSize: 20, bold: true, color: C.text2, fontFace: "Cambria" });
T(s, "Resume evidence", { x: 0.75, y: 2.25, w: 3.8, h: 0.3, fontSize: 12, color: HEX.accent6 });
T(s, "“Built interactive Tableau dashboards to track portfolio performance”", { x: 0.75, y: 2.55, w: 3.8, h: 0.65, fontSize: 14, italic: true });
T(s, [
  { text: "cos 0.15 → 0.55", options: { bold: true, color: HEX.accent1, breakLine: true } },
  { text: "when the requirement is embedded with a descriptor: “Power BI (business intelligence dashboards…)”. Calibration AUC 0.920 → 0.947." },
], { x: 0.75, y: 3.35, w: 3.8, h: 1.35, fontSize: 13 });
T(s, [
  { text: "overall = 100 · Σ W·sub / Σ W", options: { bold: true, fontFace: "Cambria", fontSize: 17, color: C.text2, breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "Required coverage 0.45 · Role alignment 0.20 · Experience 0.15 · Preferred 0.10 · Education 0.10", options: { breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "Related evidence worth ≤ 85% of direct evidence", options: { bullet: true, breakLine: true } },
  { text: "Certifications (CPA, CFA, Series 7) are exact-only", options: { bullet: true, breakLine: true } },
  { text: "Soft skills count half; smoothing stops 1-requirement postings scoring 100", options: { bullet: true, breakLine: true } },
  { text: "Every requirement shows status, method and the quoted resume line", options: { bullet: true } },
], { x: 5.1, y: 1.3, w: 4.4, h: 3.5, fontSize: 13, paraSpaceAfter: 4 });
s.addNotes("This is the semantic part. A keyword matcher would say this resume has no Power BI. We embed requirements and resume bullets with a sentence-embedding model — an upgrade of the word vectors from Module 2. A trick that mattered: a bare tool name means little to the encoder, so we embed each requirement with a short descriptor; similarity jumps from 0.15 to 0.55. The score itself is an explicit weighted formula, not a model, so every number is traceable. Related evidence gets at most 85 percent credit, and licences like CPA must match exactly.");

// 6 ── Demo
pres.addSection({ title: "Demo" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Demo" });
s.addText("Demo: ranked postings with evidence", { placeholder: "title" });
s.addImage({ path: FIG("demo_results.png"), x: 0.5, y: 1.15, w: 5.6, h: 4.2, sizing: { type: "cover", w: 5.6, h: 4.2 },
  altText: "JobFit web demo showing ranked postings with scores, sub-score bars and evidence table" });
T(s, [
  { text: `${Math.round(st.latency_ms_p50)} ms`, options: { fontSize: 40, bold: true, color: HEX.accent1, fontFace: "Cambria", breakLine: true } },
  { text: "median to score all 8,000 postings (CPU)", options: { fontSize: 13, color: HEX.accent6, breakLine: true } },
  { text: " ", options: { fontSize: 10, breakLine: true } },
  { text: "Upload PDF / DOCX / TXT or paste text", options: { bullet: true, breakLine: true } },
  { text: "Filter by level, function, location", options: { bullet: true, breakLine: true } },
  { text: "Green = matched, amber = related, red = missing", options: { bullet: true, breakLine: true } },
  { text: "docker compose up → localhost:8000", options: { bullet: true } },
], { x: 6.4, y: 1.3, w: 3.1, h: 3.9, fontSize: 13, paraSpaceAfter: 4 });
s.addNotes("[Switch to live demo or narrate the screenshot.] Here our sample student resume is ranked against all 8,000 postings in about 150 milliseconds. Each card shows the score, the sub-scores, an explanation, and every requirement colour-coded — click the evidence table and you see the exact resume line behind each match. Everything here runs in one Docker container.");

// 7 ── Ranking results
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Ranking: beats keyword overlap, not doc-embeddings", { placeholder: "title" });
const order = ["Random", "Skill Jaccard (exact)", "TF-IDF (keyword)", "JobFit (full)", "Embedding doc-cosine", "Hybrid: JobFit + doc-cosine (z-avg)"];
const lbl = { "Hybrid: JobFit + doc-cosine (z-avg)": "JobFit + doc-cosine", "Skill Jaccard (exact)": "Exact skill overlap" };
s.addChart(pres.charts.BAR, [{ name: "nDCG@10", labels: order.map((n) => lbl[n] || n), values: order.map((n) => +M[n]["nDCG@10"].toFixed(3)) }], {
  x: 0.4, y: 1.15, w: 5.7, h: 3.75, barDir: "bar", catAxisOrientation: "maxMin",
  chartColors: order.map((n) => (n.startsWith("JobFit") ? HEX.accent1 : n.startsWith("Hybrid") ? HEX.accent3 : HEX.accent5)),
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", valAxisMinVal: 0, valAxisMaxVal: 1,
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, barGapWidthPct: 40,
  showTitle: true, title: `nDCG@10 · ${rank.n_resumes} resumes × ${rank.n_jobs.toLocaleString()} postings`, titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: HEX.dk2, ...chartText,
});
card(s, 6.35, 1.2, 3.15, 3.7);
T(s, [
  { text: "Weak labels", options: { bold: true, color: HEX.dk2, breakLine: true } },
  { text: "Relevant = posting function matches resume category. They reward topic, not fit.", options: { breakLine: true } },
  { text: " ", options: { fontSize: 6, breakLine: true } },
  { text: "Takeaways", options: { bold: true, color: HEX.dk2, breakLine: true } },
  { text: `+0.03 over exact skill overlap; ≈ TF-IDF`, options: { bullet: true, breakLine: true } },
  { text: "Doc-embedding is a strong topic ranker — we report it plainly", options: { bullet: true, breakLine: true } },
  { text: `Role alignment matters most (−0.109 if removed)`, options: { bullet: true } },
], { x: 6.55, y: 1.4, w: 2.8, h: 3.4, fontSize: 12.5, paraSpaceAfter: 3 });
s.addNotes(`We scored all 8,000 postings for each of 586 resumes. With no public ground truth, we used weak labels: a posting is relevant if its LinkedIn function matches the resume's category. JobFit gets nDCG@10 of ${f3(M["JobFit (full)"]["nDCG@10"])}, beating exact skill overlap and matching TF-IDF — but plain whole-document embedding similarity scores ${f3(M["Embedding doc-cosine"]["nDCG@10"])}. We want to be upfront: these labels measure topic, and a document embedding is a great topic ranker. Doc-cosine can't tell you what's missing though. Combining the two gets ${f3(M["Hybrid: JobFit + doc-cosine (z-avg)"]["nDCG@10"])} while keeping our evidence — that's our recommended next step. In ablations, role alignment was the biggest contributor.`);

// 8 ── Matching quality
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("The matcher behaves as a transparent score should", { placeholder: "title" });
const stats = [
  [`${pct(para.keyword.recall)} → ${pct(mini.combined_recall)}`, "recall on paraphrased evidence (keywords → JobFit), 0% false positives", `${para.n_cases} hand-written cases`],
  [`${pct(cf.remove_matched_skill.pct_decreased)}`, "of scores drop when evidence for a matched required skill is deleted", `n = ${cf.remove_matched_skill.n}, mean ${cf.remove_matched_skill.mean_delta.toFixed(1)} pts`],
  [`${pct(st.identical_rankings_pct)}`, "identical rankings on repeated runs — zero score variance", `${st.n_resumes} resumes × 3 runs`],
];
stats.forEach(([big, lab, sub], i) => {
  const x = 0.5 + i * 3.05;
  card(s, x, 1.3, 2.85, 3.3);
  T(s, big, { x: x + 0.25, y: 1.55, w: 2.4, h: 0.9, fontSize: big.length > 6 ? 30 : 44, bold: true, color: HEX.accent1, fontFace: "Cambria" });
  T(s, lab, { x: x + 0.25, y: 2.55, w: 2.4, h: 1.2, fontSize: 14 });
  T(s, sub, { x: x + 0.25, y: 3.95, w: 2.4, h: 0.4, fontSize: 11, color: HEX.accent6, italic: true });
});
T(s, `Adding a missing required skill raised the score in ${pct(cf.add_missing_skill.pct_increased)} of ${cf.add_missing_skill.n} cases.`, { x: 0.5, y: 4.8, w: 9, h: 0.35, fontSize: 13, color: HEX.accent6 });
s.addNotes("Because the category labels can't test requirement matching, we built targeted tests. A paraphrase benchmark: 42 requirement/evidence pairs written without shared keywords, each with a hard negative. Keywords find 38 percent; JobFit finds 69 with zero false positives. Counterfactuals: delete the evidence for a matched required skill and the score drops every time; add a missing skill and it rises every time. And repeated runs give identical rankings.");

// 9 ── Value added
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Value added over a plain LLM chat", { placeholder: "title" });
const rows = [["Scale", "One posting per chat", "All 8,000 per request, filterable"],
  ["Repeatability", "Score varies by run", "Identical output (tested)"],
  ["Transparency", "Opaque number", "Per-requirement evidence + formula"],
  ["Explanations", "Can say anything", "Only from the match output"],
  ["Cost", "1 LLM call per pair", "JDs extracted once; ranking needs no LLM"]];
const hdr = (t, fill, color) => ({ text: t, options: { bold: true, fill: { color: fill }, color, fontSize: 14 } });
s.addTable([[hdr("", HEX.lt1, HEX.dk1), hdr("Plain LLM chat", HEX.lt2, HEX.dk2), hdr("JobFit", HEX.accent1, HEX.lt1)],
  ...rows.map(([a, b, c]) => [{ text: a, options: { bold: true, color: HEX.dk2 } }, { text: b, options: { color: HEX.accent6 } }, { text: c, options: { color: HEX.dk1 } }])],
  { x: 0.5, y: 1.25, w: 9, colW: [1.9, 3.2, 3.9], fontSize: 14, fontFace: "Calibri", border: { type: "solid", pt: 0.75, color: "D5D9E0" }, rowH: 0.52, valign: "middle" });
T(s, "Measured plain-LLM variance & latency: eval/llm_baseline.py  [insert results once the Gemini key is active]", { x: 0.5, y: 4.75, w: 9, h: 0.35, fontSize: 12, color: HEX.accent4, italic: true });
s.addNotes("This is our value-add over off-the-shelf functionality. Scale: one request ranks everything. Repeatability: identical output for identical input. Transparency: every score decomposes into requirements and quoted evidence. Explanations are generated only from the structured match, so they can't invent qualifications. And cost: we call the LLM once per job description, not once per resume–job pair. [If you ran llm_baseline.py, add: the plain LLM's score varied by X points across five runs.]");

// 10 ── Limitations & next steps
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Close" });
T(s, "Limitations & next steps", { x: 0.6, y: 0.55, w: 8.8, h: 0.8, fontSize: 36, bold: true, color: C.background1, fontFace: "Cambria", valign: "middle" });
const cols = [["Limitations", ["Weak labels measure topic, not fit", "Kaggle resumes are mostly experienced", "Rule extractor limited to its taxonomy", "Weights set by hand, not learned"]],
  ["Next steps", ["Human-labelled fit set → fit the weights", "Two-stage: doc-similarity retrieval + JobFit", "LLM-extracted index + measured LLM baseline", "Resume suggestions from missing requirements"]]];
cols.forEach(([h, items], i) => {
  const x = 0.6 + i * 4.5;
  T(s, h, { x, y: 1.65, w: 4.1, h: 0.45, fontSize: 20, bold: true, color: HEX.accent2, fontFace: "Cambria" });
  T(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })),
    { x, y: 2.2, w: 4.1, h: 2.4, fontSize: 15, color: "E3E8F1", paraSpaceAfter: 8 });
});
T(s, "Thank you — questions?   ·   github: [repository URL]", { x: 0.6, y: 4.85, w: 8.8, h: 0.35, fontSize: 13, color: "9AA6BD" });
s.addNotes("To wrap up: the main limitation is evaluation — our labels measure topic, and the resumes skew experienced — so the most valuable next step is a small human-labelled set of resume–job pairs, which would let us fit the weights instead of setting them. We'd also add document-similarity retrieval in front of JobFit, run the LLM-extracted index, and turn missing requirements into resume suggestions. Thanks — happy to take questions.");

(async () => {
  const out = path.join(ROOT, "docs/JobFit_Presentation.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
