// Source-verified regression guards for the bilingual reading edition.
// Source: https://arxiv.org/html/2610.13455v1 (v1).
// Does NOT fetch the source, compare rendered PDF pages, or reproduce experiments.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import assert from "node:assert/strict";

const root = resolve(import.meta.dirname, "..");
const paths = {
  14: "14-case-study-3-multi-agent-diagnosis",
  15: "15-studying-behavior-of-agentic-llms",
  16: "16-role-based-interaction-in-multi-llm-systems",
  17: "17-emergent-behavior-in-multi-llm-systems",
  18: "18-case-study-4-simulating-societies",
  19: "19-synthetic-data-generation",
  20: "20-advanced-agentic-architectures",
  21: "21-mechanistic-interpretability-for-llm-agents",
};
const read = (chapter, lang) =>
  readFileSync(resolve(root, "docs", paths[chapter], paths[chapter] + "-" + lang + ".md"), "utf8");
const tick = String.fromCharCode(96);
let checks = 0;

for (const lang of ["en", "zh"]) {
  const chapter14 = read(14, lang);
  for (const filename of ["0_train_models.py", "1_basic_mcp.py", "2_llm_mcp.py"]) {
    assert.ok(chapter14.includes(tick + filename + tick),
      "Chapter 14 must use the original prefixed filename: " + lang + " " + filename);
    checks++;
  }
  for (const wrong of ["train_models.py", "basic_mcp.py", "llm_mcp.py"]) {
    assert.ok(!chapter14.includes(tick + wrong + tick),
      "Unprefixed chapter 14 filename reintroduced: " + lang + " " + wrong);
    checks++;
  }
  assert.ok(chapter14.includes(tick + "make_case_text" + tick),
    "Chapter 14 function name must be exact in " + lang);
  checks++;

  const chapter18 = read(18, lang);
  assert.ok(chapter18.includes("\\tag{18.1}"), "Missing Chapter 18 formula " + lang);
  checks++;

  const chapter21 = read(21, lang);
  assert.ok(chapter21.includes("\\tag{21.7}"), "Missing Chapter 21 formula " + lang);
  checks++;
  const marker = lang === "en" ? "**Corrupted or counterfactual run**" : "**受扰或反事实运行**";
  const start = chapter21.indexOf(marker);
  const end = chapter21.indexOf("{10}------------------------------------------------", start);
  assert.ok(start >= 0 && end > start, "Missing Chapter 21 source page boundary " + lang);
  checks++;
  const passage = chapter21.slice(start, end);
  if (lang === "en") {
    assert.ok(passage.includes("*The Colosseum is in Rome*"), "English counterfactual drifted");
    assert.ok(!passage.includes("*The Eiffel Tower is in Rome*"), "English source corruption reappeared");
    assert.ok(chapter21.includes("**28 heads**"), "ICLR 2023 variant absent from editor note");
  } else {
    assert.ok(passage.includes("*罗马斗兽场位于罗马*"), "Chinese counterfactual drifted");
    assert.ok(!passage.includes("*埃菲尔铁塔位于罗马*"), "Chinese source corruption reappeared");
    assert.ok(chapter21.includes("**28 个注意力头**"), "ICLR 2023 variant absent from editor note");
  }
  checks += 3;
}

for (const lang of ["en", "zh"]) {
  const ch19 = read(19, lang);
  const ch20 = read(20, lang);
  const claims19 = lang === "en"
    ? ["**61 tasks and approximately 193,000 instances**", "**cross-task generalization**", "ACL 2022 Natural Instructions paper"]
    : ["**61 个任务、约 19.3 万条任务实例**", "**跨任务泛化**", "ACL 2022 Natural Instructions 原论文"];
  const claims20 = lang === "en"
    ? ["fixed random subset of 50 SWE-bench Verified problems", "iteration 15 it reports 51%", "only GSM8K", "not independently reproduced"]
    : ["固定随机 50 题子集", "迭代 15 的结果为 51%", "默认仅启用 GSM8K", "非独立实验复现"];
  for (const assertion of claims19) {
    assert.ok(ch19.includes(assertion), "Chapter 19 source qualifier missing (" + lang + "): " + assertion);
    checks++;
  }
  for (const assertion of claims20) {
    assert.ok(ch20.includes(assertion), "Chapter 20 sample/config qualifier missing (" + lang + "): " + assertion);
    checks++;
  }
}
// Static guards only: these strings were separately checked against primary sources,
// but this offline script cannot itself verify their external truth.

for (const lang of ["en", "zh"]) {
  const c15 = read(15, lang);
  const c16 = read(16, lang);
  const c17 = read(17, lang);
  const claims15 = lang === "en"
    ? ["**six** categories", "2,860", "**504 non-standard", "first **10 samples per file**"]
    : ["**六类能力", "2,860", "**504 条非标准", "**每个文件前 10 条**"];
  const claims16 = lang === "en"
    ? ["**198 CoordQA questions", "66 manually curated scenarios", "pages 8053–8072"]
    : ["**198 道 CoordQA 题**", "**66 个手工设计场景", "pages 8053–8072"];
  const claims17 = lang === "en"
    ? ["**27 A100 GPUs", "18 hours per simulated time step"]
    : ["**27 块 A100 GPU", "每个模拟时间步约 18 小时"];
  for (const [chapter, claims, body] of [
    [15, claims15, c15], [16, claims16, c16], [17, claims17, c17],
  ]) {
    for (const claim of claims) {
      assert.ok(body.includes(claim),
        "Source-corrected Chapter " + chapter + " fact drift (" + lang + "): " + claim);
      checks++;
    }
  }
  // The original bibliography error must not be reintroduced by future conversions.
  assert.ok(!c16.includes("pages 8038–8057"),
    "Incorrect original NAACL page range returned: " + lang);
  checks++;
}

for (const lang of ["en", "zh"]) {
  const ch15 = read(15, lang);
  const ch16 = read(16, lang);
  const expected15 = lang === "en"
    ? ["11.0 percentage points behind humans", "**11.0-point difference**", "**7.7 points**"]
    : ["11.0 个百分点", "**11.0 分差（百分点）**", "**7.7 分差**"];
  for (const phrase of expected15) {
    assert.ok(ch15.includes(phrase), "Chapter 15 numeric unit/version missing (" + lang + "): " + phrase);
    checks++;
  }
  assert.ok(!ch15.includes(lang === "en" ? "lag at least 11% behind humans" : "至少落后于人类 11%"),
    "Chapter 15 inaccurate percentage unit reintroduced: " + lang);
  checks++;
  for (const phrase of ["pages 8053–8072, 2025.", "198", "8053–8072"]) {
    assert.ok(ch16.includes(phrase), "Chapter 16 primary citation and QA scope missing (" + lang + "): " + phrase);
    checks++;
  }
  assert.ok(!ch16.includes("pages 8038–8057, 2025."),
    "Chapter 16 wrong published pagination reintroduced: " + lang);
  checks++;
}



for (const lang of ["en", "zh"]) {
  const ch17 = read(17, lang);
  const confirmed = lang === "en" ? [
    "599 + 599 + 594 = 1,792",
    "1,847 items", "55 fewer",
    "40/40/27/20 runs",
    "not been rerun here",
    "CASA released-data verification",
  ] : [
    "599、599、594 条，共 1,792 条",
    "1,847 条", "少 55 条",
    "40、40、27、20",
    "尚未运行复现实验",
    "CASA 公开数据实核",
  ];
  for (const phrase of confirmed) {
    assert.ok(ch17.includes(phrase),
      "Chapter 17 primary-source qualifier missing (" + lang + "): " + phrase);
    checks++;
  }
}


for (const lang of ["en", "zh"]) {
  const introPath = resolve(root, "docs", "01-introduction", "01-introduction-" + lang + ".md");
  const intro = readFileSync(introPath, "utf8");
  const label = lang === "en" ? "## Bibliography" : "## 参考文献";
  assert.ok(intro.includes(label), "Chapter 1 bibliography is missing: " + lang);
  const bibliography = intro.slice(intro.indexOf(label) + label.length);
  const pageGroups = bibliography.split(/<span id="page-(?:8|9)-0"><\/span>/);
  const actualCounts = pageGroups.map((group) => (group.match(/^- /gm) ?? []).length);
  assert.deepEqual(actualCounts, [7, 13, 4],
    "Chapter 1 references must remain 24 distinct source-aligned entries across source pages: " + lang);
  checks++;
  assert.ok(bibliography.includes("LLaMEA: A large language model evolutionary algorithm"),
    "Chapter 1 LLaMEA bibliography extraction corruption returned: " + lang);
  assert.ok(bibliography.includes("URL <https://web.stanford.edu/~jurafsky/slp3/>"),
    "Chapter 1 bibliography URL corruption returned: " + lang);
  assert.ok(!/[¨¸]/.test(bibliography) && !bibliography.includes("URL [https:]"),
    "Chapter 1 bibliography contains broken PDF encoding: " + lang);
  assert.ok(!/^####\s+(?:Agentic LLM|智能体大语言模型)\s*=/m.test(intro),
    "Chapter 1 display equation must not be a section heading: " + lang);
  checks += 4;
}
// Chapter 1 bibliography counts/order are compared to arXiv v1 HTML References.
// Static checks are not original-PDF visual verification or full citation validation.


for (const chapter of [
  { id: "02-foundations-of-language-modeling", expected: 59, first: "Josh Achiam", last: "Aston Zhang" },
  { id: "03-llm-training-pipeline", expected: 52, first: "Josh Achiam", last: "Chunting Zhou" },
]) {
  const bodies = {};
  for (const lang of ["en", "zh"]) {
    const text = readFileSync(resolve(root, "docs", chapter.id, chapter.id + "-" + lang + ".md"), "utf8");
    const heading = lang === "en" ? "## Bibliography" : "## 参考文献";
    assert.ok(text.includes(heading), "Missing chapter bibliography: " + chapter.id + " (" + lang + ")");
    const body = text.slice(text.lastIndexOf(heading) + heading.length);
    bodies[lang] = body;
    const entries = body.split("\n").filter((line) => line.startsWith("- "));
    assert.equal(entries.length, chapter.expected,
      "Source-aligned bibliography count for " + chapter.id + " (" + lang + ")");
    assert.ok(entries[0].includes(chapter.first) && entries.at(-1).includes(chapter.last),
      "Bibliography reference order drifted: " + chapter.id + " (" + lang + ")");
    const maxLine = Math.max(...body.split("\n").map((line) => line.length));
    assert.ok(maxLine < 1500,
      "Likely merged references (line exceeds 1500 characters) in " + chapter.id + " (" + lang + ")");
    assert.ok(!/\]\(#page-\d+-\d+\) [A-Z]/.test(body),
      "Merged reference boundary reappeared in " + chapter.id + " (" + lang + ")");
    assert.ok(!/URL \[[^\]]+\]\(https?:\/\/[^)\n]+\) \[[^\]]+\]\(/.test(body),
      "Split PDF-extracted bibliography URL reappeared in " + chapter.id + " (" + lang + ")");
    checks += 5;
  }
  assert.equal(bodies.en, bodies.zh,
    "Bibliography contents diverged across Chinese and English: " + chapter.id);
  checks++;
}
for (const lang of ["en", "zh"]) {
  const content = readFileSync(resolve(root, "docs", "03-llm-training-pipeline", "03-llm-training-pipeline-" + lang + ".md"), "utf8");
  assert.ok(content.includes(lang === "en"
    ? "Editorial clarification (not in the source)"
    : "编者校勘（非原文）"), "Chapter 3 Pass@K source-ambiguity note missing");
  checks++;
}
// Chapter 2/3 reference counts are checked against arXiv:2610.13455v1 HTML.
// These are static source-integrity guards, not PDF image inspection or experiment replication.


for (const lang of ["en", "zh"]) {
  const chapter = "04-scaling-laws-and-efficiency";
  const source = readFileSync(resolve(root, "docs", chapter, chapter + "-" + lang + ".md"), "utf8");
  const marker = lang === "en" ? "## Bibliography" : "## 参考文献";
  assert.ok(source.includes(marker), "Chapter 4 bibliography heading missing: " + lang);
  const bibliography = source.slice(source.indexOf(marker) + marker.length);
  const references = bibliography.split("\n").filter(line => line.startsWith("- "));
  assert.equal(references.length, 72, "Chapter 4 source-aligned reference count: " + lang);
  assert.ok(references[0].startsWith("- Marah Abdin") && references.at(-1).startsWith("- Chunting Zhou"),
    "Chapter 4 bibliography source order drifted: " + lang);
  assert.ok(Math.max(...bibliography.split("\n").map(x => x.length)) < 1500,
    "Chapter 4 merged bibliography reference line: " + lang);
  assert.ok(bibliography.includes("10.18653/v1/2023.trustnlp-1.5"),
    "Chapter 4 ACL reference DOI broken across PDF lines: " + lang);
  assert.ok(bibliography.includes("https://openreview.net/forum?id=aVSxwicpAk") &&
    !bibliography.includes("[net/forum?id=aVSxwicpAk]"),
    "Chapter 4 OpenReview URL broken across PDF page boundary: " + lang);
  for (const number of ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6"]) {
    assert.ok(source.includes("(" + number + ")") || source.includes("（" + number + "）"),
      "Chapter 4 source equation label missing: " + number + " (" + lang + ")");
  }
  checks += 12;
}
// The fitted Chinchilla parameters give the displayed rounded compute-optimal exponents.
// Arithmetic sanity check only; not an experimental replication.
const a = 0.28 / (0.34 + 0.28);
const b = 0.34 / (0.34 + 0.28);
assert.ok(Math.abs(a - 0.45) < 0.005 && Math.abs(b - 0.55) < 0.005 &&
  Math.abs(a + b - 1) < 1e-12, "Chapter 4 Chinchilla exponents are inconsistent");
checks++;

console.log("PASS " + checks + " manuscript provenance checks");
console.log("These are static guards, NOT original-PDF rendering or research replication.");
