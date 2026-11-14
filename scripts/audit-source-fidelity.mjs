// Source-verified regression guards for the bilingual reading edition.
// Source: https://arxiv.org/html/2610.13455v1 (v1).
// Does NOT fetch the source, compare rendered PDF pages, or reproduce experiments.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import assert from "node:assert/strict";

const root = resolve(import.meta.dirname, "..");
const paths = {
  14: "14-case-study-3-multi-agent-diagnosis",
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

console.log("PASS " + checks + " manuscript provenance checks");
console.log("These are static guards, NOT original-PDF rendering or research replication.");
