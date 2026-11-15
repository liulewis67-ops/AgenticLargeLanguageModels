#!/usr/bin/env node
// Offline preflight for a locally cloned CASA dataset. No model calls or downloads.
// Audited source: SalesforceAIResearch/CASA@3ddd1b568a7a0e906a17fc8219d7000a83e268ba.
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const files = [
  ["S1-A", "s1_adhere_data/online_shopping.json"],
  ["S1-A", "s1_adhere_data/social_discussion_forum.json"],
  ["S1-V", "s1_violate_data/online_shopping.json"],
  ["S1-V", "s1_violate_data/social_discussion_forum.json"],
  ["S2", "s2_data/online_shopping.json"],
  ["S2", "s2_data/social_discussion_forum.json"],
];
const paperCounts = { "S1-A": 607, "S1-V": 618, S2: 622 };
const auditedReleaseCounts = { "S1-A": 599, "S1-V": 599, S2: 594 };

export function inspectCASA(root) {
  const groups = { "S1-A": 0, "S1-V": 0, S2: 0 };
  const byFile = [];
  const indices = new Map();
  let invalidRecords = 0;
  for (const [group, relative] of files) {
    const values = JSON.parse(readFileSync(resolve(root, relative), "utf8"));
    assert.ok(Array.isArray(values), relative + ": expected a JSON array");
    const seen = new Set();
    let missingRequired = 0;
    let duplicateLocalIds = 0;
    for (const row of values) {
      const valid = row && typeof row === "object" &&
        ["index", "user_query", "violated_norm"].every(key =>
          typeof row[key] === "string" && row[key].trim().length > 0);
      if (!valid) {
        missingRequired++;
        continue;
      }
      if (seen.has(row.index)) duplicateLocalIds++;
      seen.add(row.index);
    }
    const count = values.length;
    invalidRecords += missingRequired + duplicateLocalIds;
    groups[group] += count;
    byFile.push({ group, path: relative, records: count,
      missingRequired, duplicateLocalIds });
    indices.set(relative, seen);
  }
  const pairing = [];
  for (const type of ["online_shopping.json", "social_discussion_forum.json"]) {
    const adhere = indices.get("s1_adhere_data/" + type);
    const violate = indices.get("s1_violate_data/" + type);
    const observed = indices.get("s2_data/" + type);
    const setDifference = (a, b) => [...a].filter(key => !b.has(key)).length;
    const mismatch = setDifference(adhere, violate) + setDifference(violate, adhere);
    const s2MissingParent = setDifference(observed,
      type === "online_shopping.json" ? violate : adhere);
    pairing.push({ type, s1SymmetricDifference: mismatch, s2Unmatched: s2MissingParent });
    invalidRecords += mismatch + s2MissingParent;
  }
  const report = {
    source: "SalesforceAIResearch/CASA@3ddd1b568a7a0e906a17fc8219d7000a83e268ba",
    paperCounts, auditedReleaseCounts,
    groups,
    paperTotal: Object.values(paperCounts).reduce((a, b) => a + b, 0),
    releaseTotal: Object.values(groups).reduce((a, b) => a + b, 0),
    missingAgainstPaper: Object.fromEntries(Object.keys(groups).map(g => [g, paperCounts[g] - groups[g]])),
    invalidRecords, byFile, pairing,
  };
  return report;
}

function selfTest() {
  const dir = mkdtempSync(join(tmpdir(), "casa-preflight-"));
  try {
    for (const [group, relative] of files) {
      const file = join(dir, relative);
      mkdirSync(resolve(file, ".."), { recursive: true });
      const example = { index: "China_0", user_query: "A sample question",
        violated_norm: "A sample norm" };
      writeFileSync(file, JSON.stringify([example]), "utf8");
    }
    const report = inspectCASA(dir);
    assert.deepEqual(report.groups, { "S1-A": 2, "S1-V": 2, S2: 2 });
    assert.equal(report.paperTotal, 1847);
    assert.equal(report.releaseTotal, 6);
    assert.equal(report.invalidRecords, 0);
    assert.equal(report.pairing[0].s1SymmetricDifference, 0);
    assert.equal(report.pairing[1].s2Unmatched, 0);
    const affected = join(dir, "s1_violate_data", "online_shopping.json");
    writeFileSync(affected, JSON.stringify([{ index: "China_1",
      user_query: "A sample question", violated_norm: "A sample norm" }]));
    const bad = inspectCASA(dir);
    assert.ok(bad.invalidRecords > 0, "paired-ID mismatch must be detected");
    console.log("PASS: 7 fixture assertions for CASA inventory and paired IDs; no upstream data or model execution");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

const args = process.argv.slice(2);
if (args.includes("--self-test")) {
  selfTest();
} else {
  const path = args.find(arg => !arg.startsWith("--"));
  if (!path) {
    console.error("Usage: node scripts/audit-casa-release.mjs /path/to/CASA [--expect-audited-release|--expect-paper]");
    console.error("For offline fixture tests: node scripts/audit-casa-release.mjs --self-test");
    process.exitCode = 64;
  } else {
    try {
      const report = inspectCASA(resolve(path));
      console.log(JSON.stringify(report, null, 2));
      console.log("Dataset inventory only; no LLM inference, GPT-4o judging or experimental replication.");
      if (report.invalidRecords > 0) process.exitCode = 2;
      if (args.includes("--expect-audited-release") &&
          Object.entries(auditedReleaseCounts).some(([group, n]) => report.groups[group] !== n))
        process.exitCode = 3;
      if (args.includes("--expect-paper") &&
          Object.entries(paperCounts).some(([group, n]) => report.groups[group] !== n))
        process.exitCode = 4;
    } catch (error) {
      console.error(String(error));
      process.exitCode = 1;
    }
  }
}
