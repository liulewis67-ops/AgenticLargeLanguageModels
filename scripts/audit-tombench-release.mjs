#!/usr/bin/env node
// Offline preflight for a *locally cloned* ToMBench/data release.
// Source reviewed at zhchen18/ToMBench@4f491f0784ed31ef93a8837615fbc48f885cf78d.
// This checks input integrity only. It does NOT run models or reproduce scores.
import assert from "node:assert/strict";
import { mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

export function inspectData(dataDir) {
  const names = readdirSync(dataDir).filter(name => name.endsWith(".jsonl")).sort();
  const summary = {
    dataDirectory: dataDir,
    files: names.length,
    records: 0,
    strictJsonRecords: 0,
    recoveredNonstandardNaN: 0,
    unparseableRecords: 0,
    noncanonicalAnswers: 0,
    missingBilingualFields: 0,
    nanAsChoiceC: 0,
    nonuniqueLocalIndices: 0,
    perFile: [],
    examples: [],
  };
  for (const name of names) {
    const lines = readFileSync(join(dataDir, name), "utf8").split(/\r?\n/).filter(s => s.trim());
    const file = {
      name, records: lines.length, strictJsonRecords: 0,
      recoveredNonstandardNaN: 0, unparseableRecords: 0,
      noncanonicalAnswers: 0, missingBilingualFields: 0,
      nanAsChoiceC: 0, nonuniqueLocalIndices: 0,
    };
    const seenIndex = new Set();
    for (let index = 0; index < lines.length; index++) {
      const source = lines[index];
      const line = index + 1;
      let datum;
      let strict = true;
      try {
        datum = JSON.parse(source);
      } catch {
        strict = false;
        // Explicit audit-only normalization of bare, unquoted, object-value NaN.
        // Do not silently use this to change the benchmark's gold labels.
        const normalized = source.replace(/(:\s*)NaN(?=\s*[,}])/g, "$1null");
        if (normalized !== source) {
          try {
            datum = JSON.parse(normalized);
            file.recoveredNonstandardNaN++;
          } catch {
            file.unparseableRecords++;
          }
        } else {
          file.unparseableRecords++;
        }
      }
      if (strict) file.strictJsonRecords++;
      if (!datum) {
        if (summary.examples.length < 12) summary.examples.push({ file: name, line, issue: "unparseable" });
        continue;
      }
      const label = String(datum["答案\nANSWER"] ?? "").trim();
      if (!/^[ABCD]$/.test(label)) {
        file.noncanonicalAnswers++;
        if (summary.examples.length < 12) summary.examples.push({
          file: name, line, issue: "noncanonical-answer", value: label,
        });
      }
      if (![datum.STORY, datum.QUESTION, datum["故事"], datum["问题"]]
        .every(value => typeof value === "string" && value.trim().length > 0)) {
        file.missingBilingualFields++;
      }
      if (/\"选项C\"\s*:\s*NaN(?=\s*[,}])/.test(source)) file.nanAsChoiceC++;
      const localIndex = String(datum["序号\nINDEX"] ?? "");
      if (seenIndex.has(localIndex)) file.nonuniqueLocalIndices++;
      seenIndex.add(localIndex);
    }
    summary.perFile.push(file);
    summary.records += file.records;
    for (const key of [
      "strictJsonRecords", "recoveredNonstandardNaN", "unparseableRecords",
      "noncanonicalAnswers", "missingBilingualFields", "nanAsChoiceC",
      "nonuniqueLocalIndices",
    ]) summary[key] += file[key];
  }
  return summary;
}

function selfTest() {
  const base = mkdtempSync(join(tmpdir(), "tombench-preflight-"));
  try {
    const record = {
      "序号\nINDEX": 1, "答案\nANSWER": "A",
      "STORY": "English story", "QUESTION": "English question",
      "故事": "中文故事", "问题": "中文问题", "选项C": null,
    };
    const okay = JSON.stringify(record);
    const nan = okay.replace('"选项C":null', '"选项C":NaN');
    const abnormal = JSON.stringify({ ...record, "答案\nANSWER": "A. " });
    writeFileSync(join(base, "fixture.jsonl"), [okay, nan, abnormal, "{broken"].join("\n") + "\n");
    const result = inspectData(base);
    assert.equal(result.files, 1);
    assert.equal(result.records, 4);
    assert.equal(result.strictJsonRecords, 2);
    assert.equal(result.recoveredNonstandardNaN, 1);
    assert.equal(result.unparseableRecords, 1);
    assert.equal(result.noncanonicalAnswers, 1);
    assert.equal(result.nanAsChoiceC, 1);
    assert.equal(result.missingBilingualFields, 0);
    assert.equal(result.nonuniqueLocalIndices, 2);
    console.log("PASS: 8 isolated ToMBench preflight fixture assertions (no upstream data or LLM run)");
  } finally {
    rmSync(base, { recursive: true, force: true });
  }
}

const args = process.argv.slice(2);
if (args.includes("--self-test")) {
  selfTest();
} else {
  const dir = args.find(arg => !arg.startsWith("--"));
  if (!dir) {
    console.error("Usage: node scripts/audit-tombench-release.mjs /path/to/ToMBench/data [--require-strict-json]");
    console.error("Run with --self-test for a dependency-free offline regression test.");
    process.exitCode = 64;
  } else {
    try {
      const result = inspectData(resolve(dir));
      console.log(JSON.stringify(result, null, 2));
      console.log("This is a source dataset inventory, not model inference or research replication.");
      if (result.records !== 2860 || result.files !== 20 ||
          result.unparseableRecords > 0 || result.missingBilingualFields > 0) process.exitCode = 2;
      if (args.includes("--require-strict-json") &&
          (result.recoveredNonstandardNaN > 0 || result.noncanonicalAnswers > 0)) process.exitCode = 3;
    } catch (error) {
      console.error(String(error));
      process.exitCode = 1;
    }
  }
}
