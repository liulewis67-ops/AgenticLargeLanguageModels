// Comprehensive, deterministic release-gate audit of the 472-page bilingual reading edition.
// Scope: all 32 manifest entries, both Markdown languages, bibliography boundaries,
// local source-page references, provenance metadata and original-page ranges.
// Does NOT inspect the 31 MB source PDF, verify remote citations, compare translations
// semantically, or reproduce reported experiments. Those remain human review gates.
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import process from "node:process";

const root = process.cwd();
const docsRoot = path.join(root, "docs");
const manifest = JSON.parse(readFileSync(path.join(root, "config/chapters.json"), "utf8"));
const knownBibliography = new Map(Object.entries({
  "01": 24, "02": 59, "03": 52, "04": 72, "05": 9,
  "06": 72, "07": 69, "08": 11, "09": 43, "10": 3,
  "11": 24, "12": 34, "13": 40, "14": 1, "15": 90,
  "16": 29, "17": 47, "18": 3, "19": 87, "20": 42,
  "21": 46, "22": 111, "90": 34
}));
const failures = [];
const warnings = [];
const documents = new Map();
const stats = { entries: 0, documents: 0, pages: 0, bibliographyEntries: 0,
  images: 0, pageAnchors: 0, localPageLinks: 0, externalLinks: 0, warnings: 0 };

const fail = (label, message) => failures.push(label + ": " + message);
const labelFor = (entry, lang) => entry.id + "-" + lang;
const sourceName = entry => entry.id.match(/^(?:\d{2}|part-[a-z]+)-/) ? entry.id : entry.id;

assert.equal(manifest.page_count, 472, "Source-version manifest physical page count changed");
assert.equal(manifest.entries.length, 32, "Source-book manifest now has different content boundaries");

let nextPage = 1;
for (const entry of manifest.entries) {
  assert.equal(entry.page_start, nextPage,
    "Source-book coverage gap or overlap at " + entry.id + ", expected page " + nextPage);
  assert.ok(entry.page_end >= entry.page_start && entry.page_end <= manifest.page_count,
    "Invalid source page range for " + entry.id);
  nextPage = entry.page_end + 1;
  stats.pages += entry.page_end - entry.page_start + 1;
  stats.entries++;

  for (const lang of ["en", "zh"]) {
    const file = path.join(docsRoot, entry.id, entry.id + "-" + lang + ".md");
    const label = labelFor(entry, lang);
    if (!existsSync(file)) { fail(label, "source document does not exist"); continue; }
    const content = readFileSync(file, "utf8");
    documents.set(file, { entry, lang, label, content,
      anchors: new Set([...content.matchAll(/\bid="(page-\d+-0)"/g)].map(m => m[1])) });
    stats.documents++;

    const origin = content.match(/Source:\s*AgenticLargeLanguageModels\.pdf;\s*physical PDF pages\s+(\d+)[-–](\d+)/);
    if (!origin || +origin[1] !== entry.page_start || +origin[2] !== entry.page_end)
      fail(label, "source-PDF page range does not match config/chapters.json");

    const anchorItems = [...content.matchAll(/\bid="(page-\d+-0)"/g)].map(m => m[1]);
    const duplicates = anchorItems.filter((v, i) => anchorItems.indexOf(v) !== i);
    if (duplicates.length) fail(label, "duplicated source-page ids: " + [...new Set(duplicates)].join(", "));
    stats.pageAnchors += anchorItems.length;

    stats.images += [...content.matchAll(/!\[[^\]\n]*\]\([^)]+\)/g)].length;
    if (content.includes("\uFFFD")) fail(label, "Unicode replacement character U+FFFD found");
    if (/^(?:<{7}|={7}|>{7})/m.test(content)) fail(label, "unresolved merge conflict marker");
    // This is a diagnostic only: some PDF dividers intentionally have inline page anchors.
    if (entry.type === "chapter" && anchorItems.length < entry.page_end - entry.page_start + 1) {
      warnings.push(label + ": not every original physical page has its own page-* marker"
        + " (" + anchorItems.length + "/" + (entry.page_end - entry.page_start + 1) + ")");
    }

    const chapterId = entry.id.slice(0, 2);
    if (entry.type === "chapter" && chapterId === "23") {
      if (/^#{1,3}\s+(?:<span id="page-\d+-0"><\/span>\s*)?(?:Bibliography|参考文献)\s*$/m.test(content))
        fail(label, "Ch23 sources must link to the standalone chapter-23 references module");
      if (!content.includes("../90-references/90-references-" + lang + ".md"))
        fail(label, "missing Chapter 23 link to standalone references");
      continue;
    }
    if (entry.type !== "chapter" && entry.type !== "references") continue;
    const expectedCount = knownBibliography.get(chapterId);
    if (expectedCount === undefined) { fail(label, "missing bibliography baseline"); continue; }
    const heading = content.match(/^#{1,3}\s+(?:<span id="page-\d+-0"><\/span>\s*)?(?:Bibliography|参考文献)\s*$/m);
    if (!heading || !heading[0].includes(lang === "en" ? "Bibliography" : "参考文献")) {
      fail(label, "bibliography missing or wrong-language heading");
      continue;
    }
    const body = content.slice(heading.index + heading[0].length);
    const lines = body.split("\n");
    const refs = lines.filter(line => /^- \S/.test(line));
    stats.bibliographyEntries += refs.length;
    if (refs.length !== expectedCount)
      fail(label, "expected " + expectedCount + " source references, found " + refs.length);
    const orphans = lines.map((line, i) => ({ line, n: i + 1 })).filter(({ line }) =>
      line.trim() &&
      !/^- \S/.test(line) &&
      !/^\{\d+\}-{3,}$/.test(line) &&
      !/^\s*<span id="page-\d+-0"><\/span>\s*$/.test(line) &&
      !/^<!-- .* -->$/.test(line));
    for (const {line, n} of orphans.slice(0, 8))
      fail(label, "PDF-split bibliography continuation at bibliography line " + n +
        ": " + line.slice(0, 90));
    if (orphans.length > 8) fail(label, "additional " + (orphans.length - 8) + " bibliography fragments");
    const unique = new Set(refs.map(s => s.replace(/\s+/g, " ").trim()));
    if (unique.size < refs.length)
      warnings.push(label + ": exact duplicate bibliographic entries (" + (refs.length - unique.size) + ")");
  }
}
assert.equal(nextPage, 473, "Source-book final page not covered by content manifest");
assert.equal(stats.pages, manifest.page_count, "Physical page coverage is not complete");

// Book-wide internal links are checked against the actual pair of source files.
// VitePress build/dist validators separately verify rendered heading slugs and resources.
for (const [file, doc] of documents) {
  for (const match of doc.content.matchAll(/!?\[[^\]\n]*\]\(([^)\n]+)\)/g)) {
    let raw = match[1].trim();
    if (raw.startsWith("<") && raw.includes(">")) raw = raw.slice(1, raw.indexOf(">"));
    const url = raw.split(/\s+"[^"]+"$/, 1)[0];
    if (/^[a-z][a-z\d+.-]*:/i.test(url)) {
      if (/^https?:/i.test(url)) stats.externalLinks++;
      continue;
    }
    if (url.startsWith("//")) { stats.externalLinks++; continue; }
    const hashIndex = url.indexOf("#");
    if (hashIndex < 0) continue;
    const fragment = url.slice(hashIndex + 1);
    if (!/^page-\d+-0$/.test(fragment)) continue;
    let target = file;
    if (hashIndex > 0) {
      const targetPart = url.slice(0, hashIndex).split("?")[0];
      if (targetPart.startsWith("/")) continue; // handled by built-site verifier
      try { target = path.resolve(path.dirname(file), decodeURIComponent(targetPart)); }
      catch { fail(doc.label, "URL-encoded local page target is invalid: " + url); continue; }
    }
    stats.localPageLinks++;
    const destination = documents.get(target);
    if (!destination) {
      fail(doc.label, "local page link resolves outside known book content: " + url);
      continue;
    }
    if (!destination.anchors.has(fragment))
      fail(doc.label, "target is missing source page anchor " + fragment +
        " (" + path.relative(docsRoot, target) + ")");
    if (target !== file && destination.lang !== doc.lang)
      warnings.push(doc.label + ": cross-language page link: " + url);
  }
}

const refPath = lang => path.join(docsRoot, "90-references", "90-references-" + lang + ".md");
for (const lang of ["en", "zh"]) {
  const standalone = documents.get(refPath(lang));
  if (!standalone || !/^## (Bibliography|参考文献)$/m.test(standalone.content))
    fail("90-references-"+lang, "standalone references did not resolve");
}

stats.warnings = warnings.length;
for (const error of failures) console.error("FAIL " + error);
for (const warning of warnings.slice(0, 35)) console.warn("WARN " + warning);
if (warnings.length > 35) console.warn("WARN " + (warnings.length - 35) + " additional nonblocking diagnostics");
console.log("Book integrity audit: " + JSON.stringify(stats));
console.log("Evidence limitations: no PDF page rendering, semantic translation check, remote DOI validation, or empirical model replication.");
if (failures.length) {
  console.error("Final book-integrity checks failed: " + failures.length);
  process.exitCode = 1;
} else console.log("PASS: 32 content units, 64 bilingual Markdown sources and 472-page manifest coverage checked.");
