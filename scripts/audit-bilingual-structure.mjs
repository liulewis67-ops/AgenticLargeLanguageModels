// Validate structural parity of Chinese/English academic chapter sources.
// Run: node scripts/audit-bilingual-structure.mjs [--all]
// Default scope is chapters 02-23; --all checks all numbered chapters.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const root = process.cwd();
const entries = JSON.parse(readFileSync(path.join(root, 'config/chapters.json'), 'utf8')).entries;
const auditAll = process.argv.includes('--all');
const selected = entries.filter((entry) => entry.type === 'chapter' &&
  (auditAll || /^(0[2-9]|1\d|2[0-3])-/.test(entry.id)));
const errors = [];
const warnings = [];
const summary = [];

function values(source, pattern, group = 0) {
  return [...source.matchAll(pattern)].map((hit) => hit[group]);
}

function sameItems(a, b) {
  return JSON.stringify([...a].sort()) === JSON.stringify([...b].sort());
}

function numberedHeadings(source, chapter) {
  const re = /^(#{1,6})[ \t]+(?:<span id="page-[^"]+"><\/span>[ \t]*)?(\d+(?:\.\d+){1,3})(?=[ \t]|$)/gm;
  return [...source.matchAll(re)]
    .filter((hit) => hit[2].split('.')[0] === chapter)
    .map((hit) => ({ number: hit[2], depth: hit[1].length }));
}

for (const entry of selected) {
  const files = {};
  for (const lang of ['zh', 'en']) {
    try {
      files[lang] = readFileSync(
        path.join(root, 'docs', entry.id, entry.id + '-' + lang + '.md'), 'utf8');
    } catch (error) {
      errors.push(entry.id + ': missing ' + lang + ' source: ' + error.message);
    }
  }
  if (!files.zh || !files.en) continue;
  // Validate the provenance metadata against the canonical chapter page map.
  // This checks declared source ranges, not that text matches the PDF itself.
  for (const [lang, content] of Object.entries(files)) {
    const source = content.match(/<!--\s*Source:\s*AgenticLargeLanguageModels\.pdf;\s*physical PDF pages (\d+)-(\d+);/);
    if (!source) {
      errors.push(entry.id + ' (' + lang + '): missing source PDF page metadata');
    } else if (Number(source[1]) !== entry.page_start || Number(source[2]) !== entry.page_end) {
      errors.push(entry.id + ' (' + lang + '): source PDF page range ' +
        source[1] + '-' + source[2] + ' differs from config/chapters.json ' +
        entry.page_start + '-' + entry.page_end);
    }
  }
  const chapter = String(Number(entry.id.split('-')[0]));
  const zh = numberedHeadings(files.zh, chapter);
  const en = numberedHeadings(files.en, chapter);
  for (const [lang, headings] of [['zh', zh], ['en', en]]) {
    if (!headings.length) errors.push(entry.id + ' (' + lang + '): no numbered headings');
    const seen = new Set();
    for (const heading of headings) {
      const expected = heading.number.split('.').length;
      if (heading.depth !== expected) errors.push(
        entry.id + ' (' + lang + '): ' + heading.number + ' has H' +
        heading.depth + ' instead of H' + expected);
      if (seen.has(heading.number)) errors.push(
        entry.id + ' (' + lang + '): repeated section ' + heading.number);
      seen.add(heading.number);
    }
  }
  if (JSON.stringify(zh.map((h) => h.number)) !==
      JSON.stringify(en.map((h) => h.number))) {
    errors.push(entry.id + ': bilingual section sequence mismatch');
  }
  const checks = [
    ['page anchors', /<span id="(page-\d+-\d+)"><\/span>/g, 1],
    ['image paths', /!\[[^\]]*\]\((assets\/[^)]+)\)/g, 1],
    ['page dividers', /^\{(\d+)\}-{8,}\s*$/gm, 1],
    ['LaTeX tags', /\\tag\{([^}]+)\}/g, 1],
    ['fenced code delimiters', /^\x60{3}/gm, 0],
  ];
  const counts = {};
  for (const [label, re, group] of checks) {
    const a = values(files.zh, re, group);
    const b = values(files.en, re, group);
    counts[label] = a.length;
    if (!sameItems(a, b)) errors.push(
      entry.id + ': ' + label + ' mismatch (zh=' + a.length + ', en=' + b.length + ')');
  }
  // Detect reordered evidence even when both languages contain the same items.
  // Earlier chapters may contain extraction-specific ordering, so strict ordering
  // begins with the newly audited 19-23 group until legacy cases are reconciled.
  if (Number(chapter) >= 19 && Number(chapter) <= 23) {
    for (const [label, re, group] of checks.slice(0, 4)) {
      const a = values(files.zh, re, group);
      const b = values(files.en, re, group);
      if (JSON.stringify(a) !== JSON.stringify(b)) errors.push(
        entry.id + ': ' + label + ' bilingual order mismatch');
    }
  }
  // Source extraction can leave some physical divider markers indented or inline.
  // Record gaps for manual PDF review; never synthesize a missing page boundary.
  const declaredPages = files.zh.match(/Marker split PDF pages 1-(\d+)/);
  if (declaredPages) {
    const expectedPages = Number(declaredPages[1]);
    const markers = new Set(values(files.zh, /^\{(\d+)\}-{8,}\s*$/gm, 1).map(Number));
    const missing = Array.from({ length: expectedPages }, (_, i) => i)
      .filter((page) => !markers.has(page));
    if (missing.length) warnings.push(
      entry.id + ': missing standalone PDF divider markers on zh pages ' +
      missing.join(', ') + '; check inline anchors/source PDF before editing');
  }
  const mathCountZh = values(files.zh, /\$\$[\s\S]*?\$\$/g).length;
  const mathCountEn = values(files.en, /\$\$[\s\S]*?\$\$/g).length;
  if (mathCountZh !== mathCountEn) errors.push(
    entry.id + ': display math count mismatch (zh=' + mathCountZh +
    ', en=' + mathCountEn + ')');
  for (const [lang, content] of Object.entries(files)) {
    if (content.includes('\uFFFD')) errors.push(
      entry.id + ' (' + lang + '): replacement character');
    if (/[^\r\n]\{\d+\}-{8,}/m.test(content)) errors.push(
      entry.id + ' (' + lang + '): page divider embedded in prose');
  }
  summary.push({
    chapter,
    sections: zh.length,
    page_anchors: counts['page anchors'],
    images: counts['image paths'],
    page_dividers: counts['page dividers'],
    display_math: mathCountZh,
  });
}
const result = {
  scope: auditAll ? 'all chapters' : 'chapters 02-23',
  checked_pairs: summary.length,
  status: errors.length ? 'failed' : 'passed',
  summary,
  warnings,
  errors,
  limitation: 'Structural comparison only; does not verify PDF text fidelity or translation semantics.',
};
console.log(JSON.stringify(result, null, 2));
if (errors.length) process.exitCode = 1;
