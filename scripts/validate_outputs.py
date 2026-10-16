"""Validate bilingual chapter coverage, links, anchors, and index mappings."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CHAPTERS = ROOT / "config" / "chapters.json"
REPORT = ROOT / "reports" / "validation.json"
LINK = re.compile(r"!?\[([^\]]*)\]\(([^)]+)\)")
PAGE_MARKER = re.compile(r"^ {0,2}\{\d+\}-{8,}\s*$", re.MULTILINE)
HTML_ID = re.compile(r"\bid=[\"']([^\"']+)[\"']")
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
HTML_TAG = re.compile(r"<[^>]*>")
MARKDOWN_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")


def heading_slug(text: str) -> str:
    text = MARKDOWN_LINK.sub(r"\1", text)
    text = HTML_TAG.sub("", text)
    text = re.sub(r"[`*_~]", "", text).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def anchors_for(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    resolved = path.resolve()
    if resolved not in cache:
        text = resolved.read_text(encoding="utf-8")
        anchors = set(HTML_ID.findall(text))
        heading_counts: Counter[str] = Counter()
        for line in text.splitlines():
            match = HEADING.match(line)
            if not match:
                continue
            base = heading_slug(match.group(1))
            occurrence = heading_counts[base]
            heading_counts[base] += 1
            anchors.add(base if occurrence == 0 else f"{base}-{occurrence}")
        cache[resolved] = anchors
    return cache[resolved]


def resolve_local_target(source_file: Path, raw_target: str) -> tuple[Path | None, str]:
    target = raw_target.strip().strip("<>")
    if not target or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
        return None, ""
    path, _, fragment = target.partition("#")
    destination = (source_file.parent / unquote(path)).resolve() if path else source_file.resolve()
    return destination, unquote(fragment)


def main() -> int:
    data = json.loads(CHAPTERS.read_text(encoding="utf-8"))
    entries = data["entries"]
    page_coverage = [False] * data["page_count"]
    page_owners = [0] * data["page_count"]
    chapter_results = []
    broken_links = {"en": [], "zh": []}
    broken_anchors = {"en": [], "zh": [], "index_en": [], "index_zh": []}
    missing_en: list[str] = []
    missing_zh: list[str] = []
    mismatched_page_markers = []
    index_link_errors = {"en": [], "zh": []}
    index_mapping_errors = {"en": [], "zh": []}
    anchor_cache: dict[Path, set[str]] = {}
    fragment_links_checked = 0
    index_targets_checked = {"en": 0, "zh": 0}
    en_bytes = zh_bytes = 0
    replacement_characters_en = replacement_characters_zh = 0

    for entry in entries:
        chapter_id = entry["id"]
        folder = DOCS / chapter_id
        en_file = folder / f"{chapter_id}-en.md"
        zh_file = folder / f"{chapter_id}-zh.md"
        if not en_file.is_file():
            missing_en.append(chapter_id)
        if not zh_file.is_file():
            missing_zh.append(chapter_id)
        en_text = en_file.read_text(encoding="utf-8") if en_file.is_file() else ""
        zh_text = zh_file.read_text(encoding="utf-8") if zh_file.is_file() else ""
        if en_file.is_file():
            en_bytes += en_file.stat().st_size
            replacement_characters_en += en_text.count("\ufffd")
            for match in LINK.finditer(en_text):
                destination, fragment = resolve_local_target(en_file, match.group(2))
                if destination is None:
                    continue
                if not destination.exists():
                    broken_links["en"].append({"file": en_file.relative_to(ROOT).as_posix(), "target": match.group(2)})
                elif fragment and destination.is_file() and destination.suffix.lower() == ".md":
                    fragment_links_checked += 1
                    if fragment not in anchors_for(destination, anchor_cache):
                        broken_anchors["en"].append({"file": en_file.relative_to(ROOT).as_posix(), "target": match.group(2)})
        if zh_file.is_file():
            zh_bytes += zh_file.stat().st_size
            replacement_characters_zh += zh_text.count("\ufffd")
            for match in LINK.finditer(zh_text):
                destination, fragment = resolve_local_target(zh_file, match.group(2))
                if destination is None:
                    continue
                if not destination.exists():
                    broken_links["zh"].append({"file": zh_file.relative_to(ROOT).as_posix(), "target": match.group(2)})
                elif fragment and destination.is_file() and destination.suffix.lower() == ".md":
                    fragment_links_checked += 1
                    if fragment not in anchors_for(destination, anchor_cache):
                        broken_anchors["zh"].append({"file": zh_file.relative_to(ROOT).as_posix(), "target": match.group(2)})
            if en_file.is_file():
                en_markers = len(PAGE_MARKER.findall(en_text))
                zh_markers = len(PAGE_MARKER.findall(zh_text))
                if en_markers != zh_markers:
                    mismatched_page_markers.append({"id": chapter_id, "en_count": en_markers, "zh_count": zh_markers})

        start, end = int(entry["page_start"]), int(entry["page_end"])
        for page in range(start, end + 1):
            if 1 <= page <= len(page_coverage):
                page_coverage[page - 1] = True
                page_owners[page - 1] += 1
        chapter_results.append({
            "id": chapter_id,
            "page_start": start,
            "page_end": end,
            "en_exists": en_file.is_file(),
            "zh_exists": zh_file.is_file(),
            "en_bytes": en_file.stat().st_size if en_file.is_file() else 0,
            "zh_bytes": zh_file.stat().st_size if zh_file.is_file() else 0,
            "en_replacement_characters": en_text.count("\ufffd"),
            "zh_replacement_characters": zh_text.count("\ufffd"),
        })

    missing_pages = [index + 1 for index, covered in enumerate(page_coverage) if not covered]
    overlapping_pages = [index + 1 for index, owners in enumerate(page_owners) if owners > 1]
    for language in ("en", "zh"):
        index_file = DOCS / f"index-{language}.md"
        if not index_file.is_file():
            index_link_errors[language].append({"missing_index": index_file.name})
            continue
        index_text = index_file.read_text(encoding="utf-8")
        found_targets: list[str] = []
        for match in LINK.finditer(index_text):
            destination, fragment = resolve_local_target(index_file, match.group(2))
            if destination is None:
                continue
            found_targets.append(unquote(match.group(2).split("#", 1)[0]))
            if not destination.exists():
                index_link_errors[language].append({"target": match.group(2)})
            elif fragment and destination.is_file() and destination.suffix.lower() == ".md":
                if fragment not in anchors_for(destination, anchor_cache):
                    broken_anchors[f"index_{language}"].append({"file": index_file.name, "target": match.group(2)})
        expected_targets = [f"{item['id']}/{item['id']}-{language}.md" for item in entries]
        index_targets_checked[language] = len(found_targets)
        counts = Counter(found_targets)
        for target in expected_targets:
            if counts[target] != 1:
                index_mapping_errors[language].append({"target": target, "occurrences": counts[target]})
        for target in counts:
            if target not in set(expected_targets):
                index_mapping_errors[language].append({"unexpected_target": target, "occurrences": counts[target]})

    fallback_path = ROOT / "work" / "04-translation" / "translation-fallbacks.json"
    fallbacks = json.loads(fallback_path.read_text(encoding="utf-8")) if fallback_path.exists() else []
    translation_coverage_path = ROOT / "reports" / "translation-coverage-review.json"
    translation_coverage = json.loads(translation_coverage_path.read_text(encoding="utf-8")) if translation_coverage_path.exists() else {"status": "not_run", "candidate_count": None}
    source_garbled = sum(1 for item in fallbacks if item.get("kind") in {"source_garbled_text", "source_garbled_math"})
    untranslated_fallbacks = sum(1 for item in fallbacks if item.get("kind") not in {"source_garbled_text"})
    passed = not any([
        missing_en, missing_zh, missing_pages, overlapping_pages,
        broken_links["en"], broken_links["zh"],
        broken_anchors["en"], broken_anchors["zh"], broken_anchors["index_en"], broken_anchors["index_zh"],
        mismatched_page_markers, index_link_errors["en"], index_link_errors["zh"],
        index_mapping_errors["en"], index_mapping_errors["zh"],
        translation_coverage.get("candidate_count") != 0,
    ])
    report = {
        "status": "passed" if passed else "incomplete",
        "source_pdf": data["source"],
        "source_pages": len(page_coverage),
        "chapter_count": len(entries),
        "english_chapter_files": len(entries) - len(missing_en),
        "chinese_chapter_files": len(entries) - len(missing_zh),
        "english_bytes": en_bytes,
        "chinese_bytes": zh_bytes,
        "missing_english": missing_en,
        "missing_chinese": missing_zh,
        "uncovered_pages": missing_pages,
        "overlapping_pages": overlapping_pages,
        "broken_english_links": broken_links["en"],
        "broken_chinese_links": broken_links["zh"],
        "broken_anchors": broken_anchors,
        "index_link_errors": index_link_errors,
        "index_mapping_errors": index_mapping_errors,
        "page_marker_mismatches": mismatched_page_markers,
        "navigation_checks": {
            "fragment_links_checked": fragment_links_checked,
            "english_index_chapter_targets": index_targets_checked["en"],
            "chinese_index_chapter_targets": index_targets_checked["zh"],
        },
        "translation_fallback_count": len(fallbacks),
        "source_garbled_fallback_count": source_garbled,
        "untranslated_fallback_count": untranslated_fallbacks,
        "source_replacement_characters": {"english": replacement_characters_en, "chinese": replacement_characters_zh},
        "content_review_required": True,
        "extraction_corruption_review_required": bool(replacement_characters_en or replacement_characters_zh or fallbacks),
        "human_translation_review_pending": True,
        "translation_coverage_audit": {
            "status": translation_coverage.get("status"),
            "candidate_count": translation_coverage.get("candidate_count"),
            "report": "reports/translation-coverage-review.json",
        },
        "chapters": chapter_results,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "chapters"}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
