"""Locate Marker replacement characters and attach source PDF page text."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "config" / "chapters.json"
SOURCE = ROOT / "AgenticLargeLanguageModels.pdf"
REPORT = ROOT / "reports" / "extraction-quality-review.json"
DIVIDER = re.compile(r"^\{(\d+)\}-{8,}\s*$")
PAGE_ANCHOR = re.compile(r'<span id="page-(\d+)-\d+"></span>')


def main() -> int:
    chapter_data = json.loads(CHAPTERS.read_text(encoding="utf-8"))
    reader = PdfReader(SOURCE)
    entries = []
    for chapter in chapter_data["entries"]:
        chapter_id = chapter["id"]
        markdown = ROOT / "docs" / chapter_id / f"{chapter_id}-en.md"
        if not markdown.exists():
            continue
        current_local_page: int | None = None
        issues = []
        for line_number, line in enumerate(markdown.read_text(encoding="utf-8").splitlines(), start=1):
            divider = DIVIDER.match(line)
            if divider:
                current_local_page = int(divider.group(1))
                continue
            anchors = PAGE_ANCHOR.findall(line)
            if anchors:
                current_local_page = int(anchors[-1])
            count = line.count("\ufffd")
            if not count:
                continue
            physical_page = (
                int(chapter["page_start"]) + current_local_page
                if current_local_page is not None
                else None
            )
            source_text = ""
            if physical_page is not None and 1 <= physical_page <= len(reader.pages):
                source_text = reader.pages[physical_page - 1].extract_text() or ""
            issues.append(
                {
                    "markdown_line": line_number,
                    "replacement_characters": count,
                    "physical_pdf_page": physical_page,
                    "markdown_text": line,
                    "source_pdf_text": source_text,
                }
            )
        if issues:
            entries.append({"id": chapter_id, "issues": issues})

    report = {
        "source_pdf": chapter_data["source"],
        "source_pages": chapter_data["page_count"],
        "method": "Map Marker page divider/anchor to physical PDF page; attach pypdf text for visual/manual comparison.",
        "warning": "Page text is contextual evidence, not a character-level alignment. Verify formulas against the rendered source page before editing.",
        "chapters_with_issues": len(entries),
        "replacement_characters": sum(
            issue["replacement_characters"] for chapter in entries for issue in chapter["issues"]
        ),
        "chapters": entries,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "chapters"
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
