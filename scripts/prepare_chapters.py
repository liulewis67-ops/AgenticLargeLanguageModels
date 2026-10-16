"""Build a stable chapter/page map from PDF bookmarks."""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "AgenticLargeLanguageModels.pdf"
OUTPUT = ROOT / "config" / "chapters.json"


def slugify(value: str) -> str:
    value = value.replace("*", "")
    value = value.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "section"


def flatten(items: list, depth: int = 0):
    for item in items:
        if isinstance(item, list):
            yield from flatten(item, depth + 1)
        else:
            yield {
                "page": reader.get_destination_page_number(item) + 1,
                "title": item.title,
                "depth": depth,
            }


reader = PdfReader(str(SOURCE), strict=False)
marks = list(flatten(reader.outline))
roots = [mark for mark in marks if mark["depth"] == 0]
entries: list[dict] = [
    {
        "id": "00-front-matter",
        "title": "Front Matter",
        "type": "front_matter",
        "page_start": 1,
        "page_end": roots[0]["page"] - 1,
        "parent": None,
    }
]
chapter_number = 0

for root_index, root in enumerate(roots):
    root_end = (
        roots[root_index + 1]["page"] - 1
        if root_index + 1 < len(roots)
        else len(reader.pages)
    )
    children = [
        mark
        for mark in marks
        if mark["depth"] == 1
        and root["page"] <= mark["page"] <= root_end
    ]

    if root["title"].casefold() == "introduction":
        chapter_number += 1
        entries.append(
            {
                "id": f"{chapter_number:02d}-introduction",
                "title": root["title"],
                "type": "chapter",
                "page_start": root["page"],
                "page_end": root_end,
                "parent": "00-front-matter",
            }
        )
        continue

    part_id = f"part-{slugify(root['title'])}"
    part_start = children[0]["page"] if children else root_end + 1
    entries.append(
        {
            "id": part_id,
            "title": root["title"],
            "type": "part",
            "page_start": root["page"],
            "page_end": part_start - 1,
            "parent": None,
        }
    )
    for child_index, child in enumerate(children):
        chapter_number += 1
        next_page = (
            children[child_index + 1]["page"]
            if child_index + 1 < len(children)
            else root_end + 1
        )
        entries.append(
            {
                "id": f"{chapter_number:02d}-{slugify(child['title'])}",
                "title": child["title"],
                "type": "chapter",
                "page_start": child["page"],
                "page_end": next_page - 1,
                "parent": part_id,
            }
        )

# Keep the bibliography boundary explicit for review after extraction.
final_chapter = next(entry for entry in reversed(entries) if entry["type"] == "chapter")
if final_chapter["page_end"] >= 469:
    final_chapter["page_end"] = 468
    entries.append(
        {
            "id": "90-references",
            "title": "Bibliography",
            "type": "references",
            "page_start": 469,
            "page_end": len(reader.pages),
            "parent": None,
        }
    )

payload = {
    "source": SOURCE.name,
    "page_numbering": "1-based physical PDF pages; Marker page_range uses zero-based pages",
    "page_count": len(reader.pages),
    "entries": entries,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8")
print(f"Wrote {len(entries)} entries to {OUTPUT}")
for entry in entries:
    print(
        f"{entry['page_start']:>3}-{entry['page_end']:<3} "
        f"{entry['type']:<12} {entry['id']} {entry['title']}"
    )
