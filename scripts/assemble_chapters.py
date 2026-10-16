"""Assemble Marker outputs into per-section English Markdown documents."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = ROOT / "config" / "chapters.json"
MANIFEST = ROOT / "work" / "02-extraction" / "split-manifest.json"
MARKER_OUTPUT = ROOT / "work" / "02-extraction" / "marker-output"
DOCS = ROOT / "docs"


def folder_for(entry: dict) -> Path:
    return DOCS / entry["id"]


chapter_data = json.loads(CHAPTERS.read_text(encoding="utf-8"))
split_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
manifest_by_id = {entry["id"]: entry for entry in split_manifest["entries"]}
index_rows: list[tuple[str, str]] = []

for entry in chapter_data["entries"]:
    if entry["id"] not in manifest_by_id:
        continue
    output_folder = MARKER_OUTPUT / entry["id"]
    source_md = output_folder / f"{entry['id']}.md"
    if not source_md.exists():
        raise FileNotFoundError(f"Missing Marker output: {source_md}")

    target_folder = folder_for(entry)
    assets = target_folder / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    for image in output_folder.iterdir():
        if image.is_file() and image.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            shutil.copy2(image, assets / image.name)

    content = source_md.read_text(encoding="utf-8")
    content = re.sub(
        r"\]\((_page_[^)]+)\)",
        r"](assets/\1)",
        content,
    )
    split = manifest_by_id[entry["id"]]
    source_note = (
        f"<!-- Source: {chapter_data['source']}; physical PDF pages "
        f"{entry['page_start']}-{entry['page_end']}; Marker split PDF pages "
        f"1-{split['split_page_end']}. -->\n\n"
    )
    target_file = target_folder / f"{entry['id']}-en.md"
    target_file.write_text(source_note + content, encoding="utf-8")
    index_rows.append((entry["title"], target_file.relative_to(DOCS).as_posix()))

index = [
    "# Agentic Large Language Models",
    "",
    "English Markdown extracted from the source PDF with Marker.",
    "",
    f"Source: {chapter_data['source']} ({chapter_data['page_count']} pages).",
    "",
]
last_parent = object()
for entry, (title, relative_path) in zip(chapter_data["entries"], index_rows):
    parent = entry.get("parent")
    if parent != last_parent:
        index.append("")
        last_parent = parent
    index.append(f"- [{title}]({relative_path}) (PDF pages {entry['page_start']}-{entry['page_end']})")

(DOCS / "index-en.md").write_text("\n".join(index) + "\n", encoding="utf-8")
print(f"Assembled {len(index_rows)} English Markdown sections under {DOCS}")
