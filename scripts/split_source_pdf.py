"""Create temporary per-section PDFs while preserving source page ranges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "AgenticLargeLanguageModels.pdf"
CHAPTERS = ROOT / "config" / "chapters.json"
SPLITS = ROOT / "work" / "02-extraction" / "input-pdfs"
MANIFEST = ROOT / "work" / "02-extraction" / "split-manifest.json"


reader = PdfReader(str(SOURCE), strict=False)
config = json.loads(CHAPTERS.read_text(encoding="utf-8"))
SPLITS.mkdir(parents=True, exist_ok=True)
manifest = {
    "source": SOURCE.name,
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "source_pages": len(reader.pages),
    "page_numbers": "original PDF pages are 1-based; split PDF pages restart at 1",
    "entries": [],
}

for entry in config["entries"]:
    start = entry["page_start"]
    end = entry["page_end"]
    if end < start:
        continue
    path = SPLITS / f"{entry['id']}.pdf"
    writer = PdfWriter()
    for page_index in range(start - 1, end):
        writer.add_page(reader.pages[page_index])
    writer.add_metadata(
        {
            "/Title": entry["title"],
            "/Subject": f"Source PDF pages {start}-{end}",
        }
    )
    with path.open("wb") as stream:
        writer.write(stream)
    manifest["entries"].append(
        {
            **entry,
            "split_pdf": path.name,
            "split_page_start": 1,
            "split_page_end": end - start + 1,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )

MANIFEST.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + chr(10), encoding="utf-8"
)
print(f"Wrote {len(manifest['entries'])} section PDFs to {SPLITS}")
print(f"Manifest: {MANIFEST}")
