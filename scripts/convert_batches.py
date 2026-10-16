"""Convert selected chapter ranges with Marker, in resumable page batches."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "AgenticLargeLanguageModels.pdf"
CHAPTERS = ROOT / "config" / "chapters.json"
WORK = ROOT / "work" / "02-extraction"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", help="entry ID; repeat to select several")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--mode", choices=("fast", "balanced"), default="fast")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    chapter_data = json.loads(CHAPTERS.read_text(encoding="utf-8"))
    selected = [
        entry
        for entry in chapter_data["entries"]
        if not args.entry or entry["id"] in args.entry
    ]
    found_ids = {entry["id"] for entry in selected}
    unknown = set(args.entry or ()) - found_ids
    if unknown:
        parser.error(f"unknown entry IDs: {', '.join(sorted(unknown))}")

    for entry in selected:
        start = entry["page_start"]
        end = entry["page_end"]
        if end < start:
            continue
        for batch_number, page_start in enumerate(
            range(start, end + 1, args.batch_size), start=1
        ):
            page_end = min(page_start + args.batch_size - 1, end)
            batch_dir = WORK / entry["id"] / f"batch-{batch_number:03d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            output_files = list(batch_dir.glob("*.md"))
            marker_done = bool(output_files) and not args.force
            status_path = batch_dir / "status.json"
            if marker_done:
                print(f"skip existing {entry['id']} pages {page_start}-{page_end}")
                continue

            # chapters.json uses 1-based physical pages; Marker page_range is
            # zero-based and inclusive.
            marker_range = f"{page_start - 1}-{page_end - 1}"
            command = [
                "uv",
                "run",
                "--locked",
                "--no-sync",
                "marker_single",
                str(SOURCE),
                "--page_range",
                marker_range,
                "--output_format",
                "markdown",
                "--output_dir",
                str(batch_dir),
                "--mode",
                args.mode,
                "--disable_ocr",
                "--paginate_output",
            ]
            print(f"convert {entry['id']} pages {page_start}-{page_end}", flush=True)
            started = datetime.now(timezone.utc).isoformat()
            result = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            (batch_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
            (batch_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
            status = {
                "entry": entry["id"],
                "pdf_pages": [page_start, page_end],
                "marker_page_range": marker_range,
                "mode": args.mode,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "returncode": result.returncode,
            }
            status_path.write_text(
                json.dumps(status, indent=2) + "\n", encoding="utf-8"
            )
            if result.returncode:
                print(result.stderr[-3000:])
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
