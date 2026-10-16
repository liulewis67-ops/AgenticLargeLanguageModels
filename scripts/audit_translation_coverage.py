"""Find long English prose lines copied unchanged into paired Chinese Markdown."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CHAPTERS = ROOT / "config" / "chapters.json"
REPORT = ROOT / "reports" / "translation-coverage-review.json"
BIBLIOGRAPHY = re.compile(
    r"^#{1,6}\s*(?:<span[^>]*></span>\s*)?(?:Bibliography|References|Works Cited)\b",
    re.IGNORECASE,
)
WORD = re.compile(r"[A-Za-z]{3,}")
CJK = re.compile(r"[\u3400-\u9fff]")


def main() -> int:
    entries = json.loads(CHAPTERS.read_text(encoding="utf-8"))["entries"]
    candidates: list[dict[str, object]] = []
    for entry in entries:
        chapter_id = entry["id"]
        folder = DOCS / chapter_id
        english = (folder / f"{chapter_id}-en.md").read_text(encoding="utf-8").splitlines()
        chinese_lines = set((folder / f"{chapter_id}-zh.md").read_text(encoding="utf-8").splitlines())
        in_code = False
        in_bibliography = False
        for line_number, line in enumerate(english, start=1):
            if line.lstrip().startswith("```"):
                in_code = not in_code
            if BIBLIOGRAPHY.match(line):
                in_bibliography = True
            if in_code or in_bibliography or line not in chinese_lines:
                continue
            prose = re.sub(r"!?\[[^\]]*\]\([^)]*\)|https?://\S+|<[^>]*>|`[^`]*`|\$[^$]*\$", " ", line)
            words = WORD.findall(prose)
            if len(words) >= 18 and len(" ".join(words)) > 130 and len(CJK.findall(prose)) < 6:
                candidates.append(
                    {
                        "chapter": chapter_id,
                        "english_line": line_number,
                        "word_count": len(words),
                        "text_excerpt": prose[:180],
                    }
                )
    report = {
        "status": "passed" if not candidates else "review_required",
        "method": "Exact long-prose line comparison between English and Chinese chapter pairs; code and bibliography sections excluded.",
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
