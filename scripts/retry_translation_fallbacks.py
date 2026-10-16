"""Retry only Markdown lines that the translation backend previously rejected."""

from __future__ import annotations

import json
from pathlib import Path

from translate_sections import ROOT, Translator

FALLBACKS = ROOT / "work" / "04-translation" / "translation-fallbacks.json"
GLOSSARY = ROOT / "config" / "glossary.json"


def save_fallbacks(items: list[dict[str, str]]) -> None:
    temporary = FALLBACKS.with_suffix(".tmp")
    temporary.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(FALLBACKS)


def main() -> int:
    fallbacks: list[dict[str, str]] = json.loads(FALLBACKS.read_text(encoding="utf-8"))
    if not fallbacks:
        print("No translation fallbacks to retry.")
        return 0
    glossary = json.loads(GLOSSARY.read_text(encoding="utf-8"))
    translator = Translator(glossary)
    remaining = list(fallbacks)
    for item in fallbacks:
        entry_id = item["entry"]
        source_path = ROOT / "docs" / entry_id / f"{entry_id}-en.md"
        target_path = ROOT / "docs" / entry_id / f"{entry_id}-zh.md"
        source_text = source_path.read_text(encoding="utf-8")
        target_text = target_path.read_text(encoding="utf-8")
        original = item["text"]
        if original not in source_text:
            item["retry_error"] = "Original fallback line no longer matches the English source."
            continue
        if original not in target_text:
            item["retry_error"] = "Fallback line is not present verbatim in the Chinese file."
            continue
        translator.current_entry = entry_id
        try:
            translated = (
                translator.translate_line(original)
                if item.get("kind") in {"list_item", "heading"}
                else translator.translate_inline(original)
            )
        except Exception as exc:  # Keep the original line and report the backend error.
            item["retry_error"] = f"{type(exc).__name__}: {exc}"
            continue
        if not translated.strip() or translated == original:
            item["retry_error"] = "Retry returned unchanged English."
            continue
        target_path.write_text(target_text.replace(original, translated, 1), encoding="utf-8")
        remaining.remove(item)
        save_fallbacks(remaining)
        translator._save_cache()
        print(f"translated fallback in {entry_id}", flush=True)
    save_fallbacks(remaining)
    print(json.dumps({"retried": len(fallbacks) - len(remaining), "remaining": len(remaining)}, indent=2))
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
