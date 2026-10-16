"""Translate section Markdown while preserving Markdown links and code."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from mintrans import BingTranslator

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CHAPTERS = ROOT / "config" / "chapters.json"
GLOSSARY = ROOT / "config" / "glossary.json"
CACHE = ROOT / "work" / "04-translation" / "bing-translation-cache.json"
MAX_REQUEST_CHARS = 850
MAX_GROUP_CHARS = 650
MASK = re.compile(r"ZXQ\d{4}XZ")


class MarkerLossError(RuntimeError):
    """Raised when the remote translator changes a protected Markdown token."""

FENCED = re.compile(r"^\s*([\u0060]{3,}|~{3,})")
LINK = re.compile(
    r"!? \[ (?:\\.|[^\]])* \] \( (?:[^()]|\([^()]*\))* \)",
    re.VERBOSE,
)
PROTECTED = re.compile(
    r"([\u0060]+[^\u0060]*[\u0060]+|(?<!\\)\${1,2}[^$]+(?<!\\)\${1,2}|\\\[.*?\\\]|\\\(.*?\\\)|<[^>]+>|https?://\S+)"
)
LIST_PREFIX = re.compile(r"^(\s*(?:[-*+]\s+|\d+[.)]\s+|>\s*))")
TABLE_RULE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
SENTENCES = re.compile(r"(?<=[.!?。！？])(\s+)")


class Translator:
    def __init__(self, glossary: dict[str, str], delay: float = 0.15) -> None:
        self.terms = {key.casefold(): value for key, value in glossary.items()}
        alternatives = sorted(glossary, key=len, reverse=True)
        self.term_pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(term) for term in alternatives) + r")(?!\w)",
            re.IGNORECASE,
        )
        self.protected_pattern = re.compile(
            rf"(?P<link>{LINK.pattern})|(?P<protected>{PROTECTED.pattern})",
            re.IGNORECASE | re.VERBOSE,
        )
        self.mask_pattern = re.compile(
            rf"(?P<link>{LINK.pattern})|(?P<protected>{PROTECTED.pattern})|(?P<term>{self.term_pattern.pattern})",
            re.IGNORECASE | re.VERBOSE,
        )
        self.delay = delay
        self.client = BingTranslator()
        self.cache: dict[str, str] = {}
        if CACHE.exists():
            self.cache = json.loads(CACHE.read_text(encoding="utf-8"))
        self.pending: list[str] = []
        self.request_count = 0
        self.line_count = 0
        fallback_report = ROOT / "work" / "04-translation" / "translation-fallbacks.json"
        self.translation_fallbacks: list[dict[str, str]] = (
            json.loads(fallback_report.read_text(encoding="utf-8"))
            if fallback_report.exists()
            else []
        )

    def _save_cache(self) -> None:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        temporary = CACHE.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(CACHE)

    def _translate_remote(self, text: str) -> str:
        if not text or not text.strip():
            return text
        if text in self.cache:
            return self.cache[text]
        for attempt in range(6):
            try:
                payload = self.client.translate(text, "en", "zh-Hans")
                if isinstance(payload, list) and payload:
                    payload = payload[0]
                translated = payload["translations"][0]["text"]
                if not translated:
                    raise ValueError("Bing Translator returned an empty or unchanged result")
                if translated == text and len(re.findall(r"\b[a-z]{3,}\b", text)) > 1:
                    raise ValueError("Bing Translator returned unchanged English prose")
                self.cache[text] = translated
                self.request_count += 1
                if self.request_count % 20 == 0:
                    self._save_cache()
                if self.delay:
                    time.sleep(self.delay)
                return translated
            except (KeyError, ValueError, TypeError, IndexError, RuntimeError) as exc:
                if attempt == 5:
                    raise RuntimeError(
                        f"Bing translation failed after 6 attempts for {text[:120]!r}"
                    ) from exc
                time.sleep(min(2**attempt, 20))
        raise AssertionError("unreachable")

    def translate_text(self, text: str, use_glossary: bool = True) -> str:
        if (
            text.count("\ufffd") >= 4
            and re.search(r"[=∑∏√∫]|\blog\b|\bexp\b", text)
            and len(re.findall(r"\b[A-Za-z]{2,}\b", text)) <= 3
        ):
            self.translation_fallbacks.append(
                {
                    "entry": getattr(self, "current_entry", ""),
                    "kind": "source_garbled_math",
                    "text": text[:240],
                    "reason": "Marker source contains multiple replacement characters in a formula",
                }
            )
            return text
        mask_pattern = self.mask_pattern if use_glossary else self.protected_pattern
        placeholders: dict[str, str] = {}

        def mask(match: re.Match[str]) -> str:
            token = f"ZXQ{len(placeholders):04d}XZ"
            value = (
                self.terms[match.group("term").casefold()]
                if "term" in match.re.groupindex and match.group("term") is not None
                else match.group(0)
            )
            placeholders[token] = value
            return token

        masked = mask_pattern.sub(mask, text)
        translated = self._translate_chunked(masked) if re.search(r"[A-Za-z]", MASK.sub("", masked)) else masked
        if any(token not in translated for token in placeholders):
            # If the service drops a placeholder, translate prose between protected spans.
            pieces: list[str] = []
            cursor = 0
            for match in mask_pattern.finditer(text):
                prose = text[cursor : match.start()]
                pieces.append(
                    self._translate_chunked(prose)
                    if re.search(r"[A-Za-z]", prose)
                    else prose
                )
                if "term" in match.re.groupindex and match.group("term") is not None:
                    pieces.append(self.terms[match.group("term").casefold()])
                else:
                    pieces.append(match.group(0))
                cursor = match.end()
            prose = text[cursor:]
            pieces.append(
                self._translate_chunked(prose)
                if re.search(r"[A-Za-z]", prose)
                else prose
            )
            translated = "".join(pieces)
        else:
            for token, value in placeholders.items():
                translated = translated.replace(token, value)
        translated = re.sub(r"\s+([，。；：！？、])", r"\1", translated)
        translated = re.sub(r"([，。；：！？、])\s+", r"\1", translated)
        return re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", translated)

    def _translate_chunked(self, text: str) -> str:
        if not text.strip():
            return text
        parts = SENTENCES.split(text)
        units = ["".join(parts[index : index + 2]) for index in range(0, len(parts), 2)]
        chunks: list[str] = []
        current = ""
        for unit in units:
            while len(unit) > MAX_REQUEST_CHARS:
                split_at = unit.rfind(" ", 0, MAX_REQUEST_CHARS + 1)
                if split_at <= 0:
                    split_at = MAX_REQUEST_CHARS
                prefix, unit = unit[:split_at], unit[split_at:]
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(prefix)
            if current and len(current) + len(unit) > MAX_REQUEST_CHARS:
                chunks.append(current)
                current = ""
            current += unit
        if current:
            chunks.append(current)
        return "".join(self._translate_remote(chunk) for chunk in chunks)

    def translate_inline(self, text: str) -> str:
        return self.translate_text(text)

    def translate_line(self, line: str) -> str:
        stripped = line.strip()
        if not stripped or stripped.startswith("<!--") or FENCED.match(line):
            return line
        if re.fullmatch(r"\s*\{\d+\}-{8,}\s*", line) or TABLE_RULE.fullmatch(line):
            return line
        heading = re.match(r"^(#{1,6}\s+)(.*)$", line)
        if heading:
            try:
                return heading.group(1) + self.translate_inline(heading.group(2))
            except MarkerLossError as exc:
                self.translation_fallbacks.append({"entry": getattr(self, "current_entry", ""), "kind": "heading", "text": line, "reason": str(exc)})
                return line
        if "|" in line:
            placeholders: dict[str, str] = {}
            parts: list[str] = []
            for char in line:
                if char == "|":
                    token = f"<QZXPIPE{len(placeholders):04d}Q>"
                    placeholders[token] = "|"
                    parts.append(token)
                else:
                    parts.append(char)
            try:
                translated = self.translate_text("".join(parts), use_glossary=False)
                if any(token not in translated for token in placeholders):
                    raise MarkerLossError("table delimiter was altered")
                for token, value in placeholders.items():
                    translated = translated.replace(token, value)
            except MarkerLossError as exc:
                cells = line.split("|")
                translated_cells = []
                for cell in cells:
                    if not cell.strip():
                        translated_cells.append(cell)
                        continue
                    try:
                        translated_cells.append(self.translate_inline(cell))
                    except MarkerLossError:
                        translated_cells.append(cell)
                        self.translation_fallbacks.append(
                            {"entry": getattr(self, "current_entry", ""), "kind": "table_cell", "text": cell, "reason": str(exc)}
                        )
                translated = "|".join(translated_cells)
            self.line_count += 1
            if self.line_count % 50 == 0:
                print(f"translated markdown lines {self.line_count}", flush=True)
            return translated
        prefix = LIST_PREFIX.match(line)
        if prefix:
            try:
                translated = prefix.group(1) + self.translate_inline(line[prefix.end() :])
            except MarkerLossError as exc:
                self.translation_fallbacks.append(
                    {"entry": getattr(self, "current_entry", ""), "kind": "list_item", "text": line, "reason": str(exc)}
                )
                return line
        else:
            try:
                translated = self.translate_inline(line)
            except MarkerLossError as exc:
                self.translation_fallbacks.append(
                    {"entry": getattr(self, "current_entry", ""), "kind": "line", "text": line, "reason": str(exc)}
                )
                return line
        self.line_count += 1
        if self.line_count % 50 == 0:
            print(f"translated markdown lines {self.line_count}", flush=True)
        return translated

    def translate_markdown(self, source: str) -> str:
        output: list[str] = []
        buffer: list[str] = []
        in_fence = False
        in_bibliography = False
        fence_char = ""
        fence_size = 0

        def flush_buffer() -> None:
            if not buffer:
                return
            translated = self.translate_inline("\n".join(buffer))
            output.append(translated)
            self.line_count += len(buffer)
            if self.line_count and self.line_count % 50 < len(buffer):
                print(f"translated markdown lines {self.line_count}", flush=True)
            buffer.clear()

        for line in source.splitlines():
            fence = FENCED.match(line)
            if fence:
                flush_buffer()
                marker = fence.group(1)
                if not in_fence:
                    in_fence = True
                    fence_char, fence_size = marker[0], len(marker)
                elif marker[0] == fence_char and len(marker) >= fence_size:
                    in_fence = False
                output.append(line)
            elif in_fence:
                output.append(line)
            elif in_bibliography:
                output.append(line)
            elif line.count("\ufffd") >= 4 and len(line) < 600:
                output.append(line)
                self.translation_fallbacks.append(
                    {
                        "entry": getattr(self, "current_entry", ""),
                        "kind": "source_garbled_text",
                        "text": line[:240],
                        "reason": "Marker source contains multiple replacement characters; preserved without translation",
                    }
                )
            elif line.strip() == "":
                if buffer and any(item.strip() for item in buffer):
                    if sum(map(len, buffer)) + 16 < MAX_GROUP_CHARS:
                        buffer.append(line)
                    else:
                        flush_buffer()
                        output.append(line)
                else:
                    flush_buffer()
                    output.append(line)
            elif (
                line.lstrip().startswith("<!--")
                or re.fullmatch(r"\s*\{\d+\}-{8,}\s*", line)
                or TABLE_RULE.fullmatch(line)
                or re.match(r"^#{1,6}\s+", line)
                or LIST_PREFIX.match(line)
                or line.lstrip().startswith(">")
                or "|" in line
            ):
                flush_buffer()
                translated = self.translate_line(line)
                output.append(translated)
                if re.match(r"^#{1,6}\s+(?:Bibliography|References|Works Cited)\s*$", line, re.IGNORECASE):
                    in_bibliography = True
            else:
                if buffer and sum(map(len, buffer)) + len(line) + 16 > MAX_GROUP_CHARS:
                    flush_buffer()
                buffer.append(line)
        flush_buffer()
        return "\n".join(output) + ("\n" if source.endswith("\n") else "")

    def translate_many(self, texts: list[str]) -> list[str]:
        results = []
        for index, text in enumerate(texts, start=1):
            results.append(self.translate_text(text))
            if index % 20 == 0 or index == len(texts):
                print(f"translated headings {index}/{len(texts)}", flush=True)
        self._save_cache()
        return results


def write_chinese_index(translator: Translator, entries: list[dict]) -> None:
    titles = [entry.get("title_zh") for entry in entries]
    missing = [index for index, title in enumerate(titles) if not title]
    if missing:
        translated = translator.translate_many([entries[index]["title"] for index in missing])
        for index, title in zip(missing, translated):
            titles[index] = title
    rows = [
        "# 智能体大语言模型",
        "",
        "本索引链接到按章节保存的简体中文 Markdown 译文。",
        "",
    ]
    for entry, title in zip(entries, titles):
        relative = f"{entry['id']}/{entry['id']}-zh.md"
        rows.append(
            f"- [{title}]({relative})（PDF 第 {entry['page_start']}–{entry['page_end']} 页）"
        )
    (DOCS / "index-zh.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", help="entry ID; repeat to select several")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--request-delay", type=float, default=0.15)
    args = parser.parse_args()
    if args.request_delay < 0:
        parser.error("--request-delay cannot be negative")

    data = json.loads(CHAPTERS.read_text(encoding="utf-8"))
    glossary = json.loads(GLOSSARY.read_text(encoding="utf-8"))
    entries = [
        entry
        for entry in data["entries"]
        if not args.entry or entry["id"] in args.entry
    ]
    found = {entry["id"] for entry in entries}
    unknown = set(args.entry or ()) - found
    if unknown:
        parser.error(f"unknown entry IDs: {', '.join(sorted(unknown))}")

    translator = Translator(glossary, delay=args.request_delay)
    for entry in entries:
        folder = DOCS / entry["id"]
        source = folder / f"{entry['id']}-en.md"
        target = folder / f"{entry['id']}-zh.md"
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists() and not args.force:
            print(f"skip existing {target.relative_to(ROOT)}")
            continue
        print(
            f"translate {entry['id']} PDF pages "
            f"{entry['page_start']}-{entry['page_end']}",
            flush=True,
        )
        translator.current_entry = entry["id"]
        translator.translation_fallbacks = [
            item
            for item in translator.translation_fallbacks
            if item.get("entry") != entry["id"]
        ]
        english = source.read_text(encoding="utf-8")
        chinese = translator.translate_markdown(english)
        target.write_text(chinese, encoding="utf-8")
        translator._save_cache()
        fallback_report = ROOT / "work" / "04-translation" / "translation-fallbacks.json"
        fallback_report.write_text(
            json.dumps(translator.translation_fallbacks, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    write_chinese_index(translator, data["entries"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
