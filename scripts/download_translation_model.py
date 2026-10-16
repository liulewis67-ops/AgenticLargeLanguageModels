"""Download a public Hugging Face model file in resumable HTTP range chunks."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "facebook/nllb-200-distilled-600M"
MODEL_FILE = "pytorch_model.bin"
MODEL_DIR = ROOT / "work" / "04-translation" / "models" / "nllb-200-distilled-600M"
PART_DIR = ROOT / "work" / "04-translation" / "model-parts" / "nllb-200-distilled-600M"
URL = f"https://huggingface.co/{MODEL_ID}/resolve/main/{MODEL_FILE}"
CHUNK_SIZE = 64 * 1024 * 1024
EXPECTED_SHA256 = "44e942c185321f9ee96e51eb31c9cf8c4f613215fa0c5a9035e9b71ede516fba"


def download_part(index: int, size: int) -> Path:
    start = index * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, size) - 1
    destination = PART_DIR / f"part-{index:03d}.bin"
    if destination.exists() and destination.stat().st_size == end - start + 1:
        return destination
    for attempt in range(6):
        try:
            with requests.get(
                URL,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, 180),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"HTTP {response.status_code} for bytes {start}-{end}")
                response.raise_for_status()
                written = 0
                with destination.with_suffix(".tmp").open("wb") as output:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            output.write(block)
                            written += len(block)
                if written != end - start + 1:
                    raise RuntimeError(f"short chunk {index}: {written} bytes")
                destination.with_suffix(".tmp").replace(destination)
                return destination
        except Exception:
            if attempt == 5:
                raise
    raise AssertionError("unreachable")


def main() -> int:
    PART_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    head = requests.head(URL, allow_redirects=True, timeout=30)
    head.raise_for_status()
    size = int(head.headers["content-length"])
    remote_sha = head.headers.get("etag", "").strip('"')
    if remote_sha != EXPECTED_SHA256:
        raise RuntimeError(f"unexpected model hash from Hub: {remote_sha}")
    count = math.ceil(size / CHUNK_SIZE)
    print(f"Downloading {MODEL_ID}/{MODEL_FILE}: {size:,} bytes in {count} chunks", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(download_part, index, size): index for index in range(count)}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            future.result()
            completed += 1
            print(f"downloaded chunk {index + 1}/{count} ({completed}/{count} complete)", flush=True)

    target = MODEL_DIR / MODEL_FILE
    digest = hashlib.sha256()
    with target.with_suffix(".assembling").open("wb") as output:
        for index in range(count):
            part = PART_DIR / f"part-{index:03d}.bin"
            with part.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
    actual = digest.hexdigest()
    if actual != EXPECTED_SHA256:
        target.with_suffix(".assembling").unlink(missing_ok=True)
        raise RuntimeError(f"assembled model checksum mismatch: {actual}")
    target.with_suffix(".assembling").replace(target)
    info = {
        "repository": MODEL_ID,
        "revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
        "file": "models/nllb-200-distilled-600M/pytorch_model.bin",
        "size_bytes": size,
        "sha256": actual,
        "source": "Hugging Face Hub",
    }
    (ROOT / "work" / "04-translation" / "nllb-model-info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    print(f"verified SHA256 {actual}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
