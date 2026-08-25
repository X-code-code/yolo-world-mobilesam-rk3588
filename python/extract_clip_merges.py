#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract the embedded CLIP BPE merges text from clip_vocab.h."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


EXPECTED_HEADER = "#version: 0.2"
EXPECTED_MERGE_COUNT = 48_894


def extract_bytes(header_path: Path) -> bytes:
    source = header_path.read_text(encoding="ascii")
    match = re.search(
        r"RKNN_DEMO_CLIP_VOCAB_BIN_BUF\[\]\s*=\s*\{(.*?)\};",
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"CLIP vocab byte array not found in {header_path}")

    hex_bytes = re.findall(r"0x([0-9a-fA-F]{2})", match.group(1))
    if not hex_bytes:
        raise ValueError(f"CLIP vocab byte array is empty in {header_path}")
    return bytes(int(value, 16) for value in hex_bytes)


def validate_merges(data: bytes) -> None:
    text = data.decode("utf-8")
    if not data.endswith(b"\n"):
        raise ValueError("merges asset must end with a newline")
    lines = text.splitlines()
    if not lines or lines[0] != EXPECTED_HEADER:
        raise ValueError(f"unexpected merges header: {lines[:1]!r}")
    merge_count = len(lines) - 1
    if merge_count != EXPECTED_MERGE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_MERGE_COUNT} merges, found {merge_count}"
        )


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    examples_dir = script_dir.parents[1]
    default_header = examples_dir / "yolo_world/cpp/tokenizer/clip_vocab.h"
    default_output = script_dir / "bpe_simple_vocab_16e6.txt"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, default=default_header)
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()

    data = extract_bytes(args.header)
    validate_merges(data)
    args.output.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(
        f"wrote {len(data)} bytes and {EXPECTED_MERGE_COUNT} merges to "
        f"{args.output} (sha256={digest})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
