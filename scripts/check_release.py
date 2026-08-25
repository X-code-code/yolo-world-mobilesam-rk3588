#!/usr/bin/env python3
"""Fail when a repository checkout contains unsafe or non-release artifacts."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
SKIP_TEXT_SCAN = {
    Path("python/bpe_simple_vocab_16e6.txt"),
    Path("tests/tokenizer/clip_vocab.h"),
    Path("LICENSE"),
}
FORBIDDEN_SUFFIXES = {
    ".rknn",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".engine",
    ".tflite",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
}
REQUIRED_FILES = {
    Path("README.md"),
    Path("LICENSE"),
    Path("THIRD_PARTY_NOTICES.md"),
    Path("MODEL_SHA256SUMS"),
    Path("python/rknn_target_pipeline.py"),
    Path("python/realtime_target_ui.py"),
    Path("python/verify_dynamic_targets.py"),
    Path("docs/DEPLOYMENT.md"),
    Path("docs/MODELS.md"),
}
PRIVATE_PATTERNS = {
    "Windows user profile": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
    "development home path": re.compile("/home/" + "zxyy" + r"(?:/|\b)"),
    "private board address": re.compile(r"(?<![0-9])" + "172" + r"\.16\.11\.51(?![0-9])"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "hard-coded credential": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)"
        r"\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]"
    ),
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def iter_files() -> Iterable[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        yield path


def check_markdown_links(path: Path, text: str, errors: list[str]) -> None:
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().strip("<>").split()[0]
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        link_path = target.split("#", 1)[0]
        if not link_path:
            continue
        resolved = (path.parent / link_path).resolve()
        if not resolved.exists():
            errors.append(f"broken Markdown link in {path.relative_to(ROOT)}: {target}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for required in sorted(REQUIRED_FILES):
        if not (ROOT / required).is_file():
            errors.append(f"missing required file: {required}")

    for path in iter_files():
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if size >= 100 * 1024 * 1024:
            errors.append(f"file is at least 100 MiB: {relative} ({size} bytes)")
        elif size >= 50 * 1024 * 1024:
            warnings.append(f"large file: {relative} ({size} bytes)")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden release artifact: {relative}")
        if relative in SKIP_TEXT_SCAN or size > 2 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} found in {relative}")
        if path.suffix.lower() in {".md", ".markdown"}:
            check_markdown_links(path, text, errors)

    benchmark = ROOT / "benchmarks/lubancat4_rk3588_2026-08-25.json"
    try:
        json.loads(benchmark.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"invalid benchmark JSON: {exc}")

    checksum_lines = [
        line
        for line in (ROOT / "MODEL_SHA256SUMS").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checksum_pattern = re.compile(r"^[0-9a-f]{64}  model/[A-Za-z0-9_.-]+\.rknn$")
    if len(checksum_lines) != 5 or any(not checksum_pattern.fullmatch(line) for line in checksum_lines):
        errors.append("MODEL_SHA256SUMS must contain exactly five sha256sum-compatible entries")

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    file_count = sum(1 for _ in iter_files())
    print(f"release check passed: {file_count} files, no bundled models or private paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
