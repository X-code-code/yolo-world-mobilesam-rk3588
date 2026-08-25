#!/usr/bin/env python3
"""Fail when a repository checkout contains unsafe or non-release artifacts."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
SKIP_TEXT_SCAN = {
    Path("python/bpe_simple_vocab_16e6.txt"),
    Path("tests/tokenizer/clip_vocab.h"),
    Path("LICENSE"),
    Path("LICENSES/OpenAI-CLIP-MIT.txt"),
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
    ".zip",
}
REQUIRED_FILES = {
    Path("VERSION"),
    Path("CHANGELOG.md"),
    Path("README.md"),
    Path("LICENSE"),
    Path("THIRD_PARTY_NOTICES.md"),
    Path("MODEL_LICENSES.md"),
    Path("MODEL_PROVENANCE.json"),
    Path("MODEL_RELEASES.json"),
    Path("MODEL_SHA256SUMS"),
    Path("conversion/download_onnx.py"),
    Path("conversion/convert_models.py"),
    Path("conversion/VERIFIED_RUN.md"),
    Path("python/rknn_target_pipeline.py"),
    Path("python/realtime_target_ui.py"),
    Path("python/verify_dynamic_targets.py"),
    Path("docs/DEPLOYMENT.md"),
    Path("docs/MODELS.md"),
    Path("docs/releases/v0.1.0.md"),
    Path("scripts/download_models.py"),
    Path("scripts/build_model_release.py"),
}
PRIVATE_PATTERNS = {
    "Windows user profile": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
    "development home path": re.compile(
        r"/home/(?!cat(?:/|\b))[A-Za-z0-9._-]+(?:/|\b)", re.IGNORECASE
    ),
    "private RFC1918 address": re.compile(
        r"(?<![0-9])(?:"
        r"10(?:\.[0-9]{1,3}){3}|"
        r"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2}|"
        r"192\.168(?:\.[0-9]{1,3}){2}"
        r")(?![0-9])"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "hard-coded credential": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)"
        r"\s*[:=]\s*['\"][^'\"\r\n]{8,}['\"]"
    ),
}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def iter_files() -> Iterable[Path]:
    """Yield tracked and non-ignored candidate files, excluding local model/data outputs."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"failed to enumerate Git release candidates: {exc}") from exc
    for raw_relative in result.stdout.split(b"\0"):
        if not raw_relative:
            continue
        relative = Path(raw_relative.decode("utf-8"))
        path = ROOT / relative
        if path.is_file():
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

    try:
        release_files = list(iter_files())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for path in release_files:
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

    checksum_by_name: dict[str, str] = {}
    for line in checksum_lines:
        if checksum_pattern.fullmatch(line):
            digest, relative = line.split("  ", 1)
            checksum_by_name[Path(relative).name] = digest

    try:
        provenance = json.loads((ROOT / "MODEL_PROVENANCE.json").read_text(encoding="utf-8"))
        releases = json.loads((ROOT / "MODEL_RELEASES.json").read_text(encoding="utf-8"))
        version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        if not re.fullmatch(
            r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)",
            version,
        ):
            errors.append("VERSION must be a three-component semantic version")
        if provenance.get("schema_version") != 1:
            errors.append("MODEL_PROVENANCE.json has unsupported schema_version")
        if releases.get("schema_version") != 1:
            errors.append("MODEL_RELEASES.json has unsupported schema_version")
        if releases.get("repository") != "X-code-code/yolo-world-mobilesam-rk3588":
            errors.append("MODEL_RELEASES.json repository is not the public project")
        release = releases.get("release", {})
        if release.get("tag") != f"v{version}":
            errors.append("MODEL_RELEASES.json tag does not match VERSION")
        if not str(release.get("download_base_url", "")).endswith(f"/releases/download/v{version}"):
            errors.append("MODEL_RELEASES.json download URL does not match VERSION")
        outputs = provenance.get("outputs", {})
        release_models = releases.get("models", {})
        if set(outputs) != set(checksum_by_name):
            errors.append("MODEL_PROVENANCE.json outputs do not match MODEL_SHA256SUMS")
        if set(release_models) != set(checksum_by_name):
            errors.append("MODEL_RELEASES.json models do not match MODEL_SHA256SUMS")
        for name, expected_hash in checksum_by_name.items():
            output = outputs.get(name, {})
            release_model = release_models.get(name, {})
            if output.get("sha256") != expected_hash:
                errors.append(f"provenance hash mismatch for {name}")
            if release_model.get("sha256") != expected_hash:
                errors.append(f"release manifest hash mismatch for {name}")
            if output.get("bytes") != release_model.get("bytes"):
                errors.append(f"model size mismatch across manifests for {name}")

        statuses = {
            name: entry.get("distribution", {}).get("status")
            for name, entry in release_models.items()
        }
        expected_released = {
            "mobilesam_encoder_fp16.rknn",
            "mobilesam_decoder_fp16.rknn",
        }
        if {name for name, status in statuses.items() if status == "release_bundle"} != expected_released:
            errors.append("only the two MobileSAM artifacts may be public Release assets in v0.1.0")
        if any(status not in {"release_bundle", "convert_locally"} for status in statuses.values()):
            errors.append("MODEL_RELEASES.json contains an unknown distribution status")

        bundles = releases.get("release", {}).get("bundles", {})
        if set(bundles) != {"mobilesam-rk3588-fp16"}:
            errors.append("v0.1.0 must define exactly one MobileSAM Release bundle")
        sha_pattern = re.compile(r"^[0-9a-f]{64}$")
        for bundle_name, bundle in bundles.items():
            if not bundle.get("asset", "").endswith(".zip"):
                errors.append(f"release bundle must be a zip: {bundle_name}")
            if not isinstance(bundle.get("bytes"), int) or bundle["bytes"] <= 0:
                errors.append(f"release bundle has invalid byte size: {bundle_name}")
            if not sha_pattern.fullmatch(str(bundle.get("sha256", ""))):
                errors.append(f"release bundle has invalid SHA-256: {bundle_name}")
        for name, entry in release_models.items():
            distribution = entry.get("distribution", {})
            if distribution.get("status") == "release_bundle":
                bundle_name = distribution.get("bundle")
                if bundle_name not in bundles:
                    errors.append(f"model references missing Release bundle: {name}")
                if distribution.get("member") != f"model/{name}":
                    errors.append(f"Release bundle member mismatch for {name}")
            elif distribution.get("status") == "convert_locally":
                instructions = distribution.get("instructions")
                if not instructions or not (ROOT / instructions).is_file():
                    errors.append(f"missing local-conversion instructions for {name}")
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, KeyError) as exc:
        errors.append(f"invalid model manifest JSON: {exc}")

    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in sorted(set(errors)):
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    file_count = len(release_files)
    print(
        f"release check passed: {file_count} files, model manifests consistent, "
        "no tracked model binaries or private paths"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
