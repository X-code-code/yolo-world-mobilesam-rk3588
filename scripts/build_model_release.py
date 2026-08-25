#!/usr/bin/env python3
"""Build the deterministic, license-carrying MobileSAM GitHub Release archive."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import BinaryIO


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "mobilesam-rk3588-fp16-v0.1.0"
FIXED_TIMESTAMP = (2026, 8, 25, 0, 0, 0)
MODEL_NAMES = ("mobilesam_encoder_fp16.rknn", "mobilesam_decoder_fp16.rknn")
BUNDLE_KEY = "mobilesam-rk3588-fp16"


class ReleaseBuildError(RuntimeError):
    """Raised when source artifacts do not match the release manifest."""


def sha256_stream(handle: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def zip_info(relative: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def add_bytes(archive: zipfile.ZipFile, relative: str, payload: bytes) -> None:
    archive.writestr(zip_info(relative), payload)


def add_file(archive: zipfile.ZipFile, relative: str, source: Path) -> None:
    with source.open("rb") as source_handle, archive.open(zip_info(relative), "w") as target_handle:
        shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)


def validate_models(model_dir: Path, manifest: dict) -> None:
    for name in MODEL_NAMES:
        path = model_dir / name
        if not path.is_file():
            raise ReleaseBuildError(f"missing model: {path}")
        entry = manifest["models"][name]
        if path.stat().st_size != int(entry["bytes"]):
            raise ReleaseBuildError(f"size mismatch for {name}")
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise ReleaseBuildError(f"SHA-256 mismatch for {name}: {actual}")


def verify_bundle_manifest(output: Path, result: dict[str, object], manifest: dict) -> None:
    try:
        bundle = manifest["release"]["bundles"][BUNDLE_KEY]
    except (KeyError, TypeError) as exc:
        raise ReleaseBuildError(f"release manifest is missing bundle {BUNDLE_KEY}") from exc
    mismatches = []
    if bundle.get("asset") != output.name:
        mismatches.append(f"asset={output.name!r}")
    if bundle.get("bytes") != result["bytes"]:
        mismatches.append(f"bytes={result['bytes']}")
    if bundle.get("sha256") != result["sha256"]:
        mismatches.append(f"sha256={result['sha256']}")
    if mismatches:
        raise ReleaseBuildError(
            "archive does not match MODEL_RELEASES.json; update the reviewed manifest and "
            f"rerun the strict build ({', '.join(mismatches)})"
        )


def build_archive(
    model_dir: Path,
    output: Path,
    force: bool = False,
    *,
    verify_manifest: bool = True,
) -> dict[str, object]:
    manifest_path = ROOT / "MODEL_RELEASES.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_models(model_dir, manifest)
    if output.exists() and not force:
        raise ReleaseBuildError(f"output already exists: {output}; pass --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.part")
    if temporary.exists():
        temporary.unlink()

    checksums = "".join(
        f"{manifest['models'][name]['sha256']}  model/{name}\n" for name in MODEL_NAMES
    )
    readme = "# MobileSAM RK3588 FP16 model artifacts\n\n"
    readme += "This package contains the two MobileSAM RKNN artifacts verified on LubanCat 4 / RK3588.\n\n"
    readme += "- Built with RKNN Toolkit2 2.3.2 for target `rk3588`.\n"
    readme += "- Converted from the ONNX inputs identified in `MODEL_PROVENANCE.json`.\n"
    readme += "- Verify with `sha256sum -c SHA256SUMS` from this directory.\n"
    readme += "- These two files only provide segmentation; the dynamic YOLO-World pipeline also requires locally converted CLIP and YOLO models.\n\n"
    readme += "See `MODEL_LICENSES.md` and `LICENSE` before redistribution. No upstream endorsement or warranty is provided.\n"

    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name in MODEL_NAMES:
                add_file(archive, f"model/{name}", model_dir / name)
            add_bytes(archive, "SHA256SUMS", checksums.encode("utf-8"))
            add_bytes(archive, "README.md", readme.encode("utf-8"))
            for name in ("LICENSE", "MODEL_LICENSES.md", "MODEL_PROVENANCE.json"):
                add_file(archive, name, ROOT / name)
        result: dict[str, object] = {
            "archive": str(output),
            "bytes": temporary.stat().st_size,
            "sha256": sha256_file(temporary),
            "models": list(MODEL_NAMES),
        }
        if verify_manifest:
            verify_bundle_manifest(output, result, manifest)
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--candidate",
        action="store_true",
        help=(
            "report a candidate hash without matching MODEL_RELEASES.json; "
            "never upload until a later strict build passes"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_archive(
            args.model_dir,
            args.output,
            force=args.force,
            verify_manifest=not args.candidate,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ReleaseBuildError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
