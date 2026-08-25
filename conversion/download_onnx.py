#!/usr/bin/env python3
"""Download the four pinned ONNX inputs used by the RK3588 conversion.

Only Python's standard library is used.  Every payload is downloaded to a
temporary file, checked for both byte size and SHA-256, and then moved into
place.  Existing files are never replaced unless ``--force`` is explicit.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Mapping, Optional, Sequence
from urllib.request import Request, urlopen


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_RECORD_SCHEMA = 1
DEFAULT_CHUNK_SIZE = 1024 * 1024


class ArtifactError(RuntimeError):
    """Base class for download and verification failures."""


class ArtifactVerificationError(ArtifactError):
    """Raised when a local or downloaded artifact does not match its pin."""


@dataclass(frozen=True)
class OnnxArtifact:
    key: str
    filename: str
    url: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DownloadResult:
    key: str
    path: Path
    status: str
    size_bytes: int
    sha256: str
    url: str

    def as_record(self) -> dict[str, object]:
        return {
            "key": self.key,
            "filename": self.path.name,
            "status": self.status,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "source_url": self.url,
        }


# URLs are the ones in the corresponding rknn_model_zoo download_model.sh
# files.  The filenames intentionally match the locally verified deployment.
ONNX_ARTIFACTS: tuple[OnnxArtifact, ...] = (
    OnnxArtifact(
        key="clip_text",
        filename="clip_text.onnx",
        url=(
            "https://ftrg.zbox.filez.com/v2/delivery/data/"
            "95f00b0fc900458ba134f8b180b3f7a1/examples/clip/clip_text.onnx"
        ),
        size_bytes=254_185_587,
        sha256="9cd686c57d874b4f7ec8e539fecc9ac2367c4252be18d662cd7bd23e872fddee",
    ),
    OnnxArtifact(
        key="yolo_world_v2s",
        filename="yolo_world_v2s.onnx",
        url=(
            "https://ftrg.zbox.filez.com/v2/delivery/data/"
            "95f00b0fc900458ba134f8b180b3f7a1/examples/yolo_world/yolo_world_v2s.onnx"
        ),
        size_bytes=51_068_366,
        sha256="5c057c5c8c9354a261a4e2d217abc8b8a50bcbac738623f3e914fc942941cd8c",
    ),
    OnnxArtifact(
        key="mobilesam_encoder",
        filename="mobilesam_encoder.onnx",
        url=(
            "https://ftrg.zbox.filez.com/v2/delivery/data/"
            "95f00b0fc900458ba134f8b180b3f7a1/examples/mobilesam/"
            "mobilesam_encoder_tiny.onnx"
        ),
        size_bytes=27_931_848,
        sha256="09ad167250443d5cd9d94ee4e46ab4c5163e97e9bfbaf1834039862307242489",
    ),
    OnnxArtifact(
        key="mobilesam_decoder",
        filename="mobilesam_decoder.onnx",
        url=(
            "https://ftrg.zbox.filez.com/v2/delivery/data/"
            "95f00b0fc900458ba134f8b180b3f7a1/examples/mobilesam/mobilesam_decoder.onnx"
        ),
        size_bytes=16_483_151,
        sha256="56671c3f7f1efdf7a64d97d9db3883388a5a37ee3dd986f5411e0c0e249694ed",
    ),
)
ONNX_BY_KEY: Mapping[str, OnnxArtifact] = {item.key: item for item in ONNX_ARTIFACTS}


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return a streaming SHA-256 digest for ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: os.PathLike[str] | str, spec: OnnxArtifact) -> tuple[int, str]:
    """Verify size and SHA-256, returning the observed values."""
    candidate = Path(path)
    if not candidate.is_file():
        raise ArtifactVerificationError(f"missing artifact: {candidate}")
    observed_size = candidate.stat().st_size
    if observed_size != spec.size_bytes:
        raise ArtifactVerificationError(
            f"size mismatch for {candidate}: expected {spec.size_bytes}, got {observed_size}"
        )
    observed_sha256 = sha256_file(candidate)
    if observed_sha256.lower() != spec.sha256.lower():
        raise ArtifactVerificationError(
            f"SHA-256 mismatch for {candidate}: expected {spec.sha256}, got {observed_sha256}"
        )
    return observed_size, observed_sha256


def _open_url(request: Request):
    return urlopen(request, timeout=60)


def download_artifact(
    spec: OnnxArtifact,
    output_dir: os.PathLike[str] | str,
    *,
    force: bool = False,
    opener: Callable[[Request], BinaryIO] = _open_url,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> DownloadResult:
    """Download one artifact atomically and verify it before publication."""
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / spec.filename

    if destination.exists():
        try:
            observed_size, observed_sha256 = verify_artifact(destination, spec)
        except ArtifactVerificationError as exc:
            if not force:
                raise FileExistsError(
                    f"refusing to replace invalid existing file without --force: {destination}"
                ) from exc
        else:
            return DownloadResult(
                key=spec.key,
                path=destination,
                status="verified-existing",
                size_bytes=observed_size,
                sha256=observed_sha256,
                url=spec.url,
            )

    request = Request(spec.url, headers={"User-Agent": "yolo-world-mobilesam-rk3588/1"})
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{spec.filename}.",
            suffix=".part",
            dir=str(destination_dir),
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with opener(request) as response:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    temporary.write(chunk)
            temporary.flush()
            os.fsync(temporary.fileno())

        observed_size, observed_sha256 = verify_artifact(temporary_path, spec)
        if destination.exists() and not force:
            raise FileExistsError(f"destination appeared during download: {destination}")
        os.replace(temporary_path, destination)
        temporary_path = None
        return DownloadResult(
            key=spec.key,
            path=destination,
            status="downloaded",
            size_bytes=observed_size,
            sha256=observed_sha256,
            url=spec.url,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def download_selected(
    keys: Iterable[str],
    output_dir: os.PathLike[str] | str,
    *,
    force: bool = False,
    record_path: Optional[os.PathLike[str] | str] = None,
    opener: Callable[[Request], BinaryIO] = _open_url,
) -> list[DownloadResult]:
    """Download selected keys in caller-provided order after full preflight."""
    selected_specs: list[OnnxArtifact] = []
    selected_keys = list(keys)
    seen: set[str] = set()
    for key in selected_keys:
        if key in seen:
            raise ValueError(f"duplicate model key: {key}")
        seen.add(key)
        try:
            selected_specs.append(ONNX_BY_KEY[key])
        except KeyError as exc:
            raise ValueError(f"unknown model key: {key}") from exc

    destination_dir = Path(output_dir).expanduser().resolve()
    destinations: dict[str, Path] = {}
    for spec in selected_specs:
        destination = (destination_dir / spec.filename).resolve()
        path_key = os.path.normcase(str(destination))
        previous = destinations.get(path_key)
        if previous is not None:
            raise ValueError(
                f"download output path collision: {previous} and {destination}"
            )
        destinations[path_key] = destination

    if record_path is not None:
        record = Path(record_path).expanduser().resolve()
        if os.path.normcase(str(record)) in destinations:
            raise ValueError(
                f"download record path collides with an ONNX output: {record}"
            )
        if record.exists() and not force:
            raise FileExistsError(f"refusing to overwrite existing record: {record}")

    results: list[DownloadResult] = []
    for spec in selected_specs:
        results.append(download_artifact(spec, output_dir, force=force, opener=opener))
    return results


def _environment_record() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }


def build_download_record(results: Sequence[DownloadResult]) -> dict[str, object]:
    return {
        "schema_version": DOWNLOAD_RECORD_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": _environment_record(),
        "artifacts": [item.as_record() for item in results],
    }


def write_json_record(
    path: os.PathLike[str] | str,
    payload: object,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write JSON, refusing implicit replacement."""
    destination = Path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing record: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(f"destination appeared while writing record: {destination}")
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "model" / "onnx",
        help="download directory (default: repository model/onnx)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(ONNX_BY_KEY),
        default=list(ONNX_BY_KEY),
        help="model keys to download (default: all four)",
    )
    parser.add_argument("--force", action="store_true", help="explicitly replace an invalid existing file")
    parser.add_argument("--record", type=Path, help="optional JSON download record path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        results = download_selected(
            args.models,
            args.output_dir,
            force=args.force,
            record_path=args.record,
        )
        if args.record is not None:
            write_json_record(args.record, build_download_record(results), overwrite=args.force)
    except (ArtifactError, FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(f"{result.status}: {result.path} ({result.size_bytes} bytes, sha256={result.sha256})")
    if args.record is not None:
        print(f"record: {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
