#!/usr/bin/env python3
"""Convert the pinned ONNX inputs into five RK3588 RKNN artifacts.

The RKNN import is deliberately delayed until after paths, hashes, dataset and
overwrite policy have been validated.  Conversion calls reproduce the local
rknn_model_zoo profiles that were verified with RKNN Toolkit2 2.3.2.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shlex
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

try:
    from .download_onnx import ONNX_BY_KEY, OnnxArtifact, sha256_file, verify_artifact, write_json_record
except ImportError:  # Support ``python conversion/convert_models.py``.
    from download_onnx import ONNX_BY_KEY, OnnxArtifact, sha256_file, verify_artifact, write_json_record


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLKIT_VERSION = "2.3.2"
CONVERSION_RECORD_SCHEMA = 1
OFFICIAL_DATASET_RELATIVE = Path("examples/yolo_world/model/dataset.txt")
OFFICIAL_DATASET_SHA256 = "1b568c09557d06ef6e0f50787b534eb866a95305e9c26d5713c7050dc60b488c"
OFFICIAL_DATASET_ROWS = 20
OFFICIAL_TEXT_EMBEDDING_SHA256 = (
    "e603be500f346dc04f5fd99808a7ac8fe1e31ad248b74c03482bbc4087ace1c2"
)
# Exact COCO subset shipped by rknn_model_zoo v2.3.2.  Pinning dataset.txt is
# not sufficient: a file may keep the same name while its pixels are changed.
OFFICIAL_COCO_IMAGE_SHA256: Mapping[str, str] = {
    "datasets/COCO/subset/000000005001.jpg": "99600c56dc25c72dc1cc1d1f2b09aa33b97142560b54d034fa83a38fcf3c5b0f",
    "datasets/COCO/subset/000000038829.jpg": "73f5b0fe070aaec65bbb7263d16d3d717a75304333fe939dcf086bd85d6ff2f9",
    "datasets/COCO/subset/000000052891.jpg": "e55b86a1cda874117ba1bd855e38dbc9b5427b840c362d72330019676ec47a44",
    "datasets/COCO/subset/000000075612.jpg": "ad55b7fe3105db47f033892e97ffe87a92c3545ae303f129d4bd59c5f211148a",
    "datasets/COCO/subset/000000098261.jpg": "6d73f5967db3ef19680f17791dd7252e1acf189ddb0cf6590ed211f8da123475",
    "datasets/COCO/subset/000000181542.jpg": "c0fdc24e1516a871e5dd195ff34a0da7b2227ef6486492545fb2548743ba4bf5",
    "datasets/COCO/subset/000000215245.jpg": "8f26d81db7cb960105b0dbfdd5e2846a4aace8d2fb352b0698f00afb3fe6ebb4",
    "datasets/COCO/subset/000000277005.jpg": "3cfea124dbcca946c927f9fefe4314b1a41ff8374adb316bdd969aca100b3d85",
    "datasets/COCO/subset/000000288685.jpg": "cd33f1aae3ea110b6de7b35f2225bce018cf30248a78652955678d67b4c19a21",
    "datasets/COCO/subset/000000301421.jpg": "a68205cde5ae859ac50c6cb563a0a3e5f7d780f28625afddeb8db153c7dcbb71",
    "datasets/COCO/subset/000000334371.jpg": "92974746d650d1562788bde333d087df29ca666662b5b35855cac1f9e62f21b8",
    "datasets/COCO/subset/000000348481.jpg": "7b4434ecb88a7379cd6d8327692ce2a1255100ef66bebd602b680e00b5a4206a",
    "datasets/COCO/subset/000000373353.jpg": "bc8f393230d375ddce63522d13740ccac6419e646560c97695a12ea3cee0e6fa",
    "datasets/COCO/subset/000000397681.jpg": "2d0a0055517147f3a0be7a13ab375f2ad3d41194f8a9d33fb510ca058c5e4662",
    "datasets/COCO/subset/000000414673.jpg": "6eda0abfc23d8612800893b44c5a1fe8a4ceb0c43cfc86a3c0deac9a0681c53e",
    "datasets/COCO/subset/000000419312.jpg": "7b953b528b2ae9fdc1c768710a8581c397ea552651ceeab6b59d2603f3223417",
    "datasets/COCO/subset/000000465822.jpg": "f90159f0241e3358ef8eed729247385f2e682eb48e803ddad9134f41b9d558f3",
    "datasets/COCO/subset/000000475732.jpg": "9f2c46374153ac27594475e3b42be01838eb35295068108a89134ad6c73532e7",
    "datasets/COCO/subset/000000559707.jpg": "50ae104992c88d5356fa5087bd62c51904134cb6ffa507e4c503024c75efe403",
    "datasets/COCO/subset/000000574315.jpg": "b4f4f99c9f1231ed99efa94744f4ae8ef9657ff69ce14a8d0ca60c7f5974dc13",
}


class ConversionError(RuntimeError):
    """Raised for a failed preflight or RKNN conversion stage."""


@dataclass(frozen=True)
class ConversionProfile:
    key: str
    input_key: str
    output_filename: str
    verbose: bool
    config_kwargs: Mapping[str, object]
    load_kwargs: Mapping[str, object]
    build_kwargs: Mapping[str, object]
    needs_yolo_dataset: bool = False
    reference_sha256: Optional[str] = None

    def parameters_record(self, dataset_included: bool) -> dict[str, object]:
        build_kwargs = dict(self.build_kwargs)
        if dataset_included:
            build_kwargs["dataset"] = "examples/yolo_world/model/dataset.txt"
        return {
            "constructor": {"verbose": self.verbose},
            "config": dict(self.config_kwargs),
            "load_onnx": dict(self.load_kwargs),
            "build": build_kwargs,
            "output_filename": self.output_filename,
        }


PROFILES: tuple[ConversionProfile, ...] = (
    ConversionProfile(
        key="clip_text_fp16",
        input_key="clip_text",
        output_filename="clip_text_fp16.rknn",
        verbose=False,
        config_kwargs={"target_platform": "rk3588"},
        load_kwargs={"inputs": ["input_ids"], "input_size_list": [[1, 20]]},
        build_kwargs={"do_quantization": False},
        reference_sha256="872765bb5f9813d96d57888d97ab3599270264aeeead2a97e505bffbed466563",
    ),
    ConversionProfile(
        key="yolo_world_v2s_fp16",
        input_key="yolo_world_v2s",
        output_filename="yolo_world_v2s_fp16.rknn",
        verbose=False,
        config_kwargs={
            "target_platform": "rk3588",
            "mean_values": [[0, 0, 0]],
            "std_values": [[255, 255, 255]],
        },
        load_kwargs={
            "inputs": ["images", "texts"],
            "input_size_list": [[1, 3, 640, 640], [1, 80, 512]],
        },
        build_kwargs={"do_quantization": False},
        reference_sha256="bc03e95b31b9bd73ce308a981e76b97b3c7cd11cdd44955671fd594477965fc7",
    ),
    ConversionProfile(
        key="yolo_world_v2s_i8",
        input_key="yolo_world_v2s",
        output_filename="yolo_world_v2s_i8.rknn",
        verbose=False,
        config_kwargs={
            "target_platform": "rk3588",
            "mean_values": [[0, 0, 0]],
            "std_values": [[255, 255, 255]],
        },
        load_kwargs={
            "inputs": ["images", "texts"],
            "input_size_list": [[1, 3, 640, 640], [1, 80, 512]],
        },
        build_kwargs={"do_quantization": True},
        needs_yolo_dataset=True,
        reference_sha256="c2af5058828ff62f39910d1f84284df5644ec46c05317aec54fa5235b04fa61a",
    ),
    ConversionProfile(
        key="mobilesam_encoder_fp16",
        input_key="mobilesam_encoder",
        output_filename="mobilesam_encoder_fp16.rknn",
        verbose=True,
        config_kwargs={
            "target_platform": "rk3588",
            "mean_values": [[123.675, 116.28, 103.53]],
            "std_values": [[58.395, 57.12, 57.375]],
        },
        load_kwargs={},
        build_kwargs={"do_quantization": False},
        reference_sha256="d1c3104934967cc488c83471bd645fee783f2609f90f446e1dfaf77e532875d9",
    ),
    ConversionProfile(
        key="mobilesam_decoder_fp16",
        input_key="mobilesam_decoder",
        output_filename="mobilesam_decoder_fp16.rknn",
        verbose=True,
        config_kwargs={"target_platform": "rk3588"},
        load_kwargs={
            "inputs": [
                "image_embeddings",
                "point_coords",
                "point_labels",
                "mask_input",
                "has_mask_input",
            ],
            "input_size_list": [
                [1, 256, 28, 28],
                [1, 2, 2],
                [1, 2],
                [1, 1, 112, 112],
                [1],
            ],
            "outputs": ["iou_predictions", "low_res_masks"],
        },
        build_kwargs={"do_quantization": False},
        reference_sha256="5f509ea393396ac33cb4c3805492227ca4f6b89c533dcbf6ef5a94808fd89660",
    ),
)
PROFILE_BY_KEY: Mapping[str, ConversionProfile] = {item.key: item for item in PROFILES}


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "size_bytes": self.size_bytes, "sha256": self.sha256}


@dataclass(frozen=True)
class CalibrationRecord:
    root: Path
    dataset_path: Path
    files: tuple[FileRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "model_zoo_dataset": OFFICIAL_DATASET_RELATIVE.as_posix(),
            "dataset_sha256": OFFICIAL_DATASET_SHA256,
            "files": [item.as_dict() for item in self.files],
        }


def _relative_record(path: Path, root: Path) -> FileRecord:
    return FileRecord(
        path=path.resolve().relative_to(root.resolve()).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
    )


def validate_official_dataset(model_zoo_root: os.PathLike[str] | str) -> CalibrationRecord:
    """Validate the exact Model Zoo YOLO-World calibration manifest and inputs."""
    root = Path(model_zoo_root).expanduser().resolve()
    if not root.is_dir():
        raise ConversionError(f"Model Zoo root is not a directory: {root}")
    dataset_path = root / OFFICIAL_DATASET_RELATIVE
    if not dataset_path.is_file():
        raise ConversionError(
            f"official YOLO-World dataset is missing under Model Zoo root: {dataset_path}"
        )
    dataset_sha256 = sha256_file(dataset_path)
    if dataset_sha256 != OFFICIAL_DATASET_SHA256:
        raise ConversionError(
            "official dataset.txt hash mismatch: "
            f"expected {OFFICIAL_DATASET_SHA256}, got {dataset_sha256}"
        )

    lines = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != OFFICIAL_DATASET_ROWS:
        raise ConversionError(
            f"official dataset must contain {OFFICIAL_DATASET_ROWS} rows, got {len(lines)}"
        )

    referenced: dict[Path, FileRecord] = {}
    image_paths: set[str] = set()
    text_embedding_paths: set[Path] = set()
    for line_number, line in enumerate(lines, 1):
        fields = shlex.split(line)
        if len(fields) != 2:
            raise ConversionError(f"dataset row {line_number} must contain image and text input")
        for field_index, field in enumerate(fields):
            candidate = (dataset_path.parent / field).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ConversionError(
                    f"dataset row {line_number} escapes Model Zoo root: {field}"
                ) from exc
            if not candidate.is_file():
                raise ConversionError(f"dataset input is missing: {candidate}")
            if candidate not in referenced:
                referenced[candidate] = _relative_record(candidate, root)
            if field_index == 0:
                relative_path = referenced[candidate].path
                expected_sha256 = OFFICIAL_COCO_IMAGE_SHA256.get(relative_path)
                if expected_sha256 is None:
                    raise ConversionError(
                        f"dataset row {line_number} references an unpinned COCO image: "
                        f"{relative_path}"
                    )
                observed_sha256 = referenced[candidate].sha256
                if observed_sha256 != expected_sha256:
                    raise ConversionError(
                        f"COCO image hash mismatch for {relative_path}: "
                        f"expected {expected_sha256}, got {observed_sha256}"
                    )
                image_paths.add(relative_path)
            else:
                text_embedding_paths.add(candidate)

    expected_image_paths = set(OFFICIAL_COCO_IMAGE_SHA256)
    if image_paths != expected_image_paths:
        missing = sorted(expected_image_paths - image_paths)
        unexpected = sorted(image_paths - expected_image_paths)
        raise ConversionError(
            "official dataset COCO image set mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    if len(text_embedding_paths) != 1:
        raise ConversionError("official dataset must reuse one coco_text_outp.npy input")
    text_embedding = next(iter(text_embedding_paths))
    observed_text_sha256 = referenced[text_embedding].sha256
    if observed_text_sha256 != OFFICIAL_TEXT_EMBEDDING_SHA256:
        raise ConversionError(
            "coco_text_outp.npy hash mismatch: "
            f"expected {OFFICIAL_TEXT_EMBEDDING_SHA256}, got {observed_text_sha256}"
        )

    dataset_record = _relative_record(dataset_path, root)
    files = (dataset_record, *tuple(referenced[path] for path in sorted(referenced)))
    return CalibrationRecord(root=root, dataset_path=dataset_path, files=files)


def resolve_profiles(keys: Iterable[str]) -> list[ConversionProfile]:
    selected: list[ConversionProfile] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            raise ConversionError(f"duplicate conversion profile: {key}")
        seen.add(key)
        try:
            selected.append(PROFILE_BY_KEY[key])
        except KeyError as exc:
            raise ConversionError(f"unknown conversion profile: {key}") from exc
    if not selected:
        raise ConversionError("at least one conversion profile is required")
    return selected


def _check_ret(stage: str, result: object) -> None:
    if result != 0:
        raise ConversionError(f"RKNN {stage} failed with return value {result!r}")


def run_rknn_conversion(
    profile: ConversionProfile,
    onnx_path: os.PathLike[str] | str,
    output_path: os.PathLike[str] | str,
    *,
    dataset_path: Optional[os.PathLike[str] | str] = None,
    rknn_factory: Callable[..., object],
) -> None:
    """Execute one exact RKNN API profile using an injectable factory."""
    if profile.needs_yolo_dataset and dataset_path is None:
        raise ConversionError(f"{profile.key} requires the official Model Zoo dataset")

    rknn = rknn_factory(verbose=profile.verbose)
    try:
        _check_ret("config", rknn.config(**dict(profile.config_kwargs)))
        load_kwargs = {"model": str(Path(onnx_path)), **dict(profile.load_kwargs)}
        _check_ret("load_onnx", rknn.load_onnx(**load_kwargs))
        build_kwargs = dict(profile.build_kwargs)
        # The upstream YOLO converter passes DATASET for both FP and I8.  It is
        # ignored by Toolkit2 for FP, but included whenever the root is given.
        if profile.input_key == "yolo_world_v2s" and dataset_path is not None:
            build_kwargs["dataset"] = str(Path(dataset_path))
        _check_ret("build", rknn.build(**build_kwargs))
        _check_ret("export_rknn", rknn.export_rknn(str(Path(output_path))))
    finally:
        rknn.release()


def _load_production_rknn() -> tuple[Callable[..., object], str]:
    """Import RKNN only when a real conversion is about to begin."""
    try:
        toolkit_version = importlib.metadata.version("rknn-toolkit2")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ConversionError("rknn-toolkit2 is not installed in this Python environment") from exc
    if toolkit_version != EXPECTED_TOOLKIT_VERSION:
        raise ConversionError(
            f"rknn-toolkit2 {EXPECTED_TOOLKIT_VERSION} is required; found {toolkit_version}"
        )
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise ConversionError("failed to import rknn.api.RKNN") from exc
    return RKNN, toolkit_version


def _input_record(path: Path, spec: OnnxArtifact, onnx_dir: Path) -> FileRecord:
    observed_size, observed_sha256 = verify_artifact(path, spec)
    return FileRecord(
        path=path.resolve().relative_to(onnx_dir.resolve()).as_posix(),
        size_bytes=observed_size,
        sha256=observed_sha256,
    )


def _environment_record(toolkit_version: str) -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "rknn_toolkit2_version": toolkit_version,
    }


def _path_identity(path: os.PathLike[str] | str) -> str:
    """Return a platform-aware identity for collision checks."""
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _preflight_conversion_paths(
    profiles: Sequence[ConversionProfile],
    source_dir: Path,
    destination_dir: Path,
    destination_record: Path,
    *,
    overwrite: bool,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Resolve every path and reject collisions before RKNN is constructed."""
    if destination_dir.exists() and not destination_dir.is_dir():
        raise ConversionError(f"RKNN output directory is not a directory: {destination_dir}")

    input_paths: dict[str, Path] = {}
    for profile in profiles:
        if profile.input_key in input_paths:
            continue
        spec = ONNX_BY_KEY[profile.input_key]
        source = (source_dir / spec.filename).resolve()
        try:
            source.relative_to(source_dir)
        except ValueError as exc:
            raise ConversionError(f"ONNX input escapes --onnx-dir: {source}") from exc
        input_paths[profile.input_key] = source

    output_paths: dict[str, Path] = {}
    output_identities: dict[str, Path] = {}
    input_identities = {_path_identity(path): path for path in input_paths.values()}
    for profile in profiles:
        destination = (destination_dir / profile.output_filename).resolve()
        try:
            destination.relative_to(destination_dir)
        except ValueError as exc:
            raise ConversionError(f"RKNN output escapes --output-dir: {destination}") from exc
        identity = _path_identity(destination)
        if identity in input_identities:
            raise ConversionError(
                f"RKNN output path collides with ONNX input: {destination}"
            )
        previous = output_identities.get(identity)
        if previous is not None:
            raise ConversionError(
                f"RKNN output path collision: {previous} and {destination}"
            )
        output_identities[identity] = destination
        output_paths[profile.key] = destination

    record_identity = _path_identity(destination_record)
    if record_identity in input_identities:
        raise ConversionError(
            f"conversion record path collides with ONNX input: {destination_record}"
        )
    if record_identity in output_identities:
        raise ConversionError(
            f"conversion record path collides with RKNN output: {destination_record}"
        )

    targets = [*output_paths.values(), destination_record]
    for target in targets:
        if target.exists() and target.is_dir():
            raise ConversionError(f"publication target is a directory: {target}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite publication target: {target}")
    return input_paths, output_paths


def _preflight_calibration_paths(
    calibration: CalibrationRecord,
    output_paths: Mapping[str, Path],
    destination_record: Path,
) -> None:
    """Prevent publication from replacing any verified calibration input."""
    calibration_paths = {
        _path_identity(calibration.root / item.path): calibration.root / item.path
        for item in calibration.files
    }
    if _path_identity(destination_record) in calibration_paths:
        raise ConversionError(
            f"conversion record path collides with calibration input: {destination_record}"
        )
    for destination in output_paths.values():
        if _path_identity(destination) in calibration_paths:
            raise ConversionError(
                f"RKNN output path collides with calibration input: {destination}"
            )


def _replace_path(source: Path, destination: Path) -> None:
    """Replace one path; kept separate so rollback behavior is testable."""
    os.replace(source, destination)


def _publish_staged_files(
    publications: Sequence[tuple[Path, Path]],
    *,
    overwrite: bool,
) -> None:
    """Publish a group of staged files, restoring every old target on failure."""
    for staged, destination in publications:
        if not staged.is_file():
            raise ConversionError(f"staged publication is missing: {staged}")
        if destination.exists() and destination.is_dir():
            raise ConversionError(f"publication target is a directory: {destination}")
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"publication target appeared while converting: {destination}"
            )

    backups: dict[Path, Path] = {}
    published: set[Path] = set()
    try:
        if overwrite:
            for _, destination in publications:
                if not destination.exists():
                    continue
                backup = destination.parent / (
                    f".{destination.name}.{uuid.uuid4().hex}.rollback"
                )
                backups[destination] = backup
                _replace_path(destination, backup)

        for staged, destination in publications:
            published.add(destination)
            _replace_path(staged, destination)
    except BaseException as exc:
        rollback_errors: list[str] = []
        for _, destination in reversed(publications):
            backup = backups.get(destination)
            try:
                if backup is not None and backup.exists():
                    _replace_path(backup, destination)
                elif backup is None and destination in published:
                    destination.unlink(missing_ok=True)
            except BaseException as rollback_exc:
                backup_restored = (
                    backup is not None
                    and destination.exists()
                    and not backup.exists()
                )
                new_file_removed = backup is None and not destination.exists()
                if not (backup_restored or new_file_removed):
                    rollback_errors.append(f"{destination}: {rollback_exc}")
        if rollback_errors:
            raise ConversionError(
                "publication failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from exc
        raise
    else:
        for backup in backups.values():
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                # Publication is already complete and internally consistent.
                # A stale hidden backup is safer than reporting a false
                # conversion failure after the commit point.
                pass


def convert_selected(
    profile_keys: Sequence[str],
    onnx_dir: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    *,
    model_zoo_root: Optional[os.PathLike[str] | str] = None,
    record_path: Optional[os.PathLike[str] | str] = None,
    overwrite: bool = False,
    rknn_factory: Optional[Callable[..., object]] = None,
    toolkit_version: Optional[str] = None,
) -> dict[str, object]:
    """Validate, stage all conversions, then publish outputs and record together."""
    profiles = resolve_profiles(profile_keys)
    source_dir = Path(onnx_dir).expanduser().resolve()
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_record = (
        Path(record_path).expanduser().resolve()
        if record_path is not None
        else destination_dir / "conversion-record.json"
    )
    input_paths, output_paths = _preflight_conversion_paths(
        profiles,
        source_dir,
        destination_dir,
        destination_record,
        overwrite=overwrite,
    )

    requires_i8_dataset = any(profile.needs_yolo_dataset for profile in profiles)
    if requires_i8_dataset and model_zoo_root is None:
        raise ConversionError(
            "yolo_world_v2s_i8 requires --model-zoo-root pointing to the official dataset"
        )
    calibration = (
        validate_official_dataset(model_zoo_root)
        if model_zoo_root is not None
        else None
    )
    if calibration is not None:
        _preflight_calibration_paths(
            calibration,
            output_paths,
            destination_record,
        )

    input_records: dict[str, FileRecord] = {}
    for profile in profiles:
        if profile.input_key in input_records:
            continue
        spec = ONNX_BY_KEY[profile.input_key]
        source = input_paths[profile.input_key]
        input_records[profile.input_key] = _input_record(source, spec, source_dir)

    if rknn_factory is None:
        rknn_factory, observed_toolkit_version = _load_production_rknn()
    else:
        observed_toolkit_version = toolkit_version or "injected-test-double"
    if observed_toolkit_version != EXPECTED_TOOLKIT_VERSION:
        raise ConversionError(
            f"conversion profile requires Toolkit2 {EXPECTED_TOOLKIT_VERSION}; "
            f"got {observed_toolkit_version}"
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_record.parent.mkdir(parents=True, exist_ok=True)
    staged_record = destination_record.parent / (
        f".{destination_record.name}.{uuid.uuid4().hex}.staged"
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix=".conversion-staging-",
            dir=str(destination_dir),
        ) as staging_directory:
            staging_dir = Path(staging_directory)
            output_records: list[dict[str, object]] = []
            staged_outputs: list[tuple[Path, Path]] = []
            for profile in profiles:
                source = input_paths[profile.input_key]
                destination = output_paths[profile.key]
                staged_output = staging_dir / profile.output_filename
                staged_output.parent.mkdir(parents=True, exist_ok=True)
                run_rknn_conversion(
                    profile,
                    source,
                    staged_output,
                    dataset_path=calibration.dataset_path if calibration is not None else None,
                    rknn_factory=rknn_factory,
                )
                if not staged_output.is_file() or staged_output.stat().st_size <= 0:
                    raise ConversionError(
                        f"RKNN exporter did not create a non-empty file: {staged_output}"
                    )
                output_sha256 = sha256_file(staged_output)
                output_size = staged_output.stat().st_size
                output_records.append(
                    {
                        "profile": profile.key,
                        "path": destination.name,
                        "size_bytes": output_size,
                        "sha256": output_sha256,
                        "reference_sha256": profile.reference_sha256,
                        "matches_verified_reference": output_sha256 == profile.reference_sha256,
                        "parameters": profile.parameters_record(
                            dataset_included=(
                                profile.input_key == "yolo_world_v2s"
                                and calibration is not None
                            )
                        ),
                    }
                )
                staged_outputs.append((staged_output, destination))

            record: dict[str, object] = {
                "schema_version": CONVERSION_RECORD_SCHEMA,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "target_platform": "rk3588",
                "profile_source": (
                    "airockchip/rknn_model_zoo parameters verified with Toolkit2 2.3.2"
                ),
                "environment": _environment_record(observed_toolkit_version),
                "inputs": {
                    key: item.as_dict()
                    for key, item in sorted(input_records.items())
                },
                "calibration": calibration.as_dict() if calibration is not None else None,
                "outputs": output_records,
            }
            write_json_record(staged_record, record)
            _publish_staged_files(
                [*staged_outputs, (staged_record, destination_record)],
                overwrite=overwrite,
            )
            return record
    finally:
        staged_record.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=REPOSITORY_ROOT / "model" / "onnx",
        help="directory containing the four pinned ONNX files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "model",
        help="RKNN output directory",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(PROFILE_BY_KEY),
        default=list(PROFILE_BY_KEY),
        help="conversion profiles (default: all five)",
    )
    parser.add_argument(
        "--model-zoo-root",
        type=Path,
        help="rknn_model_zoo root; mandatory for I8 official calibration",
    )
    parser.add_argument("--record", type=Path, help="JSON conversion record path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace existing outputs and record",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        record = convert_selected(
            args.models,
            args.onnx_dir,
            args.output_dir,
            model_zoo_root=args.model_zoo_root,
            record_path=args.record,
            overwrite=args.overwrite,
        )
    except (ConversionError, FileExistsError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for output in record["outputs"]:
        print(
            f"converted: {output['path']} "
            f"({output['size_bytes']} bytes, sha256={output['sha256']})"
        )
    record_path = args.record or (args.output_dir / "conversion-record.json")
    print(f"record: {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
