#!/usr/bin/env python3
"""Offline tests for model download and RKNN conversion tooling."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from conversion import convert_models
from conversion import download_onnx


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class FakeRKNN:
    instances: list["FakeRKNN"] = []
    export_payload = b"fake-rknn"
    fail_stage: str | None = None
    fail_instance_number: int | None = None

    def __init__(self, *, verbose: bool):
        self.verbose = verbose
        self.instance_number = len(type(self).instances) + 1
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.released = False
        type(self).instances.append(self)

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.fail_stage = None
        cls.fail_instance_number = None

    def _fails(self, stage: str) -> bool:
        return self.fail_stage == stage and (
            self.fail_instance_number is None
            or self.fail_instance_number == self.instance_number
        )

    def config(self, **kwargs):
        self.calls.append(("config", kwargs))
        return 1 if self._fails("config") else 0

    def load_onnx(self, **kwargs):
        self.calls.append(("load_onnx", kwargs))
        return 1 if self._fails("load_onnx") else 0

    def build(self, **kwargs):
        self.calls.append(("build", kwargs))
        return 1 if self._fails("build") else 0

    def export_rknn(self, path: str):
        self.calls.append(("export_rknn", {"path": path}))
        if self._fails("export_rknn"):
            return 1
        Path(path).write_bytes(self.export_payload)
        return 0

    def release(self):
        self.released = True
        self.calls.append(("release", {}))


class DownloadManifestTest(unittest.TestCase):
    def test_manifest_pins_all_four_verified_onnx_files(self):
        self.assertEqual(
            [item.key for item in download_onnx.ONNX_ARTIFACTS],
            ["clip_text", "yolo_world_v2s", "mobilesam_encoder", "mobilesam_decoder"],
        )
        self.assertEqual(len({item.sha256 for item in download_onnx.ONNX_ARTIFACTS}), 4)
        for item in download_onnx.ONNX_ARTIFACTS:
            self.assertEqual(len(item.sha256), 64)
            int(item.sha256, 16)
            self.assertGreater(item.size_bytes, 0)
            self.assertTrue(item.url.startswith("https://"))

    def test_download_uses_temp_file_and_verifies_payload(self):
        payload = b"verified model bytes"
        spec = download_onnx.OnnxArtifact(
            key="test",
            filename="test.onnx",
            url="https://example.invalid/test.onnx",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = download_onnx.download_artifact(
                spec,
                temporary,
                opener=lambda request: FakeResponse(payload),
            )
            self.assertEqual(result.status, "downloaded")
            self.assertEqual((Path(temporary) / spec.filename).read_bytes(), payload)
            self.assertEqual(list(Path(temporary).glob("*.part")), [])

    def test_existing_invalid_file_is_not_silently_replaced(self):
        payload = b"right"
        spec = download_onnx.OnnxArtifact(
            key="test",
            filename="test.onnx",
            url="https://example.invalid/test.onnx",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / spec.filename
            destination.write_bytes(b"wrong")
            with self.assertRaises(FileExistsError):
                download_onnx.download_artifact(
                    spec,
                    temporary,
                    opener=lambda request: FakeResponse(payload),
                )
            self.assertEqual(destination.read_bytes(), b"wrong")

            result = download_onnx.download_artifact(
                spec,
                temporary,
                force=True,
                opener=lambda request: FakeResponse(payload),
            )
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(destination.read_bytes(), payload)

    def test_valid_existing_file_is_reused_without_network(self):
        payload = b"already here"
        spec = download_onnx.OnnxArtifact(
            key="test",
            filename="test.onnx",
            url="https://example.invalid/test.onnx",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / spec.filename).write_bytes(payload)
            result = download_onnx.download_artifact(
                spec,
                temporary,
                opener=lambda request: self.fail("network must not be used"),
            )
            self.assertEqual(result.status, "verified-existing")

    def test_record_collision_is_rejected_before_download(self):
        payload = b"never downloaded"
        spec = download_onnx.OnnxArtifact(
            key="test",
            filename="test.onnx",
            url="https://example.invalid/test.onnx",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "onnx"
            patched_specs = {"test": spec}
            with patch.object(download_onnx, "ONNX_BY_KEY", patched_specs):
                with self.assertRaisesRegex(ValueError, "record path collides"):
                    download_onnx.download_selected(
                        ["test"],
                        output_dir,
                        record_path=output_dir / spec.filename,
                        opener=lambda request: self.fail("network must not be used"),
                    )
            self.assertFalse(output_dir.exists())

    def test_existing_record_is_rejected_before_download(self):
        payload = b"never downloaded"
        spec = download_onnx.OnnxArtifact(
            key="test",
            filename="test.onnx",
            url="https://example.invalid/test.onnx",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = root / "record.json"
            record.write_text("old", encoding="utf-8")
            patched_specs = {"test": spec}
            with patch.object(download_onnx, "ONNX_BY_KEY", patched_specs):
                with self.assertRaises(FileExistsError):
                    download_onnx.download_selected(
                        ["test"],
                        root / "onnx",
                        record_path=record,
                        opener=lambda request: self.fail("network must not be used"),
                    )
            self.assertEqual(record.read_text(encoding="utf-8"), "old")


class CalibrationDatasetTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, dict[str, str], str, str]:
        model_dir = root / convert_models.OFFICIAL_DATASET_RELATIVE.parent
        model_dir.mkdir(parents=True)
        embedding = b"pinned-text-embedding"
        (model_dir / "coco_text_outp.npy").write_bytes(embedding)

        image_hashes: dict[str, str] = {}
        rows: list[str] = []
        for index in range(convert_models.OFFICIAL_DATASET_ROWS):
            filename = f"{index:012d}.jpg"
            relative = f"datasets/COCO/subset/{filename}"
            payload = f"image-{index}".encode("ascii")
            image_path = root / relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(payload)
            image_hashes[relative] = hashlib.sha256(payload).hexdigest()
            rows.append(
                f"../../../datasets/COCO/subset/{filename} coco_text_outp.npy"
            )
        dataset_payload = ("\n".join(rows) + "\n").encode("utf-8")
        (model_dir / "dataset.txt").write_bytes(dataset_payload)
        return (
            model_dir,
            image_hashes,
            hashlib.sha256(dataset_payload).hexdigest(),
            hashlib.sha256(embedding).hexdigest(),
        )

    def test_v232_manifest_pins_exactly_twenty_coco_images(self):
        pins = convert_models.OFFICIAL_COCO_IMAGE_SHA256
        self.assertEqual(len(pins), 20)
        self.assertEqual(
            sorted(Path(path).name for path in pins),
            [
                "000000005001.jpg",
                "000000038829.jpg",
                "000000052891.jpg",
                "000000075612.jpg",
                "000000098261.jpg",
                "000000181542.jpg",
                "000000215245.jpg",
                "000000277005.jpg",
                "000000288685.jpg",
                "000000301421.jpg",
                "000000334371.jpg",
                "000000348481.jpg",
                "000000373353.jpg",
                "000000397681.jpg",
                "000000414673.jpg",
                "000000419312.jpg",
                "000000465822.jpg",
                "000000475732.jpg",
                "000000559707.jpg",
                "000000574315.jpg",
            ],
        )
        for digest in pins.values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_tampered_coco_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_dir, image_hashes, dataset_sha256, embedding_sha256 = (
                self._write_fixture(root)
            )
            with (
                patch.object(
                    convert_models,
                    "OFFICIAL_DATASET_SHA256",
                    dataset_sha256,
                ),
                patch.object(
                    convert_models,
                    "OFFICIAL_COCO_IMAGE_SHA256",
                    image_hashes,
                ),
                patch.object(
                    convert_models,
                    "OFFICIAL_TEXT_EMBEDDING_SHA256",
                    embedding_sha256,
                ),
            ):
                record = convert_models.validate_official_dataset(root)
                self.assertEqual(len(record.files), 22)
                tampered = root / next(iter(image_hashes))
                tampered.write_bytes(b"tampered")
                with self.assertRaisesRegex(
                    convert_models.ConversionError,
                    "COCO image hash mismatch",
                ):
                    convert_models.validate_official_dataset(root)
            self.assertTrue((model_dir / "dataset.txt").is_file())


class RKNNProfileTest(unittest.TestCase):
    def setUp(self):
        FakeRKNN.reset()

    def _run(self, key: str, *, dataset: Path | None = None) -> FakeRKNN:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.onnx"
            output = Path(temporary) / "output.rknn"
            source.write_bytes(b"onnx")
            convert_models.run_rknn_conversion(
                convert_models.PROFILE_BY_KEY[key],
                source,
                output,
                dataset_path=dataset,
                rknn_factory=FakeRKNN,
            )
            self.assertEqual(output.read_bytes(), FakeRKNN.export_payload)
        return FakeRKNN.instances[-1]

    def test_clip_profile_exact_parameters(self):
        instance = self._run("clip_text_fp16")
        self.assertFalse(instance.verbose)
        self.assertEqual(instance.calls[0], ("config", {"target_platform": "rk3588"}))
        self.assertEqual(
            instance.calls[1][1]["inputs"],
            ["input_ids"],
        )
        self.assertEqual(instance.calls[1][1]["input_size_list"], [[1, 20]])
        self.assertEqual(instance.calls[2], ("build", {"do_quantization": False}))
        self.assertTrue(instance.released)

    def test_yolo_fp_and_i8_profiles_exact_parameters(self):
        dataset = Path("/model-zoo/examples/yolo_world/model/dataset.txt")
        fp = self._run("yolo_world_v2s_fp16", dataset=dataset)
        i8 = self._run("yolo_world_v2s_i8", dataset=dataset)
        for instance in (fp, i8):
            self.assertEqual(
                instance.calls[0][1],
                {
                    "target_platform": "rk3588",
                    "mean_values": [[0, 0, 0]],
                    "std_values": [[255, 255, 255]],
                },
            )
            self.assertEqual(instance.calls[1][1]["inputs"], ["images", "texts"])
            self.assertEqual(
                instance.calls[1][1]["input_size_list"],
                [[1, 3, 640, 640], [1, 80, 512]],
            )
            self.assertEqual(instance.calls[2][1]["dataset"], str(dataset))
        self.assertFalse(fp.calls[2][1]["do_quantization"])
        self.assertTrue(i8.calls[2][1]["do_quantization"])

    def test_mobilesam_profiles_exact_parameters(self):
        encoder = self._run("mobilesam_encoder_fp16")
        decoder = self._run("mobilesam_decoder_fp16")
        self.assertTrue(encoder.verbose)
        self.assertEqual(
            encoder.calls[0][1],
            {
                "target_platform": "rk3588",
                "mean_values": [[123.675, 116.28, 103.53]],
                "std_values": [[58.395, 57.12, 57.375]],
            },
        )
        self.assertEqual(set(encoder.calls[1][1]), {"model"})
        self.assertTrue(decoder.verbose)
        self.assertEqual(decoder.calls[0][1], {"target_platform": "rk3588"})
        self.assertEqual(
            decoder.calls[1][1]["inputs"],
            [
                "image_embeddings",
                "point_coords",
                "point_labels",
                "mask_input",
                "has_mask_input",
            ],
        )
        self.assertEqual(
            decoder.calls[1][1]["input_size_list"],
            [[1, 256, 28, 28], [1, 2, 2], [1, 2], [1, 1, 112, 112], [1]],
        )
        self.assertEqual(
            decoder.calls[1][1]["outputs"],
            ["iou_predictions", "low_res_masks"],
        )

    def test_i8_rejects_missing_dataset_before_constructing_rknn(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(convert_models.ConversionError):
                convert_models.run_rknn_conversion(
                    convert_models.PROFILE_BY_KEY["yolo_world_v2s_i8"],
                    Path(temporary) / "input.onnx",
                    Path(temporary) / "output.rknn",
                    rknn_factory=FakeRKNN,
                )
        self.assertEqual(FakeRKNN.instances, [])

    def test_release_is_called_when_rknn_stage_fails(self):
        FakeRKNN.fail_stage = "build"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(convert_models.ConversionError):
                convert_models.run_rknn_conversion(
                    convert_models.PROFILE_BY_KEY["clip_text_fp16"],
                    Path(temporary) / "input.onnx",
                    Path(temporary) / "output.rknn",
                    rknn_factory=FakeRKNN,
                )
        self.assertTrue(FakeRKNN.instances[-1].released)

    def test_config_return_value_is_checked_and_release_is_called(self):
        FakeRKNN.fail_stage = "config"
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(convert_models.ConversionError, "RKNN config failed"):
                convert_models.run_rknn_conversion(
                    convert_models.PROFILE_BY_KEY["clip_text_fp16"],
                    Path(temporary) / "input.onnx",
                    Path(temporary) / "output.rknn",
                    rknn_factory=FakeRKNN,
                )
        instance = FakeRKNN.instances[-1]
        self.assertTrue(instance.released)
        self.assertEqual([name for name, _ in instance.calls], ["config", "release"])


class ConversionRecordTest(unittest.TestCase):
    def setUp(self):
        FakeRKNN.reset()

    def test_i8_requires_explicit_model_zoo_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(convert_models.ConversionError):
                convert_models.convert_selected(
                    ["yolo_world_v2s_i8"],
                    temporary,
                    Path(temporary) / "out",
                    rknn_factory=FakeRKNN,
                    toolkit_version=convert_models.EXPECTED_TOOLKIT_VERSION,
                )
        self.assertEqual(FakeRKNN.instances, [])

    def test_conversion_record_contains_input_and_output_sha256(self):
        payload = b"small pinned onnx"
        pinned = download_onnx.OnnxArtifact(
            key="clip_text",
            filename="clip_text.onnx",
            url="https://example.invalid/clip.onnx",
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            onnx_dir = root / "onnx"
            output_dir = root / "model"
            onnx_dir.mkdir()
            (onnx_dir / pinned.filename).write_bytes(payload)
            record_path = root / "record.json"
            patched_specs = dict(convert_models.ONNX_BY_KEY)
            patched_specs["clip_text"] = pinned
            with patch.object(convert_models, "ONNX_BY_KEY", patched_specs):
                record = convert_models.convert_selected(
                    ["clip_text_fp16"],
                    onnx_dir,
                    output_dir,
                    record_path=record_path,
                    rknn_factory=FakeRKNN,
                    toolkit_version=convert_models.EXPECTED_TOOLKIT_VERSION,
                )

            saved = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(saved, record)
            self.assertEqual(saved["inputs"]["clip_text"]["sha256"], pinned.sha256)
            expected_output_sha = hashlib.sha256(FakeRKNN.export_payload).hexdigest()
            self.assertEqual(saved["outputs"][0]["sha256"], expected_output_sha)
            self.assertEqual(
                saved["environment"]["rknn_toolkit2_version"],
                convert_models.EXPECTED_TOOLKIT_VERSION,
            )
            self.assertTrue((output_dir / "clip_text_fp16.rknn").is_file())

    def test_existing_output_is_refused_before_rknn_is_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "model"
            output_dir.mkdir()
            (output_dir / "clip_text_fp16.rknn").write_bytes(b"keep")
            with self.assertRaises(FileExistsError):
                convert_models.convert_selected(
                    ["clip_text_fp16"],
                    root,
                    output_dir,
                    rknn_factory=FakeRKNN,
                    toolkit_version=convert_models.EXPECTED_TOOLKIT_VERSION,
                )
            self.assertEqual(
                (output_dir / "clip_text_fp16.rknn").read_bytes(),
                b"keep",
            )
            self.assertEqual(FakeRKNN.instances, [])

    def test_record_path_cannot_collide_with_input_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "model"
            collisions = (
                root / "clip_text.onnx",
                output_dir / "clip_text_fp16.rknn",
            )
            for record_path in collisions:
                with self.subTest(record_path=record_path):
                    with self.assertRaisesRegex(
                        convert_models.ConversionError,
                        "record path collides",
                    ):
                        convert_models.convert_selected(
                            ["clip_text_fp16"],
                            root,
                            output_dir,
                            record_path=record_path,
                            rknn_factory=FakeRKNN,
                            toolkit_version=convert_models.EXPECTED_TOOLKIT_VERSION,
                        )
            self.assertEqual(FakeRKNN.instances, [])

    def test_output_path_cannot_collide_with_input(self):
        profile = convert_models.ConversionProfile(
            key="clip_text_fp16",
            input_key="clip_text",
            output_filename="clip_text.onnx",
            verbose=False,
            config_kwargs={"target_platform": "rk3588"},
            load_kwargs={},
            build_kwargs={"do_quantization": False},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patched_profiles = dict(convert_models.PROFILE_BY_KEY)
            patched_profiles[profile.key] = profile
            with patch.object(convert_models, "PROFILE_BY_KEY", patched_profiles):
                with self.assertRaisesRegex(
                    convert_models.ConversionError,
                    "output path collides with ONNX input",
                ):
                    convert_models.convert_selected(
                        [profile.key],
                        root,
                        root,
                        record_path=root / "record.json",
                        rknn_factory=FakeRKNN,
                        toolkit_version=convert_models.EXPECTED_TOOLKIT_VERSION,
                    )
            self.assertEqual(FakeRKNN.instances, [])

    def test_record_path_cannot_replace_a_calibration_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration_input = root / "dataset.txt"
            calibration_input.write_bytes(b"calibration")
            calibration = convert_models.CalibrationRecord(
                root=root,
                dataset_path=calibration_input,
                files=(
                    convert_models.FileRecord(
                        path="dataset.txt",
                        size_bytes=calibration_input.stat().st_size,
                        sha256=hashlib.sha256(b"calibration").hexdigest(),
                    ),
                ),
            )
            with patch.object(
                convert_models,
                "validate_official_dataset",
                return_value=calibration,
            ):
                with self.assertRaisesRegex(
                    convert_models.ConversionError,
                    "record path collides with calibration input",
                ):
                    convert_models.convert_selected(
                        ["clip_text_fp16"],
                        root,
                        root / "model",
                        model_zoo_root=root,
                        record_path=calibration_input,
                        overwrite=True,
                        rknn_factory=FakeRKNN,
                        toolkit_version=convert_models.EXPECTED_TOOLKIT_VERSION,
                    )
            self.assertEqual(calibration_input.read_bytes(), b"calibration")
            self.assertEqual(FakeRKNN.instances, [])

    def test_multi_model_failure_does_not_publish_partial_outputs(self):
        payloads = {
            "clip_text": b"clip-input",
            "mobilesam_encoder": b"encoder-input",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            onnx_dir = root / "onnx"
            output_dir = root / "model"
            onnx_dir.mkdir()
            output_dir.mkdir()
            patched_specs = dict(convert_models.ONNX_BY_KEY)
            for key, payload in payloads.items():
                original = patched_specs[key]
                patched_specs[key] = download_onnx.OnnxArtifact(
                    key=key,
                    filename=original.filename,
                    url=original.url,
                    size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
                (onnx_dir / original.filename).write_bytes(payload)

            clip_output = output_dir / "clip_text_fp16.rknn"
            encoder_output = output_dir / "mobilesam_encoder_fp16.rknn"
            record_path = output_dir / "conversion-record.json"
            clip_output.write_bytes(b"old-clip")
            encoder_output.write_bytes(b"old-encoder")
            record_path.write_bytes(b"old-record")
            FakeRKNN.fail_stage = "build"
            FakeRKNN.fail_instance_number = 2

            with patch.object(convert_models, "ONNX_BY_KEY", patched_specs):
                with self.assertRaises(convert_models.ConversionError):
                    convert_models.convert_selected(
                        ["clip_text_fp16", "mobilesam_encoder_fp16"],
                        onnx_dir,
                        output_dir,
                        record_path=record_path,
                        overwrite=True,
                        rknn_factory=FakeRKNN,
                        toolkit_version=convert_models.EXPECTED_TOOLKIT_VERSION,
                    )

            self.assertEqual(clip_output.read_bytes(), b"old-clip")
            self.assertEqual(encoder_output.read_bytes(), b"old-encoder")
            self.assertEqual(record_path.read_bytes(), b"old-record")
            self.assertEqual(list(output_dir.glob(".conversion-staging-*")), [])

    def test_overwrite_publication_failure_rolls_back_outputs_and_record(self):
        payloads = {
            "clip_text": b"clip-input",
            "mobilesam_encoder": b"encoder-input",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            onnx_dir = root / "onnx"
            output_dir = root / "model"
            onnx_dir.mkdir()
            output_dir.mkdir()
            patched_specs = dict(convert_models.ONNX_BY_KEY)
            for key, payload in payloads.items():
                original = patched_specs[key]
                patched_specs[key] = download_onnx.OnnxArtifact(
                    key=key,
                    filename=original.filename,
                    url=original.url,
                    size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
                (onnx_dir / original.filename).write_bytes(payload)

            clip_output = output_dir / "clip_text_fp16.rknn"
            encoder_output = output_dir / "mobilesam_encoder_fp16.rknn"
            record_path = output_dir / "conversion-record.json"
            clip_output.write_bytes(b"old-clip")
            encoder_output.write_bytes(b"old-encoder")
            record_path.write_bytes(b"old-record")
            original_replace = convert_models._replace_path
            failed = False

            def fail_second_publication(source: Path, destination: Path) -> None:
                nonlocal failed
                source = Path(source)
                destination = Path(destination)
                if (
                    not failed
                    and destination == encoder_output
                    and source.parent.name.startswith(".conversion-staging-")
                ):
                    failed = True
                    raise OSError("simulated publication failure")
                original_replace(source, destination)

            with (
                patch.object(convert_models, "ONNX_BY_KEY", patched_specs),
                patch.object(convert_models, "_replace_path", fail_second_publication),
            ):
                with self.assertRaisesRegex(OSError, "simulated publication failure"):
                    convert_models.convert_selected(
                        ["clip_text_fp16", "mobilesam_encoder_fp16"],
                        onnx_dir,
                        output_dir,
                        record_path=record_path,
                        overwrite=True,
                        rknn_factory=FakeRKNN,
                        toolkit_version=convert_models.EXPECTED_TOOLKIT_VERSION,
                    )

            self.assertTrue(failed)
            self.assertEqual(clip_output.read_bytes(), b"old-clip")
            self.assertEqual(encoder_output.read_bytes(), b"old-encoder")
            self.assertEqual(record_path.read_bytes(), b"old-record")
            self.assertEqual(list(output_dir.glob("*.rollback")), [])
            self.assertEqual(list(output_dir.glob(".conversion-staging-*")), [])

    def test_publish_interrupt_after_backup_move_restores_old_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            output = root / "model"
            stage.mkdir()
            output.mkdir()
            staged = stage / "model.rknn"
            destination = output / "model.rknn"
            staged.write_bytes(b"new-model")
            destination.write_bytes(b"old-model")
            original_replace = convert_models._replace_path
            interrupted = False

            def interrupt_after_backup(source: Path, target: Path) -> None:
                nonlocal interrupted
                source = Path(source)
                target = Path(target)
                original_replace(source, target)
                if (
                    not interrupted
                    and source == destination
                    and target.parent == output
                    and target.name.startswith(".model.rknn.")
                    and target.name.endswith(".rollback")
                ):
                    interrupted = True
                    raise KeyboardInterrupt("interrupt after backup move")

            with patch.object(
                convert_models, "_replace_path", interrupt_after_backup
            ):
                with self.assertRaises(KeyboardInterrupt):
                    convert_models._publish_staged_files(
                        [(staged, destination)], overwrite=True
                    )
            self.assertTrue(interrupted)
            self.assertEqual(destination.read_bytes(), b"old-model")
            self.assertEqual(list(output.glob(".*.rollback")), [])

    def test_publish_interrupt_after_first_move_removes_partial_new_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            output = root / "model"
            stage.mkdir()
            output.mkdir()
            publications = []
            for name in ("first.rknn", "second.rknn"):
                staged = stage / name
                staged.write_bytes(f"new-{name}".encode("ascii"))
                publications.append((staged, output / name))
            first_staged, first_destination = publications[0]
            original_replace = convert_models._replace_path
            interrupted = False

            def interrupt_after_publish(source: Path, target: Path) -> None:
                nonlocal interrupted
                source = Path(source)
                target = Path(target)
                original_replace(source, target)
                if (
                    not interrupted
                    and source == first_staged
                    and target == first_destination
                ):
                    interrupted = True
                    raise KeyboardInterrupt("interrupt after first publication")

            with patch.object(
                convert_models, "_replace_path", interrupt_after_publish
            ):
                with self.assertRaises(KeyboardInterrupt):
                    convert_models._publish_staged_files(
                        publications, overwrite=False
                    )
            self.assertTrue(interrupted)
            self.assertFalse(first_destination.exists())
            self.assertFalse(publications[1][1].exists())
            self.assertEqual(list(output.glob(".*.rollback")), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
