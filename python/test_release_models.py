#!/usr/bin/env python3
"""Offline tests for GitHub Release model download and packaging."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_model_release
from scripts import download_models


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ReleaseDownloaderTest(unittest.TestCase):
    def make_bundle(self, root: Path) -> tuple[dict, dict[str, bytes]]:
        payloads = {
            "encoder.rknn": b"encoder-model",
            "decoder.rknn": b"decoder-model",
        }
        archive_root = "test-bundle"
        archive_path = root / "test-bundle.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name, payload in payloads.items():
                archive.writestr(f"{archive_root}/model/{name}", payload)
        archive_payload = archive_path.read_bytes()
        manifest = {
            "schema_version": 1,
            "release": {
                "download_base_url": root.as_uri(),
                "bundles": {
                    "sam": {
                        "asset": archive_path.name,
                        "bytes": len(archive_payload),
                        "sha256": digest(archive_payload),
                        "archive_root": archive_root,
                    }
                },
            },
            "models": {
                name: {
                    "bytes": len(payload),
                    "sha256": digest(payload),
                    "distribution": {
                        "status": "release_bundle",
                        "bundle": "sam",
                        "member": f"model/{name}",
                    },
                }
                for name, payload in payloads.items()
            },
        }
        manifest["models"]["restricted.rknn"] = {
            "bytes": 1,
            "sha256": "0" * 64,
            "distribution": {
                "status": "convert_locally",
                "reason": "test restriction",
            },
        }
        return manifest, payloads

    def test_download_extracts_and_verifies_bundle_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, payloads = self.make_bundle(root)
            output = root / "output"
            selected = download_models.released_model_names(manifest)
            download_models.download_selected(
                selected,
                manifest,
                output,
                root.as_uri(),
            )
            self.assertEqual(
                {path.name: path.read_bytes() for path in output.glob("*.rknn")},
                payloads,
            )
            self.assertEqual(list(output.glob("model-download-*")), [])

    def test_convert_only_model_is_rejected_with_reason(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = self.make_bundle(Path(temporary))
            with self.assertRaisesRegex(download_models.ModelDownloadError, "test restriction"):
                download_models.select_models(manifest, ["restricted.rknn"])

    def test_invalid_existing_model_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_bundle(root)
            output = root / "output"
            output.mkdir()
            destination = output / "encoder.rknn"
            destination.write_bytes(b"keep-this-invalid-file")
            with self.assertRaisesRegex(download_models.ModelDownloadError, "--force"):
                download_models.download_selected(
                    ["encoder.rknn"], manifest, output, root.as_uri()
                )
            self.assertEqual(destination.read_bytes(), b"keep-this-invalid-file")

    def test_manifest_rejects_path_traversal_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_bundle(root)
            entry = manifest["models"].pop("encoder.rknn")
            manifest["models"]["../outside.rknn"] = entry
            with self.assertRaisesRegex(download_models.ModelDownloadError, "unsafe model name"):
                download_models.validate_manifest(manifest, root / "manifest.json")

    def test_manifest_rejects_absolute_or_nested_asset_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_bundle(root)
            manifest["release"]["bundles"]["sam"]["asset"] = "../bundle.zip"
            with self.assertRaisesRegex(download_models.ModelDownloadError, "unsafe bundle asset"):
                download_models.validate_manifest(manifest, root / "manifest.json")

    def test_extract_publish_failure_rolls_back_the_whole_model_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_bundle(root)
            output = root / "output"
            output.mkdir()
            old_payloads = {
                "decoder.rknn": b"old-decoder",
                "encoder.rknn": b"old-encoder",
            }
            for name, payload in old_payloads.items():
                (output / name).write_bytes(payload)
            bundle = manifest["release"]["bundles"]["sam"]
            real_replace = os.replace
            failed = False

            def fail_second_publish(source, destination):
                nonlocal failed
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    not failed
                    and source_path.name == "encoder.rknn"
                    and destination_path == output / "encoder.rknn"
                ):
                    failed = True
                    raise OSError("injected publish failure")
                return real_replace(source, destination)

            with patch.object(download_models.os, "replace", side_effect=fail_second_publish):
                with self.assertRaisesRegex(download_models.ModelDownloadError, "rolled back"):
                    download_models.extract_models(
                        root / bundle["asset"],
                        bundle,
                        ["decoder.rknn", "encoder.rknn"],
                        manifest,
                        output,
                        force=True,
                    )
            self.assertTrue(failed)
            self.assertEqual(
                {name: (output / name).read_bytes() for name in old_payloads},
                old_payloads,
            )
            self.assertEqual(list(output.glob(".*.rollback")), [])

    def test_extract_interrupt_rolls_back_the_whole_model_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_bundle(root)
            output = root / "output"
            output.mkdir()
            old_payloads = {
                "decoder.rknn": b"old-decoder",
                "encoder.rknn": b"old-encoder",
            }
            for name, payload in old_payloads.items():
                (output / name).write_bytes(payload)
            bundle = manifest["release"]["bundles"]["sam"]
            real_replace = os.replace
            interrupted = False

            def interrupt_second_publish(source, destination):
                nonlocal interrupted
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    not interrupted
                    and source_path.name == "encoder.rknn"
                    and destination_path == output / "encoder.rknn"
                ):
                    interrupted = True
                    raise KeyboardInterrupt("injected publication interruption")
                return real_replace(source, destination)

            with patch.object(
                download_models.os, "replace", side_effect=interrupt_second_publish
            ):
                with self.assertRaises(KeyboardInterrupt):
                    download_models.extract_models(
                        root / bundle["asset"],
                        bundle,
                        ["decoder.rknn", "encoder.rknn"],
                        manifest,
                        output,
                        force=True,
                    )
            self.assertTrue(interrupted)
            self.assertEqual(
                {name: (output / name).read_bytes() for name in old_payloads},
                old_payloads,
            )
            self.assertEqual(list(output.glob(".*.rollback")), [])

    def test_extract_interrupt_after_backup_move_restores_old_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_bundle(root)
            output = root / "output"
            output.mkdir()
            old_payloads = {
                "decoder.rknn": b"old-decoder",
                "encoder.rknn": b"old-encoder",
            }
            for name, payload in old_payloads.items():
                (output / name).write_bytes(payload)
            bundle = manifest["release"]["bundles"]["sam"]
            real_replace = os.replace
            interrupted = False

            def interrupt_after_backup(source, destination):
                nonlocal interrupted
                source_path = Path(source)
                destination_path = Path(destination)
                real_replace(source, destination)
                if (
                    not interrupted
                    and source_path == output / "decoder.rknn"
                    and destination_path.parent == output
                    and destination_path.name.startswith(".decoder.rknn.")
                    and destination_path.name.endswith(".rollback")
                ):
                    interrupted = True
                    raise KeyboardInterrupt("interrupt after backup move")

            with patch.object(
                download_models.os, "replace", side_effect=interrupt_after_backup
            ):
                with self.assertRaises(KeyboardInterrupt):
                    download_models.extract_models(
                        root / bundle["asset"],
                        bundle,
                        ["decoder.rknn", "encoder.rknn"],
                        manifest,
                        output,
                        force=True,
                    )
            self.assertTrue(interrupted)
            self.assertEqual(
                {name: (output / name).read_bytes() for name in old_payloads},
                old_payloads,
            )
            self.assertEqual(list(output.glob(".*.rollback")), [])

    def test_extract_interrupt_after_first_publish_removes_partial_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = self.make_bundle(root)
            output = root / "output"
            output.mkdir()
            bundle = manifest["release"]["bundles"]["sam"]
            real_replace = os.replace
            interrupted = False

            def interrupt_after_publish(source, destination):
                nonlocal interrupted
                source_path = Path(source)
                destination_path = Path(destination)
                real_replace(source, destination)
                if (
                    not interrupted
                    and source_path.name == "decoder.rknn"
                    and source_path.parent.name.startswith("model-extract-")
                    and destination_path == output / "decoder.rknn"
                ):
                    interrupted = True
                    raise KeyboardInterrupt("interrupt after first publication")

            with patch.object(
                download_models.os, "replace", side_effect=interrupt_after_publish
            ):
                with self.assertRaises(KeyboardInterrupt):
                    download_models.extract_models(
                        root / bundle["asset"],
                        bundle,
                        ["decoder.rknn", "encoder.rknn"],
                        manifest,
                        output,
                        force=False,
                    )
            self.assertTrue(interrupted)
            self.assertFalse((output / "decoder.rknn").exists())
            self.assertFalse((output / "encoder.rknn").exists())
            self.assertEqual(list(output.glob(".*.rollback")), [])


class ReleaseBuilderTest(unittest.TestCase):
    def test_archive_is_deterministic_and_carries_notices(self):
        names = ("encoder.rknn", "decoder.rknn")
        payloads = {"encoder.rknn": b"encoder", "decoder.rknn": b"decoder"}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            models.mkdir()
            for name, payload in payloads.items():
                (models / name).write_bytes(payload)
            manifest = {
                "models": {
                    name: {"bytes": len(payload), "sha256": digest(payload)}
                    for name, payload in payloads.items()
                }
            }
            (root / "MODEL_RELEASES.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
            (root / "MODEL_LICENSES.md").write_text("# Notices\n", encoding="utf-8")
            (root / "MODEL_PROVENANCE.json").write_text("{}\n", encoding="utf-8")
            first = root / "first.zip"
            second = root / "second.zip"
            with patch.object(build_model_release, "ROOT", root), patch.object(
                build_model_release, "MODEL_NAMES", names
            ):
                first_result = build_model_release.build_archive(
                    models, first, verify_manifest=False
                )
                manifest["release"] = {
                    "bundles": {
                        build_model_release.BUNDLE_KEY: {
                            "asset": second.name,
                            "bytes": first_result["bytes"],
                            "sha256": first_result["sha256"],
                        }
                    }
                }
                (root / "MODEL_RELEASES.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                second_result = build_model_release.build_archive(models, second)
            self.assertEqual(first_result["sha256"], second_result["sha256"])
            with zipfile.ZipFile(first) as archive:
                members = archive.namelist()
                self.assertTrue(any(name.endswith("/LICENSE") for name in members))
                self.assertTrue(any(name.endswith("/MODEL_LICENSES.md") for name in members))
                self.assertTrue(any(name.endswith("/MODEL_PROVENANCE.json") for name in members))
                for name, payload in payloads.items():
                    member = next(item for item in members if item.endswith(f"/model/{name}"))
                    self.assertEqual(archive.read(member), payload)

    def test_archive_manifest_mismatch_is_a_build_failure(self):
        names = ("encoder.rknn", "decoder.rknn")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / "models"
            models.mkdir()
            manifest = {"models": {}, "release": {"bundles": {}}}
            for name in names:
                payload = name.encode("ascii")
                (models / name).write_bytes(payload)
                manifest["models"][name] = {
                    "bytes": len(payload),
                    "sha256": digest(payload),
                }
            manifest["release"]["bundles"][build_model_release.BUNDLE_KEY] = {
                "asset": "bundle.zip",
                "bytes": 1,
                "sha256": "0" * 64,
            }
            (root / "MODEL_RELEASES.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "LICENSE").write_text("license\n", encoding="utf-8")
            (root / "MODEL_LICENSES.md").write_text("notice\n", encoding="utf-8")
            (root / "MODEL_PROVENANCE.json").write_text("{}\n", encoding="utf-8")
            output = root / "bundle.zip"
            old_output = b"previous reviewed release asset"
            output.write_bytes(old_output)
            with patch.object(build_model_release, "ROOT", root), patch.object(
                build_model_release, "MODEL_NAMES", names
            ):
                with self.assertRaisesRegex(
                    build_model_release.ReleaseBuildError,
                    "does not match MODEL_RELEASES",
                ):
                    build_model_release.build_archive(models, output, force=True)
            self.assertEqual(output.read_bytes(), old_output)
            self.assertFalse((root / ".bundle.zip.part").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
