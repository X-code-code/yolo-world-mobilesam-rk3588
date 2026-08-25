#!/usr/bin/env python3
"""Download redistributable model assets from this project's GitHub Release."""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "MODEL_RELEASES.json"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ModelDownloadError(RuntimeError):
    """Raised when a model cannot be selected, downloaded, or verified."""


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ModelDownloadError(f"missing or non-string {label}")
    component = value
    if not SAFE_COMPONENT.fullmatch(component):
        raise ModelDownloadError(f"unsafe {label}: {component!r}")
    return component


def validate_manifest(manifest: dict[str, Any], path: Path) -> None:
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), dict):
        raise ModelDownloadError(f"unsupported model manifest: {path}")
    release = manifest.get("release")
    if not isinstance(release, dict) or not isinstance(release.get("bundles"), dict):
        raise ModelDownloadError(f"manifest has no Release bundles: {path}")
    bundles = release["bundles"]
    for bundle_name, bundle in bundles.items():
        _safe_component(bundle_name, "bundle name")
        if not isinstance(bundle, dict):
            raise ModelDownloadError(f"invalid bundle entry: {bundle_name}")
        asset = _safe_component(bundle.get("asset"), "bundle asset")
        if not asset.endswith(".zip"):
            raise ModelDownloadError(f"bundle asset is not a zip: {asset}")
        _safe_component(bundle.get("archive_root"), "archive root")
    for name, entry in manifest["models"].items():
        safe_name = _safe_component(name, "model name")
        if not safe_name.endswith(".rknn") or not isinstance(entry, dict):
            raise ModelDownloadError(f"invalid model entry: {name}")
        distribution = entry.get("distribution")
        if not isinstance(distribution, dict):
            raise ModelDownloadError(f"missing distribution entry for {name}")
        if distribution.get("status") == "release_bundle":
            bundle_name = _safe_component(distribution.get("bundle"), "model bundle")
            if bundle_name not in bundles:
                raise ModelDownloadError(f"model references unknown bundle: {name}")
            if distribution.get("member") != f"model/{safe_name}":
                raise ModelDownloadError(f"unsafe or inconsistent archive member for {name}")


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest, path)
    return manifest


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(path: Path, entry: dict[str, Any]) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    expected_size = int(entry["bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        return False, f"size mismatch: expected {expected_size}, got {actual_size}"
    actual_hash = sha256_file(path)
    expected_hash = str(entry["sha256"]).lower()
    if actual_hash != expected_hash:
        return False, f"SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
    return True, "verified"


def released_model_names(manifest: dict[str, Any]) -> list[str]:
    return sorted(
        name
        for name, entry in manifest["models"].items()
        if entry.get("distribution", {}).get("status") == "release_bundle"
    )


def select_models(manifest: dict[str, Any], requested: Iterable[str] | None) -> list[str]:
    if requested:
        names = list(dict.fromkeys(requested))
    else:
        names = released_model_names(manifest)
    unknown = [name for name in names if name not in manifest["models"]]
    if unknown:
        raise ModelDownloadError(f"unknown model name(s): {', '.join(unknown)}")
    unavailable = []
    for name in names:
        distribution = manifest["models"][name].get("distribution", {})
        if distribution.get("status") != "release_bundle":
            reason = distribution.get("reason", "not published as a Release asset")
            unavailable.append(f"{name}: {reason}")
    if unavailable:
        raise ModelDownloadError(
            "requested model(s) must be converted locally:\n  " + "\n  ".join(unavailable)
        )
    return names


def download_bundle(
    bundle: dict[str, Any],
    destination: Path,
    base_url: str,
    *,
    timeout: float = 120.0,
) -> Path:
    asset = bundle["asset"]
    url = f"{base_url.rstrip('/')}/{asset}"
    request = urllib.request.Request(url, headers={"User-Agent": "yolo-world-mobilesam-rk3588/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        if destination.stat().st_size != int(bundle["bytes"]):
            raise ModelDownloadError(f"downloaded bundle size mismatch: {asset}")
        actual_hash = sha256_file(destination)
        if actual_hash != bundle["sha256"]:
            raise ModelDownloadError(
                f"downloaded bundle SHA-256 mismatch: expected {bundle['sha256']}, got {actual_hash}"
            )
    except (OSError, urllib.error.URLError) as exc:
        raise ModelDownloadError(f"failed to download {url}: {exc}") from exc
    print(f"downloaded and verified bundle {asset}")
    return destination


def extract_models(
    archive_path: Path,
    bundle: dict[str, Any],
    names: list[str],
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    force: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_root = bundle["archive_root"].strip("/")
    try:
        with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(
            prefix="model-extract-", dir=output_dir
        ) as stage_directory:
            stage_root = Path(stage_directory)
            pending_names: list[str] = []
            for name in names:
                entry = manifest["models"][name]
                destination = output_dir / name
                if destination.exists() and not force:
                    ok, detail = verify_model(destination, entry)
                    if ok:
                        print(f"verified existing {destination}")
                        continue
                    raise ModelDownloadError(
                        f"refusing to replace {destination}: {detail}; pass --force"
                    )
                pending_names.append(name)
                member = entry["distribution"]["member"].lstrip("/")
                archive_member = f"{archive_root}/{member}"
                if archive_member not in archive.namelist():
                    raise ModelDownloadError(f"bundle is missing expected member: {archive_member}")
                staged = stage_root / name
                with archive.open(archive_member) as source, staged.open("xb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                ok, detail = verify_model(staged, entry)
                if not ok:
                    raise ModelDownloadError(f"extracted {name} failed verification: {detail}")

            backups: dict[Path, Path] = {}
            published: set[Path] = set()
            try:
                for name in pending_names:
                    destination = output_dir / name
                    if destination.exists():
                        backup = output_dir / f".{name}.{uuid.uuid4().hex}.rollback"
                        backups[destination] = backup
                        os.replace(destination, backup)
                    published.add(destination)
                    os.replace(stage_root / name, destination)
            except BaseException as exc:
                rollback_errors: list[str] = []
                for name in reversed(pending_names):
                    destination = output_dir / name
                    backup = backups.get(destination)
                    try:
                        if backup is not None and backup.exists():
                            os.replace(backup, destination)
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
                    retained = [str(path) for path in backups.values() if path.exists()]
                    raise ModelDownloadError(
                        "model publication failed and rollback was incomplete; "
                        f"retained backups={retained}; errors={rollback_errors}"
                    ) from exc
                if isinstance(exc, Exception):
                    raise ModelDownloadError(
                        "model publication failed; previous model set was rolled back"
                    ) from exc
                raise
            else:
                for backup in backups.values():
                    try:
                        backup.unlink(missing_ok=True)
                    except OSError:
                        # The complete new set is already published. Retaining
                        # a hidden backup is safer than reporting a false
                        # download failure after the commit point.
                        pass
            for name in pending_names:
                print(f"extracted and verified {output_dir / name}")
    except zipfile.BadZipFile as exc:
        raise ModelDownloadError(f"invalid Release bundle: {archive_path}") from exc


def download_selected(
    selected: list[str],
    manifest: dict[str, Any],
    output_dir: Path,
    base_url: str,
    *,
    force: bool = False,
) -> None:
    validate_manifest(manifest, Path("<in-memory manifest>"))
    pending: dict[str, list[str]] = {}
    for name in selected:
        destination = output_dir / name
        entry = manifest["models"][name]
        if destination.exists() and not force:
            ok, detail = verify_model(destination, entry)
            if ok:
                print(f"verified existing {destination}")
                continue
            raise ModelDownloadError(f"refusing to replace {destination}: {detail}; pass --force")
        bundle_name = entry["distribution"]["bundle"]
        pending.setdefault(bundle_name, []).append(name)
    if not pending:
        return

    bundles = manifest["release"]["bundles"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="model-download-", dir=output_dir) as temp_dir:
        temp_root = Path(temp_dir)
        for bundle_name, names in pending.items():
            if bundle_name not in bundles:
                raise ModelDownloadError(f"manifest references unknown bundle: {bundle_name}")
            bundle = bundles[bundle_name]
            archive_path = temp_root / bundle["asset"]
            download_bundle(bundle, archive_path, base_url)
            extract_models(archive_path, bundle, names, manifest, output_dir, force=force)


def print_inventory(manifest: dict[str, Any]) -> None:
    for name, entry in manifest["models"].items():
        distribution = entry.get("distribution", {})
        status = distribution.get("status", "unknown")
        print(f"{name}\t{status}\t{entry['bytes']} bytes\t{entry['sha256']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and SHA-256 verify model assets that this project is allowed to publish. "
            "Models not present in the Release must be converted locally."
        )
    )
    parser.add_argument("--model", action="append", dest="models", help="model filename; repeat as needed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base-url", help="override the Release download base URL")
    parser.add_argument("--verify-only", action="store_true", help="do not use the network")
    parser.add_argument("--force", action="store_true", help="replace a local file that fails verification")
    parser.add_argument("--list", action="store_true", help="show all known artifacts and distribution status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        if args.list:
            print_inventory(manifest)
            return 0
        selected = select_models(manifest, args.models)
        base_url = args.base_url or manifest["release"]["download_base_url"]
        for name in selected:
            entry = manifest["models"][name]
            destination = args.output_dir / name
            if args.verify_only:
                ok, detail = verify_model(destination, entry)
                if not ok:
                    raise ModelDownloadError(f"{destination}: {detail}")
                print(f"verified {destination}")
        if not args.verify_only:
            download_selected(selected, manifest, args.output_dir, base_url, force=args.force)
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ModelDownloadError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
