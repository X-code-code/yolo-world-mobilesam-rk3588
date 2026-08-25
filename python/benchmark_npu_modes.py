#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark serial and three-core RK3588 execution on one static image."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from rknn_target_pipeline import RknnTargetPipeline


BENCHMARK_MODES = ("serial-auto", "three-core")
TIMING_FIELDS = {
    "total": "total_ms",
    "parallel": "parallel_region_ms",
    "yolo": "yolo_inference_ms",
    "encoder": "sam_encoder_ms",
    "decoder": "sam_decoder_ms",
}
FLOAT_TOLERANCE = 1e-6


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile without a NumPy API dependency."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty sample")
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _timing_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("at least one measured run is required")
    return {
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": float(_percentile(values, 0.50)),
        "p90_ms": float(_percentile(values, 0.90)),
    }


def _optional_float_equal(left: Optional[float], right: Optional[float]) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=FLOAT_TOLERANCE)


def _result_signature(result) -> Dict[str, object]:
    detection = result.detection
    mask = result.mask
    mask_bytes = None
    if mask is not None:
        mask_bytes = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
    return {
        "confidence": None if detection is None else float(detection.confidence),
        "bbox": None if detection is None else [int(value) for value in detection.bbox],
        "mask_iou": None if result.mask_iou is None else float(result.mask_iou),
        "mask_pixels": None if mask is None else int(np.count_nonzero(mask)),
        "mask_sha256": mask_bytes,
    }


def _signatures_equal(left: Dict[str, object], right: Dict[str, object]) -> bool:
    return (
        _optional_float_equal(left["confidence"], right["confidence"])
        and left["bbox"] == right["bbox"]
        and _optional_float_equal(left["mask_iou"], right["mask_iou"])
        and left["mask_pixels"] == right["mask_pixels"]
        and left["mask_sha256"] == right["mask_sha256"]
    )


def _run_mode(args, image: np.ndarray, mode: str) -> Tuple[Dict[str, object], Dict[str, object]]:
    pipeline = None
    try:
        pipeline = RknnTargetPipeline(
            args.clip_model,
            args.yolo_model,
            args.sam_encoder_model,
            args.sam_decoder_model,
            args.merges,
            args.conf,
            args.nms,
            npu_mode=mode,
        )

        for _ in range(args.warmup):
            pipeline.process(image, args.target, enable_sam=not args.no_sam)

        samples: Dict[str, List[float]] = {name: [] for name in TIMING_FIELDS}
        signatures: List[Dict[str, object]] = []
        measured_start = time.perf_counter()
        for _ in range(args.runs):
            result = pipeline.process(image, args.target, enable_sam=not args.no_sam)
            for output_name, timing_attribute in TIMING_FIELDS.items():
                samples[output_name].append(float(getattr(result.timings, timing_attribute)))
            signatures.append(_result_signature(result))
        measured_wall_ms = (time.perf_counter() - measured_start) * 1000.0

        reference = signatures[0]
        within_mode_consistent = all(
            _signatures_equal(reference, signature) for signature in signatures[1:]
        )
        total_mean_ms = statistics.fmean(samples["total"])
        mode_output: Dict[str, object] = {
            "runs": args.runs,
            "total": _timing_summary(samples["total"]),
            "parallel": _timing_summary(samples["parallel"]),
            "yolo": _timing_summary(samples["yolo"]),
            "encoder": _timing_summary(samples["encoder"]),
            "decoder": _timing_summary(samples["decoder"]),
            "fps": float(1000.0 / total_mean_ms) if total_mean_ms > 0.0 else None,
            "wall_fps": float(args.runs * 1000.0 / measured_wall_ms) if measured_wall_ms > 0.0 else None,
            "confidence": reference["confidence"],
            "bbox": reference["bbox"],
            "mask_iou": reference["mask_iou"],
            "mask_pixels": reference["mask_pixels"],
            "mask_sha256": reference["mask_sha256"],
            "within_mode_consistent": within_mode_consistent,
        }
        return mode_output, reference
    finally:
        if pipeline is not None:
            pipeline.release()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark serial-auto versus three-core YOLO-World -> MobileSAM execution"
    )
    deploy = Path(__file__).resolve().parent.parent / "model"
    parser.add_argument("--clip-model", default=str(deploy / "clip_text_fp16.rknn"))
    parser.add_argument("--yolo-model", default=str(deploy / "yolo_world_v2s_i8.rknn"))
    parser.add_argument("--sam-encoder-model", default=str(deploy / "mobilesam_encoder_fp16.rknn"))
    parser.add_argument("--sam-decoder-model", default=str(deploy / "mobilesam_decoder_fp16.rknn"))
    parser.add_argument("--merges", default=str(Path(__file__).with_name("bpe_simple_vocab_16e6.txt")))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--image", required=True, help="fixed image used for every warm-up and measured run")
    parser.add_argument("--target", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--no-sam", action="store_true")
    parser.add_argument("--output", help="write the JSON report to this path (recommended because RKNN logs use stdout)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("warmup must be non-negative")
    if args.runs < 1:
        parser.error("runs must be at least 1")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("conf must be between 0 and 1")
    if not 0.0 <= args.nms <= 1.0:
        parser.error("nms must be between 0 and 1")

    image_path = Path(args.image).expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"cannot read image: {image_path}")

    payload: Dict[str, object] = {
        "image": str(image_path),
        "target": RknnTargetPipeline.normalize_target(args.target),
        "warmup": args.warmup,
        "runs": args.runs,
        "sam_enabled": not args.no_sam,
        "models": {
            "clip": str(Path(args.clip_model).expanduser().resolve()),
            "yolo": str(Path(args.yolo_model).expanduser().resolve()),
            "sam_encoder": str(Path(args.sam_encoder_model).expanduser().resolve()),
            "sam_decoder": str(Path(args.sam_decoder_model).expanduser().resolve()),
        },
        "modes": {},
    }

    references: Dict[str, Dict[str, object]] = {}
    for mode in BENCHMARK_MODES:
        mode_output, reference = _run_mode(args, image, mode)
        payload["modes"][mode] = mode_output
        references[mode] = reference

    serial_reference = references["serial-auto"]
    cross_mode_consistent = _signatures_equal(serial_reference, references["three-core"])
    payload["result_consistency"] = {
        "serial_auto_within_mode": payload["modes"]["serial-auto"]["within_mode_consistent"],
        "three_core_within_mode": payload["modes"]["three-core"]["within_mode_consistent"],
        "three_core_matches_serial_auto": cross_mode_consistent,
        "all_results_consistent": bool(
            payload["modes"]["serial-auto"]["within_mode_consistent"]
            and payload["modes"]["three-core"]["within_mode_consistent"]
            and cross_mode_consistent
        ),
        "float_tolerance": FLOAT_TOLERANCE,
    }
    report = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
