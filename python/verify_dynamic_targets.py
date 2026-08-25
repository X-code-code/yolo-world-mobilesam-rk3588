#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify target changes and CLIP embedding reuse in one RKNNLite process."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2

from rknn_target_pipeline import RknnTargetPipeline, add_model_arguments, save_static_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--image", required=True)
    parser.add_argument("--targets", nargs="+", default=["cup", "cat", "cup"])
    parser.add_argument("--output-dir", default="results/dynamic_targets")
    parser.add_argument("--with-sam", action="store_true")
    args = parser.parse_args()

    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"cannot read image: {args.image}")
    output_root = Path(args.output_dir)
    summaries = []
    with RknnTargetPipeline(
        args.clip_model,
        args.yolo_model,
        args.sam_encoder_model,
        args.sam_decoder_model,
        args.merges,
        args.conf,
        args.nms,
        npu_mode=args.npu_mode,
    ) as pipeline:
        for index, target in enumerate(args.targets, start=1):
            result = pipeline.process(image, target, enable_sam=args.with_sam)
            destination = output_root / f"{index:02d}_{result.target.replace(' ', '_')}"
            paths = save_static_results(destination, image, result)
            summaries.append({
                "index": index,
                "target": result.target,
                "npu_mode": result.npu_mode,
                "detected": result.detection is not None,
                "detection": None if result.detection is None else asdict(result.detection),
                "mask_iou": result.mask_iou,
                "embedding_cache_hit": result.timings.embedding_cache_hit,
                "clip_text_ms": result.timings.clip_text_ms,
                "total_ms": result.timings.total_ms,
                "outputs": paths,
            })
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "dynamic_targets.json"
    report_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"npu_mode": args.npu_mode, "runs": summaries, "report": str(report_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
