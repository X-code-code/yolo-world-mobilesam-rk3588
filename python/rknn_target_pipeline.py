#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""YOLO-World text -> bbox -> MobileSAM mask pipeline for RK3588.

This module intentionally depends only on NumPy, OpenCV and RKNNLite on the
board.  The CLIP BPE implementation lives in ``clip_tokenizer.py`` so that the
runtime does not require transformers/torch.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from clip_tokenizer import CLIPTokenizer


YOLO_SIZE = 640
SAM_SIZE = 448
SAM_LOW_RES = 112
TEXT_SLOTS = 80
TEXT_DIM = 512
TEXT_CONTEXT = 20
NPU_MODES = ("three-core", "serial-auto")


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    x_pad: int
    y_pad: int
    resized_width: int
    resized_height: int
    original_width: int
    original_height: int


@dataclass(frozen=True)
class Detection:
    target: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class StageTimings:
    clip_text_ms: float = 0.0
    yolo_preprocess_ms: float = 0.0
    yolo_inference_ms: float = 0.0
    yolo_postprocess_ms: float = 0.0
    sam_preprocess_ms: float = 0.0
    sam_encoder_ms: float = 0.0
    sam_decoder_ms: float = 0.0
    mask_postprocess_ms: float = 0.0
    parallel_region_ms: float = 0.0
    total_ms: float = 0.0
    embedding_cache_hit: bool = False


@dataclass
class PipelineResult:
    target: str
    detection: Optional[Detection]
    mask: Optional[np.ndarray]
    mask_iou: Optional[float]
    contour: Optional[np.ndarray]
    timings: StageTimings = field(default_factory=StageTimings)
    npu_mode: str = "serial-auto"


@dataclass(frozen=True)
class SamImageContext:
    """Immutable per-frame MobileSAM state used to prevent cross-frame masks."""

    embedding: np.ndarray
    image_shape: Tuple[int, int]
    resized_shape: Tuple[int, int]


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def yolo_letterbox(image_bgr: np.ndarray, size: int = YOLO_SIZE) -> Tuple[np.ndarray, LetterboxMeta]:
    """Match rknn_model_zoo letterbox geometry and alignment.

    The vendor helper truncates the non-limiting side, aligns width down to a
    multiple of four and height down to a multiple of two, then centres it with
    an even x/y offset. Padding colour is 114. OpenCV's half-pixel bilinear
    sampler can differ slightly from the vendor C fallback's pixel sampling;
    the scale, padding and inverse box transform are the same.
    """
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image must be a non-empty HxWx3 BGR array")
    src_h, src_w = image_bgr.shape[:2]
    if src_h < 1 or src_w < 1:
        raise ValueError("image dimensions must be positive")

    scale_w = size / float(src_w)
    scale_h = size / float(src_h)
    resize_w = size
    resize_h = size
    if scale_w < scale_h:
        scale = scale_w
        resize_h = int(src_h * scale)
    else:
        scale = scale_h
        resize_w = int(src_w * scale)

    if resize_w % 4:
        resize_w -= resize_w % 4
    if resize_h % 2:
        resize_h -= resize_h % 2
    resize_w = max(1, resize_w)
    resize_h = max(1, resize_h)

    pad_w = size - resize_w
    pad_h = size - resize_h
    x_pad = 0
    y_pad = 0
    if scale_w < scale_h:
        y_pad = pad_h // 2
        y_pad -= y_pad % 2
    else:
        x_pad = pad_w // 2
        x_pad -= x_pad % 2

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    resized = cv2.resize(image_bgr, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    canvas[y_pad : y_pad + resize_h, x_pad : x_pad + resize_w] = resized
    meta = LetterboxMeta(
        scale=scale,
        x_pad=x_pad,
        y_pad=y_pad,
        resized_width=resize_w,
        resized_height=resize_h,
        original_width=src_w,
        original_height=src_h,
    )
    return canvas, meta


def sam_resize_longest_side(image_bgr: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Match the official MobileSAM resize + bottom/right padding."""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image must be a non-empty HxWx3 BGR array")
    old_h, old_w = image_bgr.shape[:2]
    scale = SAM_SIZE / float(max(old_h, old_w))
    new_h = int(old_h * scale + 0.5)
    new_w = int(old_w * scale + 0.5)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((SAM_SIZE, SAM_SIZE, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    return padded, new_h, new_w


def sam_transform_bbox(
    bbox: Sequence[float], original_height: int, original_width: int, new_height: int, new_width: int
) -> np.ndarray:
    """Convert original-image xyxy to official MobileSAM decoder coordinates."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain x1,y1,x2,y2")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1 = min(max(x1, 0.0), original_width - 1.0)
    x2 = min(max(x2, 0.0), original_width - 1.0)
    y1 = min(max(y1, 0.0), original_height - 1.0)
    y2 = min(max(y2, 0.0), original_height - 1.0)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid bbox after clipping: {(x1, y1, x2, y2)}")
    return np.asarray(
        [[[x1 * new_width / original_width, y1 * new_height / original_height],
          [x2 * new_width / original_width, y2 * new_height / original_height]]],
        dtype=np.float32,
    )


def _canonical_nchw(array: np.ndarray, channels: int) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 4:
        raise ValueError(f"expected 4D tensor, got shape={arr.shape}")
    if arr.shape[1] == channels:
        return arr
    if arr.shape[-1] == channels:
        return arr.transpose(0, 3, 1, 2)
    raise ValueError(f"cannot find channel dimension {channels} in shape={arr.shape}")


def _canonical_nhwc(array: np.ndarray, channels: int) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 4:
        raise ValueError(f"expected 4D tensor, got shape={arr.shape}")
    if arr.shape[-1] == channels:
        return arr
    if arr.shape[1] == channels:
        return arr.transpose(0, 2, 3, 1)
    raise ValueError(f"cannot find channel dimension {channels} in shape={arr.shape}")


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1 + 1.0) * np.maximum(0.0, yy2 - yy1 + 1.0)
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[np.where(iou <= threshold)[0]]
    return keep


def yolo_postprocess(
    outputs: Sequence[np.ndarray],
    meta: LetterboxMeta,
    target: str,
    confidence_threshold: float,
    nms_threshold: float,
) -> List[Detection]:
    """Decode FP16 (6 outputs) or quantized (9 outputs) for one text class.

    Quantized model-zoo exports add one ``score_sum`` output to each of the
    three branches.  It is only an accelerator-side filtering aid; Python uses
    the score and box tensors at offsets 0 and 1 and ignores that third tensor.
    """
    if len(outputs) not in (6, 9):
        raise ValueError(f"YOLO-World expected 6 or 9 outputs, got {len(outputs)}")
    outputs_per_branch = len(outputs) // 3
    raw_boxes: List[Tuple[float, float, float, float]] = []
    raw_scores: List[float] = []

    for branch in range(3):
        offset = branch * outputs_per_branch
        score_tensor = _canonical_nchw(np.asarray(outputs[offset]), TEXT_SLOTS)
        box_tensor = _canonical_nchw(np.asarray(outputs[offset + 1]), 4)
        grid_h, grid_w = box_tensor.shape[2:4]
        stride = YOLO_SIZE // grid_h
        score_map = score_tensor[0, 0]
        ys, xs = np.where(score_map > confidence_threshold)
        for y, x in zip(ys.tolist(), xs.tolist()):
            score = float(score_map[y, x])
            distances = box_tensor[0, :, y, x].astype(np.float32)
            left = (-float(distances[0]) + x + 0.5) * stride
            top = (-float(distances[1]) + y + 0.5) * stride
            right = (float(distances[2]) + x + 0.5) * stride
            bottom = (float(distances[3]) + y + 0.5) * stride
            raw_boxes.append((left, top, right, bottom))
            raw_scores.append(score)

    if not raw_boxes:
        return []
    boxes = np.asarray(raw_boxes, dtype=np.float32)
    scores = np.asarray(raw_scores, dtype=np.float32)
    keep = _nms(boxes, scores, nms_threshold)
    detections: List[Detection] = []
    for index in keep:
        x1, y1, x2, y2 = boxes[index]
        x1 = (np.clip(x1 - meta.x_pad, 0, YOLO_SIZE) / meta.scale)
        y1 = (np.clip(y1 - meta.y_pad, 0, YOLO_SIZE) / meta.scale)
        x2 = (np.clip(x2 - meta.x_pad, 0, YOLO_SIZE) / meta.scale)
        y2 = (np.clip(y2 - meta.y_pad, 0, YOLO_SIZE) / meta.scale)
        ix1 = int(np.clip(x1, 0, meta.original_width - 1))
        iy1 = int(np.clip(y1, 0, meta.original_height - 1))
        ix2 = int(np.clip(x2, 0, meta.original_width - 1))
        iy2 = int(np.clip(y2, 0, meta.original_height - 1))
        if ix2 > ix1 and iy2 > iy1:
            detections.append(Detection(target, float(scores[index]), (ix1, iy1, ix2, iy2)))
    detections.sort(key=lambda item: item.confidence, reverse=True)
    return detections


def largest_external_contour(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


class MobileSAM:
    """Encapsulated set_image()/segment() MobileSAM API."""

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self._context: Optional[SamImageContext] = None

    def encode(self, image_bgr: np.ndarray, timings: StageTimings) -> SamImageContext:
        start = time.perf_counter()
        padded_rgb, new_h, new_w = sam_resize_longest_side(image_bgr)
        timings.sam_preprocess_ms = _elapsed_ms(start)
        start = time.perf_counter()
        outputs = self.encoder.inference(inputs=[padded_rgb[np.newaxis, ...]])
        timings.sam_encoder_ms = _elapsed_ms(start)
        if not outputs:
            raise RuntimeError("MobileSAM encoder returned no output")
        # Copy because RKNNLite may reuse its output buffer on a later frame.
        embedding = _canonical_nhwc(np.asarray(outputs[0]), 256).astype(np.float32, copy=False).copy()
        return SamImageContext(embedding, image_bgr.shape[:2], (new_h, new_w))

    def decode(
        self,
        context: SamImageContext,
        bbox: Sequence[float],
        timings: StageTimings,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        old_h, old_w = context.image_shape
        new_h, new_w = context.resized_shape
        coords = sam_transform_bbox(bbox, old_h, old_w, new_h, new_w)
        labels = np.asarray([[2.0, 3.0]], dtype=np.float32)
        mask_input = np.zeros((1, SAM_LOW_RES, SAM_LOW_RES, 1), dtype=np.float32)
        has_mask_input = np.zeros((1,), dtype=np.float32)

        start = time.perf_counter()
        outputs = self.decoder.inference(
            inputs=[context.embedding, coords, labels, mask_input, has_mask_input]
        )
        timings.sam_decoder_ms = _elapsed_ms(start)
        if outputs is None or len(outputs) != 2:
            raise RuntimeError(f"MobileSAM decoder expected 2 outputs, got {0 if outputs is None else len(outputs)}")

        start = time.perf_counter()
        iou = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        low_res = _canonical_nchw(np.asarray(outputs[1]), 4).astype(np.float32, copy=False)
        best = int(np.argmax(iou))
        logits = low_res[0, best]
        mask_448 = cv2.resize(logits, (SAM_SIZE, SAM_SIZE), interpolation=cv2.INTER_LINEAR)
        unpadded = mask_448[:new_h, :new_w]
        original_logits = cv2.resize(unpadded, (old_w, old_h), interpolation=cv2.INTER_LINEAR)
        mask = (original_logits > 0.0).astype(np.uint8)
        contour = largest_external_contour(mask)
        timings.mask_postprocess_ms = _elapsed_ms(start)
        return mask, float(iou[best]), contour

    def set_image(self, image_bgr: np.ndarray, timings: StageTimings) -> None:
        """Compatibility API; the pipeline itself uses explicit contexts."""
        self._context = self.encode(image_bgr, timings)

    def segment(self, bbox: Sequence[float], timings: StageTimings) -> Tuple[np.ndarray, float, np.ndarray]:
        if self._context is None:
            raise RuntimeError("set_image() must be called before segment()")
        return self.decode(self._context, bbox, timings)


class RknnTargetPipeline:
    def __init__(
        self,
        clip_model: os.PathLike,
        yolo_model: os.PathLike,
        sam_encoder_model: os.PathLike,
        sam_decoder_model: os.PathLike,
        merges_path: os.PathLike,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        npu_mode: str = "serial-auto",
        runtime_type=None,
    ) -> None:
        if npu_mode not in NPU_MODES:
            raise ValueError(f"npu_mode must be one of {NPU_MODES}, got {npu_mode!r}")
        if runtime_type is None:
            try:
                from rknnlite.api import RKNNLite
            except ImportError as exc:
                raise RuntimeError("rknn-toolkit-lite2 is required on the RK3588 board") from exc
            runtime_type = RKNNLite

        self._rknn_lite_type = runtime_type
        self.npu_mode = npu_mode
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.tokenizer = CLIPTokenizer(merges_path)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._encoder_executor: Optional[ThreadPoolExecutor] = None
        self._decoder_executor: Optional[ThreadPoolExecutor] = None
        self._yolo_executor: Optional[ThreadPoolExecutor] = None
        self.clip = None
        self.yolo = None
        self.sam_encoder = None
        self.sam_decoder = None
        self.sam: Optional[MobileSAM] = None
        try:
            if self.npu_mode == "three-core":
                self._encoder_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rknn-core0-encoder")
                self._decoder_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rknn-core1-text-decoder")
                self._yolo_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rknn-core2-yolo")
                core0 = self._rknn_lite_type.NPU_CORE_0
                core1 = self._rknn_lite_type.NPU_CORE_1
                core2 = self._rknn_lite_type.NPU_CORE_2
                self.sam_encoder = self._encoder_executor.submit(
                    self._load_model, sam_encoder_model, "MobileSAM encoder", core0
                ).result()
                self.clip = self._decoder_executor.submit(
                    self._load_model, clip_model, "CLIP text", core1
                ).result()
                self.sam_decoder = self._decoder_executor.submit(
                    self._load_model, sam_decoder_model, "MobileSAM decoder", core1
                ).result()
                self.yolo = self._yolo_executor.submit(
                    self._load_model, yolo_model, "YOLO-World", core2
                ).result()
            else:
                self.clip = self._load_model(clip_model, "CLIP text")
                self.yolo = self._load_model(yolo_model, "YOLO-World")
                self.sam_encoder = self._load_model(sam_encoder_model, "MobileSAM encoder")
                self.sam_decoder = self._load_model(sam_decoder_model, "MobileSAM decoder")
        except Exception:
            self.release()
            raise
        self.sam = MobileSAM(self.sam_encoder, self.sam_decoder)

    def _load_model(self, path: os.PathLike, label: str, core_mask=None):
        model_path = str(Path(path).expanduser().resolve())
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"{label} model not found: {model_path}")
        runtime = self._rknn_lite_type(verbose=False)
        ret = runtime.load_rknn(model_path)
        if ret != 0:
            runtime.release()
            raise RuntimeError(f"load_rknn failed for {label}: ret={ret}, path={model_path}")
        ret = runtime.init_runtime() if core_mask is None else runtime.init_runtime(core_mask=core_mask)
        if ret != 0:
            runtime.release()
            raise RuntimeError(f"init_runtime failed for {label}: ret={ret}, path={model_path}")
        return runtime

    def _release_owned(self, attributes: Sequence[str]) -> None:
        for attribute in attributes:
            runtime = getattr(self, attribute)
            if runtime is not None:
                try:
                    runtime.release()
                finally:
                    setattr(self, attribute, None)

    def release(self) -> None:
        if self.npu_mode == "three-core":
            owner_groups = (
                ("_yolo_executor", ("yolo",)),
                ("_encoder_executor", ("sam_encoder",)),
                ("_decoder_executor", ("sam_decoder", "clip")),
            )
            for executor_attribute, runtime_attributes in owner_groups:
                executor = getattr(self, executor_attribute)
                if executor is None:
                    self._release_owned(runtime_attributes)
                    continue
                try:
                    executor.submit(self._release_owned, runtime_attributes).result()
                finally:
                    executor.shutdown(wait=True, cancel_futures=False)
                    setattr(self, executor_attribute, None)
        else:
            self._release_owned(("sam_decoder", "sam_encoder", "yolo", "clip"))
        self.sam = None

    def __enter__(self) -> "RknnTargetPipeline":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()

    @staticmethod
    def normalize_target(target: str) -> str:
        normalized = " ".join(str(target).strip().split())
        if not normalized:
            raise ValueError("target must not be empty")
        return normalized

    def _text_embedding(self, target: str, timings: StageTimings) -> np.ndarray:
        if target in self._embedding_cache:
            timings.embedding_cache_hit = True
            return self._embedding_cache[target]
        token_ids = self.tokenizer.tokenize(target, context_length=TEXT_CONTEXT)
        token_array = np.asarray(token_ids, dtype=np.int32).reshape(1, TEXT_CONTEXT)
        start = time.perf_counter()
        outputs = self.clip.inference(inputs=[token_array])
        timings.clip_text_ms = _elapsed_ms(start)
        if not outputs:
            raise RuntimeError("CLIP text model returned no output")
        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(1, TEXT_DIM)
        self._embedding_cache[target] = embedding.copy()
        return embedding

    def _detect_with_embedding(
        self,
        image_bgr: np.ndarray,
        target: str,
        embedding: np.ndarray,
        timings: StageTimings,
    ) -> List[Detection]:
        texts = np.zeros((1, TEXT_SLOTS, TEXT_DIM), dtype=np.float32)
        texts[0, 0] = embedding[0]
        start = time.perf_counter()
        padded_bgr, meta = yolo_letterbox(image_bgr)
        image_rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
        timings.yolo_preprocess_ms = _elapsed_ms(start)
        start = time.perf_counter()
        outputs = self.yolo.inference(inputs=[image_rgb[np.newaxis, ...], texts])
        timings.yolo_inference_ms = _elapsed_ms(start)
        start = time.perf_counter()
        detections = yolo_postprocess(
            outputs,
            meta,
            target,
            self.confidence_threshold,
            self.nms_threshold,
        )
        timings.yolo_postprocess_ms = _elapsed_ms(start)
        return detections

    def detect(self, image_bgr: np.ndarray, target: str, timings: StageTimings) -> List[Detection]:
        target = self.normalize_target(target)
        if self.npu_mode == "three-core":
            embedding = self._decoder_executor.submit(self._text_embedding, target, timings).result()
            return self._yolo_executor.submit(
                self._detect_with_embedding, image_bgr, target, embedding, timings
            ).result()
        embedding = self._text_embedding(target, timings)
        return self._detect_with_embedding(image_bgr, target, embedding, timings)

    def process(self, image_bgr: np.ndarray, target: str, enable_sam: bool = True) -> PipelineResult:
        total_start = time.perf_counter()
        target = self.normalize_target(target)
        timings = StageTimings()
        sam_context: Optional[SamImageContext] = None
        if self.npu_mode == "three-core":
            parallel_start = time.perf_counter()
            encoder_future = None
            if enable_sam:
                encoder_future = self._encoder_executor.submit(self.sam.encode, image_bgr, timings)
            embedding_future = self._decoder_executor.submit(self._text_embedding, target, timings)
            try:
                embedding = embedding_future.result()
            except Exception:
                if encoder_future is not None:
                    wait((encoder_future,))
                raise
            yolo_future = self._yolo_executor.submit(
                self._detect_with_embedding, image_bgr, target, embedding, timings
            )
            active_futures = [yolo_future]
            if encoder_future is not None:
                active_futures.append(encoder_future)
            wait(active_futures)
            detections = yolo_future.result()
            if encoder_future is not None:
                sam_context = encoder_future.result()
            timings.parallel_region_ms = _elapsed_ms(parallel_start)
        else:
            detections = self.detect(image_bgr, target, timings)
        selected = detections[0] if detections else None
        mask: Optional[np.ndarray] = None
        mask_iou: Optional[float] = None
        contour: Optional[np.ndarray] = None
        if selected is not None and enable_sam:
            if self.npu_mode == "three-core":
                mask, mask_iou, contour = self._decoder_executor.submit(
                    self.sam.decode, sam_context, selected.bbox, timings
                ).result()
            else:
                self.sam.set_image(image_bgr, timings)
                mask, mask_iou, contour = self.sam.segment(selected.bbox, timings)
        timings.total_ms = _elapsed_ms(total_start)
        return PipelineResult(target, selected, mask, mask_iou, contour, timings, self.npu_mode)


def draw_detection(image_bgr: np.ndarray, result: PipelineResult) -> np.ndarray:
    output = image_bgr.copy()
    if result.detection is None:
        cv2.putText(output, f"{result.target}: not found", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output
    x1, y1, x2, y2 = result.detection.bbox
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{result.target} {result.detection.confidence:.3f}"
    cv2.putText(output, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return output


def draw_mask(image_bgr: np.ndarray, result: PipelineResult) -> np.ndarray:
    output = draw_detection(image_bgr, result)
    if result.mask is None:
        return output
    coloured = np.zeros_like(output)
    coloured[:, :, 1] = 180
    coloured[:, :, 2] = 60
    active = result.mask.astype(bool)
    output[active] = cv2.addWeighted(output, 0.45, coloured, 0.55, 0)[active]
    return output


def draw_contour(image_bgr: np.ndarray, result: PipelineResult) -> np.ndarray:
    output = image_bgr.copy()
    if result.contour is not None:
        cv2.drawContours(output, [result.contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
    label = result.target
    if result.mask_iou is not None:
        label += f" mask_iou={result.mask_iou:.3f}"
    cv2.putText(output, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return output


def save_static_results(output_dir: os.PathLike, image_bgr: np.ndarray, result: PipelineResult) -> Dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "detection": str(destination / "01_yolo_detection.png"),
        "mask": str(destination / "02_mobilesam_mask.png"),
        "contour": str(destination / "03_target_contour.png"),
        "metrics": str(destination / "metrics.json"),
    }
    images = (draw_detection(image_bgr, result), draw_mask(image_bgr, result), draw_contour(image_bgr, result))
    for key, rendered in zip(("detection", "mask", "contour"), images):
        if not cv2.imwrite(paths[key], rendered):
            raise RuntimeError(f"failed to write {paths[key]}")
    payload = {
        "target": result.target,
        "npu_mode": result.npu_mode,
        "detection": None if result.detection is None else asdict(result.detection),
        "mask_iou": result.mask_iou,
        "mask_pixels": None if result.mask is None else int(np.count_nonzero(result.mask)),
        "contour_area": None if result.contour is None else float(cv2.contourArea(result.contour)),
        "timings_ms": asdict(result.timings),
        "outputs": paths,
    }
    Path(paths["metrics"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    deploy = Path(__file__).resolve().parent.parent / "model"
    parser.add_argument("--clip-model", default=str(deploy / "clip_text_fp16.rknn"))
    parser.add_argument("--yolo-model", default=str(deploy / "yolo_world_v2s_i8.rknn"))
    parser.add_argument("--sam-encoder-model", default=str(deploy / "mobilesam_encoder_fp16.rknn"))
    parser.add_argument("--sam-decoder-model", default=str(deploy / "mobilesam_decoder_fp16.rknn"))
    parser.add_argument("--merges", default=str(Path(__file__).with_name("bpe_simple_vocab_16e6.txt")))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument(
        "--npu-mode",
        choices=NPU_MODES,
        default="three-core",
        help="three-core pins YOLO/Encoder/Decoder to separate RK3588 cores; serial-auto is the compatibility fallback",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Static YOLO-World -> MobileSAM RKNN validation")
    add_model_arguments(parser)
    parser.add_argument("--image", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-sam", action="store_true")
    args = parser.parse_args(argv)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"cannot read image: {args.image}")
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
        result = pipeline.process(image, args.target, enable_sam=not args.no_sam)
        outputs = save_static_results(args.output_dir, image, result)
    print(json.dumps({
        "target": result.target,
        "detection": None if result.detection is None else asdict(result.detection),
        "mask_iou": result.mask_iou,
        "npu_mode": result.npu_mode,
        "timings_ms": asdict(result.timings),
        "outputs": outputs,
    }, ensure_ascii=False, indent=2))
    return 0 if result.detection is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
