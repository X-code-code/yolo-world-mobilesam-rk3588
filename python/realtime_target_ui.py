#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live camera UI for dynamic YOLO-World targets and MobileSAM masks.

Window mode is intended for a terminal on the LubanCat desktop. Web mode is
intended for SSH use and exposes an MJPEG page on loopback by default.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from dataclasses import asdict
from enum import Enum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Sequence
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from rknn_target_pipeline import (
    PipelineResult,
    RknnTargetPipeline,
    add_model_arguments,
    draw_mask,
)


def _ascii_overlay(text: str) -> str:
    return text.encode("ascii", errors="replace").decode("ascii")


def _safe_print(*values, **kwargs) -> None:
    """Keep a closed SSH stdout pipe from becoming an inference error."""
    try:
        print(*values, **kwargs)
    except (BrokenPipeError, OSError, ValueError):
        pass


class PublishOutcome(str, Enum):
    """Result publication state; stale work is not a terminal condition."""

    PUBLISHED = "published"
    STALE = "stale"
    STOPPED = "stopped"


class LiveState:
    def __init__(self, initial_target: str, sam_enabled: bool, npu_mode: str = "unknown") -> None:
        self.condition = threading.Condition(threading.RLock())
        self.stop_event = threading.Event()
        self.capture_frame: Optional[np.ndarray] = None
        self.capture_sequence = 0
        self.capture_timestamp = 0.0
        self.capture_fps = 0.0
        self.display_frame: Optional[np.ndarray] = None
        self.display_sequence = 0
        self.render_sequence = 0
        self.display_timestamp = 0.0
        self.inference_fps = 0.0
        self.model_ready = False
        self.active_target = initial_target
        self.pending_target: Optional[str] = initial_target
        self.sam_enabled = sam_enabled
        self.config_generation = 0
        self.npu_mode = npu_mode
        self.status_message = "starting"
        self.error: Optional[str] = None
        self.fatal_error: Optional[str] = None
        self.last_result: Optional[Dict] = None
        self.inference_count = 0
        self.inference_error_count = 0

    def stop(self) -> None:
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()

    def publish_capture(self, frame: np.ndarray, now: float) -> None:
        with self.condition:
            if self.capture_timestamp > 0:
                instantaneous = 1.0 / max(now - self.capture_timestamp, 1e-6)
                self.capture_fps = instantaneous if self.capture_fps == 0 else 0.9 * self.capture_fps + 0.1 * instantaneous
            self.capture_frame = frame
            self.capture_timestamp = now
            self.capture_sequence += 1
            if self.display_frame is None:
                self.render_sequence += 1
            self.condition.notify_all()

    def wait_for_capture(self, after_sequence: int, timeout: float = 0.5):
        with self.condition:
            self.condition.wait_for(
                lambda: self.stop_event.is_set() or self.capture_sequence > after_sequence,
                timeout=timeout,
            )
            if self.stop_event.is_set() or self.capture_frame is None or self.capture_sequence <= after_sequence:
                return None
            target = self.pending_target if self.pending_target is not None else self.active_target
            return (
                self.capture_frame.copy(),
                self.capture_sequence,
                target,
                self.sam_enabled,
                self.config_generation,
            )

    def submit_target(self, target: str) -> str:
        normalized = RknnTargetPipeline.normalize_target(target)
        if len(normalized.encode("utf-8")) > 256:
            raise ValueError("target is too long (maximum 256 UTF-8 bytes)")
        with self.condition:
            self.config_generation += 1
            self.pending_target = normalized
            self.status_message = f"target pending: {normalized}"
            self.condition.notify_all()
        return normalized

    def set_sam_enabled(self, enabled: bool) -> None:
        with self.condition:
            enabled = bool(enabled)
            if enabled != self.sam_enabled:
                self.config_generation += 1
            self.sam_enabled = enabled
            self.status_message = "MobileSAM enabled" if enabled else "MobileSAM disabled"
            self.condition.notify_all()

    def publish_result(
        self,
        frame: np.ndarray,
        target: str,
        result: PipelineResult,
        config_generation: int,
    ) -> PublishOutcome:
        now = time.monotonic()
        with self.condition:
            # A camera/model fatal error is terminal.  In-flight RKNN work may
            # finish after stop() was requested, but it must not clear the
            # terminal error or turn a failed run into a successful one.
            if self.stop_event.is_set() or self.fatal_error is not None:
                return PublishOutcome.STOPPED
            if config_generation != self.config_generation:
                return PublishOutcome.STALE
            if self.display_timestamp > 0:
                instantaneous = 1.0 / max(now - self.display_timestamp, 1e-6)
                self.inference_fps = instantaneous if self.inference_fps == 0 else 0.8 * self.inference_fps + 0.2 * instantaneous
            self.display_frame = frame
            self.display_timestamp = now
            self.display_sequence += 1
            self.render_sequence += 1
            self.inference_count += 1
            self.active_target = target
            if self.pending_target == target:
                self.pending_target = None
            self.error = None
            if result.detection is None:
                self.status_message = f"{target}: not found"
            elif result.mask is None:
                self.status_message = f"{target}: detected"
            else:
                self.status_message = f"{target}: detected + segmented"
            self.last_result = {
                "target": target,
                "detection": None if result.detection is None else asdict(result.detection),
                "mask_iou": result.mask_iou,
                "npu_mode": result.npu_mode,
                "parallel_region_ms": result.timings.parallel_region_ms,
                "timings_ms": asdict(result.timings),
            }
            self.npu_mode = result.npu_mode
            self.model_ready = True
            self.condition.notify_all()
            return PublishOutcome.PUBLISHED

    def publish_inference_error(
        self,
        frame: np.ndarray,
        target: str,
        message: str,
        config_generation: int,
    ) -> PublishOutcome:
        """Publish a current camera frame with an error overlay.

        This keeps MJPEG/window output live when a prompt or output-contract
        error repeats after an earlier successful inference.
        """
        rendered = frame.copy()
        text = _ascii_overlay(str(message))
        cv2.rectangle(rendered, (0, 0), (rendered.shape[1], 64), (0, 0, 0), -1)
        cv2.putText(rendered, f"target: {_ascii_overlay(target)}", (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(rendered, text[:100], (8, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        now = time.monotonic()
        with self.condition:
            if self.stop_event.is_set() or self.fatal_error is not None:
                return PublishOutcome.STOPPED
            if config_generation != self.config_generation:
                return PublishOutcome.STALE
            self.display_frame = rendered
            self.display_timestamp = now
            self.display_sequence += 1
            self.render_sequence += 1
            self.inference_error_count += 1
            self.active_target = target
            if self.pending_target == target:
                self.pending_target = None
            self.error = str(message)
            self.status_message = f"{target}: inference error"
            self.last_result = {"target": target, "error": str(message)}
            self.model_ready = True
            self.condition.notify_all()
            return PublishOutcome.PUBLISHED

    def set_ready(self) -> None:
        with self.condition:
            if self.stop_event.is_set() or self.fatal_error is not None:
                return
            self.model_ready = True
            self.status_message = "models ready; waiting for camera frame"
            self.condition.notify_all()

    def set_error(self, message: str, fatal: bool = False) -> None:
        with self.condition:
            self.error = str(message)
            if fatal:
                self.fatal_error = str(message)
            self.status_message = "error"
            self.condition.notify_all()
        if fatal:
            self.stop()

    def frame_for_display(self) -> Optional[np.ndarray]:
        with self.condition:
            if self.display_frame is not None:
                return self.display_frame.copy()
            if self.capture_frame is not None:
                return self.capture_frame.copy()
            return None

    def wait_for_display(self, after_sequence: int, timeout: float = 2.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.stop_event.is_set() or self.render_sequence > after_sequence,
                timeout=timeout,
            )
            if self.stop_event.is_set():
                return None
            if self.render_sequence <= after_sequence:
                return None
            if self.display_frame is not None:
                return self.display_frame.copy(), self.render_sequence
            if self.capture_frame is not None:
                return self.capture_frame.copy(), self.render_sequence
            return None

    def snapshot(self) -> Dict:
        with self.condition:
            return {
                "model_ready": self.model_ready,
                "active_target": self.active_target,
                "pending_target": self.pending_target,
                "sam_enabled": self.sam_enabled,
                "config_generation": self.config_generation,
                "npu_mode": self.npu_mode,
                "status": self.status_message,
                "error": self.error,
                "fatal_error": self.fatal_error,
                "capture_fps": round(self.capture_fps, 3),
                "inference_fps": round(self.inference_fps, 3),
                "capture_sequence": self.capture_sequence,
                "display_sequence": self.display_sequence,
                "render_sequence": self.render_sequence,
                "inference_count": self.inference_count,
                "inference_error_count": self.inference_error_count,
                "last_result": self.last_result,
            }


class CaptureWorker(threading.Thread):
    def __init__(self, state: LiveState, device: str, width: int, height: int, fps: float) -> None:
        super().__init__(name="camera-capture", daemon=True)
        self.state = state
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps

    def run(self) -> None:
        backend = cv2.CAP_V4L2 if self.device.startswith("/dev/video") else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.device, backend)
        try:
            if not capture.isOpened():
                self.state.set_error(f"cannot open camera: {self.device}", fatal=True)
                return
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if self.fps > 0:
                capture.set(cv2.CAP_PROP_FPS, self.fps)
            consecutive_failures = 0
            while not self.state.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        self.state.set_error(f"camera read failed 30 times: {self.device}", fatal=True)
                        return
                    self.state.stop_event.wait(0.02)
                    continue
                consecutive_failures = 0
                self.state.publish_capture(frame, time.monotonic())
        except Exception as exc:
            self.state.set_error(f"camera worker failed: {exc}", fatal=True)
        finally:
            capture.release()


def _render_result(frame: np.ndarray, result: PipelineResult, state: LiveState) -> np.ndarray:
    rendered = draw_mask(frame, result)
    if result.contour is not None:
        cv2.drawContours(rendered, [result.contour], -1, (0, 255, 255), 2, cv2.LINE_AA)
    status = state.snapshot()
    lines = [
        f"target: {_ascii_overlay(result.target)}",
        f"capture {status['capture_fps']:.1f} FPS | inference {status['inference_fps']:.1f} FPS",
        f"NPU {result.npu_mode} | parallel {result.timings.parallel_region_ms:.1f} ms",
        f"total {result.timings.total_ms:.1f} ms | yolo {result.timings.yolo_inference_ms:.1f} ms",
    ]
    if result.mask is not None:
        lines.append(
            f"sam encoder {result.timings.sam_encoder_ms:.1f} ms | decoder {result.timings.sam_decoder_ms:.1f} ms"
        )
    for index, line in enumerate(lines):
        y = frame.shape[0] - 12 - (len(lines) - 1 - index) * 22
        cv2.putText(rendered, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(rendered, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return rendered


class InferenceWorker(threading.Thread):
    def __init__(self, state: LiveState, args: argparse.Namespace) -> None:
        super().__init__(name="rknn-inference", daemon=True)
        self.state = state
        self.args = args

    def run(self) -> None:
        pipeline: Optional[RknnTargetPipeline] = None
        try:
            pipeline = RknnTargetPipeline(
                self.args.clip_model,
                self.args.yolo_model,
                self.args.sam_encoder_model,
                self.args.sam_decoder_model,
                self.args.merges,
                self.args.conf,
                self.args.nms,
                npu_mode=self.args.npu_mode,
            )
            self.state.set_ready()
            last_sequence = 0
            last_logged_target: Optional[str] = None
            last_detection_signature = object()
            local_result_count = 0
            consecutive_errors = 0
            while not self.state.stop_event.is_set():
                task = self.state.wait_for_capture(last_sequence)
                if task is None:
                    continue
                frame, sequence, target, sam_enabled, config_generation = task
                last_sequence = sequence
                try:
                    result = pipeline.process(frame, target, enable_sam=sam_enabled)
                    rendered = _render_result(frame, result, self.state)
                    publish_outcome = self.state.publish_result(
                        rendered,
                        result.target,
                        result,
                        config_generation,
                    )
                    if publish_outcome is PublishOutcome.STOPPED:
                        break
                    if publish_outcome is PublishOutcome.STALE:
                        continue
                    consecutive_errors = 0
                    local_result_count += 1
                    detection_signature = None if result.detection is None else (
                        result.detection.bbox,
                        round(result.detection.confidence, 3),
                    )
                    should_log = (
                        result.target != last_logged_target
                        or detection_signature != last_detection_signature
                        or (self.args.log_every > 0 and local_result_count % self.args.log_every == 0)
                    )
                    if should_log:
                        detection = "none" if result.detection is None else (
                            f"{result.detection.confidence:.4f}@{result.detection.bbox}"
                        )
                        _safe_print(
                            f"frame={sequence} target={result.target!r} detection={detection} "
                            f"mask_iou={result.mask_iou} cache_hit={result.timings.embedding_cache_hit} "
                            f"total_ms={result.timings.total_ms:.3f}",
                            flush=True,
                        )
                        last_logged_target = result.target
                    last_detection_signature = detection_signature
                except Exception as exc:
                    consecutive_errors += 1
                    message = f"inference failed for {target!r}: {exc}"
                    publish_outcome = self.state.publish_inference_error(
                        frame,
                        target,
                        message,
                        config_generation,
                    )
                    if publish_outcome is PublishOutcome.STOPPED:
                        break
                    if publish_outcome is PublishOutcome.STALE:
                        continue
                    if consecutive_errors == 1 or (
                        self.args.log_every > 0 and consecutive_errors % self.args.log_every == 0
                    ):
                        _safe_print(f"inference_error target={target!r}: {exc}", flush=True)
                    self.state.stop_event.wait(min(0.05 * consecutive_errors, 0.5))
                if self.args.max_inferences > 0 and self.state.inference_count >= self.args.max_inferences:
                    self.state.stop()
                    break
        except Exception as exc:
            self.state.set_error(f"model initialization failed: {exc}", fatal=True)
        finally:
            if pipeline is not None:
                pipeline.release()


def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


WEB_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RK3588 YOLO-World + MobileSAM</title>
<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:20px auto;padding:0 14px;background:#101418;color:#eef}img{width:100%;height:auto;border:1px solid #52606d;background:#000}form{display:flex;gap:8px;margin:14px 0}input[type=text]{flex:1;font-size:18px;padding:10px}button{padding:8px 16px;font-size:16px}pre{white-space:pre-wrap;background:#182028;padding:12px;border-radius:6px}.row{display:flex;gap:18px;align-items:center}</style>
</head><body><h2>YOLO-World → MobileSAM 实时目标定位</h2>
<img src="/stream.mjpg" alt="live camera">
<form id="target-form"><input id="target" name="target" type="text" autocomplete="off" placeholder="输入目标名，例如 cup" required><button>定位</button></form>
<div class="row"><label><input id="sam" type="checkbox" checked> MobileSAM mask + contour</label><span id="message"></span></div>
<div id="performance">NPU: waiting for models...</div>
<pre id="status">connecting...</pre>
<script>
const form=document.getElementById('target-form'), target=document.getElementById('target'), sam=document.getElementById('sam'), statusBox=document.getElementById('status'), message=document.getElementById('message'), performance=document.getElementById('performance');
form.addEventListener('submit',async e=>{e.preventDefault();const r=await fetch('/target',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded;charset=UTF-8'},body:new URLSearchParams({target:target.value})});const j=await r.json();message.textContent=j.error||('已提交: '+j.target);});
sam.addEventListener('change',async()=>{await fetch('/sam',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({enabled:sam.checked?'1':'0'})});});
async function poll(){try{const j=await (await fetch('/status',{cache:'no-store'})).json();sam.checked=j.sam_enabled;const p=j.last_result&&j.last_result.parallel_region_ms;performance.textContent='NPU: '+j.npu_mode+' | parallel region: '+(p==null?'waiting':Number(p).toFixed(1)+' ms');statusBox.textContent=JSON.stringify(j,null,2);}catch(e){statusBox.textContent=String(e);}setTimeout(poll,500);}poll();
</script></body></html>"""


class LiveHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, state: LiveState, jpeg_quality: int):
        self.state = state
        self.jpeg_quality = jpeg_quality
        super().__init__(address, LiveRequestHandler)


class LiveRequestHandler(BaseHTTPRequestHandler):
    server: LiveHTTPServer

    def log_message(self, format: str, *args) -> None:
        return

    def _send_json(self, payload: Dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_form(self) -> Dict[str, str]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 4096:
            raise ValueError("request body must contain 1..4096 bytes")
        raw = self.rfile.read(length)
        content_type = self.headers.get("Content-Type", "")
        if content_type.startswith("application/json"):
            payload = json.loads(raw.decode("utf-8"))
            return {str(key): str(value) for key, value in payload.items()}
        parsed = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        return {key: values[-1] for key, values in parsed.items()}

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            data = WEB_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/status":
            self._send_json(self.server.state.snapshot())
            return
        if path == "/snapshot.jpg":
            frame = self.server.state.frame_for_display()
            if frame is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "no camera frame yet")
                return
            data = _encode_jpeg(frame, self.server.jpeg_quality)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/stream.mjpg":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = 0
            try:
                while not self.server.state.stop_event.is_set():
                    item = self.server.state.wait_for_display(sequence)
                    if item is None:
                        continue
                    frame, sequence = item
                    data = _encode_jpeg(frame, self.server.jpeg_quality)
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                    self.wfile.write(str(len(data)).encode("ascii"))
                    self.wfile.write(b"\r\n\r\n")
                    self.wfile.write(data)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            values = self._read_form()
            if path == "/target":
                target = self.server.state.submit_target(values.get("target", ""))
                self._send_json({"ok": True, "target": target})
                return
            if path == "/sam":
                enabled = values.get("enabled", "").lower() in {"1", "true", "yes", "on"}
                self.server.state.set_sam_enabled(enabled)
                self._send_json({"ok": True, "sam_enabled": enabled})
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)


def _overlay_window_editor(frame: np.ndarray, edit_buffer: str, state: LiveState) -> np.ndarray:
    output = frame.copy()
    snapshot = state.snapshot()
    line = f"Edit: {_ascii_overlay(edit_buffer)}  [Enter=apply, Backspace=delete, Esc=quit]"
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(output, line, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    if snapshot["error"]:
        cv2.putText(output, _ascii_overlay(snapshot["error"]), (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 2)
    return output


def run_window(state: LiveState, window_name: str) -> None:
    edit_buffer = ""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        while not state.stop_event.is_set():
            frame = state.frame_for_display()
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for camera...", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, _overlay_window_editor(frame, edit_buffer, state))
            key = cv2.waitKeyEx(15)
            if key < 0:
                continue
            low = key & 0xFF
            if low == 27 or low == 17:  # Escape or Ctrl+Q
                state.stop()
                break
            if low in (10, 13):
                try:
                    edit_buffer = state.submit_target(edit_buffer)
                except ValueError as exc:
                    state.set_error(str(exc))
                continue
            if low in (8, 127):
                edit_buffer = edit_buffer[:-1]
                continue
            if 32 <= low <= 126:
                edit_buffer += chr(low)
    finally:
        cv2.destroyWindow(window_name)


def save_last_frame(state: LiveState, path: Optional[str]) -> None:
    if not path:
        return
    frame = state.frame_for_display()
    if frame is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"failed to save final frame: {destination}")
    _safe_print(f"saved_last_frame={destination}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--ui", choices=("window", "web"), default="web")
    parser.add_argument("--device", default="/dev/video11")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--capture-fps", type=float, default=30.0)
    parser.add_argument("--target", default="cup")
    parser.add_argument("--no-sam", action="store_true", help="start with MobileSAM disabled")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--window-name", default="YOLO-World + MobileSAM")
    parser.add_argument("--max-inferences", type=int, default=0, help="stop after N results; 0 runs until interrupted")
    parser.add_argument("--log-every", type=int, default=30, help="log every N results in addition to target changes/detections; 0 disables periodic logs")
    parser.add_argument("--save-last", help="save the last rendered frame when stopping")
    args = parser.parse_args(argv)
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if not 30 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be in 30..100")
    if args.max_inferences < 0:
        parser.error("max-inferences must be non-negative")
    if args.log_every < 0:
        parser.error("log-every must be non-negative")
    args.target = RknnTargetPipeline.normalize_target(args.target)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    state = LiveState(args.target, sam_enabled=not args.no_sam, npu_mode=args.npu_mode)
    capture = CaptureWorker(state, args.device, args.width, args.height, args.capture_fps)
    inference = InferenceWorker(state, args)

    def request_stop(signum=None, frame=None):
        state.stop()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    capture.start()
    inference.start()
    server: Optional[LiveHTTPServer] = None
    server_thread: Optional[threading.Thread] = None
    try:
        if args.ui == "window":
            run_window(state, args.window_name)
        else:
            server = LiveHTTPServer((args.bind, args.port), state, args.jpeg_quality)
            server_thread = threading.Thread(target=server.serve_forever, name="mjpeg-http", daemon=True)
            server_thread.start()
            _safe_print(f"web_url=http://{args.bind}:{args.port}", flush=True)
            while not state.stop_event.wait(0.25):
                pass
    except KeyboardInterrupt:
        state.stop()
    finally:
        state.stop()
        if server is not None:
            server.shutdown()
            server.server_close()
        capture.join(timeout=3.0)
        inference.join(timeout=10.0)
        if capture.is_alive():
            state.set_error("camera worker did not stop within 3 seconds", fatal=True)
        if inference.is_alive():
            state.set_error("RKNN inference worker did not stop within 10 seconds", fatal=True)
        if server_thread is not None:
            server_thread.join(timeout=2.0)
        save_last_frame(state, args.save_last)
        _safe_print(json.dumps(state.snapshot(), ensure_ascii=False), flush=True)
    return 1 if state.fatal_error or (state.error and state.inference_count == 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
