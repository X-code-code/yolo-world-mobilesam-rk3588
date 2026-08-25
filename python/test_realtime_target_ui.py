#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from rknn_target_pipeline import PipelineResult, StageTimings
from realtime_target_ui import (
    WEB_PAGE,
    InferenceWorker,
    LiveHTTPServer,
    LiveState,
    PublishOutcome,
    _safe_print,
)


class LiveStateTest(unittest.TestCase):
    def test_closed_stdout_does_not_become_an_inference_error(self):
        with patch("builtins.print", side_effect=ValueError("I/O operation on closed file")):
            _safe_print("frame status", flush=True)

    def test_capture_slot_keeps_only_latest_frame(self):
        state = LiveState("cup", True)
        state.publish_capture(np.zeros((8, 8, 3), dtype=np.uint8), 1.0)
        state.publish_capture(np.full((8, 8, 3), 7, dtype=np.uint8), 1.1)
        task = state.wait_for_capture(0, timeout=0.01)
        self.assertIsNotNone(task)
        frame, sequence, target, sam_enabled, config_generation = task
        self.assertEqual(sequence, 2)
        self.assertTrue(np.all(frame == 7))
        self.assertEqual(target, "cup")
        self.assertTrue(sam_enabled)
        self.assertEqual(config_generation, 0)

    def test_target_submission_normalizes_unicode(self):
        state = LiveState("cup", True)
        self.assertEqual(state.submit_target("  红色  杯子 "), "红色 杯子")
        self.assertEqual(state.snapshot()["pending_target"], "红色 杯子")
        with self.assertRaises(ValueError):
            state.submit_target("   ")

    def test_fatal_error_cannot_be_cleared_by_late_result(self):
        state = LiveState("cup", True)
        state.set_error("camera disconnected", fatal=True)
        result = PipelineResult("cup", None, None, None, None)
        outcome = state.publish_result(np.zeros((8, 8, 3), dtype=np.uint8), "cup", result, 0)
        self.assertIs(outcome, PublishOutcome.STOPPED)
        self.assertEqual(state.snapshot()["fatal_error"], "camera disconnected")
        self.assertEqual(state.snapshot()["inference_count"], 0)

    def test_inference_error_publishes_a_fresh_frame(self):
        state = LiveState("cup", True)
        first = np.zeros((24, 32, 3), dtype=np.uint8)
        state.publish_capture(first, 1.0)
        before = state.snapshot()["render_sequence"]
        outcome = state.publish_inference_error(first, "cat", "decoder mismatch", 0)
        self.assertIs(outcome, PublishOutcome.PUBLISHED)
        snapshot = state.snapshot()
        self.assertGreater(snapshot["render_sequence"], before)
        self.assertEqual(snapshot["active_target"], "cat")
        self.assertEqual(snapshot["inference_error_count"], 1)
        self.assertIn("decoder mismatch", snapshot["error"])

    def test_target_change_discards_stale_result_without_stopping(self):
        state = LiveState("cup", True, npu_mode="three-core")
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        state.publish_capture(frame, 1.0)
        task = state.wait_for_capture(0, timeout=0.01)
        self.assertIsNotNone(task)
        _, _, _, _, old_generation = task

        state.submit_target("cat")
        stale_result = PipelineResult("cup", None, None, None, None, npu_mode="three-core")
        outcome = state.publish_result(frame, "cup", stale_result, old_generation)
        self.assertIs(outcome, PublishOutcome.STALE)
        snapshot = state.snapshot()
        self.assertFalse(state.stop_event.is_set())
        self.assertEqual(snapshot["pending_target"], "cat")
        self.assertEqual(snapshot["active_target"], "cup")
        self.assertEqual(snapshot["inference_count"], 0)

        current_generation = snapshot["config_generation"]
        current_result = PipelineResult("cat", None, None, None, None, npu_mode="three-core")
        outcome = state.publish_result(frame, "cat", current_result, current_generation)
        self.assertIs(outcome, PublishOutcome.PUBLISHED)
        snapshot = state.snapshot()
        self.assertEqual(snapshot["active_target"], "cat")
        self.assertIsNone(snapshot["pending_target"])
        self.assertEqual(snapshot["inference_count"], 1)

    def test_sam_change_discards_stale_error(self):
        state = LiveState("cup", True)
        old_generation = state.snapshot()["config_generation"]
        state.set_sam_enabled(False)
        outcome = state.publish_inference_error(
            np.zeros((8, 8, 3), dtype=np.uint8),
            "cup",
            "old configuration failed",
            old_generation,
        )
        self.assertIs(outcome, PublishOutcome.STALE)
        snapshot = state.snapshot()
        self.assertFalse(state.stop_event.is_set())
        self.assertFalse(snapshot["sam_enabled"])
        self.assertIsNone(snapshot["error"])
        self.assertEqual(snapshot["inference_error_count"], 0)

    def test_result_status_exposes_npu_mode_and_parallel_region(self):
        state = LiveState("cup", True, npu_mode="three-core")
        timings = StageTimings(parallel_region_ms=123.5, total_ms=140.0)
        result = PipelineResult("cup", None, None, None, None, timings, "three-core")
        outcome = state.publish_result(
            np.zeros((8, 8, 3), dtype=np.uint8),
            "cup",
            result,
            state.snapshot()["config_generation"],
        )
        self.assertIs(outcome, PublishOutcome.PUBLISHED)
        snapshot = state.snapshot()
        self.assertEqual(snapshot["npu_mode"], "three-core")
        self.assertEqual(snapshot["last_result"]["npu_mode"], "three-core")
        self.assertEqual(snapshot["last_result"]["parallel_region_ms"], 123.5)
        self.assertIn('id="performance"', WEB_PAGE)
        self.assertIn("parallel region", WEB_PAGE)

    def test_inference_worker_passes_npu_mode_to_pipeline(self):
        state = LiveState("cup", False, npu_mode="three-core")
        state.publish_capture(np.zeros((16, 16, 3), dtype=np.uint8), 1.0)
        args = SimpleNamespace(
            clip_model="clip.rknn",
            yolo_model="yolo.rknn",
            sam_encoder_model="encoder.rknn",
            sam_decoder_model="decoder.rknn",
            merges="merges.txt",
            conf=0.25,
            nms=0.45,
            npu_mode="three-core",
            max_inferences=1,
            log_every=0,
        )
        pipeline = MagicMock()
        pipeline.process.return_value = PipelineResult(
            "cup",
            None,
            None,
            None,
            None,
            StageTimings(parallel_region_ms=88.0),
            "three-core",
        )
        with patch("realtime_target_ui.RknnTargetPipeline", return_value=pipeline) as constructor:
            InferenceWorker(state, args).run()

        constructor.assert_called_once_with(
            "clip.rknn",
            "yolo.rknn",
            "encoder.rknn",
            "decoder.rknn",
            "merges.txt",
            0.25,
            0.45,
            npu_mode="three-core",
        )
        pipeline.process.assert_called_once()
        pipeline.release.assert_called_once()
        self.assertEqual(state.snapshot()["last_result"]["parallel_region_ms"], 88.0)


class HTTPTest(unittest.TestCase):
    def setUp(self):
        self.state = LiveState("cup", True)
        self.state.publish_capture(np.zeros((24, 32, 3), dtype=np.uint8), time.monotonic())
        self.server = LiveHTTPServer(("127.0.0.1", 0), self.state, 75)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.state.stop()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_status_snapshot_and_unicode_target(self):
        with urlopen(self.base + "/status", timeout=2) as response:
            status = json.load(response)
        self.assertEqual(status["active_target"], "cup")

        data = urlencode({"target": "红色杯子"}).encode("utf-8")
        request = Request(
            self.base + "/target",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
        )
        with urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertEqual(payload["target"], "红色杯子")
        self.assertEqual(self.state.snapshot()["pending_target"], "红色杯子")

        with urlopen(self.base + "/snapshot.jpg", timeout=2) as response:
            jpeg = response.read()
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        self.assertTrue(jpeg.endswith(b"\xff\xd9"))

    def test_sam_toggle(self):
        request = Request(
            self.base + "/sam",
            data=b"enabled=0",
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urlopen(request, timeout=2) as response:
            payload = json.load(response)
        self.assertFalse(payload["sam_enabled"])
        self.assertFalse(self.state.snapshot()["sam_enabled"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
