#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from rknn_target_pipeline import (
    LetterboxMeta,
    RknnTargetPipeline,
    largest_external_contour,
    sam_resize_longest_side,
    sam_transform_bbox,
    yolo_letterbox,
    yolo_postprocess,
)


class FakeRKNNLite:
    """Thread-aware RKNNLite fake for the three-core ownership contract."""

    NPU_CORE_AUTO = 0
    NPU_CORE_0 = 1
    NPU_CORE_1 = 2
    NPU_CORE_2 = 4

    instances = []
    by_role = {}
    encoder_started = threading.Event()
    yolo_started = threading.Event()

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.by_role = {}
        cls.encoder_started = threading.Event()
        cls.yolo_started = threading.Event()

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.role = None
        self.owner_thread = threading.get_ident()
        self.owner_thread_name = threading.current_thread().name
        self.load_thread = None
        self.init_thread = None
        self.init_core_mask = None
        self.inference_threads = []
        self.inference_intervals = []
        self.release_thread = None
        self.release_thread_name = None
        self.release_count = 0
        self.__class__.instances.append(self)

    def load_rknn(self, model_path):
        self.role = Path(model_path).stem
        self.load_thread = threading.get_ident()
        self.__class__.by_role[self.role] = self
        return 0

    def init_runtime(self, core_mask=NPU_CORE_AUTO):
        self.init_thread = threading.get_ident()
        self.init_core_mask = core_mask
        return 0

    @staticmethod
    def _yolo_outputs():
        outputs = []
        for branch in range(3):
            scores = np.zeros((1, 80, 1, 1), dtype=np.float32)
            boxes = np.zeros((1, 4, 1, 1), dtype=np.float32)
            if branch == 0:
                scores[0, 0, 0, 0] = 0.9
                boxes[0, :, 0, 0] = (0.0, 0.0, 0.4, 0.4)
            outputs.extend((scores, boxes))
        return outputs

    def inference(self, inputs):
        started = time.perf_counter()
        self.inference_threads.append(threading.get_ident())
        try:
            if self.role == "clip":
                return [np.ones((1, 512), dtype=np.float32)]
            if self.role == "yolo":
                self.__class__.yolo_started.set()
                if not self.__class__.encoder_started.wait(timeout=1.0):
                    raise AssertionError("YOLO did not overlap the same-frame encoder")
                time.sleep(0.02)
                return self._yolo_outputs()
            if self.role == "sam_encoder":
                self.__class__.encoder_started.set()
                if not self.__class__.yolo_started.wait(timeout=1.0):
                    raise AssertionError("encoder did not overlap the same-frame YOLO")
                time.sleep(0.02)
                return [np.zeros((1, 256, 28, 28), dtype=np.float32)]
            if self.role == "sam_decoder":
                iou = np.asarray([[0.1, 0.9, 0.2, 0.3]], dtype=np.float32)
                masks = np.zeros((1, 4, 112, 112), dtype=np.float32)
                masks[0, 1, 20:80, 30:90] = 1.0
                return [iou, masks]
            raise AssertionError(f"unexpected fake model role: {self.role!r}")
        finally:
            self.inference_intervals.append((started, time.perf_counter()))

    def release(self):
        self.release_thread = threading.get_ident()
        self.release_thread_name = threading.current_thread().name
        self.release_count += 1


class ThreeCorePipelineTest(unittest.TestCase):
    def setUp(self):
        FakeRKNNLite.reset()

    @contextmanager
    def three_core_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_paths = {
                role: root / f"{role}.rknn"
                for role in ("clip", "yolo", "sam_encoder", "sam_decoder")
            }
            for model_path in model_paths.values():
                model_path.touch()
            pipeline = RknnTargetPipeline(
                model_paths["clip"],
                model_paths["yolo"],
                model_paths["sam_encoder"],
                model_paths["sam_decoder"],
                Path(__file__).with_name("bpe_simple_vocab_16e6.txt"),
                npu_mode="three-core",
                runtime_type=FakeRKNNLite,
            )
            try:
                yield pipeline
            finally:
                pipeline.release()

    @staticmethod
    def exercise(pipeline):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = pipeline.process(image, "cup", enable_sam=True)
        if result.detection is None or result.mask is None:
            raise AssertionError("fake pipeline did not exercise the complete bbox-to-mask path")
        return result

    def test_three_core_runtime_binding(self):
        with self.three_core_pipeline():
            roles = FakeRKNNLite.by_role
            self.assertEqual(roles["sam_encoder"].init_core_mask, FakeRKNNLite.NPU_CORE_0)
            self.assertEqual(roles["clip"].init_core_mask, FakeRKNNLite.NPU_CORE_1)
            self.assertEqual(roles["sam_decoder"].init_core_mask, FakeRKNNLite.NPU_CORE_1)
            self.assertEqual(roles["yolo"].init_core_mask, FakeRKNNLite.NPU_CORE_2)

    def test_inference_runs_on_each_runtime_owner_thread(self):
        with self.three_core_pipeline() as pipeline:
            self.exercise(pipeline)
            roles = FakeRKNNLite.by_role
            for role, runtime in roles.items():
                self.assertEqual(runtime.load_thread, runtime.owner_thread, role)
                self.assertEqual(runtime.init_thread, runtime.owner_thread, role)
                self.assertTrue(runtime.inference_threads, role)
                self.assertTrue(
                    all(thread_id == runtime.owner_thread for thread_id in runtime.inference_threads),
                    role,
                )
            self.assertEqual(roles["clip"].owner_thread, roles["sam_decoder"].owner_thread)
            self.assertNotEqual(roles["sam_encoder"].owner_thread, roles["yolo"].owner_thread)
            self.assertNotEqual(roles["sam_encoder"].owner_thread, roles["clip"].owner_thread)
            self.assertNotEqual(roles["yolo"].owner_thread, roles["clip"].owner_thread)
            self.assertTrue(roles["sam_encoder"].owner_thread_name.startswith("rknn-core0-encoder"))
            self.assertTrue(roles["clip"].owner_thread_name.startswith("rknn-core1-text-decoder"))
            self.assertTrue(roles["yolo"].owner_thread_name.startswith("rknn-core2-yolo"))

    def test_same_frame_encoder_and_yolo_really_overlap(self):
        with self.three_core_pipeline() as pipeline:
            self.exercise(pipeline)
            encoder_interval = FakeRKNNLite.by_role["sam_encoder"].inference_intervals[0]
            yolo_interval = FakeRKNNLite.by_role["yolo"].inference_intervals[0]
            overlap = min(encoder_interval[1], yolo_interval[1]) - max(
                encoder_interval[0], yolo_interval[0]
            )
            self.assertGreater(overlap, 0.0)

    def test_release_runs_once_on_each_runtime_owner_thread(self):
        with self.three_core_pipeline() as pipeline:
            self.exercise(pipeline)
        for role, runtime in FakeRKNNLite.by_role.items():
            self.assertEqual(runtime.release_count, 1, role)
            self.assertEqual(runtime.release_thread, runtime.owner_thread, role)
            self.assertEqual(runtime.release_thread_name, runtime.owner_thread_name, role)

    def test_invalid_npu_mode_is_rejected_before_runtime_creation(self):
        with self.assertRaisesRegex(ValueError, "npu_mode must be one of"):
            RknnTargetPipeline(
                "clip.rknn",
                "yolo.rknn",
                "sam_encoder.rknn",
                "sam_decoder.rknn",
                "merges.txt",
                npu_mode="all-at-once",
                runtime_type=FakeRKNNLite,
            )
        self.assertEqual(FakeRKNNLite.instances, [])


class GeometryTest(unittest.TestCase):
    def test_yolo_letterbox_matches_vendor_alignment(self):
        image = np.zeros((500, 375, 3), dtype=np.uint8)
        padded, meta = yolo_letterbox(image)
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(meta.scale, 1.28, places=6)
        self.assertEqual((meta.x_pad, meta.y_pad), (80, 0))
        self.assertEqual((meta.resized_width, meta.resized_height), (480, 640))
        self.assertTrue(np.all(padded[:, :80] == 114))

    def test_sam_resize_and_bbox_match_official_example(self):
        image = np.zeros((770, 769, 3), dtype=np.uint8)
        padded, new_h, new_w = sam_resize_longest_side(image)
        self.assertEqual((new_h, new_w), (448, 447))
        self.assertEqual(padded.shape, (448, 448, 3))
        coords = sam_transform_bbox((190, 70, 460, 280), 770, 769, new_h, new_w)
        expected = np.asarray(
            [[[190 * 447 / 769, 70 * 448 / 770], [460 * 447 / 769, 280 * 448 / 770]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(coords, expected, rtol=0, atol=1e-5)

    def test_synthetic_yolo_decode_matches_rknnlite_nchw(self):
        score0 = np.zeros((1, 80, 80, 80), dtype=np.float32)
        box0 = np.zeros((1, 4, 80, 80), dtype=np.float32)
        score0[0, 0, 10, 20] = 0.9
        box0[0, :, 10, 20] = (1, 2, 3, 4)
        outputs = [
            score0,
            box0,
            np.zeros((1, 80, 40, 40), dtype=np.float32),
            np.zeros((1, 4, 40, 40), dtype=np.float32),
            np.zeros((1, 80, 20, 20), dtype=np.float32),
            np.zeros((1, 4, 20, 20), dtype=np.float32),
        ]
        meta = LetterboxMeta(1.0, 0, 0, 640, 640, 640, 640)
        detections = yolo_postprocess(outputs, meta, "cup", 0.25, 0.45)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (156, 68, 188, 116))
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=6)

    def test_quantized_nine_output_layout_ignores_score_sum(self):
        score0 = np.zeros((1, 80, 80, 80), dtype=np.float32)
        box0 = np.zeros((1, 4, 80, 80), dtype=np.float32)
        score0[0, 0, 10, 20] = 0.9
        box0[0, :, 10, 20] = (1, 2, 3, 4)
        outputs = [
            score0,
            box0,
            np.full((1, 1, 80, 80), 999.0, dtype=np.float32),
            np.zeros((1, 80, 40, 40), dtype=np.float32),
            np.zeros((1, 4, 40, 40), dtype=np.float32),
            np.zeros((1, 1, 40, 40), dtype=np.float32),
            np.zeros((1, 80, 20, 20), dtype=np.float32),
            np.zeros((1, 4, 20, 20), dtype=np.float32),
            np.zeros((1, 1, 20, 20), dtype=np.float32),
        ]
        meta = LetterboxMeta(1.0, 0, 0, 640, 640, 640, 640)
        detections = yolo_postprocess(outputs, meta, "cup", 0.25, 0.45)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (156, 68, 188, 116))
        self.assertAlmostEqual(detections[0].confidence, 0.9, places=6)

    def test_contour_returns_largest_external_component(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (5, 5), (15, 15), 1, -1)
        cv2.rectangle(mask, (30, 30), (80, 80), 1, -1)
        contour = largest_external_contour(mask)
        self.assertIsNotNone(contour)
        self.assertAlmostEqual(cv2.contourArea(contour), 2500.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
