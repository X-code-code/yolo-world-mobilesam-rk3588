# Contributing

Contributions are welcome, especially reproducible fixes for other RK3588 boards, cameras, RKNN runtime versions, and YOLO-World output layouts.

## Before opening a pull request

1. Keep model files, shared libraries, camera captures, logs, credentials and local paths out of Git.
2. Add or update tests for behavioral changes.
3. Run:

   ```bash
   python3 -m pip install -r requirements-dev.txt
   cd python
   python3 -m unittest discover -v
   cd ..
   python3 scripts/check_release.py
   ```

4. For performance claims, include hardware, RKNNLite/Runtime, driver, model hash, warm-up count, measured runs and whether the number is pure inference or end-to-end camera FPS.
5. Document third-party code or data and preserve its license notices.

## Code expectations

- Keep the board runtime limited to Python, NumPy, OpenCV and RKNNLite unless a new dependency has a clear deployment benefit.
- Do not share one RKNNLite context across concurrent threads. Each NPU role owns its runtime context and executor.
- Preserve configuration-generation checks when changing the live UI; stale results must not overwrite a newly selected target.
- Treat the MobileSAM decoder as dependent on the current frame's bbox and image embedding.
- Prefer deterministic unit tests with fake runtimes, then report separate real-board verification.
