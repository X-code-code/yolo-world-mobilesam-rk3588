# Contributing

Contributions are welcome, especially reproducible fixes for other RK3588 boards, cameras, RKNN runtime versions, and YOLO-World output layouts.

## Before opening a pull request

1. Keep model files, shared libraries, camera captures, logs, credentials and local paths out of ordinary Git. Approved model assets use a separate GitHub Release bundle with hashes, provenance and applicable licenses.
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

## Model artifact changes

- Update `MODEL_PROVENANCE.json`, `MODEL_RELEASES.json`, `MODEL_SHA256SUMS`, `MODEL_LICENSES.md` and the model guide together.
- Do not infer a weight license from a source-code license. Re-audit the exact upstream model revision before adding a public asset.
- Build approved MobileSAM packages with `scripts/build_model_release.py`; its final strict run must match the bundle hash in `MODEL_RELEASES.json`. Never add `.rknn`, `.onnx`, Toolkit wheels or vendor runtime libraries to Git history.
- Verify an uploaded asset by downloading it again and checking both the bundle hash and contained model hashes before publishing the Release.

## Code expectations

- Keep the board runtime limited to Python, NumPy, OpenCV and RKNNLite unless a new dependency has a clear deployment benefit.
- Do not share one RKNNLite context across concurrent threads. Each NPU role owns its runtime context and executor.
- Preserve configuration-generation checks when changing the live UI; stale results must not overwrite a newly selected target.
- Treat the MobileSAM decoder as dependent on the current frame's bbox and image embedding.
- Prefer deterministic unit tests with fake runtimes, then report separate real-board verification.
