# Third-Party Notices

This repository contains original integration work plus adapted source and data from third-party projects. The repository-level `LICENSE` applies only where the relevant file or component does not state a different license. Ordinary Git contains no model or vendor runtime binary. The v0.1.0 GitHub Release distributes only two MobileSAM RKNN artifacts in a separate license-carrying package; CLIP and YOLO-World RKNN artifacts are identified for reproducibility but are not public assets.

## Rockchip RKNN Model Zoo

- Project: [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)
- License: Apache License 2.0
- Use in this repository: RKNN YOLO-World and MobileSAM preprocessing, postprocessing, model I/O conventions, and the C++ CLIP tokenizer reference under `tests/tokenizer/` were used as the direct implementation base.

Changes include a Python CLIP tokenizer, a unified bbox-to-mask pipeline, text-embedding caching, live camera UIs, explicit per-core RKNNLite contexts, same-frame parallel scheduling, benchmarks, and tests.

## OpenAI CLIP

- Project: [openai/CLIP](https://github.com/openai/CLIP)
- License: MIT
- Use in this repository: CLIP byte-pair encoding behavior and `python/bpe_simple_vocab_16e6.txt`. The embedded reference vocabulary in `tests/tokenizer/clip_vocab.h` represents the same BPE merge data for cross-implementation testing.

Copyright © OpenAI. A copy of the OpenAI CLIP MIT license is included at [`LICENSES/OpenAI-CLIP-MIT.txt`](LICENSES/OpenAI-CLIP-MIT.txt).

The `clip_text_fp16.rknn` source model is associated with `openai/clip-vit-base-patch32`. Its official model card does not state an explicit weight redistribution license. The source-code MIT license is therefore not represented here as permission to publish the converted weight artifact; v0.1.0 provides a local conversion path instead.

## YOLO-World

- Upstream project: [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- Rockchip adaptation: [airockchip/YOLO-World](https://github.com/airockchip/YOLO-World)
- Upstream repository license: GPL-3.0; the upstream project also documents separate commercial licensing.
- Pinned upstream commit: `b4fd87838d7f53adc0dbf5844313b92d9e3124c7`
- Pinned Rockchip adaptation commit: `b8b0fe9beffa9564306a798f6e443c9fe88057af`
- Use in this repository: external ONNX/RKNN model architecture and weights only. YOLO-World model binaries and GPL corresponding source are not included in v0.1.0.

The repository records exact local conversion parameters without copying GPL export source into the Apache-licensed project. Public YOLO RKNN distribution is deferred until a complete GPL corresponding-source bundle or a separate commercial license is available.

## MobileSAM

- Upstream project: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- Rockchip adaptation: [airockchip/MobileSAM](https://github.com/airockchip/MobileSAM)
- License: Apache License 2.0
- Pinned upstream commit: `c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed`
- Pinned Rockchip adaptation commit: `e6aceeb93a08d75c39dbca073266d8447290f330`
- Use in this repository: external encoder/decoder model architecture and weights. The v0.1.0 Release package contains the two RK3588 FP16 RKNN conversions together with the Apache-2.0 license, provenance, checksums and a 2026-08-25 modification statement.

The upstream checkpoint, Rockchip export changes, Model Zoo ONNX entry points and exact downloaded ONNX hashes are linked in `MODEL_LICENSES.md` and `MODEL_PROVENANCE.json`; the distribution decision is not based only on a repository badge.

## RKNN Toolkit2 and RKNN Runtime

- Project: [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)
- Use in this repository: external model conversion tooling, RKNNLite Python package, target runtime and device driver.

These vendor packages and shared libraries are not included. Obtain them from Rockchip or the board vendor and comply with their own distribution terms.

This notice is an engineering inventory, not legal advice.
