# Third-Party Notices

This repository contains original integration work plus adapted source and data from third-party projects. The repository-level `LICENSE` applies only where the relevant file or component does not state a different license. Model files and vendor runtime binaries are deliberately not distributed here.

## Rockchip RKNN Model Zoo

- Project: [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)
- License: Apache License 2.0
- Use in this repository: RKNN YOLO-World and MobileSAM preprocessing, postprocessing, model I/O conventions, and the C++ CLIP tokenizer reference under `tests/tokenizer/` were used as the direct implementation base.

Changes include a Python CLIP tokenizer, a unified bbox-to-mask pipeline, text-embedding caching, live camera UIs, explicit per-core RKNNLite contexts, same-frame parallel scheduling, benchmarks, and tests.

## OpenAI CLIP

- Project: [openai/CLIP](https://github.com/openai/CLIP)
- License: MIT
- Use in this repository: CLIP byte-pair encoding behavior and `python/bpe_simple_vocab_16e6.txt`. The embedded reference vocabulary in `tests/tokenizer/clip_vocab.h` represents the same BPE merge data for cross-implementation testing.

Copyright © OpenAI. The OpenAI CLIP MIT license is available in its upstream repository.

## YOLO-World

- Upstream project: [AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- Rockchip adaptation: [airockchip/YOLO-World](https://github.com/airockchip/YOLO-World)
- Upstream repository license: GPL-3.0; the upstream project also documents separate commercial licensing.
- Use in this repository: external ONNX/RKNN model architecture and weights only. YOLO-World model binaries and GPL source are not included.

Users who download, convert, redistribute, or commercially use YOLO-World models must evaluate the current upstream license and model terms themselves.

## MobileSAM

- Upstream project: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- Rockchip adaptation: [airockchip/MobileSAM](https://github.com/airockchip/MobileSAM)
- License: Apache License 2.0
- Use in this repository: external encoder/decoder model architecture and weights. Model binaries are not included.

## RKNN Toolkit2 and RKNN Runtime

- Project: [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)
- Use in this repository: external model conversion tooling, RKNNLite Python package, target runtime and device driver.

These vendor packages and shared libraries are not included. Obtain them from Rockchip or the board vendor and comply with their own distribution terms.

This notice is an engineering inventory, not legal advice.
