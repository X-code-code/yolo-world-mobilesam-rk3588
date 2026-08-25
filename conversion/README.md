# Model download and RKNN conversion

This directory reproduces the conversion that produced the five verified
RK3588 artifacts. It starts from Rockchip-provided ONNX files; it does not claim
to reproduce the upstream PyTorch-to-ONNX export.

The downloader uses only the Python standard library. The converter imports
`rknn.api.RKNN` only after every path, input hash, calibration file and output
collision has passed preflight.

## Requirements

- x86_64 Ubuntu 22.04 or a compatible Linux host
- Python 3.10
- `rknn-toolkit2==2.3.2`
- An `airockchip/rknn_model_zoo` checkout containing the official YOLO-World
  calibration data when converting `yolo_world_v2s_i8`

The locally verified conversion also used ONNX 1.14.1, NumPy 1.26.4,
PyTorch 2.4.0+cpu, protobuf 4.25.4 and OpenCV 4.8.1.78. The script enforces the
Toolkit2 version because RKNN graph generation is vendor-version-sensitive.

## 1. Download and verify the four ONNX inputs

From the repository root:

```bash
python3 conversion/download_onnx.py \
  --output-dir model/onnx \
  --record model/onnx/download-record.json
```

The four URLs are the ones used by the corresponding Model Zoo
`download_model.sh` files. Each file has a pinned byte count and SHA-256. A
valid existing file is reused; an invalid existing file stops the command.
Only an explicit `--force` may replace an invalid file or an existing record.

## 2. Convert all five RKNN models

```bash
python3 conversion/convert_models.py \
  --onnx-dir model/onnx \
  --output-dir model \
  --model-zoo-root /path/to/rknn_model_zoo \
  --record model/conversion-record.json
```

Outputs:

```text
model/clip_text_fp16.rknn
model/yolo_world_v2s_fp16.rknn
model/yolo_world_v2s_i8.rknn
model/mobilesam_encoder_fp16.rknn
model/mobilesam_decoder_fp16.rknn
```

No output is silently overwritten. Rerun with `--overwrite` only when replacing
all selected outputs and their JSON record is intentional. Conversions are
exported into one staging area first. Only after every selected RKNN is
non-empty and hashed are the complete output set and JSON record published. If
publication fails, `--overwrite` restores every previous model and record, so a
new model set cannot be paired with an old record.

To convert only selected artifacts:

```bash
python3 conversion/convert_models.py \
  --models clip_text_fp16 mobilesam_encoder_fp16 mobilesam_decoder_fp16 \
  --onnx-dir model/onnx \
  --output-dir model
```

`--model-zoo-root` is mandatory whenever `yolo_world_v2s_i8` is selected. The
converter accepts only the official file at
`examples/yolo_world/model/dataset.txt`, verifies its pinned hash and 20 rows,
verifies the fixed SHA-256 of each of the 20 referenced COCO images, and
verifies the shared `coco_text_outp.npy`. It will not substitute an arbitrary
calibration set. Input, RKNN output and JSON record paths are also required to
be distinct before Toolkit2 is loaded.

## Exact RKNN API profiles

| Output | RKNN constructor/config | ONNX crop/input contract | Build |
| --- | --- | --- | --- |
| `clip_text_fp16.rknn` | `verbose=False`, `target_platform=rk3588` | `input_ids`, `[1,20]` | no quantization |
| `yolo_world_v2s_fp16.rknn` | mean `[0,0,0]`, std `[255,255,255]` | `images [1,3,640,640]`, `texts [1,80,512]` | no quantization; official dataset is passed and Toolkit2 ignores it |
| `yolo_world_v2s_i8.rknn` | same YOLO config | same YOLO inputs | I8 with official Model Zoo dataset |
| `mobilesam_encoder_fp16.rknn` | `verbose=True`, mean `[123.675,116.28,103.53]`, std `[58.395,57.12,57.375]` | uncropped ONNX, fixed 448 × 448 graph | no quantization |
| `mobilesam_decoder_fp16.rknn` | `verbose=True`, `target_platform=rk3588` | five inputs with two bbox points; outputs `iou_predictions` and `low_res_masks` | no quantization |

The JSON conversion record contains the host environment, exact per-model API
parameters, every input SHA-256, all calibration dependency hashes, every
output SHA-256 and whether an output matches the previously board-verified
reference artifact.

The sanitized evidence from the conversion that produced the reference hashes
is recorded in [`VERIFIED_RUN.md`](VERIFIED_RUN.md). Raw terminal logs are not
published because they contain progress-control noise and local paths.

## Test without RKNN or network

```bash
python3 -m unittest -v python/test_model_distribution.py
```

Tests inject a fake downloader and fake RKNN implementation. They verify the
fixed download manifest, atomic overwrite policy, delayed dependency boundary,
all five RKNN call profiles (including `config` return handling), tampered COCO
image rejection, all-or-nothing multi-model publication and JSON input/output
hashes without performing network access or importing RKNN.

## Provenance and distribution

The API parameters were adapted from `airockchip/rknn_model_zoo`. The ONNX and
RKNN files are model artifacts, not original repository source. Review
`THIRD_PARTY_NOTICES.md` and the current upstream weight terms before public
redistribution, especially for YOLO-World. Model files remain ignored by Git;
publish approved binaries as separately checksummed release assets rather than
ordinary Git objects.
