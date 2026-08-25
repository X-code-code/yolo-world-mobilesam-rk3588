# Verified conversion run

This is the sanitized record of the model conversion used for the 2026-08-25 LubanCat 4 verification. It separates observed evidence from steps that were not performed.

## Scope and environment

```text
Input: Rockchip-provided ONNX files
Tool: RKNN Toolkit2 2.3.2
Host: Ubuntu 22.04 x86_64 under WSL, Python 3.10.12
Target: rk3588
Output: five RKNN files
```

The environment also contained ONNX 1.14.1, NumPy 1.26.4, PyTorch 2.4.0+cpu, protobuf 4.25.4 and OpenCV 4.8.1.78. These versions describe the observed run; `convert_models.py` strictly enforces Toolkit2 2.3.2 because that component determines RKNN graph generation.

## Evidence retained

| Output | Persistent original conversion log | Build/export result | Output identity |
| --- | --- | --- | --- |
| `clip_text_fp16.rknn` | yes | completed | matches `MODEL_PROVENANCE.json` |
| `yolo_world_v2s_fp16.rknn` | yes | completed | matches `MODEL_PROVENANCE.json` |
| `yolo_world_v2s_i8.rknn` | yes | completed | matches `MODEL_PROVENANCE.json` |
| `mobilesam_encoder_fp16.rknn` | no complete persistent log | output present and later used on board | matches `MODEL_PROVENANCE.json` |
| `mobilesam_decoder_fp16.rknn` | no complete persistent log | output present and later used on board | matches `MODEL_PROVENANCE.json` |

The three retained raw logs contain terminal progress control sequences and local machine paths, so they are not copied into the public repository. Their useful evidence was reduced to the versioned, machine-readable provenance record rather than publishing noisy or private-path-bearing output.

Observed warnings included CLIP graph corrections for extreme `Expand` values and `Unknown op target` messages during the I8 YOLO build. In both cases Toolkit2 subsequently reported build and export completion. These warnings are recorded rather than hidden; successful export alone was not treated as functional proof.

## Independent artifact check

The five output hashes were recomputed in both the conversion environment and the board deployment directory. Every pair matched. The board then completed dynamic text-prompt detection, YOLO bbox to MobileSAM mask inference, and serial versus three-core consistency tests. The de-identified benchmark evidence is in `benchmarks/lubancat4_rk3588_2026-08-25.json`.

## Explicitly not claimed

- This project did not perform a `.pt` or `.pth` to ONNX export for this run.
- A complete MobileSAM terminal log was not retained.
- Matching a known SHA-256 does not prove compatibility with every RKNN Runtime or driver version.
- Conversion success does not replace visual validation of bbox and mask quality.
