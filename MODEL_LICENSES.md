# Model artifact notices

This file records the redistribution decision for the model artifacts identified by `MODEL_RELEASES.json`. It is separate from the repository-level Apache-2.0 license. This is an engineering compliance record, not legal advice.

## Public Release assets

### MobileSAM encoder and decoder

- Artifacts: `mobilesam_encoder_fp16.rknn`, `mobilesam_decoder_fp16.rknn`
- Upstream: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- Rockchip adaptation: [airockchip/MobileSAM](https://github.com/airockchip/MobileSAM)
- Conversion example: [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/v2.3.2/examples/mobilesam)
- License stated by those repositories: Apache License 2.0

Weight-level provenance used for this decision:

- The original [`weights/mobile_sam.pt`](https://github.com/ChaoningZhang/MobileSAM/blob/c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed/weights/mobile_sam.pt) is committed inside the pinned Apache-2.0 MobileSAM repository.
- The pinned [Rockchip RKNN export record](https://github.com/airockchip/MobileSAM/blob/e6aceeb93a08d75c39dbca073266d8447290f330/RKNN_README_EN.md) identifies its upstream commit and ONNX/RKNN graph changes.
- The pinned [Model Zoo MobileSAM example](https://github.com/airockchip/rknn_model_zoo/tree/bad6c7334531becaf90a561988519b7bec34d0ab/examples/mobilesam) supplies the exact ONNX download and conversion entry points. The downloaded ONNX identities are fixed in `MODEL_PROVENANCE.json` rather than inferred from filenames.

Pinned source revisions:

- MobileSAM: `c12dd83cbe26dffdcc6a0f9e7be2f6fb024df0ed`
- Rockchip MobileSAM fork: `e6aceeb93a08d75c39dbca073266d8447290f330`
- RKNN Model Zoo: `bad6c7334531becaf90a561988519b7bec34d0ab` (`v2.3.2`)

Modification statement: on 2026-08-25, the ONNX encoder and decoder identified in `MODEL_PROVENANCE.json` were compiled for RK3588 FP16 with RKNN Toolkit2 2.3.2. The package does not claim to contain the original PyTorch-to-ONNX export process.

The Release package includes a copy of the Apache License 2.0 and this notice. Upstream copyright and file-level notices remain in force. No trademark rights, upstream endorsement, or warranty are granted.

## Verified locally, not redistributed as Release assets

### CLIP text encoder

`clip_text_fp16.rknn` is identified by hash for reproducibility but is not uploaded. The source model is associated with `openai/clip-vit-base-patch32`; its public model card does not state an explicit weight license. The MIT license in the OpenAI CLIP source repository is not treated here as automatic permission to redistribute a converted weight artifact.

### YOLO-World detector

`yolo_world_v2s_i8.rknn` and `yolo_world_v2s_fp16.rknn` are identified by hash but are not uploaded in this Release. The Rockchip YOLO-World adaptation and upstream YOLO-World repository use GPL-3.0 and also mention a separate commercial licensing route. Binary redistribution requires satisfying the applicable corresponding-source obligations; linking to an upstream page is not represented here as a completed source bundle.

The repository therefore publishes exact local-conversion instructions for these artifacts instead of the binaries.
