# Model directory

Download the redistributable MobileSAM Release bundle from the repository root:

```bash
python3 scripts/download_models.py
```

Then place the locally converted CLIP and YOLO RKNN artifacts here as well:

```text
clip_text_fp16.rknn
yolo_world_v2s_i8.rknn
yolo_world_v2s_fp16.rknn          # optional comparison model
mobilesam_encoder_fp16.rknn
mobilesam_decoder_fp16.rknn
```

Model files are ignored by ordinary Git. The two MobileSAM artifacts are distributed in a license-carrying GitHub Release archive; CLIP and YOLO artifacts are not public Release assets in v0.1.0 and must be converted locally. See [the model guide](../docs/MODELS.md), [`MODEL_RELEASES.json`](../MODEL_RELEASES.json), and [`MODEL_SHA256SUMS`](../MODEL_SHA256SUMS).
