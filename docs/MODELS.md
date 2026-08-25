# 模型、转换与校验

## 为什么仓库不带模型

五个已验证 RKNN 文件合计约 212 MiB，其中 `clip_text_fp16.rknn` 单文件约 123.5 MiB，超过 GitHub 普通 Git 的 100 MiB 单文件限制。更重要的是，模型权重、YOLO-World 许可和 RKNN 工具链条款与本仓库源代码许可不是同一件事。

因此 `.gitignore` 明确排除 `.rknn`、`.onnx`、PyTorch 权重和厂商二进制。使用者需要从上游获取 ONNX 并自行转换，或在符合上游许可的前提下从自己的制品存储下载。

## 运行时文件名

| 文件 | 用途 | 精度 | 字节数（已验证制品） | SHA-256 |
| --- | --- | --- | ---: | --- |
| `clip_text_fp16.rknn` | 提示词 token 到 512 维 embedding | FP16 | 129,536,946 | `872765bb5f9813d96d57888d97ab3599270264aeeead2a97e505bffbed466563` |
| `yolo_world_v2s_i8.rknn` | 默认文本条件检测 | I8 | 15,327,930 | `c2af5058828ff62f39910d1f84284df5644ec46c05317aec54fa5235b04fa61a` |
| `yolo_world_v2s_fp16.rknn` | 检测精度对照，可选 | FP16 | 27,844,050 | `bc03e95b31b9bd73ce308a981e76b97b3c7cd11cdd44955671fd594477965fc7` |
| `mobilesam_encoder_fp16.rknn` | 每帧 image embedding | FP16 | 38,544,118 | `d1c3104934967cc488c83471bd645fee783f2609f90f446e1dfaf77e532875d9` |
| `mobilesam_decoder_fp16.rknn` | bbox prompt 到 mask | FP16 | 11,266,702 | `5f509ea393396ac33cb4c3805492227ca4f6b89c533dcbf6ef5a94808fd89660` |

这些 hash 对应 2026-08-25 已在板端验证的本地制品。转换工具版本、图优化或量化数据变化都可能改变文件 hash。`MODEL_SHA256SUMS` 是制品身份记录，不是所有有效模型的唯一白名单。

默认运行需要 CLIP、I8 YOLO、MobileSAM Encoder 和 Decoder 四个文件。FP16 YOLO 仅在显式传入 `--yolo-model model/yolo_world_v2s_fp16.rknn` 时使用。

## 上游来源

- RKNN 示例与转换脚本：[airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)
- Rockchip YOLO-World 适配：[airockchip/YOLO-World](https://github.com/airockchip/YOLO-World)
- YOLO-World 上游：[AILab-CVC/YOLO-World](https://github.com/AILab-CVC/YOLO-World)
- Rockchip MobileSAM 适配：[airockchip/MobileSAM](https://github.com/airockchip/MobileSAM)
- MobileSAM 上游：[ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

Rockchip Model Zoo 的 `examples/yolo_world/model/download_model.sh` 与 `examples/mobilesam/model/download_model.sh` 提供对应 ONNX 下载入口。下载地址可能变化，应以当前上游仓库为准。

## 使用 RKNN Model Zoo 转换

以下命令针对已验证 checkout 中的实际嵌套脚本路径。转换在 x86_64 Linux/WSL 中使用 RKNN Toolkit2 完成，不在板端用 RKNNLite 转换。

先设置目标仓库目录，变量名不要指向系统目录：

```bash
export RKNN_REPO_DEST=/path/to/yolo-world-mobilesam-rk3588
```

### YOLO-World 与 CLIP Text

```bash
cd /path/to/rknn_model_zoo/examples/yolo_world/model
bash download_model.sh

cd ../python
python3 clip_text/convert.py \
  ../model/clip_text.onnx rk3588 fp \
  "${RKNN_REPO_DEST}/model/clip_text_fp16.rknn"

python3 yolo_world/convert.py \
  ../model/yolo_world_v2s.onnx rk3588 i8 \
  "${RKNN_REPO_DEST}/model/yolo_world_v2s_i8.rknn"

python3 yolo_world/convert.py \
  ../model/yolo_world_v2s.onnx rk3588 fp \
  "${RKNN_REPO_DEST}/model/yolo_world_v2s_fp16.rknn"
```

I8 转换使用 Model Zoo `examples/yolo_world/model/dataset.txt` 指定的量化数据。替换数据集可能改变精度与 hash，应记录数据来源和 Toolkit 版本。

CLIP Text 转换脚本只接受 `fp`；当前图固定输入为 `[1, 20]` token IDs。

### MobileSAM

```bash
cd /path/to/rknn_model_zoo/examples/mobilesam/model
bash download_model.sh

cd ../python
python3 encoder/convert.py \
  ../model/mobilesam_encoder.onnx rk3588 fp \
  "${RKNN_REPO_DEST}/model/mobilesam_encoder_fp16.rknn"

python3 decoder/convert.py \
  ../model/mobilesam_decoder.onnx rk3588 fp \
  "${RKNN_REPO_DEST}/model/mobilesam_decoder_fp16.rknn"
```

当前 MobileSAM 转换脚本只接受 `fp`。Encoder 使用 448 × 448 输入；Decoder 期望 image embeddings `[1, 256, 28, 28]`、两个 box prompt 点、112 × 112 mask input，并输出 IoU predictions 与 low-resolution masks。

## 转换后的检查顺序

1. 记录 RKNN Toolkit2 版本、Model Zoo 来源、ONNX hash、RKNN hash和量化数据。
2. 把 RKNN 复制到板端 `model/`。
3. 使用静态图片分别跑 I8 和 FP16 YOLO，视觉核对 bbox 与置信度。
4. 开启 MobileSAM，视觉核对 mask 是否贴合目标。
5. 运行动态目标验证，确认新提示词和缓存命中。
6. 最后运行串行/三核基准，并检查 `all_results_consistent=true`。

只有“转换成功”或“模型能初始化”不足以证明完整链路正确。

## 许可边界

本仓库没有分发上述模型。YOLO-World 上游仓库标示 GPL-3.0，并说明独立商业许可；MobileSAM 使用 Apache-2.0。下载、转换、分发和商用模型前，必须检查当时的上游许可与权重条款。完整清单见 [第三方许可说明](../THIRD_PARTY_NOTICES.md)。
