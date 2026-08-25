# 模型、转换、发布与校验

## 先说结论

这个项目已经有真实上板模型，不只是实现说明。2026-08-25，WSL 转换目录与鲁班猫 4 部署目录中的五个 RKNN 文件逐个 SHA-256 一致；五件套合计约 212.21 MiB，完整 YOLO-World → MobileSAM 链路和三核基准也已在板端运行。

模型二进制不进入普通 Git 历史：`clip_text_fp16.rknn` 单文件约 123.5 MiB，超过 GitHub 普通 Git 100 MiB 限制，而且模型许可与本仓库原创代码的 Apache-2.0 不是同一件事。可公开再分发的模型使用 GitHub Release；其余模型保留来源、转换脚本和已验证 hash，由用户从官方 ONNX 本地生成。

## v0.1.0 发布范围

| 文件 | 运行用途 | 发布方式 | 原因 |
| --- | --- | --- | --- |
| `mobilesam_encoder_fp16.rknn` | 每帧 image embedding | Release 压缩包 | MobileSAM 与 Rockchip fork 均为 Apache-2.0 |
| `mobilesam_decoder_fp16.rknn` | bbox prompt 到 mask | Release 压缩包 | 与许可证、变更说明和 provenance 一起交付 |
| `clip_text_fp16.rknn` | 提示词到 512 维 embedding | 本地转换 | 官方 CLIP 模型卡未明确声明权重再分发许可 |
| `yolo_world_v2s_i8.rknn` | 默认文本条件检测 | 本地转换 | GPL-3.0 二进制发布需要完整对应源码包；本版本不冒充已满足 |
| `yolo_world_v2s_fp16.rknn` | 可选精度对照 | 本地转换 | 同上 |

列出状态：

```bash
python3 scripts/download_models.py --list
```

下载并校验 Release 中的 MobileSAM 包：

```bash
python3 scripts/download_models.py
```

下载器先验证 ZIP 的大小与 SHA-256，再只提取清单中的模型，并再次验证每个 RKNN。坏文件不会被静默覆盖；确实要替换时显式使用 `--force`。

这两个文件只完成分割部分。真实动态提示词检测还需要本地生成 CLIP 和至少一个 YOLO 模型。

## 五个已验证制品

| 文件 | 精度 | 字节数 | SHA-256 |
| --- | --- | ---: | --- |
| `clip_text_fp16.rknn` | FP16 | 129,536,946 | `872765bb5f9813d96d57888d97ab3599270264aeeead2a97e505bffbed466563` |
| `yolo_world_v2s_i8.rknn` | I8 | 15,327,930 | `c2af5058828ff62f39910d1f84284df5644ec46c05317aec54fa5235b04fa61a` |
| `yolo_world_v2s_fp16.rknn` | FP16 | 27,844,050 | `bc03e95b31b9bd73ce308a981e76b97b3c7cd11cdd44955671fd594477965fc7` |
| `mobilesam_encoder_fp16.rknn` | FP16 | 38,544,118 | `d1c3104934967cc488c83471bd645fee783f2609f90f446e1dfaf77e532875d9` |
| `mobilesam_decoder_fp16.rknn` | FP16 | 11,266,702 | `5f509ea393396ac33cb4c3805492227ca4f6b89c533dcbf6ef5a94808fd89660` |

默认运行需要 CLIP、I8 YOLO、MobileSAM Encoder 和 Decoder 四个文件。FP16 YOLO 只在显式传入 `--yolo-model model/yolo_world_v2s_fp16.rknn` 时使用。

`MODEL_SHA256SUMS` 适合板端 `sha256sum -c`；`MODEL_PROVENANCE.json` 还固定了 ONNX 输入、Model Zoo 版本、转换器 hash、校准输入和环境；`MODEL_RELEASES.json` 是下载器使用的机器可读发布清单。重新转换时，只要 Toolkit、图优化或量化输入变化，输出 hash 都可能不同，因此 hash 是制品身份，不是对所有有效模型的唯一白名单。

## 可复现的真实转换边界

本项目实际完成的是：

```text
Rockchip 提供的 4 个 ONNX
  -> RKNN Toolkit2 2.3.2
  -> RK3588 的 5 个 RKNN
```

项目没有执行或伪造 `.pt/.pth -> ONNX` 阶段。源 ONNX 下载入口来自固定的 RKNN Model Zoo `v2.3.2`（commit `bad6c7334531becaf90a561988519b7bec34d0ab`），下载后还会核对本次真机版本的大小和 SHA-256。

在 x86_64 Ubuntu 22.04 / WSL 中准备官方 Model Zoo：

```bash
git clone --branch v2.3.2 --depth 1 \
  https://github.com/airockchip/rknn_model_zoo.git \
  /path/to/rknn_model_zoo
```

安装与授权条款相符的 `rknn-toolkit2==2.3.2` 后，从本仓库根目录执行：

```bash
python3 conversion/download_onnx.py \
  --output-dir model/onnx \
  --record model/onnx/download-record.json

python3 conversion/convert_models.py \
  --onnx-dir model/onnx \
  --output-dir model \
  --model-zoo-root /path/to/rknn_model_zoo \
  --record model/conversion-record.json
```

I8 YOLO 必须使用 `v2.3.2` 的官方 `examples/yolo_world/model/dataset.txt`、20 张 COCO 校准图和 `coco_text_outp.npy`。脚本会验证 dataset 与文本 embedding 的 hash、行数和图片存在性，不会悄悄拿任意图片代替。COCO 图片不复制到本仓库。

只转换运行必需的 CLIP 与 I8 YOLO：

```bash
python3 conversion/convert_models.py \
  --models clip_text_fp16 yolo_world_v2s_i8 \
  --onnx-dir model/onnx \
  --output-dir model \
  --model-zoo-root /path/to/rknn_model_zoo
```

脚本拒绝静默覆盖。每次转换会写 JSON 记录，包含精确 RKNN API 参数、主机环境、输入输出 hash、校准依赖，以及是否与本次已上板 reference hash 一致。全部参数与离线测试见 [`conversion/README.md`](../conversion/README.md)。

## 转换后的功能检查

1. 运行 `sha256sum -c MODEL_SHA256SUMS`，确认使用的是哪一组制品。
2. 分别运行 I8 与 FP16 YOLO 静态图片测试，视觉核对 bbox 和 confidence。
3. 开启 MobileSAM，核对 mask 是否贴合 bbox 中的真实目标。
4. 运行 `cup -> cat -> cup` 动态目标验证，确认提示词实际改变且第三次命中 embedding cache。
5. 最后运行串行/三核基准，并检查输出一致性。

“转换完成”或“模型能初始化”都不能代替真实视觉结果检查。

## Release 包怎样生成

维护者先把两个已核验的 MobileSAM RKNN 放在仓库外，再运行：

```bash
python3 scripts/build_model_release.py \
  --model-dir /path/to/verified-models \
  --output /path/to/mobilesam-rk3588-fp16-v0.1.0.zip
```

构建器先核对两个模型的大小与 hash，再生成确定性的 ZIP，内含 Apache-2.0 全文、模型许可说明、provenance、SHA256SUMS 和使用说明。模型和 ZIP 都不会误入 Git 历史。

正常构建还会把最终 ZIP 的文件名、字节数和 SHA-256 与 `MODEL_RELEASES.json` 再比较，不一致即失败。只有在有意更新 provenance/notice、需要先计算新 hash 时才使用 `--candidate`；把候选值人工审核写回 manifest 后，必须再跑一次不带 `--candidate` 的严格构建，严格构建通过的文件才允许上传。

## 许可边界

仓库原创代码使用 Apache-2.0，并不自动改变模型权重、Toolkit、Runtime 或训练数据条款。v0.1.0 的分发决定记录在 [`MODEL_LICENSES.md`](../MODEL_LICENSES.md)；完整第三方清单见 [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md)。这是一份工程合规记录，不替代法律意见。
