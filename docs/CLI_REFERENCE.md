# CLI 与 HTTP 参考

所有命令默认从仓库根目录运行。模型参数的默认位置是 `model/`，BPE merges 默认使用 `python/bpe_simple_vocab_16e6.txt`。

## 共用模型参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--clip-model` | `model/clip_text_fp16.rknn` | CLIP Text RKNN |
| `--yolo-model` | `model/yolo_world_v2s_i8.rknn` | YOLO-World V2-S RKNN |
| `--sam-encoder-model` | `model/mobilesam_encoder_fp16.rknn` | MobileSAM Encoder |
| `--sam-decoder-model` | `model/mobilesam_decoder_fp16.rknn` | MobileSAM Decoder |
| `--merges` | `python/bpe_simple_vocab_16e6.txt` | CLIP BPE merges |
| `--conf` | `0.25` | 置信度阈值，范围建议 0–1 |
| `--nms` | `0.45` | NMS IoU 阈值，范围建议 0–1 |
| `--npu-mode` | `three-core` | `three-core` 或 `serial-auto` |

## `rknn_target_pipeline.py`

对单张图片运行一次目标检测，可选 MobileSAM，并保存可视化与 JSON。

```bash
python3 python/rknn_target_pipeline.py \
  --image data/test.jpg \
  --target "coffee mug" \
  --output-dir results/coffee_mug
```

专有参数：

| 参数 | 必需/默认 | 说明 |
| --- | --- | --- |
| `--image` | 必需 | 输入图片路径 |
| `--target` | 必需 | 运行时目标提示词 |
| `--output-dir` | `results` | 输出目录 |
| `--no-sam` | 关闭 | 只运行 YOLO，不输出有效 mask |

检测存在时返回 `0`，未检测到目标返回 `2`。每次都会写出三张可视化；无 mask 时相关图片保留原图/检测叠加，具体指标以 `metrics.json` 为准。

## `verify_dynamic_targets.py`

在一个进程和一组常驻模型中依次切换提示词。

```bash
python3 python/verify_dynamic_targets.py \
  --image data/test.jpg \
  --targets cup bottle cup \
  --with-sam \
  --output-dir results/dynamic_targets
```

| 参数 | 必需/默认 | 说明 |
| --- | --- | --- |
| `--image` | 必需 | 所有目标共用的图片 |
| `--targets` | `cup cat cup` | 一个或多个空格分隔的目标；带空格短语需加引号 |
| `--output-dir` | `results/dynamic_targets` | 每个目标的结果和汇总 JSON |
| `--with-sam` | 关闭 | 启用 bbox 到 mask 链路 |

输出汇总字段包括 `embedding_cache_hit`、`clip_text_ms`、`total_ms`、检测与 mask 指标。

## `realtime_target_ui.py`

实时摄像头程序。

```bash
python3 python/realtime_target_ui.py --ui web --device /dev/video11 --target cup
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--ui` | `web` | `web` 或 `window` |
| `--device` | `/dev/video11` | V4L2 节点或 OpenCV 可接受的视频源 |
| `--width` | `640` | 请求采集宽度，必须大于 0 |
| `--height` | `480` | 请求采集高度，必须大于 0 |
| `--capture-fps` | `30` | 请求采集 FPS |
| `--target` | `cup` | 初始目标 |
| `--no-sam` | 关闭 | 启动时禁用 MobileSAM，可在网页重新打开 |
| `--bind` | `127.0.0.1` | Web 监听地址 |
| `--port` | `8080` | Web 端口，范围 1–65535 |
| `--jpeg-quality` | `80` | MJPEG 质量，范围 30–100 |
| `--window-name` | `YOLO-World + MobileSAM` | 窗口标题 |
| `--max-inferences` | `0` | N 次已发布推理后停止；0 表示持续运行 |
| `--log-every` | `30` | 每 N 次结果打印状态；0 关闭周期日志 |
| `--save-last` | 未设置 | 退出时保存最后显示帧 |

目标会去掉首尾空白并折叠连续空白；空字符串被拒绝；Web 提交最大 256 UTF-8 字节。

### HTTP 接口

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/` | 控制页面 |
| `GET` | `/status` | 当前 JSON 状态 |
| `GET` | `/snapshot.jpg` | 最新显示帧 JPEG |
| `GET` | `/stream.mjpg` | multipart MJPEG 流 |
| `POST` | `/target` | 表单字段 `target=<text>`，提交新目标 |
| `POST` | `/sam` | 表单字段 `enabled=1` 或 `0` |

示例：

```bash
curl http://127.0.0.1:8080/status
curl -X POST -d 'target=bottle' http://127.0.0.1:8080/target
curl -X POST -d 'enabled=0' http://127.0.0.1:8080/sam
```

状态包含采集/推理 FPS、活动和待处理目标、模型就绪状态、错误计数、NPU 模式及最近一次各阶段耗时。接口没有认证，默认只应通过 SSH tunnel 使用。

## `benchmark_npu_modes.py`

在同一图片上顺序运行 `serial-auto` 与 `three-core`，并比较输出一致性。

```bash
python3 python/benchmark_npu_modes.py \
  --image data/test.jpg \
  --target cup \
  --warmup 5 \
  --runs 50 \
  --output results/benchmark_npu_modes.json
```

| 参数 | 必需/默认 | 说明 |
| --- | --- | --- |
| `--image` | 必需 | 固定输入图片 |
| `--target` | 必需 | 固定提示词 |
| `--warmup` | `5` | 每种模式预热次数，可为 0 |
| `--runs` | `30` | 每种模式测量次数，至少 1 |
| `--no-sam` | 关闭 | 只基准检测链路 |
| `--output` | 未设置 | 纯 JSON 输出路径；建议设置，因为 RKNN 日志可能写 stdout |

输出一致性比较 confidence、bbox、mask IoU、mask 像素数和 mask SHA-256；浮点绝对容差为 `1e-6`。
