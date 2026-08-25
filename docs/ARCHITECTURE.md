# 系统架构

## 目标

本项目解决一个具体链路：摄像头帧和运行时目标名称进入系统后，由 YOLO-World 找到目标 bbox，再把 bbox 作为 MobileSAM 的 box prompt 得到像素级 mask。提示词变化不能要求重启进程，实时画面不能被旧提示词的在途结果覆盖。

![YOLO-World to MobileSAM data flow](assets/pipeline-overview.svg)

## 数据流

```text
UTF-8 target
    │
    ├─ CLIP BPE tokenizer ─ CLIP Text RKNN ─ 1 x 512 embedding ─┐
    │                                                           │
BGR frame ─ 640 letterbox ─ YOLO-World RKNN ─ NMS ─ bbox ───────┼─ MobileSAM Decoder
    │                                                           │        │
    └─ RGB resize-longest-side + 448 pad ─ MobileSAM Encoder ───┘        ▼
                                                                     mask + IoU
                                                                         │
                                                        overlay + largest contour
```

### 文本支路

`clip_tokenizer.py` 实现 CLIP byte-pair tokenizer，不依赖 PyTorch 或 Transformers。上下文固定为 20 token；文本模型输出 512 维 embedding。检测模型保留 80 个文本槽位，本项目把当前目标填入第一个槽位，其余槽位置零。

embedding 以规范化后的完整提示词为键缓存在进程内。切换到新词时运行一次 CLIP Text；再次输入已有词时直接复用。缓存不跨进程持久化。

### YOLO-World 支路

输入帧按 Rockchip 示例的几何规则 letterbox 到 640 × 640。后处理兼容当前模型的 6 输出和 9 输出布局，完成 DFL box 解码、阈值过滤、NMS，并把 bbox 映射回原图坐标。当前 UI 选择置信度最高的保留目标。

### MobileSAM 支路

原图转 RGB，最长边缩放到 448，再在右侧/底部补零。Encoder 产生每帧独立的 image embedding；YOLO bbox 被映射到 decoder 坐标，Decoder 输出低分辨率 mask logits 和 IoU 预测。logits 恢复到原图尺寸并以 0 为阈值二值化，最后保留最大外轮廓用于显示。

每帧的 `SamImageContext` 是不可变对象，绑定 image embedding、原图尺寸和缩放尺寸，防止跨帧误用 embedding。

## 三核调度

`three-core` 为每个并行角色创建独立的 RKNNLite context 和单线程 executor：

| NPU 核 | 运行角色 | 原因 |
| --- | --- | --- |
| core 0 | MobileSAM Encoder | 通常是最长阶段，可与检测重叠 |
| core 1 | CLIP Text、MobileSAM Decoder | 文本只在首次出现的提示词上运行；Decoder 依赖 bbox |
| core 2 | YOLO-World | 与当前帧 Encoder 并行 |

开启 SAM 的单帧时序：

```text
core 0: [------------ MobileSAM Encoder ------------]
core 1: [CLIP if cache miss]                 [Decoder]
core 2: [----------- YOLO-World + bbox -----------]
        <----------- parallel region ------------>
```

Decoder 必须等待当前帧的 bbox 和 image embedding，因此不能被“强行三核并行”。`serial-auto` 使用 RKNN 的自动核选择并按顺序运行，作为兼容和诊断模式。

不要在多个 Python 线程间并发调用同一个 RKNNLite context。本实现把 context 的创建、推理和释放都放在拥有它的 executor 中。

![Measured three-core scheduling timeline](assets/three-core-schedule.svg)

## 实时程序并发

`realtime_target_ui.py` 有三个相互解耦的职责：

1. `CaptureWorker` 持续读取 V4L2，只保存最新帧，避免摄像头队列不断积压。
2. `InferenceWorker` 取最新可用帧并运行统一 pipeline。
3. 窗口或 HTTP/MJPEG 服务只读取已经发布的显示帧和状态快照。

每次提示词或 SAM 开关变化时，`config_generation` 加一。推理开始时捕获 generation，结束发布前再次比较；如果配置已经变化，该结果标记为 `STALE` 并丢弃。这样旧提示词的慢结果不会覆盖新提示词画面。

错误分两类：

- 可恢复推理错误：保留摄像头画面并叠加错误信息，后续帧继续尝试。
- 摄像头/初始化等致命错误：设置停止状态，确保在途任务不能把失败改写为成功。

## HTTP 暴露面

Web 模式使用标准库 `ThreadingHTTPServer`，提供页面、MJPEG、快照、状态及两个控制接口。它没有认证和 TLS，因此默认绑定 `127.0.0.1`，通过 SSH tunnel 使用。接口明细见 [CLI 与 HTTP 参考](CLI_REFERENCE.md)。

## 测试边界

主机侧测试通过 fake RKNN runtime 验证：

- 预处理、bbox 和 mask 后处理；
- 四个模型的 NPU 核映射；
- Encoder/YOLO 并行与 Decoder 依赖顺序；
- context 在拥有线程中释放；
- CLIP tokenizer 与 C++ 参考结果一致；
- 动态配置、陈旧结果丢弃、HTTP 状态和控制接口。

这些测试不加载真实 RKNN，不能证明某个模型、驱动、摄像头或板端性能。真实硬件验证步骤见 [部署指南](DEPLOYMENT.md) 和 [性能记录](BENCHMARKS.md)。
