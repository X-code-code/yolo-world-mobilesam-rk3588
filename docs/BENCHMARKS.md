# RK3588 真机性能记录

## 结论

在固定图片的完整 YOLO-World + MobileSAM 链路中，显式三核分工把 wall throughput 从 4.83 FPS 提高到 7.42 FPS，提升约 53.5%。两种模式的检测与 mask 签名一致。

真实摄像头界面中，三核模式同样更快，但端到端 FPS 低于静态基准，因为还包含采集、颜色转换、绘制、状态同步和 MJPEG/窗口相关开销。

![Serial and three-core throughput comparison](assets/benchmark-fps.svg)

## 环境

| 字段 | 值 |
| --- | --- |
| 日期 | 2026-08-25 |
| 开发板 | 鲁班猫 4，RK3588 系列 NPU |
| RKNNLite / Runtime | 2.3.2 |
| RKNPU Driver | 0.9.8 |
| YOLO | `yolo_world_v2s_i8.rknn` |
| MobileSAM | FP16 Encoder + FP16 Decoder |
| 提示词 | `cup` |
| SAM | 开启 |

模型 hash 见 [模型说明](MODELS.md)。可机读的脱敏记录是 [`benchmarks/lubancat4_rk3588_2026-08-25.json`](../benchmarks/lubancat4_rk3588_2026-08-25.json)。仓库不包含原始测试图片或摄像头帧。

## 静态图片方法

每种模式分别：

1. 创建并常驻加载四个 RKNNLite context；
2. 预热 5 次；
3. 对同一图片连续测量 50 次完整推理；
4. 记录各阶段 `perf_counter` 墙钟时间；
5. 比较每次结果的 confidence、bbox、mask IoU、mask 非零像素数和 mask SHA-256；
6. 切换模式后重复。

复现命令：

```bash
python3 python/benchmark_npu_modes.py \
  --image data/test.jpg \
  --target cup \
  --warmup 5 \
  --runs 50 \
  --output results/benchmark_npu_modes.json
```

`wall_fps` 使用整个 50 次测量区间计算；`fps` 使用每次 `total_ms` 的平均值换算。报告结果时优先给出 `wall_fps`。

## 静态结果

| 模式 | mean total | P50 total | P90 total | mean YOLO | mean Encoder | mean Decoder | wall FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `serial-auto` | 206.62 ms | 188.85 ms | 229.96 ms | 63.46 ms | 120.94 ms | 16.28 ms | 4.83 |
| `three-core` | 134.64 ms | 133.28 ms | 142.79 ms | 66.21 ms | 117.78 ms | 14.33 ms | 7.42 |

三核模式的平均并行区间为 118.90 ms。总耗时不是各阶段简单相加，因为 YOLO 与 Encoder 重叠运行。

### 输出一致性

| 检查 | 结果 |
| --- | --- |
| 串行模式内 50 次一致 | 是 |
| 三核模式内 50 次一致 | 是 |
| 三核与串行一致 | 是 |
| 浮点比较绝对容差 | `1e-6` |

本次输出签名：confidence `0.9475963712`、bbox `[0, 70, 92, 228]`、mask IoU `0.98193359375`、mask pixels `11534`、mask SHA-256 `c8a28207e28dcb1e665e346d72101370bab585f9d7d481145af9954cfc2c8f95`。

签名用于验证两种调度未改变输出，不是通用精度指标；它只对应这张未公开测试图片和这个提示词。

## 摄像头端到端结果

同一摄像头、同一场景、相同模型和 `cup` 提示词，分别运行 30 个已发布的完整检测+分割结果：

| 模式 | 推理 FPS 状态值 | 推理错误 |
| --- | ---: | ---: |
| `serial-auto` | 约 4.75 | 0 |
| `three-core` | 约 6.40 | 0 |

这里的状态值是 UI 中的推理结果 EMA，不是摄像头传感器 FPS，也不是纯 RKNN kernel throughput。它适合说明本次端到端改善，但不适合跨设备排名。

## 正确比较方式

- 固定模型 hash、输入图片、提示词、阈值、SAM 开关、分辨率和电源/散热模式。
- 每种模式独立创建 context 并预热。
- 同时报告延迟分位数、wall FPS 和输出一致性。
- 摄像头性能另外报告，注明采集格式、节点、推理次数和错误数。
- 不把后端推理 FPS 当作最终 Viewer FPS。
- 如果模型输出不同，先停止性能结论并诊断正确性。
