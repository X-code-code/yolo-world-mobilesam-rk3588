# 部署到鲁班猫 4

## 1. 前置条件

板端需要：

- 64 位 Linux 与可用的 RK3588/RK3588S RKNPU 驱动；
- Python 3；
- 与板端 Runtime/Driver 兼容的 `rknn_toolkit_lite2` wheel；
- NumPy、OpenCV Python；
- V4L2 摄像头（实时模式需要）。

已验证组合是 RKNNLite/Runtime 2.3.2、Driver 0.9.8。不要只更新其中一个组件后假定仍兼容；先运行静态图片验证。

检查环境：

```bash
python3 - <<'PY'
import cv2
import numpy
from rknnlite.api import RKNNLite
print("OpenCV", cv2.__version__)
print("NumPy", numpy.__version__)
print("RKNNLite import OK", RKNNLite)
PY

cat /sys/kernel/debug/rknpu/version 2>/dev/null || true
```

具体 RKNN wheel 和 Runtime 安装方式随板端固件变化，请优先使用板卡厂商提供的匹配包，或参考 [RKNN Toolkit2 文档](https://github.com/airockchip/rknn-toolkit2/tree/master/doc)。

## 2. 获取仓库

在板端：

```bash
cd /home/cat
git clone https://github.com/X-code-code/yolo-world-mobilesam-rk3588.git
cd yolo-world-mobilesam-rk3588
```

也可以在电脑下载后复制到板端。连接别名按本项目环境使用 `ssh lubancat`。

## 3. 获取并放置模型

模型二进制不在普通 Git 历史中。先下载 GitHub Release 里许可明确的两个 MobileSAM RKNN，并自动验证 ZIP 和模型的 SHA-256：

```bash
python3 scripts/download_models.py
```

动态提示词检测还需要 CLIP 与 YOLO。按照 [模型与转换说明](MODELS.md) 使用 `conversion/download_onnx.py` 和 `conversion/convert_models.py` 从官方 ONNX 本地生成；完整运行目录应为：

```text
model/clip_text_fp16.rknn
model/yolo_world_v2s_i8.rknn
model/yolo_world_v2s_fp16.rknn          # 可选：精度对照
model/mobilesam_encoder_fp16.rknn
model/mobilesam_decoder_fp16.rknn
```

如果电脑或既有板端部署目录已经有全部五个已核验模型，也可以从 Windows 复制：

```powershell
scp .\model\*.rknn lubancat:/home/cat/yolo-world-mobilesam-rk3588/model/
```

板端验证本次已测五件套的 SHA-256：

```bash
cd /home/cat/yolo-world-mobilesam-rk3588
sha256sum -c MODEL_SHA256SUMS
```

如果只放置默认运行所需的四个模型，`sha256sum` 会把缺少的 FP16 YOLO 报为失败；可改为单独核对已有文件。`python3 scripts/download_models.py --verify-only` 只核对当前 Release 提供的两个 MobileSAM 文件。不同 RKNN Toolkit 版本重新转换出的文件可能具有不同 hash，不能仅因 hash 不同就判定模型错误，还应保留 `conversion-record.json` 并做功能验证。

模型来源、转换命令、大小与校验值见 [模型说明](MODELS.md)。

## 4. 静态冒烟测试

准备一张不含隐私的图片：

```bash
cp /path/to/test.jpg data/test.jpg
python3 python/rknn_target_pipeline.py \
  --image data/test.jpg \
  --target cup \
  --output-dir results/smoke
```

检查：

- 进程能初始化四个模型；
- JSON 中 `npu_mode` 为 `three-core`；
- 检测目标存在时生成 bbox、mask、轮廓和 `metrics.json`；
- 输出图中的 bbox 与 mask 确实对应原图目标，而不只是命令成功退出。

如果只想先排查 YOLO，可加 `--no-sam`。若三核初始化失败，用 `--npu-mode serial-auto` 区分“模型/运行库问题”和“多 context/核映射问题”。

## 5. 动态提示词验证

```bash
python3 python/verify_dynamic_targets.py \
  --image data/test.jpg \
  --targets cup cat cup \
  --with-sam \
  --output-dir results/dynamic_check
```

不要只看第三次缓存命中，还要分别打开各目录的检测图和 mask。`embedding_cache_hit=true` 证明重复文本没有再次运行 CLIP，不证明每个词都检测正确。

## 6. 摄像头节点

```bash
v4l2-ctl --list-devices
v4l2-ctl --device /dev/video11 --all
```

已验证命令请求 640 × 480、30 FPS。驱动可能选择最接近的格式；最终帧率还受曝光、USB/CSI 链路和颜色转换影响。

## 7. SSH 网页界面

推荐保留服务的回环绑定。Windows PowerShell：

```powershell
ssh -t -L 18080:127.0.0.1:8080 lubancat "cd /home/cat/yolo-world-mobilesam-rk3588 && exec python3 python/realtime_target_ui.py --ui web --device /dev/video11 --bind 127.0.0.1 --port 8080 --target cup"
```

打开 `http://127.0.0.1:18080/`。终端中按 `Ctrl+C` 停止。

也可以先建立 tunnel：

```powershell
ssh -L 18080:127.0.0.1:8080 lubancat
```

然后在登录后的板端 shell 运行：

```bash
cd /home/cat/yolo-world-mobilesam-rk3588
python3 python/realtime_target_ui.py \
  --ui web \
  --bind 127.0.0.1 \
  --port 8080 \
  --device /dev/video11 \
  --target cup
```

Web 服务无认证、无 TLS，而且暴露摄像头流。除非已经增加访问控制，不要把 `--bind` 改为 `0.0.0.0`。

## 8. 板端窗口模式

在有图形桌面的板端终端：

```bash
python3 python/realtime_target_ui.py \
  --ui window \
  --device /dev/video11 \
  --target cup
```

直接键入 ASCII 目标名，`Enter` 应用，`Backspace` 删除，`Esc` 或 `Ctrl+Q` 退出。OpenCV 键盘事件不适合中文输入；中文提示词使用网页界面。

## 9. 常见问题

### `ModuleNotFoundError: rknnlite`

安装与板端架构、Runtime 版本匹配的 RKNN Toolkit Lite2 wheel。不要在 x86 开发机上把 `rknn-toolkit2` 与板端 `rknn_toolkit_lite2` 混为同一个包。

### 模型初始化失败

核对模型目标平台是 `rk3588`，再记录 RKNNLite、Runtime 和驱动版本。先试 `serial-auto`，并逐个检查模型路径和 hash。

### YOLO 报输出数量错误

当前后处理支持 6 或 9 个输出。其他导出图、其他 YOLO-World 版本或重新命名输出的模型需要相应后处理适配。

### 能出框但没有 mask

确认 Encoder 和 Decoder 是同一 MobileSAM 导出组合；检查 bbox 是否有效；查看状态中的 `sam_encoder_ms`、`sam_decoder_ms` 和错误字段。低 IoU 预测不一定表示运行错误，但需要视觉核对。

### 网页打不开

确认板端日志打印 `web_url=http://127.0.0.1:8080`，SSH tunnel 仍保持连接，本机浏览器访问的是转发端口 `18080`。用 `curl http://127.0.0.1:8080/status` 在板端区分服务问题和 tunnel 问题。

### 画面帧率高于/低于推理帧率

采集线程与推理线程独立，`capture_fps` 和 `inference_fps` 不是同一指标。性能比较必须使用相同模型、提示词、SAM 开关、输入和测量方法。
