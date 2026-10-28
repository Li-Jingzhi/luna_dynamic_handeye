# Luna V2.0 双目实时画面与传输简述

记录日期：2026-10-28。本文对应当前新标定参数的实时预览，不代表此前其他视频推流方案。

## 1. 实时画面如何得到

```text
双目相机 → Orin 相机驱动 / Argus / ISP
        → GStreamer 同时读取 CSI0、CSI1（2560×1984，配置30帧/秒）
        → 每眼缩小为640×496，NVENC硬件H.264编码
        → RTP/UDP两路视频传到本机
        → OpenCV按新标定参数去畸变、极线校正 → 本机窗口
```

相机经ISP输出NV12图像，经NVMM硬件缓冲转换和编码；本机解码为BGR，不是传感器Bayer RAW。当前预览不额外旋转180°；内参K、投影矩阵P随预览分辨率同比缩放。

|组件|来源及作用|
|---|---|
|厂家相机驱动、设备树、GMSL初始化工具|当前这台 Orin 原有；本次执行原有 `cam_geac/demo/init_demo 3` 初始化相机，没有重装内核或驱动|
|JetPack Argus、ISP、GStreamer及 NVIDIA 插件|当前 Orin 已具备；通过 `nvarguscamerasrc` 取图，`nvvidconv` 转换格式|
|Orin 的 GStreamer 命令行及编码/RTP插件|当前硬件推流直接使用已有 `gst-launch-1.0`，不依赖板端Python图像编码|
|`capture_server.py`、`stereo_pairing.py`|此前棋盘格采集脚本，当前已停止其HTTP服务以释放CSI相机；不参与本次UDP预览|
|本机 Python、OpenCV、NumPy、GStreamer|已有可用环境；使用支持GStreamer的系统 `/usr/bin/python3` 和 `openh264dec` 解码；Paramiko仅用于远程部署控制|

上一台 Orin 曾修复过驱动配套问题；不能据此认为当前这台也重新安装过驱动。当前预览不依赖 ROS、NoMachine，也不使用 VPI/CUDA 校正；图像校正在本机通过 OpenCV 完成。

## 2. 图像如何传到本机，依赖什么

**Orin NVENC硬件H.264编码 → RTP封装 → UDP直接传输 → 本机解码校正**。

|环节|具体技术与依赖|
|---|---|
|Orin取图和缩放|JetPack/厂家驱动、Argus、GStreamer `nvarguscamerasrc`、`nvvidconv`；依赖正确的相机驱动、设备树、GMSL初始化及可用CSI相机|
|硬件编码|GStreamer `nvv4l2h264enc` 调用Orin硬件编码器（NVENC），不使用CPU软件H.264编码；当前每眼640×496、30fps、目标4 Mbit/s，关键帧间隔30帧|
|网络发送|`h264parse` → `rtph264pay` → `udpsink`，RTP负载类型96，UDP负载包上限配置1400字节；需要两端IP可达及UDP端口未被防火墙阻挡|
|本机接收解码|`udpsrc` → `rtpjitterbuffer` → `rtph264depay` → `h264parse` → `openh264dec` → BGR → OpenCV；本机当前是软件解码，不要求NVIDIA显卡|
|校正显示|支持GStreamer和桌面窗口的OpenCV、NumPy，以及本机新标定K/D/R/P；内参和投影矩阵按640×496同比缩放|

当前网络：Orin `6.6.7.100` → 本机 `6.6.7.101`；左CSI0发送到本机 **UDP 5100**，右CSI1发送到 **UDP 5102**。Wi-Fi也可使用，但须修改发送端目标IP，并检查路由、防火墙及带宽。两路目标编码码率合计约8 Mbit/s，实际流量会随场景与协议开销变化。

**SSH只用于登录、部署和启动/停止推流，视频本身不经过SSH隧道。** RTP/UDP当前没有加密或重传，适用于可信局域网；NoMachine、ROS、VPI均不参与这条预览链。

本次已验证Orin的 `nvv4l2h264enc`、`rtph264pay`，以及本机 `openh264dec`、`rtph264depay` 可用，未新增安装依赖。普通pip版OpenCV不保证带GStreamer支持；当前使用系统Python/OpenCV。

Orin启动脚本：`/home/nvidia/luna_stereo_calibration/start_h264_udp.sh`；日志：同目录 `h264_udp.log`。发送端运行后，在项目根目录打开本机窗口：

```bash
/usr/bin/python3 stereo_calibration/live_rectified_preview.py --udp \
  --calibration stereo_calibration/output/luna_current_20261028_fisheye_60.json
```

当前已停止原棋盘格HTTP服务。后续棋盘格无损采集仍应使用原 `/capture` PNG与元数据链；须先停止占用相机的UDP发送端，再通过 `start_capture.py` 恢复采集服务。H.264预览画面不作为标定原图。

## 3. 实时性与限制

|环节|当前情况|
|---|---|
|相机源|每眼配置2560×1984、30帧/秒；配置值不等于本次已独立测得的持续帧率|
|本机校正预览|本次新H.264/UDP链窗口计数约15.9次更新/秒，已确认画面持续更新；尚未达到30次/秒。原JPEG/SSH链约15.7–15.9次/秒，未进行严格同条件性能对比|
|端到端延迟|尚未测量曝光到屏幕的延迟，不能仅凭帧率推断延迟|
|测量/落盘速率|预览速度不能证明合格双目配对或PNG落盘能持续30组/秒|

本机接收设置30ms RTP抖动缓冲、超时缓冲丢弃，并以 `appsink drop=true max-buffers=1 sync=false` 保留最新帧。实际速度受编码、网络丢包/抖动、解码、校正与显示影响；30ms缓冲配置不是端到端延迟测量。

**预览不等于同步测量。** 当前两路UDP视频分别接收，未按曝光时刻对齐；共同启动也不等于硬件同步。原棋盘格 `/capture`要求共同源时钟差与主机到达时差均≤10 ms，且每帧只消费一次；这些时间检查仍不等于已验证的曝光硬同步。棋盘格采集须保持静止。

当前新参数为 `output/luna_current_20261028_fisheye_60.json`。入口为 `live_rectified_preview.py`，按Q关闭。尚未完成独立验证，也未把新参数替换到正式手眼采集配置。
