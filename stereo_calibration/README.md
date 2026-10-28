# 独立棋盘格双目标定

当前实机取图、SSH传输、依赖来源和实时性说明见 [双目实时画面与传输简述](REALTIME_STEREO_OVERVIEW.md)。

移植来源为 `head_eye_capt/portable_stereo_fisheye_calibration`，有效针孔备用入口来源为该项目的 `camera_calibration/fit.py`。来源提交和文件 SHA256 见 `SOURCE.json`。算法代码已包含在本目录，不导入其他项目，不读取其他机器的标定参数。原始鱼眼算法与原项目一致：SB角点检测 → 左右单目鱼眼标定 → 固定内参的双目标定 → 极线校正。

输出左右 K/D、右眼相对左眼 R/T（平移单位米）、R1/R2/P1/P2/Q、基线、训练重投影及纵向误差。局部增加数据哈希、参数检查和独立评估，不自动替换项目现有参数。

## 安装与独立运行

仅离线标定需要 Python、NumPy、OpenCV；不需要 ROS、SDK、机器人或源项目。桌面采集另需Paramiko，Orin端需JetPack GStreamer/Argus与Python gi；无需ROS2。

```bash
cd stereo_calibration
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python calibrate.py --help
```

当前工作区也可使用项目根目录 `.venv/bin/python`，或 `bash stereo_calibration/run.sh calibrate --help`。脚本路径不依赖当前工作目录；输入、输出的相对路径按调用目录解析。复制本目录到其他电脑并安装依赖后，可直接运行各 Python 文件。

## 1. 直连CSI采集（无ROS2）

当前入口与 `head_eye_capt` 的Pro棋盘格流程一致：Orin上GStreamer/Argus直连CSI → 本机SSH隧道请求仅监听localhost的HTTP服务 → 无损PNG及元数据。已移除本目录旧ROS2采集器。

先编辑本目录 `capture_config.json`：Luna示例为 `nvidia@6.6.7.100`，左CSI0、右CSI1；源项目Pro为左CSI8、右CSI0，不能直接照搬。内角点数和格长也须按实物修改。配置不存密码。

```bash
# 项目根目录执行，第一次可用--password交互输入SSH密码；也支持SSH密钥
bash stereo_calibration/run.sh start_capture --password
# 自动停稳/去重采集
bash stereo_calibration/run.sh start_capture --password --auto
# 指定设备配置；恢复自动采集时附加--resume 既有数据目录
bash stereo_calibration/run.sh start_capture --config stereo_calibration/capture_config.json --auto
```

启动器将本目录自包含服务上传到远端用户home下的 `luna_stereo_calibration/`；不调用另一项目。Orin需要JetPack的GStreamer/Argus、Python gi、OpenCV、NumPy，不需要ROS2。相机须未被其他进程占用；启动器不会终止其他采集服务，端口/传感器不匹配会拒绝复用。

每眼mode0、2560×1984、30 FPS，50Hz防频闪，约5秒自动适应后锁定并回读AE/AWB。保存未去畸变、未缩放的ISP BGR PNG（不是Bayer RAW）。预览每眼640×496，仅用于显示/自动姿态门控；保存前以全分辨率检测完整角点。

手动界面：空格保存，C标定（至少20对），Q退出。自动界面：停稳0.5秒、与历史姿态有足够差异后保存，Q或Ctrl+C结束。数据在本目录 `data/`，标定在 `output/`，状态在 `runtime/`。本机界面退出后远端服务继续运行，与源项目一致；需要停采远端时由现场结束该服务。

元数据保留两眼PTS、主机到达时间、配对时差、AE/AWB锁定状态和传感器顺序。两眼置于同一GStreamer管线并使用共同SystemClock；保存时源时间差与到达时间差均须≤10ms，配对每帧只消费一次，过期/超限帧丢弃。预览可显示最新帧，不能作为测量配对。配对检查不等于曝光同步。棋盘格和相机必须静止。两眼锁定各自自适应结果，不保证曝光时间/增益相同，也未测量实际传感器曝光参数。

另开一个独立采集会话用于验证，不将训练图片复制为验证图片。`example_capture_manifest.json` 是本地Luna配置的清单示例，实际清单由GUI生成。

## 2. 拟合和预览

```bash
.venv/bin/python calibrate.py --images data/board_train \
  --output output/stereo_fisheye.json \
  --corners 11 8 --square-mm 35 --min-pairs 20 --balance 0
.venv/bin/python rectify_preview.py --calibration output/stereo_fisheye.json \
  --images data/board_train --output output/preview --count 10
```

只接受 raw 原始鱼眼图像；不能输入已去畸变或已极线校正的图像。建议30–50对训练图，包含画面中央、边缘、近中远距离和倾斜板姿。标定文件独占创建，重复实验使用新文件名。检测失败记录保留在 rejected_pairs 中。

## 3. 独立评估

### 2026-10-28 本机重标定与实时预览

本机新采集 `data/raw_chessboard_20261028_101409` 共60对，57对用于拟合；0048–0050在离线SB完整角点检测时失败，原图保留。输出为 `output/luna_current_20261028_fisheye_60.json`，含左右K/D、右眼相对左眼R/T及校正矩阵。训练单目RMS左0.08454 / 右0.13822 px，双目RMS 0.20465 px，校正纵向差P95 0.30923 px，均为2560×1984尺度；基线60.47672 mm。尚未采集独立验证集，`approved=false`。

当前预览已切换H.264/UDP，在同一入口附加 `--udp`，依赖与端口见 [实时传输说明](REALTIME_STEREO_OVERVIEW.md)。以下为保留的旧SSH/JPEG方式：保留已启动的棋盘格相机服务，在项目根目录运行以下命令，SSH密码通过环境变量 `LUNA_CAPTURE_PASSWORD` 提供（或使用密钥）：

```bash
/usr/bin/python3 stereo_calibration/live_rectified_preview.py \
  --calibration stereo_calibration/output/luna_current_20261028_fisheye_60.json
```

窗口每眼640×496，按对应缩放的K/P进行去畸变和极线校正；Q退出。读取现有服务的最新JPEG预览，不代表同步测量配对，不保存训练数据。新参数仅在此预览使用，现有手眼测量配置尚未替换。

```bash
.venv/bin/python evaluate.py --calibration output/stereo_fisheye.json \
  --images data/board_test --output output/board_evaluation.json
```

建议另采15–20对。评估不再拟合相机参数，并按图像哈希拒绝已使用的训练图片。报告校正纵向差、相邻格边长度误差、无尺度自由度的刚体板形误差，均输出均值/P95/最大值。图像哈希不能识别同一次拍摄的不同压缩版本，仍需人为隔离会话。棋盘格尺寸与平整度误差会进入这些指标；这些指标不能替代躯干系视觉/FK验收。没有自动宣告1 mm达标。

## 已去畸变图像备用入口

`calibrate_pinhole.py` 保留原项目的 SH5 2560×1984、CSI0/CSI1 和话题契约，输入必须为 `ideal_pinhole`，且清单必须记录内角点数和实际格长。它估计有效针孔K及R/T，不能恢复原始鱼眼D：

```bash
.venv/bin/python calibrate_pinhole.py --images data/undistorted_board \
  --output output/effective_pinhole.json --corners 9 6 --square-mm 20
```

本目录采集器、预览器和独立评估器针对 raw 鱼眼路径；不能拿它们处理该备用入口的数据。备用入口的数据需由已有去畸变图像流采集并提供契约清单。

## 接回手眼项目

标定输出是独立相机文件，不等同于本项目完整 measurement contract。先独立检查，再同步更新原始相机配置、校正流、1280×992测量尺度对应的K/P和配置哈希，重新生成球心观测与手眼模型。不能仅将文件重命名覆盖旧JSON，也不能把2560×1984的P直接用于1280×992像素。新模型需重新通过头姿/空间/会话验收。

完整重采清单与顺序见 [RECAPTURE_PLAN.md](RECAPTURE_PLAN.md)。

## 已执行的软件验证（2026-10-27）

本地环境 Python 3.12、NumPy 2.5.4、OpenCV 4.14.0。依赖限制 OpenCV <5；已适配4.x双目标定额外返回逐视图外参的接口。当前项目根目录虚拟环境已安装依赖。

```bash
# 项目根目录执行
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s stereo_calibration -p 'test_*.py' -v
```

当前5项离线测试通过：图像角点检测；合成24姿双目标定/独立评估；CSI配置检查；锁定/时差/传感器顺序与断点编号检查；自动采集稳定/去重门控。整个目录复制到临时目录、清除PYTHONPATH，在项目之外运行启动器帮助及5项测试全部通过。此前项目原有20项测试也全部通过。

这些验证不包含实机CSI采集、真实相机新标定或1 mm精度验收。没有启动机器人或相机，没有替换现有运行模型。

当前棋盘格链和源项目实时链的逐项区别见 [PIPELINE_COMPARISON.md](PIPELINE_COMPARISON.md)。

## 重标定后的实时数据采集

将本目录复制到Orin（或至少复制 `capture_realtime.py` 与 `stereo_pairing.py`），在安装了JetPack GStreamer/Argus、Python gi、VPI CUDA、NumPy和OpenCV的系统Python环境运行。VPI来自JetPack，不由桌面pip依赖替代。先结束占用CSI的棋盘格服务，再启动实时采集。

```bash
python3 capture_realtime.py --calibration output/stereo_fisheye.json \
  --output data/rectified_session_001 --sensors 0 1 --seconds 60 --preview
```

这条链按参考项目 `pro_sender.py` 的共同时钟、双重10ms配帧和VPI/CUDA等距鱼眼映射实现：传感器2560×1984/30fps → nvvidconv缩放1280×992 → 严格配帧 → AE/AWB锁定后VPI CUDA线性插值校正。K和P同步按0.5缩放。保存每对1280×992未校正图和校正图、PTS/共同源时间/到达时间、相机参数哈希、清单和丢帧统计。这里未校正图已经过0.5缩放，不可冒充全分辨率棋盘格原图。

保存使用无损PNG，磁盘/传输速度会限制有效采样率；积压时保留最新合格配对并计入丢弃统计，不承诺30对/秒落盘。不会用异步预览帧作为数据。`frames.jsonl`记录实际采集时间而非文件写入时刻。

当前入口负责图像与时间证据，**不采机械臂/头角反馈，也不自动生成手眼训练JSONL**。手眼采集需将同一时钟映射下的实际反馈与pair_id关联，按RECAPTURE_PLAN保存；不能把未对时的反馈直接用于运动中验收。没有曝光硬同步证据时 `exposure_time_valid=false`。

2026-10-27本次验证：8项离线测试通过，覆盖源/到达双门限、单次消费、积压计数、半分辨率映射和既有标定/采集契约。未连接Orin，VPI/CUDA执行、曝光属性和实际采样率需现场验证。
