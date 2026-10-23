# 可迁移双目鱼眼棋盘格标定工具

本工具从双目**原始鱼眼图像**标定以下参数：左右相机内参 `K`、鱼眼畸变 `D`、右相机相对左相机的 `R/T`、极线校正 `R1/R2/P1/P2/Q`、基线长度和质量指标。棋盘格尺寸、ROS2话题、采集时长和最少图像数量均可修改。

## 1. 棋盘格参数

参数中的 `--corners COLS ROWS` 是**内角点数量**，不是黑白方格数量。例如：

| 实际棋盘格 | 命令参数 |
|---|---|
| 12×9 方格、35 mm | `--corners 11 8 --square-mm 35` |
| 9×6 内角点、20 mm | `--corners 9 6 --square-mm 20` |

必须用卡尺测量相邻角点中心距离。打印缩放误差会按相同比例进入基线和三维距离。

## 2. 环境

标定电脑：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Orin采集端需要已有ROS2、`rclpy`、`sensor_msgs`、OpenCV和左右原始NV12话题。默认话题：

```text
/camera/left_eye/image_nv12
/camera/right_eye/image_nv12
```

先启动相机ROS2/Apex节点，确认两个话题持续发布且没有其他程序独占CSI相机。

## 3. 一键采集并下载

编辑环境变量后运行：

```bash
export ORIN_IP=6.6.7.100
export ORIN_USER=nvidia
export LOCAL_IFACE=eno0
export BOARD_COLS=11
export BOARD_ROWS=8
export SQUARE_MM=35
export CAPTURE_SECONDS=1800
./capture_and_download.sh
```

SSH会询问密码。程序在Orin显示左右原始灰度画面；左右眼都识别完整棋盘格并保持稳定后自动保存。移动棋盘格到新位置继续采集，按 `q` 或终端 `Ctrl+C` 结束。

建议采集30–50对：覆盖画面中央、四角和边缘；包含近、中、远距离；绕三个方向倾斜；避免所有照片都正对相机。棋盘格必须平整、完整可见、没有运动模糊和强反光。

也可直接在Orin运行，并修改话题：

```bash
python3 capture_orin.py --output raw_images \
  --corners 11 8 --square-mm 35 \
  --left-topic /camera/left_eye/image_nv12 \
  --right-topic /camera/right_eye/image_nv12 \
  --preview
```

采集目录包含成对的 `0001_left.png`、`0001_right.png`、时间信息JSON和 `capture_manifest.json`。不要手工交换左右图文件名。

## 4. 标定

```bash
python3 calibrate.py \
  --images data/raw_chessboard_时间戳 \
  --output output/stereo_fisheye.json \
  --corners 11 8 \
  --square-mm 35 \
  --min-pairs 20 \
  --balance 0
```

`--balance 0`裁剪较多但有效区域更规整；`1`保留更多鱼眼视野，也会增加黑边。运行时CLI棋盘格参数必须与采集清单一致，防止误用尺寸。

输出的关键质量指标：

- `stereo_rms_px`：双目标定RMS，越小越好。
- `rectified_vertical_p95_px`：校正后95%角点的左右纵坐标差。
- `baseline_m`：双目基线，应与实物大致一致。
- `valid_pairs`：左右眼都完整识别的图像对数量。

本项目经验验收线为：`stereo_rms_px < 1 px`、`rectified_vertical_p95_px < 1 px`；精密任务建议尽量低于0.3 px。还必须检查基线方向、数量级和独立图像上的三维误差。

## 5. 生成校正预览

```bash
python3 rectify_preview.py \
  --calibration output/stereo_fisheye.json \
  --images data/raw_chessboard_时间戳 \
  --output output/preview --count 10
```

预览图包含水平参考线。同一棋盘格角点在左右图中应落在近似相同的水平线上；明显上下错位表示标定或左右顺序有问题。

## 6. 迁移到另一台设备

复制整个目录并安装 `requirements.txt` 即可运行标定。采集脚本中的ROS2环境路径属于当前Orin软件栈；其他设备若安装路径不同，只需修改 `capture_and_download.sh` 中三条 `source` 命令。相机、镜头焦距、安装位置、分辨率或裁剪改变后必须重新标定，不能沿用本设备参数。

## 7. 常见错误

- 一直不自动保存：检查传入的是内角点数，棋盘格是否完整出现在两眼中。
- 有效图像不足：增加角度和位置变化，避免重复姿态。
- RMS很高：删除模糊、反光、弯曲或角点误检图像后重新标定。
- 基线尺寸错误：核对 `--square-mm` 和实际打印尺寸。
- 左右校正后上下错位：核对左右话题和文件名是否交换。
- `manifest ... does not match`：采集与拟合使用了不同棋盘格参数；新建目录重新采集。

原始鱼眼模型只适用于相同分辨率和裁剪方式的图像。缩放输出时必须按比例缩放内参及投影矩阵，不能直接混用像素坐标。
