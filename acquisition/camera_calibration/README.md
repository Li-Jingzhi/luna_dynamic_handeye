# 新 SH5 双目棋盘格标定

当前棋盘格按图片标注设为 **9×6 个内角点、相邻角点间距 20 mm**，即默认 `--corners 9 6 --square-mm 20`。图片提示打印可能缩放；用卡尺核对实际方格边长，打印缩放会直接造成三维尺度误差。

```bash
cd /home/lijingzhi/Project/Learned_IK/head_2dof_capt
bash scripts/capture_chessboard.sh
# 左右眼同时识别到棋盘格并稳定 0.5 秒后自动保存；移动到不同位置/倾角，至少 20 对，按 q 结束。
python3 -m camera_calibration.fit --images data/chessboard_时间戳 \
  --output data/sh5_stereo_时间戳.json
```

输入是新 SH5 的 `image_h264_undistorted` 左右单眼原生 2560×1984 画面，不用旧相机内参。由于图像已去畸变，拟合的是**该输出像素域**的有效针孔内参 K、左右相对 R/T、F、E 和可选二次极线校正矩阵。原始鱼眼的内参/畸变不能从这些画面恢复。棋盘格必须在保存左右图的过程中保持静止；当前 DDS 到达时间并非曝光时间。标定报告含 RMS 和极线 p95，仍需另采独立棋盘格位置评估三维尺寸精度。

采集目录会自动生成 `capture_manifest.json`；拟合程序核验其中的左右传感器顺序、话题和去畸变像素域。旧的原始 NV12 棋盘格目录不能直接作为本标定输入。电脑有线口未连接时，采集脚本自动选择 Wi-Fi `wlp0s20f3` ↔ Orin `wlan0`，可用 `LUNA_TRANSPORT=ethernet` 明确指定有线。
