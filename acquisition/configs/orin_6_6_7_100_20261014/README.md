# Luna V2.0 当前双目配置

目录名为兼容既有入口保留；其中两个相机JSON已更新为2026-10-28本机标定，不再使用2026-10-14参数。

来自 `stereo_calibration/data/raw_chessboard_20261028_101409`：60对、57对有效，11×8内角点，35mm。左右CSI0/1，原图2560×1984，测量1280×992。原图训练双目RMS0.204651px，纵向P95 0.309226px，基线60.476721mm；尚无独立验证。

正式入口 `acquisition/orin_rectified_stream/start.py`，直接Argus、共同时钟配帧、VPI校正、黄球检测。旧ROS YAML标定文件已删除，不再使用旧ROS启动脚本。
