# Orin 双目配置快照（2026-10-14）

来源：`nvidia@6.6.7.100`，Apex 安装版 `gmsl_quadcam`。

| 文件 | 原路径 | SHA-256 |
|---|---|---|
| `sh5.yaml` | `/opt/kernelmind/apex/install/gmsl_quadcam/share/gmsl_quadcam/config/camera_profiles/sh5.yaml` | `736b14441f5e0097841762d2637223acf4d32881c7ee71f11b348e183739aba6` |
| `quad_quickview.yaml` | `/opt/kernelmind/apex/install/gmsl_quadcam/share/gmsl_quadcam/config/quad_quickview.yaml` | `92ff5d3ef08539d66a1595d026e7729981accb6d2c6c692c0b4c2fdbbe420b80` |
| `luna_stereo_undistorted.yaml` | `/tmp/luna_stereo_undistorted.yaml`，本次运行参数 | `279e942c20148acf0cd1902bf100de24757f73f24d912a6d8c48b69615698149` |
| `luna_raw_fisheye_stereo.json` | 2026-10-16 原始鱼眼棋盘格标定，项目固定参数 | `d865ad6cca6d1dce9d9a49c870499c115a3f46a059ccde6d80b98e5454a144ea` |

固定标定参数来自 42 组 `12×9` 方格（`11×8` 内角点、35 mm）原始双目图像，其中 37 组有效。模型为 OpenCV fisheye，双目 RMS `0.118 px`，极线校正垂直误差 P95 `0.189 px`，基线 `59.538 mm`。后续原始图像去畸变、极线校正和三角测量统一使用 `luna_raw_fisheye_stereo.json`；只有相机安装、镜头、分辨率或裁剪发生变化时才允许替换并重新验证。

部署接线：左眼 `csi://0`，右眼 `csi://1`；SH5 传感器输入 `2560×1984`、30 FPS。当前 ROS2/Apex 日志显示去畸变 H.264 链路为 `NV12 2560x1984 -> H264 1280x992`，因此 ROS2 传回的实际图像细节是每眼 `1280×992`。历史脚本再将检测坐标乘 2 写成 `2560×1984` 坐标域，但这不会恢复已在编码前丢失的图像细节。

历史归档中的 `1280×720` 鱼眼模型和传感器 `[1,8]` 已停用，禁止用于本项目。新数据统一使用Apex ROS2原始H.264在本机按`luna_raw_fisheye_stereo.json`校正得到的每眼`1280×992`像素，分别使用各眼左上角坐标系，并标记`pixel_domain=ideal_pinhole`。禁止将坐标乘2或乘4后声称为全分辨率测量。

建议由 Orin 上的 Apex 相机组件保持 CSI 唯一采集，并在 Orin 本机解码上述两个去畸变话题、检测黄球和配对关节反馈；本机预览不参与像素测量。每条记录保留逐眼帧号、相机时间戳及其时间基准、左右帧配对误差、去畸变像素、图像尺寸、相机槽位、实际头角、FK标签和参数文件哈希。当前曝光时刻与编码器的硬件同步尚未验证。旧采集包装已移入历史归档，不能用于`6.6.7.100`。
