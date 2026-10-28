# Luna 双目推流排查（2026-10-27）

双目实时预览已恢复并打开。已查看真实场景快照，验证两路帧计数持续增加；当前不启用黄球测量，不用于正式采集。

## 已验证

- 工作站只有 Integrated Camera（video0–3），目标 GMSL 相机连接在 Orin 6.6.7.100。
- Orin video0–7 为 GMSL 节点；video8–13 属于 USB RealSense，不能套用 Luna 鱼眼标定参数。
- 厂家程序映射逻辑 port0→video0、port1→video1。配置为 SHW5G RAW10；左右目物理身份尚须用实际图像核验。
- 启动配置原 FDT 文件名错误，多了一个 kernel_。用户确认机器人安全静止后，已备份并修正路径、重启。当前运行设备树已为 2560×1984 GBRG RAW10。
- 备份：Orin `/boot/extlinux/extlinux.conf.luna-camera-before-20261027`。

## 按 head_eye_pro 方式试运行

复用 pro_sender.py 与 stereo_pairing.py，临时部署在 Orin `/home/nvidia/luna_pro_style_preview`。只将标定身份校验适配为 Luna 的 `head_left:csi://0` / `head_right:csi://1`，使用本项目原始相机参数，未使用 Marvin 标定数据。

数据链：同一 GStreamer 管线/时钟采集左右目 → 源时钟和到达时差均≤10ms的一次性配对 → Orin VPI/CUDA 校正 → H.264 RTP/UDP 5120、5122 → 本机双目窗口；5124 单独发送元数据。预览640×496，校正处理1280×992。时钟配对不等于已验证曝光硬同步。

本次不启用黄球测量，质量保持 REJECT。临时预览窗口仅用于排查，不写入正式标定采集结果。此临时试运行尚未替换项目正式采集入口。

## 故障与修复记录

修复前，原 Luna 推流和 Pro 式推流均在底层出现 `NvBufSurfaceFromFd Failed`，收到0帧；直接 V4L2 读取也超时。

原启动 Image 为2026-03-26构建，nv_imx185.ko 也与厂家配套包不同。用户另行明确授权后，已验证厂家源文件MD5，完整备份原Image、initrd、启动配置及全部内核模块，切换为2026-04-28厂家SHW5G配套Image/initrd/modules，执行depmod并重启。SSH恢复，uname确认新内核生效。原启动配置中已修正的FDT路径保留。

切换后，同一Pro式数据流正常输出：Orin源帧约30FPS，预览约12FPS；检查时两路接收计数291/288且继续增加，帧龄约15/12ms。实景快照显示实验室和机械臂，非黑帧、绿帧或旧截图。测量仍为REJECT，因为本次未启用黄球检测。

备份和回退脚本（Orin）：`/home/nvidia/luna_camera_driver_backup_20261027/`。需要回退时以sudo执行该目录的`rollback.sh`后另行重启；脚本恢复本次切换前的内核、initrd、模块及启动配置。未修改机器人SDK、MDH、手眼模型或相机标定参数。

日志：Orin `/tmp/luna_pro_style.log`；本机 `/tmp/luna_pro_style_viewer.log`。

本机窗口：`Luna Pro-style STEREO - LEFT | RIGHT - G grid - Q close`。按G显示极线参考线，Q关闭预览。快照及状态：`/tmp/luna_pro_style/runtime/rectified_pro/`。

限制：尚未通过遮挡实验独立确认物理左右目，当前沿用原Luna标定身份顺序CSI0/CSI1；未验证曝光硬同步、手眼测量精度、其他板端服务或正式采集流程。
