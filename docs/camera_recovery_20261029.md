# 本机Luna V2.0相机恢复记录（2026-10-29）

本轮Orin为6.6.7.100，machine-id为5dbfb12414a3456d9014d88183e338b1。用户明确要求重启Orin后，boot-id由4e60dd3d-326b-438b-bfd6-8773a27d281a变为9f8f3b68-1684-4de7-8304-64aba05244e6，内核仍为5.15.148-rt-tegra。未修改内核、设备树或相机标定参数。

## 故障与恢复

原正式发送器已退出；Argus日志显示关键采集超时/无效状态，后续启动出现dmabuf/NvBufSurface错误，V4L2探测无图像。重启前已完成一次GMSL初始化，但仍无法取流；最终通过用户授权的Orin重启，随后完整GMSL初始化、Argus启动和正式推流恢复。

厂家init_demo3会产生约4MB寄存器调试输出。通过SSH执行时，先等退出再读stdout可能被接收窗口堵住，导致初始化无法完成。应在Orin直接重定向初始化日志，然后读取短摘要，不要只凭初始化进程存在或部分日志判断成功。

重启后的实际成功顺序：

```bash
# Orin；sudo按现场认证，不在文件中保存口令
sudo systemctl stop nvargus-daemon
cd /home/nvidia/cam_geac
sudo timeout 75s stdbuf -oL -eL ./demo/init_demo 3 \
  > /home/nvidia/luna_camera_init_current.log 2>&1
# 要求退出0，日志出现camera init ... done
tail -10 /home/nvidia/luna_camera_init_current.log
sudo systemctl start nvargus-daemon
nohup python3 /home/nvidia/luna_measurement/head_2dof_capt/orin_rectified_stream/sender.py \
  --contract /home/nvidia/luna_measurement/head_2dof_capt/configs/yellow_measurement_v1.json \
  --host 6.6.7.101 > /home/nvidia/luna_rectified_sender.log 2>&1 < /dev/null &
```

启动前检查并停止重复的相机占用进程。上面是已验证顺序；服务无法释放时不要一边初始化一边反复创建Argus会话。

## 恢复参数与验证

- 原始标定SHA256：d0a550594981b670000e48bcb7be840e22de6962cdae28da51bf103fe36b271b。
- CSI0左/CSI1右，原始2560×1984，正式测量1280×992、预览640×496。
- 双眼固定曝光5000000ns，模拟/数字增益1，wbmode0，TNR/EE模式1、强度0.5，saturation1。双眼属性读回一致。
- 正式记录PASS、亮度稳定、锁定曝光/白平衡；恢复后的新帧号约1809，主机测量年龄约29ms（不是已验证曝光延迟）。
- 独立50帧窗口通过；视差P95抖动0.05361px，窗口三维P95抖动0.46755mm。只说明此静止位置重复性，不证明绝对精度。
- 板端测量处理约26–29fps，本机预览约11fps；左右实景快照已查看，记录于被忽略的acquisition/runtime/camera_recovered_20261029/。
- 本机预览PID876972；Orin推流保持运行。具体PID是本轮记录，后续不能直接复用为停止目标，须先核对进程命令。

右臂共享轨迹示教继续运行于头姿[0,0]，未对机器人控制器下发重启。恢复相机期间没有停止或改变机械臂模式。
