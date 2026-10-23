# 人机交互球心偏移标定

此流程用于**新 Luna 右臂法兰上的 40 mm 黄球**，使用统一 Orin 服务按最新棋盘格参数校正后的每眼 1280×992 像素。它同时估计球心相对法兰的 `p_F`，以及左相机相对右臂 MDH 链原点的刚体变换。没有经过核验的右臂 MDH 原点到**整机基座**的固定变换，因此输出不称作整机基座坐标。

先用当前统一 Orin 服务打开校正画面：

```bash
cd /home/lijingzhi/Project/Learned_IK/head_2dof_capt
python3 orin_rectified_stream/start.py --restart
```

示教与回放分开执行，均使用当前 1280×992 契约和最新棋盘格校正：

```bash
bash scripts/teach_replay_ball.sh teach
bash scripts/teach_replay_ball.sh replay data/ball_taught_实际文件.json
```

示教完成输入 `q` 并回车或按 `Ctrl+C`；自动回放前检查环境，再按提示输入大写 `RUN`。回放在每个停稳点融合 50 个唯一 PASS 帧，并要求视差抖动 P95 不超过 0.1 px。

先按 `camera_calibration/README.md` 完成新设备棋盘格标定，得到双目 JSON。该 JSON 至少含 20 对有效图、双目 RMS 与极线 p95 均不超过 3 px。确认本机能看到双目画面，黄色检测圈都落在**粘于右臂法兰的同一颗球**上。

```bash
cd /home/lijingzhi/Project/Learned_IK/head_2dof_capt

# 1. 人工连续拖动右臂示教安全轨迹；程序按关节变化自动取点，输入 q 结束。
bash scripts/teach_replay_ball.sh teach --head-target 0 -25

# 2. 把右臂放回第一个示教姿态附近，然后低速逐点自动回放并采集。
bash scripts/teach_replay_ball.sh replay data/ball_taught_实际文件.json

# 3. 离线联合拟合，输出 p_F、相机刚体变换及未参与拟合姿态的误差。
python3 -m interactive_ball_calibration.fit \
  --stereo configs/orin_6_6_7_100_20261014/luna_rectified_stereo.json \
  --capture data/ball_auto_实际文件.jsonl \
  --output data/ball_joint_fit_$(date +%Y%m%d_%H%M%S).json

# 4. 固定头姿下，用FK球心标签拟合左右投影矩阵并做分块留出评估。
python3 -m interactive_ball_calibration.fit_projection \
  --capture data/ball_auto_短轨迹.jsonl data/ball_auto_长轨迹.jsonl \
  --ball-model data/ball_joint_fit_长轨迹.json \
  --output data/fixed_head_projection.json
```

示教至少 **18 个不同姿态**。进入拖动模式后自动记录起点；随后缓慢连续拖动，程序在关节变化达到 3° 时自动取点，单次采样间隔不得超过 5°。完成长轨迹后按 `Ctrl+C`，程序捕获中断、校验并保存最终JSON，再将右臂和头部恢复为IDLE。每个点同步写入 `.partial` 临时文件，窗口意外关闭时保留恢复依据。同时让法兰绕至少两个不同方向明显改变朝向，并覆盖不同球心位置。示教文件只记录安全关节轨迹，双目像素在自动回放停稳后采集；拖动期间仍应观察预览，确保黄球始终在左右眼中。每次手动拖动的真实路径也必须在机械臂和环境中可安全回放：程序只检查关节插值邻近性和在线限位，无法证明碰撞自由。若结束时机械臂位于最后一点，回放程序会先沿示教轨迹以 5% 速度反向返回起点，再正向回放采集。头部由 SDK 缓慢移动并保持在 `[0,-25]`；身体保持不动。

自动回放前程序要求输入 `RUN`。若机械臂不在逆回放首点（示教轨迹末点），程序先以单关节不超过 5° 的插值段移动到该点，这一段不采集；随后从末点开始采集，并以 SDK 5% 速度沿示教轨迹逆序返回起点，不再进行第二次正向回放。视觉观测失败会写为 `capture_valid=false` 并继续；异常或 Ctrl-C 时尝试将右臂和头部切回 IDLE。每个有效点等待右臂停稳，核验双目时效、黄球和三角化，再取 50 个不同 PASS 帧的四维像素中位数，并要求视差抖动 P95 ≤0.1 px。输出已有文件不会被覆盖。采样时间基于 Orin 出队和本机接收估计，不是真实曝光硬件同步时间；停稳采集减轻这一误差。

联合模型为

\[
{}^{C_L}p_i = {}^{C_L}R_M\bigl({}^{M}t_{F,i}+{}^{M}R_{F,i}p_F\bigr)+{}^{C_L}t_M,
\]

其中 `M` 是 `luna_new.ini:R.A1` 的 MDH 父坐标系，`C_L` 是左相机坐标系。拟合在不同示教点划分训练与留出，报告球心偏移、三维残差和双目重投影误差。球面图像区域的中心可能与真实球心投影有偏差，因此需用已知尺寸或独立测量核验 `p_F`；此模型不会自动标记为物理精度已认证。

2026-10-18 已连接 Orin 和机器人完成静态像素验收、首批 79 点以及第二批 213 点拖动示教和低速反向回放，右臂、头部均恢复 IDLE。第二批有 209 点通过 50 帧融合，视差抖动 P95 为 0.067 px。冻结首批相机变换和 `pF=[-3.30,4.45,99.66] mm` 后在第二批做纯外推，XYZ P95 为 2.788 mm、模型像素预测重投影 P95 为 1.888 px，故该有效偏移不具备亚毫米泛化能力。第二批单独重拟合的 pF 漂移 1.415 mm且自身留出 XYZ P95 仍为 1.684 mm，下一步需检查球偏移独立标定与右臂 MDH/FK 系统残差。报告见 `reports/effective_pf_generalization_20261018_094544.json`。双目真实曝光同步和整机基座外部计量仍未验证。

## 新正式标定顺序：先真实球心，再判断是否需要修 FK

当前采集器已经改为：每个 PASS 帧先用原双目中心提供深度，再结合 40 mm 球直径和每眼内参，把球轮廓椭圆中心迭代修正为真实球心投影，再用修正后的双目像素三角化。50 帧融合后写入 `sphere_center_method=sphere_center_projection_stereo_v1`、`sphere_center_left_camera_m`、修正后双目像素、球心 3D 抖动和修正视差抖动。历史 79/209 点虽然只保存了四维亚像素中心，但这套修正不依赖历史椭圆长短轴，因此可以通过 `upgrade_sphere_centers.py` 无损升级。

新数据完成后先运行：

```bash
python3 -m interactive_ball_calibration.fit_sphere_registration \
  --capture data/ball_auto_new_A.jsonl data/ball_auto_new_B.jsonl \
  --ball-model data/ball_fixed_pf_rectified_1280x992_20261018.json \
  --output reports/fixed_head_sphere_registration.json
```

只有固定 `p_F` + 单一刚体相机注册在独立留出/跨轨迹仍达不到 `<1 mm`，才运行：

```bash
python3 -m interactive_ball_calibration.fit_kinematic_correction \
  --capture data/ball_auto_new_A.jsonl data/ball_auto_new_B.jsonl \
  --ball-model data/ball_fixed_pf_rectified_1280x992_20261018.json \
  --output reports/fixed_head_kinematic_correction.json
```

第二级只拟合受限关节零位修正和固定相机刚体变换，不覆盖原 MDH；有多条轨迹时必须 leave-one-capture-out 每条都 `<1 mm` 才通过。旧 ellipse-centre 数据不能用于这一级正式修正。


## 2026-10-18 更新：历史升级、头部回差与最终 9D 路线

历史 79/209 点可以直接运行 `interactive_ball_calibration.upgrade_sphere_centers`：该工具只使用已有四维亚像素中心、40 mm 球直径和双目标定，通过迭代修正球心真实投影后重新三角化，不需要历史文件保存椭圆长短轴，也不会覆盖原文件。

两批数据升级后，固定 `pF=[0,0,101] mm` 的纯刚体注册 held-out P95 为 5.786 mm；再拟合受约束 q2..q7 零位后降至 1.200 mm，但 leave-one-capture-out 仍为 1.336/2.777 mm，因此不能把零位修正直接写回 MDH，也不能宣称固定头姿 `<1 mm` 已完成。

采集器同时增加头部重复定位保护：若当前反馈已在 `[0,-25]` 目标 ±0.05° 内，不再重复驱动头部；若必须移动，先到目标固定一侧约 1° 的预定位点，再从相同方向逼近目标。该策略用于降低不同会话之间的相机真实姿态漂移。

最终数值一致性使用 `fit_pose9d_consistency.py`。真实采集需要新增 `visual_xyz_camera_m` 和 `visual_rotation6d_camera`。q7 只在标定阶段生成 FK 标签，运行时模型完全由视觉 9D 驱动。只有跨轨迹以及最终第三条 untouched 轨迹的 XYZ P95 都 `<1 mm` 才通过。
