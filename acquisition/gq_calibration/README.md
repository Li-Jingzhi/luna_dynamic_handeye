# 头部两自由度 G(q) 标定

```bash
cd /home/lijingzhi/Project/Learned_IK/head_2dof_capt
bash scripts/capture_gq.sh
```

脚本先从**在线控制器**读取 yaw/pitch 限位并检查目标，随后依次到 `[0,-25]` 和 yaw=±20/±40/±60、pitch=-25/-10/0 的全部组合，共 19 个头姿。每次头到位后按 Enter，右臂进入 DragJoint；本机双目画面会框出检测到的 40 mm 黄球。持续拖动右臂时，每个新鲜双目帧都会记录对应q7，不需要停稳；按 `q` 结束该头姿并自动移动到下一头姿。Ctrl-C提前退出时程序尝试将右臂和头部切回IDLE。记录包含实际完整头关节、右臂q7、身体角度、2560×1984四维像素和采样时间。头角偏离目标超过0.3°会中止。若参考姿态已经采完，可用第二个参数从目标2开始：`bash scripts/capture_gq.sh data/gq.jsonl 2`。

现有 `capt_head.calibrate.fit_reference` 和 `fit_head` 已实现固定头姿投影矩阵及两轴刚体运动 G(q) 的拟合。本目录 `fit.py` 复用它们：

```bash
python3 -m gq_calibration.fit --data data/已标注基座XYZ.jsonl \
  --reference-output data/sh5_q0_model.json --head-output data/sh5_gq_model.json
```

原始采集只有 q7 和像素，**还没有基座三维真值**。拟合输入每条还需 `xyz_base_m`、实测 `yellow_center_flange_m` 及 `fk_frame_independently_verified=true`。必须用这台设备的 `luna_new.ini`、实际身体/肩部基座变换、球心法兰偏移生成并核验 XYZ；不能把 20 mm 假定偏移或旧设备参数当成已测真值。参考头姿至少需要 20 个分散球位、30 条训练记录和 8 条留出记录，多头姿总计至少 40 个分散球位、60 条训练记录和 12 条留出记录。脚本将同头姿、关节角相距不超过 1° 的相邻视频帧归为同组，再按组划分训练/验证/测试。最终模型仍是研究用途，精度需用独立测量验证。

本次未连接机器人，头部指令、拖动操作、双目实时同步需要上机核验。移动前清空头部和机械臂周围空间，并确认设备急停可用。
