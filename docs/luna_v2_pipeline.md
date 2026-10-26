# 旧版 Luna V2.0：双目、双臂动态手眼标定

本入口仅适用于旧版V2.0，头链为旧版Yaw→Pitch混合链，计算Rz(yaw) Ry(-pitch)。新版Pitch→Yaw不在本次实现范围。历史无版本字段的记录/模型按旧版兼容；新模型显式记录`robot_version=luna_v2_0`，标注新版的输入被拒绝。无版本字段的文件仍须人工确认机型来源。

## 运动链

- 双臂：`T_S_F(arm,q) = T_S_M(arm) @ MDH_arm(q)`，左右各7关节，包含固定法兰段。`T_S_M`读取`configs/robot/torso_mounts.json`完整矩阵。
- 头部：`Trans(MJCF名义中心) @ Rz(yaw) @ Ry(-pitch)`，角度顺序`[yaw,pitch]`；不使用控制器头MDH、D=640mm或原MJCF的Pitch→Yaw链。控制器配置只为双臂MDH、原始溯源与反馈限位提供依据。
- 双目：原始/校正像素域保持既有契约，C为校正左眼系；左右相对外参固定。
- 标定：当前默认`P_S = F_fixed @ K(q_head) @ X @ P_C`，只拟合X的6参数。F_fixed使用head_mount_v2.json的固定名义安装约定，不再拟合A；H2预留接口不增加自由度。

旧12参数结果只作历史对照；原模型不覆盖。固定几何、准确公式和最新误差见[六参数说明](six_parameter_calibration.md)。目前名义固定几何尚不能解释现有实机数据，模型未批准。推理不需要机械臂角度。

## 双臂FK

项目根目录运行（全部离线，无SDK或硬件访问）：

```bash
.venv/bin/python -m dynamic_handeye fk --left 0 0 0 0 0 0 0 --right 0 0 0 0 0 0 0
```

返回左右法兰`S_MODEL`位姿，长度m、输入角度deg。安装矩阵为名义CAD对齐，不是独立测量。

## 标定数据导入与拟合

```bash
.venv/bin/python -m dynamic_handeye import --capture datasets/raw/ball_point_head_20261018_v1.jsonl --output /tmp/luna_v2_dataset.json
.venv/bin/python -m dynamic_handeye fit --dataset /tmp/luna_v2_dataset.json --output reports/my_v2_fit
```

左右臂日志使用相同原始契约，`arm_side`明确为left/right，`q7_deg`是对应侧实际反馈。导入左臂记录时必须额外传`--left-ball-offset X Y Z`，为该左法兰上**实测**球心偏置（m），不得默认复用右臂。相机/MDH哈希、质量门槛、曝光及停稳检查保持不变。

左右参考球心都经躯干转换到右臂MDH父系参与共同拟合。导入结果保留`arm_side`、偏置、`target_s_m`与统一参考`target_m`。分组包含臂侧，避免两侧同编号球位冲突。报告包含实际参与拟合臂侧及各臂训练误差；训练误差不代表左/右臂独立验收。

现有硬件自动采集入口保持原右臂流程，本次新增的是左臂日志导入和联合标定支持，未增加未经验证的左臂自动运动。

## 推理与验证

现有`predict`静止入口和`runtime.predict_moving`异步曝光射线入口均输出S_MODEL。正式推理保留模型批准、覆盖范围、停稳/时间同步门禁；`--diagnostic`仅离线诊断。原始模型文件不覆盖。

旧版实机方向、零位仍须按反馈核验，不能把不同机型的符号约定直接写入原控制器MDH。连续运动精度须单独验收；历史采集仅3个停稳头姿、仅右臂，无法据此宣称双臂或运动中精度合格。
