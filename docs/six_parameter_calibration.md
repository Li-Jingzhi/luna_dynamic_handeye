# 旧Luna V2.0混合六参数手眼标定

按《Luna_头部混合运动学与动态手眼标定方案_20261026》式(1)，当前默认fit仅优化X=H1末端←校正左眼的旋转3参数与平移3参数。**新头链完全不使用控制器D=640mm，不使用其逆变换，不进行640mm抵消。** 控制器原文件保留作溯源，双臂仍使用其MDH。

## 固定头部几何

`configs/robot/head_mount_v2.json`定义旧V2.0 yaw→pitch，输入`[head[0],head[1]]=[yaw,pitch]`、角度deg。选定前X、左Y、上Z，配置零位旋转R=I；该轴向及零位是待独立核验的名义假设。

公共中心位置p=[-0.013496,0,0.49898]m从保留MJCF的Head_J1_Link相对Leg_J6_Link的pos读取。只取该安装位置，不使用MJCF的Pitch→Yaw顺序或其首关节四元数。共心名义几何是否适合旧版实机仍需核实。

```
F_fixed = [R, p; 0, 1]
K(q) = Rz(yaw) @ Ry(-pitch)
T_S_H1(q) = F_fixed @ K(q)
T_S_CL(q) = F_fixed @ K(q) @ X
```

R=I时，零位H1与S同方向、原点在p；负yaw向右、负pitch低头，组合角先yaw再pitch。H2为预留接口，不增加第三个自由度。该链名`luna_v2_hybrid`；新fit不能选择旧`controller_mdh`。

## 标定方法

数据target_m内部仍保留右臂MDH父系参考；左右臂球心先通过完整安装矩阵统一。拟合时：

```
P_S_ref = T_S_MR @ P_MR_ref
Y_i = inverse(T_S_H1(q_i)) @ P_S_ref_i
```

用Kabsch/SVD配准P_C到Y初始化X，再在S系中对`T_S_H1 X P_C - P_S_ref`做Huber阻尼最小二乘，XYZ各分量阈值3mm。数值雅可比只有6列，要求秩6并检查收敛梯度；固定几何不参加优化，不读取或冻结历史拟合A。

单头姿非退化点集即可求X，不再使用旧十二参数模型的二维头姿激励前置条件；训练仍要求至少36个有限、空间分散的对应点。跨头姿、球位和新会话验证仍保留，不因能求解就自动批准。

## 数据、模型与运行

新模型schema=`luna_dynamic_head_model_v2`，`chain=luna_v2_hybrid`，`parameter_count=6`；保存固定`torso_from_head_root_T`（此处root指新H_ref）与`head_tip_from_rectified_camera_T`，不保存自由A。安装配置与双臂安装文件SHA256绑定到结果；变更后拒绝混用旧结果并要求重新标定。

静止、异步曝光运动推理及离线诊断共享同一相机位姿函数，输出S_MODEL。旧v1/controller_mdh模型只兼容读取，原文件不覆盖。正式入口仍要求独立验收、覆盖和停稳/同步门禁。修改旧D即使会触发源文件完整性校验，也不会改变新链的运动学计算；测试直接改变内存head_rows的D验证其不参与新链。

```
.venv/bin/python -m dynamic_handeye import --capture datasets/raw/ball_point_head_20261018_v1.jsonl --output /tmp/my_hybrid_points.json
.venv/bin/python -m dynamic_handeye fit --dataset /tmp/my_hybrid_points.json --output reports/my_hybrid_fit
```

## 同一实机数据的结果

最新结果为`reports/luna_v2_hybrid_6dof_20261026/`，使用884条有效右臂观测、311个球位组、3个头姿、单会话。

| 方案 | 训练P95，mm | 轨迹留出P95，mm |
|---|---:|---:|
| 历史十二参数诊断 | 5.250 | 3.388 |
| 被替换的固定旧MDH六参数实验 | 97.114 | 96.542 |
| 当前文档式混合六参数 | 13.598 | 8.086 |

当前训练RMS=8.065mm。中间的97mm结果属于用户纠正前的旧MDH实验，不是当前模型。最新结果也没有通过实机精度验收，不能宣称亚毫米或连续运动合格。

fit_metrics的XYZ分量在S中，分组验证的XYZ分量在右臂MDH父系中，报告分别注明；三维距离指标不受坐标旋转影响。左臂实机样本、跨会话和真实运动同步验收仍缺少。源模型、原控制器/相机参数、原始记录和历史外参均未改写。
