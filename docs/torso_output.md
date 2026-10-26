# 躯干坐标输出（当前6参数）

当前fit入口只拟合相机安装6参数，输出`P_S=F_fixed K(q) X P_C`，S为S_MODEL，见[六参数说明](six_parameter_calibration.md)。旧v1模型按以下历史公式兼容推理，文件不覆盖。

## 历史十二参数输出说明

# 躯干坐标输出

推理的stationary、moving及CLI diagnostic均输出`frame_id=S_MODEL`，即`Gento Luna_Leg_J6_Link`模型原点；不是上移182mm的CAD接口参考。

输出公式：`P_S = T_S_M @ A @ K(q_yaw,q_pitch) @ X @ P_C`。

`T_S_M`从`configs/robot/torso_mounts.json`的`right_arm_mdh_parent`关系读取，检查模型SHA256。A/X仍是现有诊断标定的两端变换。新拟合模型标明`frame_id=S_MODEL`与`calibration_reference_frame=right_arm_mdh_parent`；旧模型经相同输出转换兼容，不改写旧模型文件。模型矩阵字段`mdh_from_head_root_T`保留原右臂参考语义。评价误差仍在原右臂系计算，其欧氏距离不受刚体输出转换影响。

H0/H1是yaw/pitch，H2为预留接口；程序K只含两段MDH，无H2接口变换。模型头部安装描述的是MJCF关节零位，而其轴顺序与控制器不同，不能直接将该矩阵与控制器K连乘。目前输出用固定臂安装关系转换，并未宣称已得到独立确认的控制器头根安装矩阵。

本次仅完成输出基准转换，不将十二参数模型冒充六参数H1T_C标定。相机安装及控制器头根独立验证、未见头姿/新会话验收仍未完成；正式推理保留原验收门禁。
