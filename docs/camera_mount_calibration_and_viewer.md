# H2←C安装标定与MuJoCo坐标视图

## 标定的是哪个矩阵

`X = ^H2 T_C`把校正左眼三维点变换到控制器头部末端系。若躯干到臂MDH父系`^S T_M`和躯干到控制器头根`^S T_H0`已确认，只拟合X六个自由度。

对每个停稳观测：

1. 保留相机参数/MDH哈希、q7、实际头角qh、球面投影修正后的相机球心P_C。
2. 用控制器MDH和固定已测pF生成球心`P_M = t_F(q7)+R_F(q7)pF`。
3. 将参考球心转到当时头部末端系：
   `Y_H2 = inverse(^S T_H0 · K(qh)) · ^S T_M · P_M`。
4. 求解 `Y_H2 ≈ X · P_C`。本工程已有`dynamic_handeye.registration.rigid_fit(P_C, Y_H2)`，返回R,t；组装成X。非共线的空间点即可约束刚体旋转和平移；实机建议用分散空间和多法兰构型、多头姿数据检验一致性。
5. 冻结X，留出整球位、整头姿和新会话，分别报告XYZ误差、重投影和同球跨头姿漂移。需要时再做鲁棒或像素域细化。

实际流程优先先12个分散球位×9头姿、每组合50对唯一合格静止双目；重复返回参考头姿。现有控制器几何保持不变。

左右臂各自用其`^S T_M`和pF提供相同躯干系标签，理论上共用一个X。当前只有右臂数据和右臂pF来源，不能假装已有左臂验收。

原始左眼到校正左眼的R1已包含在X所连接的坐标定义中；不要重复乘R1。若需要原始光学系安装矩阵，则`^H2 T_Craw = X · diag(R1,1)`。

当前工程的两端12参数诊断模型仍保留；其H0/H2与相机位置默认隐藏，可按3/4显示。**不是已经执行了上述固定几何的单X标定**。

## 打开MuJoCo

项目根目录：

```bash
./view_frames.sh
```

可选环境已位于`.venv`；重建时可用Python虚拟环境安装`requirements-viewer.txt`。只在本地显示，不导入SDK、不建立UDP/SSH连接，也不执行物理积分。

- 鼠标左键拖动旋转，滚轮缩放，鼠标右键平移。
- 按1切换模型原点S_MODEL；2切换双臂父系；3切换诊断拟合H0/H2；4切换诊断双目；6切换CAD接口参考与模型头关节系。3/4默认隐藏。
- 每套轴均红=X、绿=Y、蓝=Z。深色引线连接坐标原点与文字标签；文字位置不是坐标原点，只有三轴汇聚处的深色球是原点。
- S_MODEL来自模型第6躯干关节后的body原点，不代表已确认的实物接口。S_INTERFACE仅按胸部STL最低平面上移182mm建立显示参考。
- M_L/M_R标NOMINAL：假设零位首关节轴与CAD一致，由模型首关节安装乘控制器首段MDH变换逆求得；已扣除158.6mm。未独立核验为实机安装。
- H0_CTRL/H2_CTRL和C_L/C_R标FIT：使用当前未批准12参数诊断模型，不能作为精密测量真值。黄色线段为真实标定基线长度59.538mm，但整个相机组的位置/朝向仍是诊断拟合结果。
- 网格模型保持MJCF零位，头部轴叠加默认使用控制器yaw=0/pitch=-25°；不驱动原MJCF的不同头链。

默认浅色背景；轴线连续连到XYZ文字，字母加所属坐标系前缀。H0/H2标签显式注明FIT ONLY。

最新截图：`reports/frame_relationships_cn_20261023.png`；数值：`reports/frame_relationships_20261023.json`。固定偏置出处与父子关系见[来源复核](frame_offset_sources_20261023.md)。

```bash
./view_frames.sh --check
./view_frames.sh --snapshot reports/my_frames.png --snapshot-only
.venv/bin/python -m unittest discover -s tests -q
```

MuJoCo查看器与自定义坐标箭头采用官方passive viewer/user_scn接口：https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
