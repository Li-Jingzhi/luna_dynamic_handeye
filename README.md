# Gento Luna 动态头姿双目手眼标定

2026-10-30：新增[仅用现有黄球的自动诊断采集](data_collection/diagnostics/README.md)，无需手工摆位或夹具，自动完成光学一致性、视觉保持球心、双向返回与冻结模型验收。默认固定头姿[0,-25]、213个50帧窗口；默认离线检查，显式`--run`接机执行。当前未连接机器人，实机精度尚未验证。

2026-10-29更新：A/B共1661条有效观测、44候选/16折的[算法比较](reports/algorithm_search_20261029/README.md)已完成。当前开发候选“相对头链＋联合仿射/60mm高斯场”最差折P95=7.557mm，仍未达到1mm目标。新建[统一采集目录](data_collection/README.md)，提供拖动示教→检查→多头姿自动回放→50帧融合→摘要与实时进度；默认离线检查，`--run`执行完整流程。

2026-10-28：当前Luna V2.0相机已切换为本机新棋盘格标定（60对/57有效，基线60.476721mm）。旧相机参数已替换，旧参数的原始手眼观测和处理数据已删除；删除清单见 `reports/camera_migration_20261028/deleted_files.json`。保留的历史研究报告与模型只作算法记录，其相机哈希已过期，不能用于新相机预测。新数据必须重新拟合手眼模型。相机尚未完成独立验证。

四组360点实机采集现已完成：[0,-25]、[0,-20]、[10,-20]、[10,-25]，有效点数分别为353、347、292、262。完整索引、参数域和无效记录说明见 [datasets/README.md](datasets/README.md)。正式采集采用Orin共同时钟配帧、VPI校正与球心检测，固定5ms曝光、1×增益、关闭自动白平衡；本机预览显示采集进度。`stereo_calibration/REALTIME_STEREO_OVERVIEW.md`记录的是当日临时本机OpenCV校正预览，与当前正式采集链不同。

基于这四组新相机数据、固定精密测量`pF=[0,0,101]mm`的重新标定与分块验证见[2026-10-29报告](reports/new_data_refit_20261029/README.md)：当前训练内选中“有效链12参数＋50mm/ridge1躯干RBF”，全量拟合P95为4.330mm，整头姿留出4.341–5.229mm，头姿＋空间双留出最差10.732mm，仍为`approved=false`。下一步应先采不同右臂轨迹的独立验证数据，再补负yaw和第三个pitch层。

本目录将 Luna 的真实相机/MDH/采集数据、LunaUdpTool 模型与 Marvin_pro 的通用标定方法集中到一个独立工程。离线拟合只需 NumPy 和 OpenCV，不依赖原项目路径、SciPy、MuJoCo 或机器人SDK。仅显式执行采集的 `--run` 时才调用机器人SDK。

## 当前保留的算法

按2026-10-26确认，保留以下两条路线；主研究方案为**有效运动链＋躯干补偿**，固定名义几何＋条件RBF作为对照。两者均使用旧版Luna V2.0的Yaw→Pitch，输入校正双目四维像素和头角，输出S_MODEL中的目标位置；运行预测不输入机械臂q7或黄球pF。

| 路线 | 已保存的模型 |
|---|---|
| 固定名义几何＋条件RBF | [fixed6_conditional_train_model.json](reports/fk_consistent_vision_20261026/fixed6_conditional_train_model.json) |
| 有效运动链＋躯干补偿 | [learned12_torso_rbf_train_model.json](reports/fk_consistent_vision_20261026/learned12_torso_rbf_train_model.json) |

同一训练侧冻结pF及FK参考、同一分组划分的整头姿留出P95（mm）：

| 未参与训练的头姿 | 固定名义几何＋条件RBF | 有效运动链＋躯干补偿 |
|---|---:|---:|
| −10°，−25° | 20.119 | 2.569 |
| 0°，−20° | 4.503 | 4.810* |
| 0°，−25° | 2.769 | 1.976 |
| +10°，−25° | 19.917 | 2.720 |

*留出唯一−20°组后，训练数据全部pitch=−25°，有效12参数的数值秩为10，俯仰响应不可完整辨识。各行是独立重新拟合的留出折，不是上述单个保存模型的四项独立验收。

实现及四维像素接口见[analyze.py](reports/fk_consistent_vision_20261026/analyze.py)，完整协议见[研究报告](reports/fk_consistent_vision_20261026/README.md)。[25°参数解释与俯仰可辨识性](reports/fk_consistent_vision_20261026/pitch25_analysis.md)、[RA=Ry(q1)约束试验](reports/constrained_root_pitch_20261026/README.md)仅作诊断记录，不改变上述两条保留路线。其余候选归为研究对照，不自动按更低测试分数替换模型。

保存模型均`approved=false`，像素推理须显式`diagnostic=True`；保留算法不等于通过任意头姿或运动中验收，正式运行的审批/质量门槛保持有效。

## 机型与数据来源

| 内容 | 来源 | 用途 |
|---|---|---|
| SH5 双目内参、畸变、R/T、校正矩阵 | `head_2dof_capt` | 当前唯一相机参数来源 |
| 两臂与头部控制器MDH | `head_2dof_capt/configs/luna_new.ini` | 当前Luna FK与头角定义 |
| 实测右臂球偏移 | `head_2dof_capt/data/ball_fixed_pf_rectified_1280x992_20261018.json` | 固定pF=[0,0,101]mm，来源记录为precision_instrument |
| 三头姿实机JSONL、完成清单、示教轨迹 | `head_2dof_capt/data` | 保留原字节和拒绝记录 |
| 完整MJCF、mesh及UDP示例 | `LunaUdpTool` | 提供名义安装位置；不执行其Pitch→Yaw头链 |
| 刚体注册、联合pF与补偿方法 | `head_eye_pro`（Marvin_pro） | 复用数学方法，不复用其机型配置、相机参数或权重 |

144个复制文件的原路径、用途和SHA256均记录在 `sources.json`。`reference/marvin_methods` 是方法快照，其中其它项目依赖和历史数字只供阅读，不作为本工程入口；实际复用的无设备依赖 `registration.py` 位于 `dynamic_handeye/`。

本项目Luna臂MDH相关长度为264/264mm，首段158.6mm、固定法兰87mm。此前分析中的287/314mm来自Marvin_pro，不能用于判断Luna模型。当前相机为**左CSI0、右CSI1**，原图2560×1984、测量图1280×992，基线59.538mm。Marvin_pro的CSI配置、59.742mm基线和共享pF均不用于这里。

## 目录

```text
luna_dynamic_handeye/
├── dynamic_handeye/       # 活跃代码：几何、数据导入、拟合/验证、推理、采集适配
├── configs/
│   ├── camera/           # Luna原始鱼眼与校正双目、原始测量契约
│   ├── robot/            # Luna MDH与实测球偏移
│   └── capture_plan.json # 9头姿停稳交叉采样计划
├── datasets/
│   ├── raw/              # 未改写实机JSONL、完成清单、示教轨迹
│   └── processed/        # 通过契约检查的归一化观测
├── acquisition/          # 原Luna采集、Orin发布/接收及质量链的源码快照
├── reference/
│   ├── LunaUdpTool/      # 完整模型、mesh与UDP源码
│   └── marvin_methods/   # Marvin_pro方法与数学机理文档
├── models/               # 模型索引；当前诊断模型随报告保存
├── reports/              # 真实数据结果、模型、划分与测试记录
├── docs/                 # 坐标系/头链决策与实机采集说明
├── tests/                # 离线几何、泛化、时序与契约回归
└── sources.json          # 来源与完整性清单
```

## 已实现的模型（当前默认6参数）

`P_S = F_fixed · K(q_head) · X · P_C`。

- S为躯干S_MODEL，C为校正左眼；K采用旧V2.0的Yaw→Pitch顺序，显式计算Rz(yaw) Ry(-pitch)，不调用头部MDH。
- F_fixed为头部公共旋转中心的零位安装矩阵，由MJCF头中心位置与配置的零位轴方向构造；不使用D=640mm，也不做抵消。配置见`configs/robot/head_mount_v2.json`，当前是名义几何假设，未独立测量。
- X为H1末端到校正左眼的固定外参，优化旋转3参数、平移3参数，共6个。
- 先将机械臂参考球心反变换到每次H1系，用Kabsch/SVD初始化X，再以躯干系XYZ残差进行Huber优化；不优化A，不读取旧十二参数拟合值。
- 固定几何配置哈希写入模型；修改配置后旧六参数模型拒绝推理，须重新标定。旧v1十二参数模型保留兼容读取，不由fit入口重新拟合。

当前固定假设与真实数据尚不一致：884条数据的混合六参数训练P95为13.598mm，轨迹留出P95为8.086mm，尚不可用于实机定位。详见[六参数说明](docs/six_parameter_calibration.md)。新版Pitch→Yaw不在当前链中。

## 运行离线流程

以下命令均在本目录执行：

```bash
python3 -m dynamic_handeye inspect
python3 -m unittest discover -s tests -v
```

原始数据和本次处理/拟合产物已随项目保存。输出使用独占创建，重复实验请用新文件名：

```bash
python3 -m dynamic_handeye import \
  --capture datasets/raw/ball_point_head_20261018_v1.jsonl \
  --output datasets/processed/my_points.json

python3 -m dynamic_handeye fit \
  --dataset datasets/processed/my_points.json \
  --output reports/my_run
```

拟合产生 `model.json` 与 `validation.json`，分别保存单相机外参、固定安装及配置哈希、来源哈希、头姿/空间支持集，以及完整球位留出、5个空间条带留出、完整头姿留出和多会话留出（数据足够时）。同一球位所有头姿始终属于同一空间划分。六参数在固定头根下可用单头姿非退化点集拟合，但跨头姿与新会话验证仍不可省略。支持左右臂日志联合标定，左臂须另外提供实测球偏移。

显式查看未批准模型的离线诊断值：

```bash
python3 -m dynamic_handeye predict \
  --model reports/luna_v2_hybrid_6dof_20261026/model.json \
  --pixels 613.579062 432.531789 538.597719 432.549694 \
  --head -10 -25 --head-speed 0 0 --head-window-range 0 0 --diagnostic
```

这里输入的是**已完成球面投影修正**的左右校正像素。若输入是未修正的椭圆中心，先使用 `acquisition/interactive_ball_calibration/geometry.py:stereo_sphere_center` 的既有修正流程。去掉 `--diagnostic` 会进入正式校验；当前真实模型未通过独立验收，会被拒绝。正式入口还检查头部速度、融合窗口稳定性、角度凸包和30mm邻域支持，不外推。

连续运动接口在 `dynamic_handeye/runtime.py:predict_moving`：输入四维校正像素、左右真实曝光时刻、已验证的共同clock_id/时钟映射、时间不确定度及头部反馈历史。分别在左右曝光时刻插值头角，用两眼各自的相机位姿构造空间射线并求交，支持异步曝光。合成测试验证该数学路径；真实模型的 `moving_head_validated=false`，目前不能开启。硬件曝光/时钟验证、读出与运动模糊约束、分速度实机精度验收必须另行完成，不能通过手动改布尔值代替。

## 历史十二参数真实数据结果（对照，非当前默认）

`reports/summary.json` 与 `reports/real_capture_joint_refined_20261023/validation.json`：

- 原始969条，884有效、85拒绝；有效观测覆盖311个至少有一条有效记录的球位，274个球位三头姿全部有效。
- 仅三头姿：`[-10,-25]`、`[10,-25]`、`[0,-20]`，单会话，曝光同步未经验证。
- 固定实测pF、纯12参数几何模型：拟合P95 **5.250mm**；尾段球位留出P95 **3.388mm**。
- 五个空间条带留出P95约 **6.543 / 6.408 / 4.438 / 4.454 / 3.039mm**。
- 同球跨头姿拟合漂移P95 **3.221mm**，仅为训练数据诊断。
- 所有整头姿留出均因训练头姿激励不足而不可验证；没有新会话验收。

这些结果说明工程链路已跑通，但当前名义头链、固定pF与刚体模型不足以支持亚毫米结论。应先核验真实关节顺序/轴向、相机杆臂、球偏移坐标基准和臂FK，再增加二维头姿覆盖。当前不训练高斯残差，避免用补偿掩盖机型或链定义问题。

## 新数据采集

2026-10-27下一批使用**pitch=0、yaw=-10/0/+10°、每组合50帧**，操作流程见[零俯仰左右偏转采集](docs/pitch0_yaw_capture.md)。务必显式指定`--plan configs/capture_pitch0_yaw_plan.json`；以下旧九姿示例保留供后续使用。

先做无硬件计划检查：

```bash
python3 -m dynamic_handeye capture \
  --path datasets/raw/ball_taught_20261018_visible_v1.json \
  --output datasets/raw/new_pilot.jsonl
```

它核对9头姿计划、录制时限位、轨迹与参数哈希，不连接设备。后续实机采集入口复用完整Luna SDK质量与异常恢复逻辑，需明确加`--run`并输入原流程的`RUN`；默认仅采前12个已示教点作为预实验，检查现场后再扩大。相机启动、SDK路径、反馈/曝光限制与运动命令详见 `docs/acquisition.md`。本次未执行任何设备命令。

## MuJoCo坐标系展示

运行`./view_frames.sh`，显示躯干S、双臂MDH父系候选、控制器头根/末端及左右相机轴，红X/绿Y/蓝Z；数字1–5切换坐标组。坐标安装的名义值与未批准拟合值均有标签，不视为实机测量。见[安装矩阵标定与视图说明](docs/camera_mount_calibration_and_viewer.md)。

## 旧版 Luna V2.0 实现（2026-10-26）

当前可执行链明确采用旧版Yaw→Pitch。支持双臂躯干系FK、左右臂日志联合标定，以及双目躯干系推理；详见[旧版运行与标定说明](docs/luna_v2_pipeline.md)。新版Pitch→Yaw方案尚未部署，不能复用旧外参。

## 连续头姿＋自由pF研究验证

已实现连续几何、空间RBF、头角条件RBF与加性RBF共64候选的训练分块选型、嵌套整头姿和头姿/球位双留出验证，见[完整研究报告](reports/continuous_free_pf_20261026/README.md)。合成未见头姿验证可低于1mm，现有实机数据未达标；研究接口默认拒绝正式推理，不替换当前六参数运行模型。

补入第四头姿`[0,-25]`的318条固定视角有效观测后，四头姿共1202条、321个物理球位，比较结果见[四头姿验证](reports/four_head_free_pf_20261026/README.md)。跨会话相同球位按已核对的共同轨迹及q7绑定，防止训练/测试泄漏；尚未达到P95<1mm。

## 名义几何不准时的FK一致性研究

新增[有效运动链与躯干空间补偿比较](reports/fk_consistent_vision_20261026/README.md)：每折冻结同一训练侧pF和FK参考，比较固定6参数、有效12参数、一般双转轴POE及直接回归；推理仅输入四维校正像素和头角。有效链12+躯干空间核两端整头姿留出P95约2.57/2.72mm，仍未达到1mm，且独立俯仰激励不足。该研究不修改默认模型，也不把名义MDH/URDF当作物理真值。

## 独立棋盘格双目标定与重采

从 `head_eye_capt` 移植的独立工具位于 [stereo_calibration](stereo_calibration/README.md)，包含原始鱼眼K/D/R/T标定、GStreamer/Argus直连CSI图像采集、校正预览和独立板形/尺度评估，不依赖源项目。重新标定后的数据清单、交叉头姿与独立验收划分见 [重采计划](stereo_calibration/RECAPTURE_PLAN.md)。新相机参数须同步更新测量契约并重新拟合手眼模型。

## 2026-10-29采集暂停记录

已完成九姿计划前4轮：[0,-25]、[10,-25]、[-10,-25]、[-10,-20]，共808位置，586有效、222无效，原始拒绝记录保留。当前右臂和头IDLE，后5轮未采，下一头姿[0,-20]。数据见[datasets/new_capture_20261029](datasets/new_capture_20261029/)，核验摘要见[PAUSED_SUMMARY.json](datasets/new_capture_20261029/shared_fast_grid_20261029/PAUSED_SUMMARY.json)，后续启动见[RESUME.md](datasets/new_capture_20261029/RESUME.md)。暂停不等于标定验收通过；正式相机、MDH、pF未修改。

## 2026-10-30最新离线手眼标定

基于本机当前相机参数及A/B/新九姿前四轮共2247有效观测、6头姿，冻结原FK、pF与零位，比较40个几何/仿射/联合高斯候选。按19个开发留出测试的最差P95选择相机相对头链＋联合仿射/高斯60mm/ridge3：训练P95为4.833mm，最差留出P95为9.408mm，尚未达到1mm。完整模型、训练内选型、Pro式逆向对照与四像素推理入口见[最新标定报告](reports/latest_handeye_20261030/README.md)；模型仅供显式诊断，不替换默认运行模型，需另采独立验收数据。

固定头姿第一阶段的原因诊断和软件几何校准见[固定视角零位假设报告](reports/fixed_view_zero_diagnosis_20261030/README.md)：固定[0,-25]、615个位置，在明确修正FK参考后，分块留出P95为0.811mm、跨轨迹P95为1.057mm（最大1.952mm）。这些不是原控制器FK的精度成绩；未重置编码器或修改正式MDH，当前候选需新会话验收。
