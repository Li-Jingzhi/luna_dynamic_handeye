# 仅用现有黄球的自动误差诊断

无需拖动示教、手工摆位、精密间距靶或固定球心夹具。程序自动移动右臂、停稳并采集。当前仅研究固定头姿 `[yaw,pitch]=[0,-25]°`，不重置编码器、不修改机器人参数、不在线重拟合补偿模型。机器人未连接时可完成检查与仿真；只有显式 `collect --run` 才连接机器人并运动。

## 四组自动实验

| 顺序/名称 | 自动动作 | 静止窗口数 | 主要用途 |
|---|---|---:|---|
| optical | 围绕初始球位，3个深度层、各5个横纵位置 | 15 | 比较名义40mm球的单目尺寸深度与双目深度，查看位置相关偏差和静态抖动 |
| hold | 3个球位；对J3/J4/J6/J7方向做正负零空间姿态变化，幅度2°/4° | 48 | 视觉保持球心，比较原始FK和有效零偏FK随构型的漂移 |
| return | 3个球位；4个关节分别从正负1.5°方向返回，同方向重复5次 | 120 | 检查重复性和接近方向相关残差，辅助发现回差/负载影响 |
| acceptance | 3个深度层、各10个新横纵位置，自动改变冗余姿态 | 30 | 冻结模型下独立新增采集的一致性检查 |

合计213窗口、10,650个合格双目对。每窗口50个独立、停稳后新到达的双目对；控制用图像和拒绝图像另计。hold中将关节轴投影到位置雅可比零空间，归一化后生成目标，**并非单独转动该关节**；记录实际七关节变化，要求至少有一个关节较本位置锚点变化≥1°。新位置均以会话起始构型作冗余姿态参考，避免累积漂移。

默认是首次接机用的局部实验：光学深度±20mm、横纵约±15/10mm；相机坐标总位移半径50mm，七关节各自相对会话起点不超过12°，每次下发≤0.3°，SDK速度/加速度5%。并不覆盖完整工作空间或整个视场。不同初始构型可能不可达或触发限制，程序保留原因后停止，不盲目绕行。需要扩大覆盖时，从其他已知可见且周围通畅的起始构型建立新会话。

仅有黄球时，单目尺寸检查依赖**名义直径40mm**，不能独立证明双目绝对尺度。视觉保持实验以视觉反馈为约束，不能替代机械固定球心的独立证据，也不能单凭它认定真实编码器零位错误。拟合的J7零偏与球杆偏移仍可能等效。

## 离线使用

在项目根目录运行，使用现有项目环境（NumPy/OpenCV/Paramiko；实机另外需要MuJoCo和SDK）：

```bash
# 校验当前模型、相机/几何哈希和计划；不连接硬件
.venv/bin/python -m data_collection.diagnostics.cli check

# 仅查看部署文件与哈希，不SSH、不启相机
.venv/bin/python -m data_collection.diagnostics.cli deploy-camera

# 四组实验各模拟2个窗口，合成图像明确标记SIMULATED
.venv/bin/python -m data_collection.diagnostics.cli simulate \
  --output /tmp/luna_diagnostics_smoke
.venv/bin/python -m data_collection.diagnostics.cli analyze \
  --session /tmp/luna_diagnostics_smoke

# 完整213窗口协议仿真：用于验证程序，不能证明精度
.venv/bin/python -m data_collection.diagnostics.cli simulate \
  --limit 999 --output /tmp/luna_diagnostics_full
```

默认计划已保存为 [plan.json](plan.json)。可用 `plan --output 新文件.json` 导出，再通过 `--plan 路径` 指定。程序验证50帧、步长、运动边界、头姿和球尺寸约束；不会自动扩大运动边界来完成失败目标。默认短仿真仅检查流程；输出的验收通过标志始终为false。

## 接机后执行

机器人SDK与控制器必须均为`00040701`；SDK默认位于项目同级`../00040701`。默认控制器`6.6.7.190`、Orin`6.6.7.100`、SSH用户`nvidia`。可用命令行覆盖IP和SDK路径。SSH使用现有密钥/agent；必要时通过环境变量`LUNA_DIAG_PASSWORD`提供密码，不写入计划或输出。

开始前机器人右臂和头部须为IDLE，身体/左臂保持不动；初始右臂构型应让黄球在头姿[0,-25]时双目可见且有足够边缘余量，局部动作范围保持通畅。程序自动分步调整头部，但不会在看不到球时盲搜。名义MuJoCo检查包含机器人自碰撞，未包含桌面、黄球杆和人员。当前软件未进行接机实测，应先用单组/局部会话验证。

Orin需要现有JetPack环境中的`gi`、GStreamer Argus、OpenCV、NVIDIA VPI/CUDA。无ROS2依赖。两路CSI由本程序独占；请先停止占用相机的正式采集/预览服务。程序不会结束其他服务。诊断服务绑定Orin的`127.0.0.1:8768`，通过SSH通道传输PNG和元数据。

```bash
# 离线预检，不运动、不创建会话目录
.venv/bin/python -m data_collection.diagnostics.cli collect \
  --output datasets/diagnostics/fixed_01

# 自动部署相机服务并依序完成四组；这是执行运动的命令
.venv/bin/python -m data_collection.diagnostics.cli collect \
  --output datasets/diagnostics/fixed_01 --run

# 也可先只做一组
.venv/bin/python -m data_collection.diagnostics.cli collect \
  --experiment optical --output datasets/diagnostics/optical_01 --run

# 采后分析，保留原模型，禁止用验收数据重新拟合
.venv/bin/python -m data_collection.diagnostics.cli analyze \
  --session datasets/diagnostics/fixed_01
```

`deploy-camera --run`只部署相机，`collect --no-deploy-camera --run`使用已部署的同版本同参数服务。部署不会改内核或覆盖正式服务目录；独立目录为Orin的`~/luna_auto_diagnostics/`，日志为`camera.log`。服务在采集退出后继续运行；恢复原正式采集前，应在Orin通过`pgrep -af 'luna_auto_diagnostics/camera_server.py'`确认本服务PID，并向该PID发送`kill -TERM PID`释放CSI。

Ctrl+C、识别/同步超时、反馈漂移、限位、模型碰撞或视觉伺服发散均停止后续任务并将右臂切回IDLE、断开SDK；不会自动回零或返回远处起点。头部调整结束立即回IDLE。`--resume`仅允许原计划、模型、代码、资产和机器人检查点匹配时续采；已完成窗口不重采，失败图像保留。机器人已被移动时使用新会话，不盲目续跑。

## 相机和证据合同

相机沿用当前固定参数：5ms曝光、模拟/数字增益1、关闭自动白平衡，启动后锁定AE/AWB并核验属性。两路共用GStreamer pipeline/clock，**共同源时间差与本机到达时间差均≤10ms**。这是软件配帧，元数据明确`exposure_time_valid=false`，不声称硬件曝光同步。用当前棋盘格参数做VPI/CUDA等距鱼眼校正，图像1280×992；实时链不经过ROS2。

SDK停稳后在Orin设置到达时间屏障，仅接收屏障之后且新鲜度≤200ms的图像，逐眼去重。每次读图前后记录关节反馈，窗口内七关节跨度≤0.02°，固定头姿误差≤0.02°，校正视差P95抖动≤0.1px。视觉保持额外要求球心偏离目标≤0.6mm；新采验收构型与开发集的最近七关节最大差≥1°。这些是采集门槛，**不会按视觉−FK误差筛选好看的样本**。

每个合格双目对保存4张无损PNG：左右校正前输入、左右校正图；校正前输入是ISP缩放的BGR，非Bayer RAW或2560×1984原始传感器图。拒绝检测的图像也保留，并记录原因。真实图像压缩大小依场景变化，应为完整流程预留数十GB存储；仿真的纯色占位PNG不代表实采存储量或检测算法验证。

```text
会话/                           # all时内含optical/hold/return/acceptance子目录
  suite.json                    # 四组共用计划、模型哈希
  某组/
    manifest.json               # 计划、任务、代码/资产哈希、起始反馈/运动边界
    frozen_model.json           # 采前冻结模型
    assets/                     # 相机、机器人、来源清单、名义模型快照
    observations.jsonl          # 已完成50帧窗口、任务、反馈、伺服/返回路径审计
    camera_evidence.jsonl        # 合格及拒绝图像的元数据与PNG哈希
    source_identities.jsonl      # 控制帧及采集帧的逐眼唯一标识
    images/                     # 四张PNG/双目对
    model_checks/               # 实机动作前名义自碰撞检查
    events.jsonl                # 拒绝、失败、清理异常
    status.json / completed.json
    analysis.json               # 冻结模型下的两种FK参考、各实验指标
```

分析同时报告`visual_vs_original_fk`和`visual_vs_calibrated_fk`，避免把改变FK参考后的低误差误写成原始FK精度。验收分别给两种参考的P95、最大误差、超过1mm比例和全部点<1mm标志。仅实采、完成完整计划且最大误差<1mm时，相应固定视角一致性标志才可能通过；即使通过也只覆盖本次局部采样，不代表任意头姿或绝对物理精度。模型仍为诊断用途，采集不自动升级其审批状态。

## 2026-10-30亮度诊断会话

用户要求把预览Gamma1.6用于检测和保存的测量图。新计划为plan_gamma16.json（image_gamma=1.6），新会话datasets/diagnostics/fixed_gamma16_20261030_01/。相机曝光5ms、模拟/数字增益1不变；软件映射round(255*(pixel/255)**(1/1.6))作用于校正图。

left/right_rectified.png为实际增强后检测输入，left/right_rectified_unenhanced.png保留增强前校正图，left/right_input.png保留相机ISP未校正输入（不是BayerRAW）。每对图像保存6张PNG；metadata.image_processing及会话计划记录增强方案，已有数据不覆盖、不混入。实际用增强图作亚像素边缘及椭圆拟合，不能将此变化描述为只改变显示。

首次12对静止图对照，原始及增强均12/12通过；左中心变化约0.13–0.18px，三维球心差0.08–0.62mm。只说明算法对亮度敏感，不证明哪组更接近真值。对照存于reports/brightness_diagnosis_20261030/stationary_comparison.json。当前冻结模型来自未增强数据，此会话正是检查成像变化对其一致性的影响，不在线重拟合。

Collector将上一合格球心用于局部候选追踪，拒绝相邻静止反馈样本间超过10mm的球心跳变；持续拒绝仍会超时停止，50mm整体范围及机械限位不放宽。新会话计数与之前失败会话分开。

预览使用以下命令读取已保存的实际检测图，不竞争相机服务的take_latest帧；伺服未保存图像时可能暂不更新，不应误认为实时视频流：

```bash
.venv/bin/python -m data_collection.diagnostic_saved_preview \
 --session datasets/diagnostics/fixed_gamma16_20261030_01 --saved-images
```

Gamma方案下213窗口图像可能占约80GB，需监控实际磁盘使用。单独预览相机模式会消费帧，采集期间应使用上述saved-images模式。B键仅影响显示，不改变已保存或检测图像；初始显示不重复叠加Gamma。

## 补充实验20帧模式（2026-10-30）

按用户要求，本轮静止重复30窗口、只动头99窗口使用 `--frames 20`，共2580组正式双目对；原50帧会话不改写。通用窗口采集及分析按计划中的 `samples_per_window` 校验，只支持20或50，默认仍为50。停稳、同步、视差抖动≤0.1px和运动边界保持原值。20帧窗口的统计置信度低于50帧，不应直接当作相同重复次数的精度证据。

```bash
.venv/bin/python -m data_collection.yellow_ball_extra \
 --output datasets/orin_diagnostic_storage/ball_extra_20frames_20261030_03 \
 --after-suite datasets/diagnostics/frozen_acceptance_gamma16_20261030_01 \
 --frames 20 --run
```

首次01会话因诊断相机服务未运行而停止，0窗口；恢复服务后02会话仍因Argus不出帧停止，0窗口。按用户要求重启Orin、从 `/home/nvidia/cam_geac` 执行厂家 `./demo/init_demo 3` 后，固定控制核验与配对图像正常，另建03会话。原失败记录保留。运行前确认挂载可用，并使用新输出目录。
