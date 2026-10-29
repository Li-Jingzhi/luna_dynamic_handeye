# Luna V2 数据采集入口

## 已保存的新共用轨迹与快速模式

1003点完整轨迹已整理至[独立文件夹](../trajectories/shared_replay/right_20261029_1003points/README.md)，附九头姿计划和启动命令。按用户最新选择：每隔5点采集一次并包含首尾，共202位置/头姿，每位置50帧；1003个运动连接点全部保留。九轮交替正逆回放，减少返回空跑。预览进度、完整性检查、80%有效率检查均按202个采集位置统计。默认旧计划仍逐点采集。以下旧示教记录仅供追溯，以新文件夹中的计划为准。

## 当前选择：零头姿示教一条轨迹，多头姿共用

2026-10-29按用户最新要求：头姿[0,0]时仅示教一条右臂轨迹，后续9姿统一回放这条轨迹。已停止此前逐头姿独立示教，原partial记录保留。

```bash
.venv/bin/python data_collection/head_trajectories.py teach --arm right --head 0 0 \
  --session shared_right_zero_20261029 --run
```

本机当前该命令已在可见终端运行，勿重复启动。q回车保存完整轨迹后，使用下面入口执行九头姿共用回放：

```bash
.venv/bin/python data_collection/workflow.py --session shared_grid_replay_20261029 \
  --trajectory trajectories/by_head_pose/shared_right_zero_20261029/luna_v2_right_yaw_z0_pitch_z0.json --run
```

两条命令分别为示教和后续回放；未完成轨迹不能回放。仍按原50帧门槛拒绝不可见/不合格点；共用路径不保证所有头姿都可见。`head_trajectories.py`不指定`--head`时仍支持每头姿独立路径，命名含臂侧和有符号yaw/pitch；该模式暂不用于当前采集。

`workflow.py`把拖动示教、完成检查、模型检查、多头姿逆序回放、50帧采集和进度预览串成一次流程。此前轨迹B的日期脚本已迁入`validationB_legacy.py`，旧报告路径保留兼容入口。底层Luna驱动、球心检测和50帧融合仍复用`acquisition/`，避免复制第二套检测算法。

## 下一步采什么

当前A/B已用于算法研究，应视作开发数据。先进行以下两个诊断，再扩展头姿：

1. **相机度量验证**：使用精密已知间距的靶点，在近、中、远距离及视场中心/边缘采10–20个静止位置，每位置保留至少50帧和原始双目图。检查视觉距离与实测距离、深度与视场的关系。需要实际已知尺寸靶，普通自由移动黄球不能提供绝对尺度真值。
2. **固定球心、多法兰姿态**：将黄球中心固定在支点夹具，在3–5个不同位置，各缓慢改变10–20种法兰/肘部构型并停稳。固定pF=[0,0,101]mm，检查FK球心是否随构型漂移。不可用手扶球冒充精密固定夹具。
3. **开发采集**：另教120–180点轨迹，覆盖当前A/B空间边界、近中远距离和不同肘部/腕部构型；在yaw=-10/0/+10和pitch=-25/-20/-15的完整9姿网格采集。避免只绕已有轨迹加密。每位置50个独立合格双目对。
4. **轨迹C验收**：冻结模型后另教新轨迹，以已见4头姿检查新球位；另用[-5,-22.5]、[5,-22.5]、[-5,-17.5]、[5,-17.5]检查未见头姿。当前尚未补齐负yaw/-15训练层，这些插值点暂含局部外推，只能诊断，不能宣称通过全域验收。

## 环境

使用旧Luna V2.0，Yaw→Pitch，控制器6.6.7.190、Orin6.6.7.100，本机有线6.6.7.101。当前Orin正式服务须已输出新参数双目测量；固定5ms曝光/增益1/白平衡关闭。黄球可见且检测PASS。安装现有`requirements-viewer.txt`环境（包含MuJoCo），SDK位于项目同级`00040701`。系统`/usr/bin/python3`提供GStreamer预览依赖。

先检查相机，必要时沿用已有启动工具恢复Orin服务：

```bash
python3 acquisition/orin_rectified_stream/start.py --check
# 需要启动/重启板端时，沿用既有SSH环境与设备初始化流程。
```

新流程会启动本机接收器、关闭相同sidecar的旧接收器，等待50个新鲜PASS帧后才允许继续。不会自动修改Orin内核或相机参数。

## 一次完成示教与回放

在项目根目录先离线检查（不连接控制器、不运动、不创建采集会话）：

```bash
.venv/bin/python data_collection/workflow.py --session train_grid_C1
```

实际采集：

```bash
.venv/bin/python data_collection/workflow.py --session train_grid_C1 --run
```

该命令明确授权头姿调整、右臂拖动模式和计划中的全部自动回放。拖动时保持身体/左臂不动，慢速教新轨迹；终端实时显示点数。输入`q`回车结束示教。轨迹完整、至少18点、步长/限位/绑定校验通过后自动开始回放，无需逐头姿再次输入RUN。程序不会将失败退出留下的partial文件自动升级为完成轨迹。

采集仍以5%SDK速度、相邻接近步长≤5°、每位置50个独立合格观测执行。接近逆序起点使用直接关节插值。每轮之前做在线限位/固定身体/左臂检查及名义MuJoCo检查。模型未包括桌面、球杆和人员；现场应保证所教路径及直接接近路径通畅。

若使用已有完整、当前哈希绑定的轨迹，跳过示教：

```bash
.venv/bin/python data_collection/workflow.py --session replay_C2 \
  --trajectory datasets/raw/luna_v2_validationB_taught_20261029.json --run
```

独立轨迹C计划：

```bash
.venv/bin/python data_collection/workflow.py --session untouched_C1 \
  --plan data_collection/plans/validation.json --run
```

默认`--session`必须是新名字，同名目录拒绝覆盖。`--resume`允许从经哈希核验的连续完整轮次之后续采，需显式提供原计划与轨迹；不会追加未完成轮次、修改已有记录或重采已完成头姿。资产、计划、轨迹或机器人身体/左臂/右臂预期终点变化时拒绝自动运动。

Ctrl+C向采集子进程转发，子进程按原逻辑退出模式并回IDLE，后续轮次取消。默认一轮有效率不足80%也会暂停后续采集。显式`--continue-low-valid`允许完整低有效率轮次保留拒绝记录后继续，逐点检测和50帧门槛不变，警告保存在head_XX.warning.json，续采政策记录于resume.json。本轮用户已选择继续后续8姿，使用该显式选项。

## 固定球心诊断入口

实际夹具已固定球心后，先检查命令，再加`--run`进入拖动记录：

```bash
.venv/bin/python data_collection/pivot_capture.py \
  --site fixture_01 --output datasets/raw/pivot_fixture_01.jsonl
.venv/bin/python data_collection/pivot_capture.py \
  --site fixture_01 --output datasets/raw/pivot_fixture_01.jsonl --run
```

该模式复用原`capt_head.drag_capture --pivot`，记录关节反馈，不依赖黄球视觉PASS，也不进行自动轨迹回放。Ctrl+C完成。球心是否真实固定是物理实验约束，软件无法代替验证。

## 数据目录

```text
datasets/raw/<session>/
  session.json             计划、pF、参数哈希
  trajectory.json          本次完整示教路径；--trajectory时引用原路径
  status.json              teaching / preparing / replay / complete / stopped
  initial_feedback.json
  model_check_00.json      各头姿模型检查与实际反馈
  head_00.jsonl            对应计划第一个头姿的逐点记录
  head_00.session.json
  head_00.completed.json   成功/失败、有效数、文件SHA256
  capture_00.log
  viewer.log
```

后续头姿依序编号。预览窗口按当前JSONL显示进度；无效点保留拒绝原因。完成后再做导入和分析；通过采集质量门槛不代表达到1mm。
