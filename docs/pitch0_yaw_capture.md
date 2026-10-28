# 下一批：pitch=0、yaw左右偏转、每组合50帧

本轮使用旧版Luna V2.0，计划为`configs/capture_pitch0_yaw_plan.json`：采集顺序[-10,0]→[0,0]→[+10,0]度。每个机械臂位置依次采三个头姿，每个头姿各50个唯一合格双目对（不是三个头姿合计50帧），再回[0,0]后移动机械臂。

计划角度先沿用既有±10°覆盖；如需调整角度，在新会话开始前修改计划，续采中禁止改变绑定。该计划必须显式通过--plan指定，通用capture命令的默认计划仍是原九姿二维网格。

## 与Pro保持一致的核心模式

- 右臂拖动示教，约每3°记点，至少18点；Ctrl+C结束并保存完整轨迹。
- 低速回放，SDK速度/加速度5%，机械臂和头部停稳后采样。
- 每个(球位,头姿)严格50帧；左右每只眼的源时间戳分别不重复，源字段使用Luna的source_timestamp_ns。
- 只接收采样窗口开始后的新帧，沿用Luna像素/球心、相机身份、标定哈希、曝光/质量和时间门槛。
- 原始与球心修正后视差抖动P95均≤0.1px；50帧内机械臂与头角范围≤0.02°。
- 球心/像素及q7/头角用窗口中位数；保留逐帧元数据、反馈、失败记录与拒绝原因。
- 不改用Marvin的MDH、CSI身份、轨迹schema或相机参数；保留Luna SDK/在线限位/身体不动/异常IDLE恢复检查。

新的逐眼源帧去重和融合规则仅在该专用yaw模式启用；旧九姿入口保持原兼容行为。标准Pro遇单眼重复源帧会继续等待，Luna适配器在融合前拒绝包含重复源帧的窗口，再沿用原逐点重试/失败记录流程，避免把复用源帧算成50独立观测。

## 1. 拖动示教

在本项目根目录，相机服务与SDK按docs/acquisition.md就绪后：

```bash
PYTHONPATH=acquisition .venv/bin/python -m interactive_ball_calibration.capture teach \
  --stereo configs/camera/luna_rectified_stereo.json \
  --mdh configs/robot/luna_new.ini --sdk-root ../00040701 \
  --latest acquisition/runtime/new_sh5_yellow/latest_pose.json \
  --head-target 0 0 \
  --output datasets/raw/ball_taught_20261027_pitch0.json
```

这是真实连接/拖动命令，不是离线检查。选择三个yaw角下黄球均可见的公共空间，覆盖不同距离、图像区域及法兰姿态，保持身体与相机/球安装不变。至少18个密集轨迹点；第一轮多头姿试采可只回放前12点。

## 2. 固定[0,0]逆序回放并返回首点

```bash
PYTHONPATH=acquisition .venv/bin/python -m interactive_ball_calibration.capture replay \
  --path datasets/raw/ball_taught_20261027_pitch0.json \
  --stereo configs/camera/luna_rectified_stereo.json \
  --mdh configs/robot/luna_new.ini --sdk-root ../00040701 \
  --latest acquisition/runtime/new_sh5_yellow/latest_pose.json \
  --head-target 0 0 \
  --output datasets/raw/ball_auto_20261027_pitch0_reference.jsonl
```

按原流程终端RUN后沿已教路径逆序采集，每点50帧。该固定头姿回放沿用源Luna融合规则；新逐眼严格规则应用于下一步的多头姿主采集。示教结束通常在末点，这一步也把机械臂沿路径带回首点；交叉采集不会从任意当前位置直接跳到首点，异常中断后应先核对实际位置。

## 3. pitch0多yaw采集

仅离线检查，不连接硬件：

```bash
.venv/bin/python -m dynamic_handeye capture \
  --plan configs/capture_pitch0_yaw_plan.json \
  --path datasets/raw/ball_taught_20261027_pitch0.json \
  --output datasets/raw/ball_point_head_20261027_pitch0.jsonl \
  --stop-after 12
```

轨迹通过、现场就绪后，同一命令追加`--sdk-root ../00040701 --run`启用实机流程，再输入原流程RUN确认。

12臂位×3头姿=36组合，全部有效时1800个双目对。正式批次可扩大至30–50个已示教点（不能超过轨迹长度）。每组合的50帧独立融合，不跨头姿混合。

专用适配器将头部参考姿也设为[0,0]，每个臂位结束不会回旧[0,-25]。yaw的重复接近路径可能含目标角+1°预定位；pitch保持0，不再因该接近算法偏到1°。从原低头状态到首个零俯仰参考姿仍需要一次真实俯仰运动。

## 保存与后续拟合

输出JSONL及.jsonl.plan.json绑定计划/轨迹/参数哈希。有效行记录robot_version=luna_v2_0、fusion_policy=50_unique_source_frames_per_eye。保留50帧窗口元数据、真实反馈及中位数；源时间戳缺失、重复或抖动超限时拒绝，不能用接收时间冒充源身份。接收/源时间字段仍不代表已验证的曝光同步。

此批pitch0 yaw扫描可与安装关系未变、来源兼容的旧pitch-25数据组合，显著增加俯仰跨度；单独这批仍只有一个pitch，不可独立验收任意俯仰角。新旧会话的身体/安装变化必须核查，不要仅因参数文件哈希相同就盲目合并。

固定头姿回放日志没有target_head_deg时，旧版导入默认[0,-25]；本轮[0,0]基准必须显式指定目标：

```bash
.venv/bin/python -m dynamic_handeye import \
  --capture datasets/raw/ball_auto_20261027_pitch0_reference.jsonl \
  --fixed-head-target 0 0 --output datasets/pitch0_reference_validated.json
```

多头姿主日志逐行有target_head_deg，不需要该参数。保留两条已选算法；采后先校验日志，再适配有效运动链＋躯干补偿拟合的输入。当前通用fit仍是固定名义6参数，不等同于主研究算法。

本次完成配置、采集适配器及无SDK测试，没有执行上述实机命令，没有采集新的真实观测。
