# 右臂共用轨迹：九头姿快速采集

2026-10-29在Luna V2.0、头姿[yaw,pitch]=[0,0]完成右臂拖动示教，1003点。trajectory.json为完整原件的逐字节副本，原件仍保留；manifest.json保存来源、SHA256和轨迹ID。未删改关节角或连接点。示教时997点检测合格；这不保证其他头姿下可见。

从项目根目录执行离线检查：

```bash
.venv/bin/python data_collection/workflow.py --session shared_fast_grid_20261029 \
  --trajectory trajectories/shared_replay/right_20261029_1003points/trajectory.json \
  --plan trajectories/shared_replay/right_20261029_1003points/capture_plan.json
```

执行回放采集（将实际运动机器人）：

```bash
.venv/bin/python data_collection/workflow.py --session shared_fast_grid_20261029 \
  --trajectory trajectories/shared_replay/right_20261029_1003points/trajectory.json \
  --plan trajectories/shared_replay/right_20261029_1003points/capture_plan.json --run
```

九姿顺序与回放方向：

|轮次|[yaw,pitch]，度|方向|
|---|---|---|
|1|[0,-25]|逆向|
|2|[10,-25]|正向|
|3|[-10,-25]|逆向|
|4|[-10,-20]|正向|
|5|[0,-20]|逆向|
|6|[10,-20]|正向|
|7|[10,-15]|逆向|
|8|[0,-15]|正向|
|9|[-10,-15]|逆向|

每轮沿全部1003点移动，按原始零基索引0、5、10……1000、1002采集，共202个位置。首尾均采，9轮1818条位置记录，每位置50个独立合格双目帧，合计目标90900对。非采集连接点仍按原关节路径、停稳和安全检查通过，不跨越路径，不提高SDK5%速度。正逆交替减少轮间返回路径；提速来自减少取帧等待，不承诺整体快5倍。

若需要恢复逐点采集，仅将capture_plan.json中的capture_stride改为1，使用新会话名。旧计划省略该字段也默认为1。

板端当前测量约26–29fps，仅50帧取样理论下限约52–58分钟/九轮，另加运动、停稳、换头姿和拒绝重试。实际耗时须以首轮为准。

数据保存到datasets/raw/shared_fast_grid_20261029/，每头姿一个head_XX.jsonl；同目录session文件标明头姿、方向、原始采集索引、参数哈希和轨迹ID。双目预览显示采集位置进度（分母202）、有效/拒绝数量。不可见或不合格位置保留拒绝记录；一轮有效率不足80%或运动/资产检查失败即停止后续轮次。Ctrl+C取消后续采集。输出目录已存在时拒绝覆盖，需用新会话名。

运行前仍检查实时机器人身份、限位、身体/左臂固定状态和名义MuJoCo路径。模型不含桌面、球杆和人员；离线检查不能替代现场通畅性检查。启动记录见下节；当前实机阶段以会话status.json为准。

## 本轮停止与续采

第1姿[0,-25]已完整采202位置，159有效/43拒绝，有效率78.71%，原程序因低于80%停止，机械臂与头回IDLE，相机仍运行。31条没有合格观测、12条视差抖动超限；数据哈希及完整摘要保留。预览现在会显示STOPPED及原因，避免只看到100%。

用户随后明确选择继续后续8姿，保持逐点门槛，启动命令为：

```bash
.venv/bin/python data_collection/workflow.py --session shared_fast_grid_20261029 \
  --trajectory trajectories/shared_replay/right_20261029_1003points/trajectory.json \
  --plan trajectories/shared_replay/right_20261029_1003points/capture_plan.json \
  --resume --continue-low-valid --run
```

此命令跳过已完成并核验的第1轮，从第2轮[10,-25]正向开始。同会话正在运行时勿重复启动；进程锁拒绝重复采集。数据真实位置已归入datasets/new_capture_20261029/shared_fast_grid_20261029，旧raw路径是兼容链接。
