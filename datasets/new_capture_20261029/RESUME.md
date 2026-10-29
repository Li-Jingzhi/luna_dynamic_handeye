# 本轮暂停与续采

用户要求：完成当前第4轮[-10,-20]后暂停，后续再继续。最终状态以shared_fast_grid_20261029/status.json为准；completed摘要记录完整性、有效数与SHA256。无效记录原样保留，暂停不是标定验收通过。

计划共9轮，每轮完整1003点路径、202个采集位置，每位置50帧，SDK5%速度。前4轮头姿为[0,-25]、[10,-25]、[-10,-25]、[-10,-20]；后5轮为[0,-20]、[10,-20]、[10,-15]、[0,-15]、[-10,-15]。数据真实目录为本文件同级shared_fast_grid_20261029/，旧datasets/raw同名路径是相对兼容链接。

后续重新连好相机与控制器、同一身体/左臂条件，并保持右臂在上轮结束位置后，从项目根目录先检查：

```bash
.venv/bin/python data_collection/workflow.py --session shared_fast_grid_20261029 \
  --trajectory trajectories/shared_replay/right_20261029_1003points/trajectory.json \
  --plan trajectories/shared_replay/right_20261029_1003points/capture_plan.json \
  --resume --continue-low-valid
```

确认现场可运行后，以上命令加--run会实际运动并继续剩余轮次。程序重新验证已完成轮次、文件哈希、计划、轨迹、资产、机器人状态及预期姿态，不覆盖原数据；若期间改变机械臂位置会拒绝自动接近，不应绕过检查。

分析现状见reports/acquisition_error_audit_20261029/README.md。机械臂有效零位修正是诊断假设，尚未写入正式MDH或实机；原FK下1mm目标尚未实现。本次没有替换相机、pF或运行标定模型。
