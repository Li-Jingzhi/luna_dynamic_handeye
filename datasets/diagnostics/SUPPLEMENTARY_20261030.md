# 仅黄球补充诊断采集

用户明确取消棋盘格或独立尺寸靶，仅用法兰末端偏置黄球。现有四组会话fixed_gamma16_20261030_01继续执行optical15/hold48/return120/acceptance30。附加脚本data_collection/yellow_ball_extra.py等待该真实会话completed.json为true后才申请同一采集锁并启动；失败会话不触发后续运动。

补充静止30窗口（近/中/远局部深度各10窗口，site内不再下发臂位置命令），只动头99窗口（3球位×9姿×3轮=81，另18个参考姿返回）。头姿为yaw=-10/0/+10、pitch=-25/-20/-15。每轮交错顺序，参考[0,-25]在轮前/轮后复测；site内七关节保持同一反馈q，跨度≤0.02°。每窗口50帧，Gamma1.6检测及6张PNG/双目对，保留未增强与ISP图像；限位、名义碰撞及局部臂运动门槛不放宽。

附加数据物理位置：Orin /home/nvidia/luna_diagnostic_data/ball_extra_20261030_01/。本机通过datasets/orin_diagnostic_storage/挂载访问；挂载为本会话临时只在本机配置，未向仓库加入挂载树。约需额外数十GB。mount丢失时脚本拒绝开始，不能误写入空的本地目录。元数据、图像及SHA均存同一会话；需Orin在线才可访问。后续同步仓库时应另导出实际数据，不能只提交挂载路径。

```bash
# 离线预检
.venv/bin/python data_collection/yellow_ball_extra.py \
 --after-suite datasets/diagnostics/fixed_gamma16_20261030_01 \
 --output datasets/orin_diagnostic_storage/ball_extra_20261030_01
# 实际执行需要原四组完成且挂载存在，在上面命令末尾加--run。
```

当前已启动等待任务，勿重复启动。等待终端Ctrl+C取消排队；开始后Ctrl+C按脚本清理右臂模式。程序异常后不盲目续采，保留失败原因并检查现场状态。

这些实验能检查重复性、回程差、构型参考不一致和头链重复性；没有精密独立尺寸参考，不能据此证明相机绝对尺度。视觉保持球心不等于机械夹具固定中心。
