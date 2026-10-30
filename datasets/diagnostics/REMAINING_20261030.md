# 剩余诊断的新会话

原fixed_gamma16_20261030_01保留13个合格光学窗口，以及optical13多次失败的完整图像。原会话不修改、不覆盖；因代码策略变更，另建后续会话，沿用原manifest的相机位置、关节起点、身体/左臂参考及冻结模型。

当前真实数据：Orin /home/nvidia/luna_diagnostic_data/fixed_gamma16_remaining_20261030_01/，本机挂载datasets/orin_diagnostic_storage/fixed_gamma16_remaining_20261030_01/。计划plan_gamma16_remaining.json只包含原最后2个光学位置，随后hold48、return120、acceptance30，共200次窗口尝试。原13合格窗口不重复。

continue_measurement_rejects=true：仅0.1px视差抖动失败或90秒内不足50个独立合格帧，保存capture_valid=false及原图后继续下一个位置；不会变成合格数据。限位、碰撞、反馈运动、球心跳变、局部范围、伺服发散及验收构型新颖度失败仍停止。completed.complete表示计划尝试完毕，不保证全部窗口合格；valid_windows/rejected_windows明确区分，不能据此宣称精度达标。

后续静止30与头部99窗口等待本会话尝试完毕后执行；若运动异常提前终止不会自动启动补充实验。补充任务同样保留光学质量失败，不绕过运动限制。预览来自实际已保存检测图，不消费相机帧。

```bash
.venv/bin/python -m data_collection.diagnostics.cli collect \
 --plan data_collection/diagnostics/plan_gamma16_remaining.json \
 --reference-session datasets/diagnostics/fixed_gamma16_20261030_01/optical \
 --output datasets/orin_diagnostic_storage/fixed_gamma16_remaining_20261030_01 \
 --no-deploy-camera --run
```

当前已启动，勿重复执行。后续发生失败时先查status/events，不清空已有目录。只用黄球，没有独立尺寸靶，不声称外部绝对尺度验证。

预览卡顿已改为Orin读取日志末尾、生成640×496左右拼接JPEG，经SSH传回异步显示。入口data_collection/saved_preview_stream.py，--directory使用Orin真实会话目录。读取已存帧，不是30fps实时视频；伺服期间无新存图时显示图像年龄，不消耗采集相机帧，也不重新检测。
