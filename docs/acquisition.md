# 新Luna数据采集入口

本工程复用`acquisition/`中原有Luna采集逻辑，保持原有反馈检查、50帧唯一性、视差门槛、低速密集示教路径、限位及异常IDLE恢复。新适配器只设置头姿计划、绑定续采参数并传递入口参数；不运行Marvin_pro采集脚本，也不通过LunaUdpTool的UDP发送运动。

## 先做离线检查

在工程根目录：

```bash
python3 -m dynamic_handeye inspect
python3 -m dynamic_handeye capture \
  --plan configs/capture_plan.json \
  --path datasets/raw/ball_taught_20261018_visible_v1.json \
  --output datasets/raw/new_pilot.jsonl --stop-after 12
```

不加`--run`只检查计划和轨迹，不访问网络或控制器。默认9姿态包含yaw=-10/0/+10、pitch=-25/-20/-15，均在复制的控制器限位内；实际运行仍读在线限位。录制轨迹并不证明当前环境碰撞自由，仍需沿用原采集流程的现场路径核验。

## 相机与采集依赖

离线拟合依赖`requirements.txt`中的NumPy/OpenCV。Orin部署入口额外依赖原采集环境中的Paramiko，本地预览依赖GStreamer/系统插件；Orin需要原配置的Argus/VPI链。SDK二进制继续由厂商`00040701`单独提供，未复制或下载到本工程。

相机入口：

```bash
python3 acquisition/orin_rectified_stream/start.py --check
```

`--check`按原程序只做本地检查；`--restart`会通过SSH重新部署并重启Orin相机服务，不属于离线检查。网络默认沿用Luna的Orin 6.6.7.100、用户nvidia，控制器6.6.7.190；环境变量`LUNA_ORIN_IP/LUNA_ORIN_USER/LUNA_LOCAL_IFACE`可按真实现场设置。认证使用已有SSH或终端环境变量，不将口令写入工程。

## 明确启用实机采集

相机服务就绪、左右身份/标定哈希一致，且臂位于轨迹首个安全点附近后：

```bash
python3 -m dynamic_handeye capture \
  --plan configs/capture_plan.json \
  --path datasets/raw/ball_taught_20261018_visible_v1.json \
  --output datasets/raw/new_pilot.jsonl \
  --sdk-root ../00040701 \
  --latest acquisition/runtime/new_sh5_yellow/latest_pose.json \
  --stop-after 12 --run
```

入口保留原流程的终端`RUN`确认。它进入Luna SDK后核对SDK/控制器版本、在线限位、IDLE、身体姿态、示教路径邻近性和视觉观测；随后按“每个球位遍历9头姿，再换臂位”执行。50帧窗口保留当前源实现已支持的逐帧像素、球心、反馈与质量记录。若相机/SDK环境不匹配，应先完成现场联调，不以手动删门槛方式绕过。

每个输出附`.jsonl.plan.json`绑定头姿计划、轨迹及相机/MDH/MJCF哈希。中断续采必须使用相同绑定；不能对旧三头姿文件套新九头姿计划。原程序逐行追加记录，拒绝点保持`capture_valid=false`，不根据拟合误差删点。

采后运行`dynamic_handeye import`和`fit`。当前导入器对历史缺逐帧窗口的日志保留legacy标记；新日志有窗口时核查50个唯一帧与臂/头窗口稳定性。所有模型都默认未批准，真实泛化与动态验收必须另报。

## 首次实验安排

先采12臂位×9头姿，每组合50组有效双目；确认机械轴/角度反馈、光心杆臂与跨头姿漂移，再扩展到20–40个分散球位及新会话。为未见角度测试预留训练网格之间的新头姿，不只重复9个训练角。

运动中采集需要真实左右曝光时间、反馈历史、共享时钟或已校验映射、时间不确定度、运动速度与读出/模糊信息。现有SH5日志的出队/接收时间不具备这些含义；本版交叉采集仍是**换姿后停稳**，不将其标为运动中曝光同步数据。
