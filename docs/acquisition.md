# 新Luna数据采集入口

当前批次按用户要求改为**pitch=0、yaw左右偏转**，使用[专用采集指南](pitch0_yaw_capture.md)和`configs/capture_pitch0_yaw_plan.json`。下文九姿交叉网格保留为后续二维覆盖方案，不是本轮默认执行指令。

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

采后先用`dynamic_handeye import`校验日志。`dynamic_handeye fit`是固定名义头根的6参数入口；有效12参数＋躯干补偿研究脚本的新数据输入仍需适配，不能混用为同一算法。当前导入器对历史缺逐帧窗口的日志保留legacy标记；新日志有窗口时核查50个唯一帧与臂/头窗口稳定性。所有模型都默认未批准，真实泛化与动态验收必须另报。

## 首次实验安排

先采12臂位×9头姿，每组合50组有效双目；确认机械轴/角度反馈、光心杆臂与跨头姿漂移，再扩展到20–40个分散球位及新会话。为未见角度测试预留训练网格之间的新头姿，不只重复9个训练角。

运动中采集需要真实左右曝光时间、反馈历史、共享时钟或已校验映射、时间不确定度、运动速度与读出/模糊信息。现有SH5日志的出队/接收时间不具备这些含义；本版交叉采集仍是**换姿后停稳**，不将其标为运动中曝光同步数据。

2026-10-27核对的示教→固定头姿逆序回放→交叉采集操作顺序，以及独立验证六头姿计划，见[下一轮采集指南](下一轮拖动示教与交叉头姿采集_20261027.md)。

## 2026-10-28：正式采集采用Pro式双目配对和球心门限

正式发送器 `acquisition/orin_rectified_stream/sender.py` 已改为同一GStreamer管线、共享SystemClock；使用有界队列按绝对源时钟（base_time+PTS）和到达时间双10ms门限配对。每帧仅消费一次；未消费的旧配对覆盖计数保留。原始PTS仍作为逐眼源身份，新增source_clock_ns、pair_source_clock_skew_ms和pairing_stats。该时间不是已验证曝光时间。

黄球检测仍用Luna参数与同一亚像素椭圆检测器。先按视差挑选候选，再执行40mm球体投影修正，最后检查修正后纵向差≤0.5px；不会把修正前纵向差作为0.5px硬门限。保留原椭圆中心与raw_stereo_pixels、raw_epipolar_dv_px、raw_disparity_px，同时输出corrected_stereo_pixels。下游原椭圆中心仍供原球体几何函数使用，避免重复修正同一组像素。

接收器和实机采集入口均检查新pairing_policy、修正像素域和源时间门限，旧发送器记录不能混充新版PASS。启用时须同时部署sender、stereo_pairing.py、subpixel_ball.py、measurement_contract.py、drag_capture.py和viewer更新；start.py已增加配对模块上传，远端球体几何继续复用Lunageometry.py。sources.json保留上游哈希并记录本地修改后的哈希。

本次为本地实现及离线验证，未重启现场相机、未下发机器人运动、未推送仓库。改变配对方式和球体修正门限不保证消除新机约5.85px的几何偏差；仍需现场极线质量验收。
