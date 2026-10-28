# 2026-10-28 本机Luna V2.0相机与数据迁移

新标定来源：`stereo_calibration/data/raw_chessboard_20261028_101409`，60对/57对有效。正式配置为 `configs/camera/luna_raw_fisheye_stereo.json` 和 `luna_rectified_stereo.json`，部署副本保持字节一致。测量矩阵按1280×992缩放，双目R/T重新生成，质量门槛不变。

旧相机JSON已原位替换；过时ROS标定YAML、旧参数手眼观测和处理结果已删除，文件名/哈希见 `deleted_files.json`。历史研究报告保留，但资产哈希已失效，禁止作为当前模型。新标定尚无独立验证。

108点右臂关节轨迹保留；camera_rebinding记录旧哈希与新哈希，关节样本未修改。旧相机视觉观测不复用；轨迹回放仍验证在线限位、身体姿态、SDK版本、5%速度和当前检测。旧101mm球偏置仅保留为初始化值，不能作为本机新pF的测量结论。

当前流程：先 `datasets/raw/luna_v2_newcal_reference_20261028.jsonl` 固定[0,0]逆序回放返回首点，再 `datasets/raw/luna_v2_newcal_pitch0_yaw_20261028.jsonl` 正序采三个yaw，各50帧。有效/无效逐点保留，不把失败点计入成功数。

软件检查：源文件哈希验证138项；主测试28项中22通过、6项因旧数据已删除或旧拟合模型失效而显式跳过；双目标定模块10项通过。历史数据专属回归测试不能代替本机新数据验收。

基准回放已完成108点：72有效、36无效。多yaw采集PID2612676正在后台运行（工具会话55314），日志 `/tmp/luna_newcal_multiyaw_20261028.log`。全量目标324组合；结束后摘要写入同目录 `multiyaw_summary.json` 和数据旁 `.completed.json`，complete字段区分正常完成与提前退出。中断采集可向该采集进程发送SIGINT，原程序finally将机械臂和头部恢复IDLE；勿强制SIGKILL。
