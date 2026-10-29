# 2026-10-29新采集数据

shared_fast_grid_20261029/是真实采集目录。原datasets/raw/shared_fast_grid_20261029已改为指向此处的相对目录链接，保持正在运行的回放、日志和预览路径兼容。迁移使用同一文件系统的原子目录/链接换位，已核对目录inode不变、旧入口可访问；未复制或重写正在采集的数据。

每轮head_XX.jsonl对应head_XX.session.json中的头姿；完整计划见session.json，当前进度见status.json。完成后head_XX.completed.json记录条数、有效率和SHA256。未完成的文件不能当作完整训练或验收集。analysis_snapshot_head00/为用于诊断的固定前缀副本，不随采集更新。
