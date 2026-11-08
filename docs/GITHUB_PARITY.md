# 与原公开仓库的对应关系

对照提交：[dcd90cb551ec3e59b61e8188d998a898437a709b](https://github.com/LotusRosa/building-facade-engineering-agent/tree/dcd90cb551ec3e59b61e8188d998a898437a709b)。原始文件哈希清单保留在 `GITHUB_BASELINE_FILES.json`。

| 原功能或文件 | 当前对应 |
|---|---|
| README、环境说明 | 中英文 README、SUPPORTED_ENVIRONMENTS、共享环境向导 |
| Windows/Ubuntu 启动和 GPU 配置脚本 | 同名脚本保留；新增双击入口和浏览器首次配置 |
| 独立训练 Worker | facade_next/worker.py、gpu.py、vendor、requirements.worker.txt、worker_manifest.json |
| 模型/数据/结果协议 | contracts.py、presets.json、任务与评估收据及冻结角色 |
| 图片与标注导入、模板 | data.py、intake.py、batch_intake.py、templates 中 CSV/XLSX |
| Hugging Face 公开数据 | 固定提交、对应角色清单与 prepare_bfd_ml_4k.py 转换工具 |
| 聊天、任务面板、项目状态 | 原聊天式布局、项目侧栏、SQLite 和独立队列 |
| 比较、模型采纳、回滚 | 同轮父模型/候选评估、人工理由与确认、版本记录 |
| 已有项目查看 | facade_next 同主机状态目录读取；旧 facade_agent 不自动迁移 |
| 必须先聚类/审核失败片段 | 当前流程不再作为训练前置条件 |
| 许可证、作者、引用、第三方说明 | 原 LICENSE 保留，CITATION 与当前软件元数据同步 |
| 原界面截图 | 留在历史提交中，不作为当前界面展示 |

完整旧实现保留在 Git 历史。本次通过正常追加提交更新当前默认分支，不重写旧提交。模型训练核心与最近已核对的实现保持一致；发布元数据、界面与数据配套说明已整理。
