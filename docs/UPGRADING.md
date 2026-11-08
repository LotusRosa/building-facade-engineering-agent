# 更新程序并保留项目

将新程序放在独立目录，保留旧程序、原状态目录、照片与权重。浏览器界面名称统一为 `building-facade-engineering-agent`，内部命令模块保持 `facade_next`。

## 已使用 facade_next 的项目

先结束当前请求，避免两个服务同时写同一个状态库。用新版指定原状态目录即可查看已有项目：

```bash
python -m facade_next serve --state /absolute/path/to/original/runtime --port 8766 --ui-only
```

相同主机上的数据和权重路径保持原样。截图使用“已保存任务”读取冻结配置、评估和实际决策，无需重新训练或重新采用。备份状态时应在服务停止且 Worker 完成后复制整个状态目录，而非仅复制数据库。

## 更早的 facade_agent 程序

公开仓库旧实现使用 `facade_agent`/`facade_training_worker`、不同数据库与模型格式。它的数据库不能直接当作 `facade_next` 状态库使用；这次更新不提供旧数据库的自动迁移。

旧实现仍可从 [原仓库提交](https://github.com/LotusRosa/building-facade-engineering-agent/tree/dcd90cb551ec3e59b61e8188d998a898437a709b) 获取。若继续旧项目，使用原实现；若使用当前流程，新建独立项目并重新导入已划分数据。权重导入需匹配受支持的合同和预处理，不能只改文件名。Windows 和 Ubuntu 之间也不要直接迁移冻结了绝对路径的任务。

当前流程不再要求失败片段聚类与专家审核作为训练前置步骤，策略和输入由工程师根据任务选择。旧界面截图不代表当前界面。
