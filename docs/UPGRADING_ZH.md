# 更新程序并保留项目

将新程序放在独立目录，保留旧程序、原状态目录、照片与权重。浏览器界面名称统一为 `building-facade-engineering-agent`，内部命令模块保持 `facade_next`。

## 已使用 facade_next 的项目

先结束当前请求，避免两个服务同时写同一个状态库。用新版指定原状态目录即可查看已有项目：

```bash
python -m facade_next serve --state /absolute/path/to/original/runtime --port 8766 --ui-only
```

相同主机上的数据和权重路径保持原样。截图使用“已保存任务”读取冻结配置、评估和实际决策，无需重新训练或重新采用。备份状态时应在服务停止且 Worker 完成后复制整个状态目录，而非仅复制数据库。

## 状态格式兼容

仅直接读取兼容的 `facade_next` 状态目录。其他数据库或权重格式需核对来源和受支持的转换方式，重命名文件不能完成迁移。Windows 与 Ubuntu 之间不会自动重写冻结的绝对路径；已有任务的数据与权重路径应保持有效。

当前任务使用工程师选择的适应策略、独立 Val 与验收集，每个选中的 GPU 运行一个独立 Worker。数据格式和角色规则见 [数据准备](DATA_PREPARATION_ZH.md)。
