# 读取已有实验并导出界面

使用新版 `building-facade-engineering-agent` 目录，并保留原项目所在主机的状态库、数据和权重路径。先执行只读检查：

```bash
python scripts/check_release.py
python scripts/check_saved_case.py --state /absolute/path/to/original/runtime
```

脚本优先寻找已完成的 ConvNeXt-Tiny、seed 20260921、Standard 适应任务。存在多个对应任务时，以输出记录为准，可指定 `--round round_ID`。找不到时先核对状态目录，不创建替代实验记录。

```bash
python -m facade_next serve --state /absolute/path/to/original/runtime --port 8766 --ui-only
```

在 Agent 主机浏览器打开 <http://127.0.0.1:8766>。选择原项目 → “已保存任务”，选定同一轮次和同一候选，再分别查看任务配置、候选评估和版本记录。

面板从保存的任务快照、评估收据和决策读取信息，不启动训练或重新采用。缺失数据保留“未记录”。导出按钮生成三份原生 2600 像素宽 PNG：

- `agent_task_configuration.png`
- `agent_candidate_evaluation.png`
- `agent_version_record.png`

不要用空项目或开发测试数据替代真实实验截图。当前发布的界面导出路径仍需在目标服务器核对，软件包没有附带新的论文实验截图。
