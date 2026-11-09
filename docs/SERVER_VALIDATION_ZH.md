# 目标主机验证

先查看 [支持环境](../SUPPORTED_ENVIRONMENTS.md)。首次使用可直接打开界面配置共享环境；已有项目的查看步骤见 [升级说明](UPGRADING.md)。

```bash
python scripts/check_release.py
python -m unittest discover -s tests
python -m facade_next serve --state runtime_validation --port 8765 --ui-only
```

上述命令不会训练模型。确认目标主机环境后，使用环境中对应的 Python 显式检查 CUDA/BF16：

```bash
/path/to/environment/python -m facade_next gpu-check --execute
```

使用新验证项目，提交划分好的 Train/Val/S0，首轮选择 Core 和一个主干。确认具体数据、种子及参数后再启动。初始建模从 ImageNet 预训练开始；缺缓存会提示准备，不会改成随机初始化。

完成后查看初始评价，填写理由并确认采用，随后提交新增 Train/Current Gate 进行适应轮。选择 Standard 或任务所需策略，核对父模型、Val 和累计验收集。一个 Worker 对应一张 GPU，按需启用队列；不要在另一个服务上重复运行相同状态库。

检查以下实际结果：训练/校准/验收角色互斥；初始按 Val 选择检查点；适应按协议预算更新后校准；父模型和候选在相同验收图片上比较；采用或保留与实际版本一致；已完成 Gate 仅在人工决定后累计。失败时保留日志，不修改冻结收据掩盖失败。

已有 `facade_next` 任务可直接查看保存结果。更早的 768 裁剪模型不能当作当前 1280 整图模型自动导入；受支持模型需由工程师提供其原训练、Val 和验收角色证据，通过显式 `import-model --help` 中的参数转换。完整 GPU/策略矩阵验证属于独立验证工作，不在界面检查中宣称完成。
