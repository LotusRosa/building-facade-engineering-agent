# BFD-ML-4K 与当前流程配套

数据位于 [LotusRosa/BFD-ML-4K](https://huggingface.co/datasets/LotusRosa/BFD-ML-4K)。软件不捆绑图片。下载和使用须遵守数据仓库许可证。

当前工具固定使用数据提交 `81aab6254fe29bdd23d49a36945b1aa2c5ce44c1`。已逐项核对该提交的 4,000 个图片 ID 和四个标签，与本次实验冻结清单一致。公开数据原有划分与当前论文划分不同，不能把文件夹直接改名后视为同一协议。

| 当前固定角色 | 图片数 | Agent 目录 |
|---|---:|---|
| core_train | 2000 | initial/train |
| validation | 400 | initial/val |
| test0 | 400 | initial/safety |
| diagnosis_a | 300 | round_A/train |
| test_a | 100 | round_A/gate |
| diagnosis_b | 300 | round_B/train |
| test_b | 100 | round_B/gate |
| final_test | 400 | 仅论文端点评价，不导入常规流程 |

`protocols/bfd_ml_4k/roles.csv` 记录已冻结的角色、公开下载路径、标签、匿名分组和公开图片哈希；`protocol.json` 绑定数据提交及清单哈希。没有重新随机划分。原公开划分的历史含义保持不变。

## 下载并准备初始与 A 轮

新电脑请先按 [完整 Windows／Ubuntu 部署命令](FIRST_INSTALL.md) 安装：教程会创建独立下载工具环境并安装 `huggingface_hub==2.4.0`，无需预先具备 `hf` 命令。公开下载无需 Token。准备命令在 Agent 仓库根目录运行。下载原图加默认初始／A 轮副本约需 28 GB，另留环境与训练缓存空间。

使用 [Hugging Face 官方 CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) 下载固定提交。已有 `hf` 命令时：

```bash
hf download LotusRosa/BFD-ML-4K data/ metadata/all_samples.csv metadata/dataset_manifest.json LICENSE CITATION.cff --repo-type dataset \
  --revision 81aab6254fe29bdd23d49a36945b1aa2c5ce44c1 \
  --local-dir /absolute/path/BFD-ML-4K

python scripts/prepare_bfd_ml_4k.py \
  --dataset-root /absolute/path/BFD-ML-4K \
  --output /absolute/path/FacadeAgentData
```

Windows PowerShell 中可将每条命令写成一行，并使用 `D:\Datasets\BFD-ML-4K` 等本机路径。`hf` 是下载辅助工具，不是 Agent 界面的运行依赖。

转换工具读取既定清单，校验元数据和所选图片的 SHA-256，复制到新目录并生成 `labels.csv`，保留图像 ID、标签与角色分组。不会修改下载源，不覆盖已有输出。默认生成初始与 A 轮；同时需要 B 轮时追加 `--batch initial --batch round_A --batch round_B`。Final 不被复制。复制失败时保留临时目录用于排查，不把不完整结果发布为成功目录。

准备完成后，在 Agent 中依次提交 `FacadeAgentData/initial` 和 `FacadeAgentData/round_A`，按导入卡片核对数量。这是读取已有固定协议；普通用户的数据仍由用户自行划分。

本地元数据核对没有下载或解码全部图片。转换工具会在用户已下载的数据上逐张校验真实文件；只有成功收据才能说明所选输出图片通过哈希检查。

下载命令仅选择固定提交中的 `data/`、必要元数据及许可证／引用文件，不下载其中的历史 README、工具或说明图片。操作说明以当前 Agent 仓库为准。
