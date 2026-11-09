# 一次配置，多项目共用

先使用系统现有的 Python 3.11+ 打开 Agent。界面本身不需要 PyTorch；训练环境可在浏览器首次向导或“设置 → 共享训练环境”中配置。

1. 选择“在新文件夹创建环境”，指定主机上的空文件夹；或选择“接入已有环境”，填写包含 `.venv` 的目录或环境目录。
2. 指定训练 GPU 编号，例如 `0` 或 `1`。双卡并行才填写 `0,1`，每卡一个独立 Worker。
3. 勾选需要准备的 ImageNet 主干权重，点击创建或检查。日志显示依赖安装、权重字节进度、校验结果及失败原因。
4. 出现“共享环境已就绪”后返回聊天，各项目使用此环境。项目数据和模型输出分别保存在项目状态中。

例如，Ubuntu 环境文件夹可为 `/home/test/FacadeAgentEnvironment`，Windows 可为 `D:\FacadeAgentEnvironment`。环境目录、项目状态目录和数据目录分别指定。

向导创建 `.venv`，使用 CUDA 12.8 的 PyTorch/TorchVision 组合，安装 NumPy/Pillow，并检查 CUDA 与原生 BF16。当前不会自动挑选所有 GPU/驱动版本对应的安装方案，也不安装 NVIDIA 驱动。详细范围见 [支持环境](../SUPPORTED_ENVIRONMENTS.md)。

下载中断时查看日志，恢复网络后重试同一环境目录；已完成依赖和校验通过的缓存会复用。不会为失败任务自动重训。切换共享环境前先完成队列中的训练。

主机配置默认保存在用户目录 `.facade-agent/environment.json`。`--environment-config /absolute/path/environment.json` 可指定独立配置。只看历史或预览时用 `--ui-only`，避免启动已保存配置中的 Worker。

新 Ubuntu 主机创建环境前需安装所选 Python 版本对应的 `python3-venv`；Windows 使用带 venv／ensurepip 的 64 位 Python。固定 CUDA 安装组合以 Python 3.12 为参考，见 [首次部署](FIRST_INSTALL.md)。当前交互 Worker 初始训练为 BF16，与论文精度差异见 [对应关系](MANUSCRIPT_CORRESPONDENCE.md)。
