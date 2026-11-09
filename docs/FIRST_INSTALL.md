# First installation: download data and run the Agent

These steps start a new project. Keep an existing project's runtime, data and checkpoints in place when [upgrading](UPGRADING.md).

The browser UI needs only Python 3.11+; Python 3.12 is the reference. The optional Hugging Face download tool has its own small environment. The browser wizard creates the separate shared GPU training environment. Public dataset downloads do not require an upload token or an LLM key.

## Ubuntu 24.04

Install the host tools if missing. Other Ubuntu releases must also provide Python 3.11 or newer.

```bash
sudo apt update
sudo apt install git python3 python3-venv
python3 --version
git clone https://github.com/LotusRosa/building-facade-engineering-agent.git
cd building-facade-engineering-agent
python3 scripts/check_release.py
```

To use your own already-split data, skip the next block. To use the public corpus, run this block from the repository root. The output data folder must not already exist.

```bash
python3 -m venv .download-venv
.download-venv/bin/python -m pip install huggingface_hub==2.4.0
.download-venv/bin/hf download LotusRosa/BFD-ML-4K data/ metadata/all_samples.csv metadata/dataset_manifest.json LICENSE CITATION.cff --repo-type dataset --revision 81aab6254fe29bdd23d49a36945b1aa2c5ce44c1 --local-dir "$HOME/FacadeAgentDownloads/BFD-ML-4K"
python3 scripts/prepare_bfd_ml_4k.py --dataset-root "$HOME/FacadeAgentDownloads/BFD-ML-4K" --output "$HOME/FacadeAgentData"
```

Start the local service:

```bash
python3 -m facade_next serve --state runtime --port 8765
```

## Windows PowerShell

Install 64-bit [Python 3.12](https://www.python.org/downloads/windows/) and [Git for Windows](https://git-scm.com/download/win) if missing, then open a new PowerShell window. The Python launcher must recognize `py -3.12`.

```powershell
py -3.12 --version
git clone https://github.com/LotusRosa/building-facade-engineering-agent.git
cd building-facade-engineering-agent
py -3.12 scripts/check_release.py
```

For the public corpus, run the following from the repository root. These commands do not require activating a PowerShell script. Skip them when using your own data; choose a new output path if `FacadeAgentData` already exists.

```powershell
py -3.12 -m venv .download-venv
.\.download-venv\Scripts\python.exe -m pip install huggingface_hub==2.4.0
.\.download-venv\Scripts\hf.exe download LotusRosa/BFD-ML-4K data/ metadata/all_samples.csv metadata/dataset_manifest.json LICENSE CITATION.cff --repo-type dataset --revision 81aab6254fe29bdd23d49a36945b1aa2c5ce44c1 --local-dir "$env:USERPROFILE\FacadeAgentDownloads\BFD-ML-4K"
py -3.12 scripts/prepare_bfd_ml_4k.py --dataset-root "$env:USERPROFILE\FacadeAgentDownloads\BFD-ML-4K" --output "$env:USERPROFILE\FacadeAgentData"
py -3.12 -m facade_next serve --state runtime --port 8765
```

## Shared environment and the first project

1. Open <http://127.0.0.1:8765> on the computer running the service. Keep its terminal open; closing it stops the service. Use Ctrl+C to stop it and the same state path to resume later.
2. In the first-run wizard, choose an empty environment folder and a GPU index. The NVIDIA driver must already be installed. The wizard installs the pinned CUDA 12.8 PyTorch pair, checks native BF16, and downloads the selected ImageNet weights. See [environment setup](FIRST_RUN_ENVIRONMENT.md).
3. Wait for **Shared environment ready**. A failed download can be retried in the same unfinished environment folder. CPU-only hosts can view the UI, but cannot run the current training Worker.
4. Optionally connect an LLM using **Connect model**. An LLM key is not required for deterministic project/task operation.
5. Create a project in chat. Supply the absolute `FacadeAgentData/initial` folder path on this host, then confirm Train 2000, Val 400 and S0 400. Select **Core · Initial modeling**, a backbone and seed; review and confirm the task.
6. After training, inspect evaluation and separately confirm adoption with a reason. Then start an adaptation round and supply `FacadeAgentData/round_A`; confirm new Train 300 and Current Gate 100, while reusing historical Train and Val. **Standard · Adaptation** is FT. Other strategies are optional selections.

All projects reuse the selected training environment. Data, environment and runtime are separate folders. The prepared directory names and role table are in [BFD-ML-4K instructions](BFD_ML_4K.md). Ordinary tasks do not need Final. See [paper correspondence](MANUSCRIPT_CORRESPONDENCE.md) before treating this quickstart as exact reproduction of every paper experiment.

## Storage, network and remote access

The 4,000 original JPEGs total about 15.4 GB (decimal). The default initial/A preparation copies another 12.4 GB: approximately 28 GB for image files alone. Keep additional free space for download caches, GPU dependencies, decoded training caches, weights and checkpoints. The software ZIP does not contain these files.

If a corpus download is interrupted, repeat the same `hf download` command and destination. The preparation tool verifies actual image checksums and refuses an existing output folder. A missing `hf` command means the separate download-tool installation did not finish; the commands above use its explicit executable path.

If the browser runs on another computer, `127.0.0.1` refers to that browser's computer. For SSH access, run `ssh -N -L 8765:127.0.0.1:8765 user@server` on that computer and open the local URL there. Keep the service bound to loopback.

The documentation audit checked anonymous image download endpoints with HEAD requests, package availability and model-free startup. It did not redownload the complete corpus or reinstall GPU dependencies. Full GPU acceptance remains a [target-host check](SERVER_VALIDATION.md).

The download commands deliberately select only `data/`, required metadata and license/citation files from the immutable image revision. Its historical README, tools and documentation figures are not downloaded. Current instructions remain in this Agent checkout.
