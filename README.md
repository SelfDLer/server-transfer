# Server Transfer

专门用于把本地文件通过 GitHub 传送到服务器。

## 当前文件

- `files/plot_nextqa_precision/`：[NExT-QA FSDP2 精度对比绘图](files/plot_nextqa_precision/README.md)，读取 verl console 日志，输出 reward 主图和耗时/显存表。

- `files/install_nextqa_a3_npu_manual.sh`：Ascend A3 / Qwen3-Omni 的 Conda 环境安装与验证清单。
- `files/repro_vision_wait/`：[TP=4 视觉编码器等待复现与诊断](files/repro_vision_wait/README.md)，直接使用已安装环境。

先修改脚本第 1 节的路径和安装配置，再在 Linux Bash 中逐节执行。
默认使用当前 shell 中的 `conda`，通过 `conda info --base` 查询其安装根目录，
并加载该目录下的初始化脚本。`echo "$CONDA_PREFIX"` 可查看当前激活环境的目录。
不要直接整体 `bash` 或 `source` 此文件。环境通过 `conda create -p` 创建；
安装后检查依赖、模块导入和 NPU 基础运算。原有依赖冲突说明仍保留。
`SOC_VERSION` 默认留空，由固定版本的 vLLM-Ascend 在构建时通过 `npu-smi` 自动识别；
识别失败会报错，也可在开头手动填写目标芯片型号。
Git、编译器、构建工具、音视频库、NUMA/RDMA 用户态库和 C++ 运行库优先安装到 Conda。
第 1 节集中配置 `CONDA_CHANNEL`、编译器版本及 `CONDA_PACKAGES`。
将 `VERL_OMNI_DIR` 改为你已有的 verl-omni 项目绝对路径，例如 `/mnt/share/你的目录/verl-omni`。
verl-omni 从该目录做 editable 安装，保留现有分支、提交和本地修改；其余四个源码仓库仍放在 `src/` 并固定版本。
依赖读取、导入路径检查和版本记录均使用 `VERL_OMNI_DIR`；第 4 节核对自有源码中已有的引擎 pin 文件。
该路径在第 3 节通过 `conda env config vars` 持久保存；迁移源码后需更新配置并重新执行 editable 安装。
Conda 默认使用[清华 conda-forge 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)，
创建环境时通过 [`--insecure`](https://docs.conda.io/projects/conda/en/latest/commands/create.html)
跳过 TLS 证书验证，仅作用于该次创建命令。
pip 的 [`PIP_TRUSTED_HOST`](https://pip.pypa.io/en/stable/cli/pip/#cmdoption-trusted-host)
在第 1 节集中配置，默认覆盖交大镜像及下载域名（`mirror.sjtu.edu.cn`、`s3.jcloud.sjtu.edu.cn`）、PyTorch 官方域名、华为 Ascend、清华和阿里镜像域名，对这些域名跳过证书验证。
PyTorch CPU 包默认从交大 `https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/` 安装。
`torch-npu` 的首次安装和最终版本对齐使用 `HUAWEI_PIP_INDEX_URL`，默认为华为云通用 PyPI 源
`https://mirrors.huaweicloud.com/repository/pypi/simple`，并保留交大 CPU 源作为额外索引。
其他 Ascend 包仍使用 `ASCEND_PIP_INDEX_URL` 指定的专用源。
第 6 节先从 `HUAWEI_PIP_INDEX_URL`（默认华为云通用 PyPI 源）安装并验证 `regex`，
vLLM 安装也显式使用该通用源，并以交大 CPU 源作为额外索引，避免仅从 PyTorch 专用源查找普通依赖。
所有 pip 安装命令（包括 torch 和 torch-npu）共用此设置，第 3 节通过 `conda env config vars` 保存。
更换镜像或遇到下载重定向到其他域名时，按实际下载地址调整列表，多个主机名用空格分隔。
创建命令优先采用当前 shell 的 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 和 `NO_PROXY`
（也支持小写，小写优先，显式空值同样生效）。只要其中任一变量存在，便在本次 Conda
进程内清空 `proxy_servers`，由下载后端读取 shell 设置；全部未设置时沿用 Conda 配置。
代理不写入 `.condarc` 或 `conda env config vars`，不会固定到创建后的环境。
代理包装使用 `python -c` 和主模块保护，避免多进程启动时重新加载 `<stdin>` 失败。
默认不安装 SoX 命令行程序及 Python 封装：当前视频问答安装路径不需要它们，
且本次服务器使用的清华 `linux-aarch64` 源报告缺少 `sox`。以后使用 Qwen3-TTS 时需另行补齐。
NPU 驱动、固件、CANN/ATB 和内核 RDMA 驱动使用主机安装。

`ASCEND_ROOT` 指向 Toolkit 目录，第 3 节加载其 `set_env.sh` 和 `../nnal/atb/set_env.sh`，然后通过
`conda env config vars set -p "$ENV_PREFIX"` 保存 Ascend 变量和库路径。
`LD_LIBRARY_PATH` 的第一项是当前环境的 `lib`，`PYTHONNOUSERSITE=1` 避免混入用户目录的包。
Python 包使用 editable 安装，`src/` 和自有 `VERL_OMNI_DIR` 都必须保留；不会把源码仓库拼进 `PYTHONPATH` 来掩盖安装问题。
此配置使用 Conda 自身的环境变量管理，正常 `conda activate`/`conda deactivate` 即可应用/恢复变量。
`PATH` 交给 Conda 管理，不持久化安装终端的完整 PATH。更换 CANN 路径后需重做第 3 节。

查看或修改环境变量：

```bash
conda env config vars list -p "$ENV_PREFIX"
conda env config vars set -p "$ENV_PREFIX" "LD_LIBRARY_PATH=$ENV_PREFIX/lib:<其他必要库目录>"
conda deactivate
conda activate "$ENV_PREFIX"
```

已安装的环境可在确认 `libstdcxx-ng` 存在并激活目标环境后，单独执行第 1、3 节更新配置。
若检测仍加载系统旧的 `libstdc++.so.6`，可设置 `PRELOAD_CONDA_LIBSTDCXX=1` 后重做第 3 节，
通过该环境的 `LD_PRELOAD` 指定 Conda 库；改回 `0` 并重做第 3 节会清除本脚本保存的预加载设置。
已运行的 Python/Jupyter 进程需要重启后才会使用新的动态库配置。

第 10 节生成 `$ENV_ROOT/check_environment.py`，在清理相关临时变量后通过 `conda run` 验证，
直接导入 `verl_omni` 并检查 `/proc/self/maps` 中实际加载的 `libstdc++.so.6`。
失败的完整 traceback、库路径和导入结果保存在 `$LOG_DIR/import-check.txt`；
另有 `pip-check.txt`、`pip-freeze.txt` 和 `conda-explicit.txt` 供排查依赖问题。

## 本地上传

把需要传送的文件或目录复制到 `files/`，然后在本仓库目录运行：

```powershell
git add -- files
git diff --cached --stat
git commit -m "Add files for server"
git push origin main
```

提交前检查文件列表。此仓库为公开仓库，不要放入密码、令牌、私钥或其他私密文件。
`.gitignore` 会排除部分常见的凭据与缓存，但不能替代检查。

## 服务器下载

首次下载：

```bash
git clone --depth 1 https://github.com/SelfDLer/server-transfer.git
cd server-transfer
ls files
```

以后更新，在服务器的仓库目录运行：

```bash
git pull --ff-only
```

服务器上需要修改或运行的文件建议复制到仓库外，避免后续拉取冲突。
文件直接存入 Git，不依赖 Git LFS 或其他下载服务。大型文件可先压缩或拆分后再提交；
频繁更新的大型二进制文件会使仓库历史持续增大。
