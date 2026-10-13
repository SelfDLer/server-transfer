# Server Transfer

专门用于把本地文件通过 GitHub 传送到服务器。

## 当前文件

- `files/install_nextqa_a3_npu_manual.sh`：Ascend A3 / Qwen3-Omni 的 Conda 环境安装与验证清单。

先修改脚本第 1 节的路径和安装配置，再在 Linux Bash 中逐节执行。
默认使用当前 shell 中的 `conda`，通过 `conda info --base` 查询其安装根目录，
并加载该目录下的初始化脚本。`echo "$CONDA_PREFIX"` 可查看当前激活环境的目录。
不要直接整体 `bash` 或 `source` 此文件。环境通过 `conda create -p` 创建；
安装后检查依赖、模块导入和 NPU 基础运算。原有依赖冲突说明仍保留。
`SOC_VERSION` 默认留空，由固定版本的 vLLM-Ascend 在构建时通过 `npu-smi` 自动识别；
识别失败会报错，也可在开头手动填写目标芯片型号。
Git、编译器、构建工具、音视频库、NUMA/RDMA 用户态库和 C++ 运行库优先安装到 Conda。
第 1 节集中配置 `CONDA_CHANNEL`、编译器版本及 `CONDA_PACKAGES`；源码按固定上游版本安装。
NPU 驱动、固件、CANN/ATB 和内核 RDMA 驱动使用主机安装。

第 3 节从 `ASCEND_ROOT` 加载 Toolkit/ATB 的 `set_env.sh`，然后通过
`conda env config vars set -p "$ENV_PREFIX"` 保存 Ascend 变量和库路径。
`LD_LIBRARY_PATH` 的第一项是当前环境的 `lib`，`PYTHONNOUSERSITE=1` 避免混入用户目录的包。
Python 包使用 editable 安装，`src/` 必须保留；不会把源码仓库拼进 `PYTHONPATH` 来掩盖安装问题。
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
