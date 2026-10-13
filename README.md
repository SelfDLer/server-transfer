# Server Transfer

专门用于把本地文件通过 GitHub 传送到服务器。

## 当前文件

- `files/install_nextqa_a3_npu_manual.sh`：Ascend A3 / Qwen3-Omni 的 Conda 环境安装与验证清单。

先修改脚本第 1 节的路径、Conda 安装目录和安装配置，再在 Linux Bash 中逐节执行。
不要直接整体 `bash` 或 `source` 此文件。环境通过 `conda create -p` 创建；
安装后检查依赖、模块导入和 NPU 基础运算。原有依赖冲突说明仍保留。
`SOC_VERSION` 默认留空，由固定版本的 vLLM-Ascend 在构建时通过 `npu-smi` 自动识别；
识别失败会报错，也可在开头手动填写目标芯片型号。
系统依赖默认自动选择 apt-get、dnf 或 yum，普通用户自动使用 sudo。
第 1 节可指定 `SYSTEM_PACKAGE_MANAGER=apt/dnf/yum/skip`，并调整对应包列表；
RPM 系统的音视频软件源需按机器实际情况配置。
Ascend 只配置 `ASCEND_ROOT` 基础路径，再从该路径下加载 Toolkit 和 ATB 的 `set_env.sh`；
源码按固定上游版本安装。

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
