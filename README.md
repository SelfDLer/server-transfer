# Server Transfer

专门用于把本地文件通过 GitHub 传送到服务器。

## 当前文件

- `files/install_nextqa_a3_npu_manual.sh`：Ascend A3 / Qwen3-Omni 的 Conda 环境搭建手动清单，附 NExT-QA 数据与训练示例。

先修改脚本第 1 节的路径、Conda 安装目录、硬件与训练配置，再在 Linux Bash 中逐节执行。
不要直接整体 `bash` 或 `source` 此文件。环境通过 `conda create -p` 创建；
新终端运行时执行第 1 节和第 12 节。原有依赖冲突说明仍保留，安装后需要检查实际结果。
系统依赖默认自动选择 apt-get、dnf 或 yum，普通用户自动使用 sudo。
第 1 节可指定 `SYSTEM_PACKAGE_MANAGER=apt/dnf/yum/skip`，并调整对应包列表；
RPM 系统的音视频软件源需按机器实际情况配置。jemalloc 路径支持自动查找或手动指定。
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
