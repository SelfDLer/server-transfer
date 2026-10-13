# Server Transfer

专门用于把本地文件通过 GitHub 传送到服务器。

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
