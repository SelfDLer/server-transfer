#!/usr/bin/env bash
# Qwen3-Omni / Ascend A3：Conda 环境手动安装与验证清单。
# 在 Linux Bash 中逐节复制执行；任一步失败时先解决问题，再继续下一节。
# 适用：Linux / Ascend A3，已安装 CANN 9.1.0、ATB 和 Conda。
# 工具、编译器和用户态库优先安装到 Conda；驱动、固件、CANN/ATB 使用主机安装。
# 驱动、NPU 设备、CANN 和 ATB 必须已就绪；本文件不安装驱动、CANN 或 Conda。
# 使用全新的独立环境执行安装。

# ===== 1. 集中配置（换机器、目录或环境时先修改本节） =====
set -o pipefail

# 常用本地目录调整
export ENV_ROOT=/path/to/your/env-root
export VERL_OMNI_DIR=/path/to/your/verl-omni
export ASCEND_ROOT=/usr/local/Ascend/ascend-toolkit

# 基础目录
export SRC_DIR="$ENV_ROOT/src"
export ENV_PREFIX="$ENV_ROOT/conda"
export LOG_DIR="$ENV_ROOT/logs"
export PYTHON_VERSION=3.12
export CONSTRAINTS_FILE="$ENV_ROOT/npu-constraints.txt"
export REQUIREMENTS_FILE="$ENV_ROOT/environment-requirements.txt"

# 网络配置
export CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
export ASCEND_PIP_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
export HUAWEI_PIP_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
export PYTORCH_CPU_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu/
export MOONCAKE_INDEX_URL=https://mirrors.aliyun.com/pypi/web/simple
export PIP_TRUSTED_HOST="mirror.sjtu.edu.cn s3.jcloud.sjtu.edu.cn download.pytorch.org download-r2.pytorch.org mirrors.huaweicloud.com mirrors.tuna.tsinghua.edu.cn mirrors.aliyun.com ascend.devcloud.huaweicloud.com"

# 基础组件
export GCC_VERSION=12
export CLANG_VERSION=15
CONDA_PACKAGES=(
  pip setuptools wheel packaging
  git wget curl ca-certificates cmake make ninja pkg-config unzip
  "gcc=$GCC_VERSION" "gxx=$GCC_VERSION" "clang=$CLANG_VERSION" "clangxx=$CLANG_VERSION"
  "libgcc-ng>=12" "libstdcxx-ng>=12"
  numactl libnuma rdma-core jemalloc libhiredis
  libxcb libgl libglib xorg-libsm xorg-libxext xorg-libxrender
  ffmpeg libsndfile
)
export PRELOAD_CONDA_LIBSTDCXX=0
export SOC_VERSION=""
export COMPILE_CUSTOM_KERNELS=1

# 固定上游源码版本；更换版本前重新核对上游兼容性。
# 源码版本。
export VLLM_REF=v0.30.0
export VLLM_ASCEND_REF=39fef3f8fe71e71a8b7a684ae0f62b85ff18661f
export VLLM_OMNI_REF=12e9280010afc41ce5c6a1f46900fc329cf08377
export VERL_REF=fefb080262e1c015a0ea05f958822a6a512dc795

# Python 包版本；约束文件和安装命令统一引用这里的配置。
export TORCH_VERSION=2.10.0+cpu
export TORCHVISION_VERSION=0.25.0+cpu
export TORCHAUDIO_VERSION=2.10.0+cpu
export TORCH_NPU_VERSION=2.10.0.post4
export TRANSFORMERS_VERSION=5.14.1
export ACCELERATE_VERSION=1.14.0
export PEFT_VERSION=0.19.1
export RAY_VERSION=2.56.1
export TRITON_ASCEND_VERSION=3.2.2
export MOONCAKE_VERSION=0.3.11.post1

# 安装与构建设置。
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1


# ===== 2. 用 Conda 按绝对路径创建独立环境 =====
conda_with_shell_proxy() {
  "$active_conda_base/bin/python" -c '
import os
import sys

if __name__ == "__main__":
    proxy_names = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
    # 小写优先；显式空值也代表 shell 的选择。统一大小写以兼容不同下载后端。
    use_shell_proxy = any(name in os.environ or name.upper() in os.environ for name in proxy_names)
    if use_shell_proxy:
        for name in proxy_names:
            value = os.environ.get(name, os.environ.get(name.upper()))
            if value is not None:
                os.environ[name] = os.environ[name.upper()] = value
        from conda.base.context import context
        # 仅在这个独立进程内覆盖配置，让下载后端读取 shell 的代理及 NO_PROXY。
        context.proxy_servers = {}

    from conda.cli.main import main
    sys.exit(main(*sys.argv[1:]))
' "$@"
}
if ! command -v conda >/dev/null 2>&1; then
  printf '当前 shell 找不到 conda，请先加载你要使用的 Conda，再执行本节。\n' >&2
elif ! active_conda_base="$(conda info --base)"; then
  printf '无法查询当前 Conda 的安装根目录，请检查 conda info --base。\n' >&2
elif [[ -z "$active_conda_base" || ! -r "$active_conda_base/etc/profile.d/conda.sh" ]]; then
  printf '当前 Conda 初始化脚本不可读取：%s/etc/profile.d/conda.sh\n' "$active_conda_base" >&2
elif [[ "$ENV_PREFIX" != /* ]]; then
  printf 'ENV_PREFIX 必须是绝对路径：%s\n' "$ENV_PREFIX" >&2
elif [[ -e "$ENV_PREFIX" ]]; then
  printf '目标环境已存在，请改用新目录：%s\n' "$ENV_PREFIX" >&2
else
  printf '使用当前 Conda：%s\n' "$active_conda_base"
  source "$active_conda_base/etc/profile.d/conda.sh" &&
  conda --version &&
  mkdir -p "$SRC_DIR" "$LOG_DIR" &&
  conda_with_shell_proxy create -y --insecure -p "$ENV_PREFIX" --override-channels -c "$CONDA_CHANNEL" \
    --strict-channel-priority "python=$PYTHON_VERSION" "${CONDA_PACKAGES[@]}" &&
  conda activate "$ENV_PREFIX" &&
  python - <<'PY'
import os
import pathlib
import sys
assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(os.environ['ENV_PREFIX']).resolve()
expected = tuple(map(int, os.environ['PYTHON_VERSION'].split('.')))
assert sys.version_info[:len(expected)] == expected, sys.version
print(sys.executable)
PY
fi
# 任一步失败或路径检查未通过时停止本节，不要执行后续安装章节。

# ===== 3. 将 Ascend 和动态库路径保存到 Conda 环境 =====
configure_conda_environment() (
  set -e
  if [[ "${CONDA_PREFIX:-}" != "$ENV_PREFIX" ]]; then
    printf '请先 conda activate "%s"，当前环境不匹配。\n' "$ENV_PREFIX" >&2
    return 1
  fi
  unset PYTHONPATH
  source "$ASCEND_ROOT/set_env.sh" || return 1
  source "$ASCEND_ROOT/../nnal/atb/set_env.sh" || return 1
  python - <<'PY'
import json
import os
import pathlib
import subprocess

prefix = pathlib.Path(os.environ['ENV_PREFIX']).resolve()
cxx_library = prefix / 'lib/libstdc++.so.6'
if not cxx_library.is_file():
    raise SystemExit(f'缺少 {cxx_library}，先安装 Conda 的 libstdcxx-ng。')

def prepend_path(first, previous):
    # 去重并去掉空项，避免空路径项意外指向当前工作目录。
    return ':'.join(dict.fromkeys([str(p) for p in first] + [p for p in previous.split(':') if p]))

# 保存官方 setup 导出的 Ascend 变量，不保存 pip 镜像等安装专用配置。
settings = {
    key: value for key, value in os.environ.items()
    if key.startswith(('ASCEND_', 'ATB_', 'CANN_', 'TBE_', 'TOOLCHAIN_'))
    and key != 'ASCEND_PIP_INDEX_URL'
}
settings.update({
    'LD_LIBRARY_PATH': prepend_path([prefix / 'lib'], os.environ.get('LD_LIBRARY_PATH', '')),
    'LIBRARY_PATH': prepend_path([prefix / 'lib'], os.environ.get('LIBRARY_PATH', '')),
    'CMAKE_PREFIX_PATH': prepend_path([prefix], os.environ.get('CMAKE_PREFIX_PATH', '')),
    'PKG_CONFIG_PATH': prepend_path([prefix / 'lib/pkgconfig', prefix / 'share/pkgconfig'],
                                    os.environ.get('PKG_CONFIG_PATH', '')),
    'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
    'PYTHONNOUSERSITE': '1',
    'PIP_TRUSTED_HOST': os.environ['PIP_TRUSTED_HOST'],
    'ENV_ROOT': os.environ['ENV_ROOT'],
    'SRC_DIR': os.environ['SRC_DIR'],
    'VERL_OMNI_DIR': os.environ['VERL_OMNI_DIR'],
    'LOG_DIR': os.environ['LOG_DIR'],
})
preload = os.environ.get('PRELOAD_CONDA_LIBSTDCXX', '0')
if preload not in ('0', '1'):
    raise SystemExit('PRELOAD_CONDA_LIBSTDCXX 必须为 0 或 1。')
if preload == '1':
    other_preloads = [p for p in os.environ.get('LD_PRELOAD', '').replace(':', ' ').split()
                     if 'libstdc++' not in p]
    settings['LD_PRELOAD'] = ':'.join([str(cxx_library), *other_preloads])

conda = os.environ['CONDA_EXE']
record = pathlib.Path(os.environ['ENV_ROOT']) / 'conda-env-vars.json'
# 关闭本脚本以前开启的预加载时，清除本脚本管理的旧设置。
previous = json.loads(record.read_text()) if record.exists() else {}
if preload == '0' and 'LD_PRELOAD' in previous:
    subprocess.run([conda, 'env', 'config', 'vars', 'unset', '-p', str(prefix), 'LD_PRELOAD'], check=True)
subprocess.run([conda, 'env', 'config', 'vars', 'set', '-p', str(prefix),
                *[f'{key}={value}' for key, value in sorted(settings.items())]], check=True)
record.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + '\n')
print('Conda 环境变量已保存：', record)
print('LD_LIBRARY_PATH:', settings['LD_LIBRARY_PATH'])
PY
)
configure_conda_environment && conda deactivate && conda activate "$ENV_PREFIX"
# 上一步失败时先处理，不要继续。重新激活后才会应用持久变量。
conda env config vars list -p "$ENV_PREFIX"

source "$ASCEND_ROOT/set_env.sh"
source "$ASCEND_ROOT/../nnal/atb/set_env.sh"
conda deactivate && conda activate "$ENV_PREFIX"

check_environment_tools() {
  local tool missing=0
  for tool in git gcc g++ clang cmake make ninja pkg-config ffmpeg ffprobe; do
    if ! command -v "$tool"; then missing=1; fi
  done
  return "$missing"
}
check_environment_tools
npu-smi info
ffmpeg -version
ffprobe -version
# CANN、ATB、NPU 驱动和内核 RDMA 驱动必须已在主机上可用。

# 在独立 Python 进程中检查动态链接器实际选择的 C++ 库。
python - <<'PY'
import ctypes
import os
import pathlib
ctypes.CDLL('libstdc++.so.6')
expected = (pathlib.Path(os.environ['CONDA_PREFIX']) / 'lib/libstdc++.so.6').resolve()
loaded = {pathlib.Path(line.split(maxsplit=5)[-1].removesuffix(' (deleted)')).resolve()
          for line in pathlib.Path('/proc/self/maps').read_text().splitlines()
          if '/libstdc++.so.6' in line}
print('Loaded libstdc++:', *sorted(map(str, loaded)), sep='\n')
assert loaded == {expected}, (
    '未使用 Conda C++ 库；检查 LD_PRELOAD/RPATH，必要时设置 PRELOAD_CONDA_LIBSTDCXX=1 后重做第 3 节',
    loaded, expected)
PY

# ===== 4. 检查自有 verl-omni 源码 =====
check_verl_omni_source() (
  printf '检查 VERL_OMNI_DIR=%q\n' "${VERL_OMNI_DIR:-}"
  if [[ -z "${VERL_OMNI_DIR:-}" ]]; then
    printf 'VERL_OMNI_DIR 未设置，请先执行第 1 节。\n' >&2
    return 1
  fi
  if [[ "$VERL_OMNI_DIR" != /* ]]; then
    printf 'VERL_OMNI_DIR 不是绝对路径：%q\n' "$VERL_OMNI_DIR" >&2
    return 1
  fi
  if [[ "$VERL_OMNI_DIR" == *$'\r'* || "$VERL_OMNI_DIR" == *$'\n'* ]]; then
    printf 'VERL_OMNI_DIR 含回车或换行，请重新输入路径：%q\n' "$VERL_OMNI_DIR" >&2
    return 1
  fi
  if [[ ! -d "$VERL_OMNI_DIR" ]]; then
    printf '目录不存在、符号链接失效，或当前用户无法访问：%q\n' "$VERL_OMNI_DIR" >&2
    return 1
  fi
  if [[ ! -f "$VERL_OMNI_DIR/pyproject.toml" ]]; then
    printf '目录可见，但缺少 pyproject.toml；请指定项目根目录：%q\n' "$VERL_OMNI_DIR" >&2
    return 1
  fi
  if [[ ! -r "$VERL_OMNI_DIR/pyproject.toml" ]]; then
    printf 'pyproject.toml 不可读取，请检查权限：%q\n' "$VERL_OMNI_DIR/pyproject.toml" >&2
    return 1
  fi
  if [[ ! -d "$VERL_OMNI_DIR/verl_omni" && ! -d "$VERL_OMNI_DIR/src/verl_omni" ]]; then
    printf '缺少 Python 包目录 verl_omni/ 或 src/verl_omni/，请核对源码内容：%q\n' "$VERL_OMNI_DIR" >&2
    return 1
  fi
  cd "$VERL_OMNI_DIR" || return 1
  # 有 pin 文件时核对配套版本；不对自有源码执行 clone/checkout/reset。
  for entry in "vllm_ascend:$VLLM_ASCEND_REF" "vllm_omni:$VLLM_OMNI_REF" "verl:$VERL_REF"; do
    pin_file=".github/${entry%%:*}_pin.txt"
    if [[ -f "$pin_file" ]]; then
      if [[ "$(tr -d '[:space:]' < "$pin_file")" != "${entry#*:}" ]]; then
        printf '版本配置与 %s 不一致，请核对第 1 节后再继续。\n' "$pin_file" >&2
        return 1
      fi
    else
      printf '未找到 %s，请手动核对该引擎的配套版本。\n' "$pin_file"
    fi
  done
)
check_verl_omni_source
# 检查失败时停止，修正路径或版本后再继续。

# ===== 5. 基础 Python / PyTorch / 通信依赖 =====
python -m pip install modelscope "ray[default]==$RAY_VERSION" 'protobuf>3.20.0'
python -m pip install --force-reinstall \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "$PYTORCH_CPU_INDEX_URL"
python -m pip install "torch-npu==$TORCH_NPU_VERSION" \
  --index-url "$HUAWEI_PIP_INDEX_URL" --extra-index-url "$PYTORCH_CPU_INDEX_URL"

# ===== 6. vLLM 0.30：NPU 路径使用 empty target，不安装 CUDA vLLM wheel =====
git clone --depth 1 --branch "$VLLM_REF" https://github.com/vllm-project/vllm.git "$SRC_DIR/vllm"
cd "$SRC_DIR/vllm"
python -m pip install regex --index-url "$HUAWEI_PIP_INDEX_URL" &&
python -c 'import regex; print(regex.__version__, regex.__file__)' &&
VLLM_TARGET_DEVICE=empty python -m pip install -e '.[audio]' \
  --index-url "$HUAWEI_PIP_INDEX_URL" --extra-index-url "$PYTORCH_CPU_INDEX_URL"
# 安装失败时先处理报错，不要继续后面的卸载和安装步骤。
python -m pip uninstall -y triton

# ===== 7. vLLM-Ascend：自动识别芯片并构建自定义算子 =====
git clone https://github.com/vllm-project/vllm-ascend.git "$SRC_DIR/vllm-ascend"
cd "$SRC_DIR/vllm-ascend"
git checkout --detach "$VLLM_ASCEND_REF"
git submodule update --init --recursive
VLLM_BATCH_INVARIANT=1 PIP_EXTRA_INDEX_URL="$ASCEND_PIP_INDEX_URL" \
  python -m pip install -e . \
  --extra-index-url "$ASCEND_PIP_INDEX_URL" --extra-index-url "$PYTORCH_CPU_INDEX_URL"
python -m pip uninstall -y triton triton-ascend
python -m pip install "triton-ascend==$TRITON_ASCEND_VERSION" concurrent-log-handler \
  --extra-index-url "$ASCEND_PIP_INDEX_URL"

# ===== 8. 按顺序安装 vLLM-Omni、verl、verl-omni =====
git clone https://github.com/vllm-project/vllm-omni.git "$SRC_DIR/vllm-omni"
cd "$SRC_DIR/vllm-omni"
git checkout --detach "$VLLM_OMNI_REF"
VLLM_OMNI_TARGET_DEVICE=npu python -m pip install -e . \
  --extra-index-url "$ASCEND_PIP_INDEX_URL" --extra-index-url "$PYTORCH_CPU_INDEX_URL"

git clone https://github.com/verl-project/verl.git "$SRC_DIR/verl"
cd "$SRC_DIR/verl"
git checkout --detach "$VERL_REF"
python -m pip install -e . \
  --extra-index-url "$ASCEND_PIP_INDEX_URL" --extra-index-url "$PYTORCH_CPU_INDEX_URL"

python -m pip install --no-deps -e "$VERL_OMNI_DIR"
python -m pip install Levenshtein

# ===== 9. 补全 Python 与音视频依赖，最终对齐版本 =====
# 前面安装 verl-omni 使用 --no-deps；从其元数据补齐普通依赖，但不重新解析/替换
# 已安装的三个源码引擎包。不是安装 .[gpu]/.[fa2]/.[omni]：后者要求 torchaudio>=2.11。
cat > "$CONSTRAINTS_FILE" <<EOF
torch==$TORCH_VERSION
torchvision==$TORCHVISION_VERSION
torchaudio==$TORCHAUDIO_VERSION
torch-npu==$TORCH_NPU_VERSION
transformers==$TRANSFORMERS_VERSION
accelerate==$ACCELERATE_VERSION
peft==$PEFT_VERSION
ray==$RAY_VERSION
triton-ascend==$TRITON_ASCEND_VERSION
EOF
python - <<'PY'
import os
import pathlib
import tomllib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
source = pathlib.Path(os.environ['VERL_OMNI_DIR'])
project = tomllib.loads((source / 'pyproject.toml').read_text())['project']
engine_packages = {'vllm', 'vllm-omni', 'verl'}
deps = [r for r in project['dependencies']
        if canonicalize_name(Requirement(r).name) not in engine_packages]
deps += project['optional-dependencies']['audio']
# torchvision 0.25 自带视频读取路径；不额外安装与 torch ABI 耦合的 TorchCodec。
deps += ['librosa>=0.10.2', 'soundfile>=0.13.1', 'av>=14',
         'qwen-vl-utils>=0.0.14', 'einops>=0.8.1']
pathlib.Path(os.environ['REQUIREMENTS_FILE']).write_text('\n'.join(deps) + '\n')
PY
python -m pip install -r "$REQUIREMENTS_FILE" \
  -c "$CONSTRAINTS_FILE" \
  --extra-index-url "$ASCEND_PIP_INDEX_URL" --extra-index-url "$PYTORCH_CPU_INDEX_URL"

# 最终 torch 对齐；--no-deps 保留前面已经安装的依赖。
python -m pip install --force-reinstall --no-deps \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "$PYTORCH_CPU_INDEX_URL"
python -m pip install --force-reinstall --no-deps "torch-npu==$TORCH_NPU_VERSION" \
  --index-url "$HUAWEI_PIP_INDEX_URL" --extra-index-url "$PYTORCH_CPU_INDEX_URL"
python -m pip install --upgrade onnxruntime -c "$CONSTRAINTS_FILE"
# 安装日志如出现其他依赖冲突，先记录并处理。

# ===== 10. 安装检查 =====
conda list -p "$ENV_PREFIX" --explicit > "$LOG_DIR/conda-explicit.txt"
python -m pip freeze > "$LOG_DIR/pip-freeze.txt"
python -m pip check > "$LOG_DIR/pip-check.txt" 2>&1
# 上一条的退出码非零表示检查失败；文件开头列出的冲突可能出现在这里。
cat "$LOG_DIR/pip-check.txt"
cat > "$ENV_ROOT/check_environment.py" <<'PY'
import importlib
import os
import pathlib
import sys
import traceback
from importlib.metadata import version

print('python:', sys.executable)
print('prefix:', sys.prefix)
print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', ''))
print('LD_PRELOAD:', os.environ.get('LD_PRELOAD', ''))
print('PYTHONPATH:', os.environ.get('PYTHONPATH', ''))
print('ASCEND_HOME_PATH:', os.environ.get('ASCEND_HOME_PATH', ''))
print('ATB_HOME_PATH:', os.environ.get('ATB_HOME_PATH', ''))
prefix = pathlib.Path(sys.prefix).resolve()
expected_cxx = (prefix / 'lib/libstdc++.so.6').resolve()

def report_cxx_library():
    loaded = {pathlib.Path(line.split(maxsplit=5)[-1].removesuffix(' (deleted)')).resolve()
              for line in pathlib.Path('/proc/self/maps').read_text().splitlines()
              if '/libstdc++.so.6' in line}
    print('Loaded libstdc++:', *sorted(map(str, loaded)), sep='\n')
    return loaded

# 首先直接导入 verl_omni，避免提前导入 torch 或用 ctypes 预载正确库后掩盖问题。
try:
    import verl_omni
except Exception:
    traceback.print_exc()
    report_cxx_library()
    raise SystemExit(1)
assert report_cxx_library() == {expected_cxx}, '实际加载的 libstdc++ 不属于当前 Conda 环境'
assert os.environ.get('PYTHONNOUSERSITE') == '1', 'Conda 持久变量未生效'

import torch
import torch_npu
print('torch:', torch.__version__, torch.__file__, 'git:', torch.version.git_version)
print('NPU available:', torch.npu.is_available(), 'count:', torch.npu.device_count())
# 从安装清单执行时核对配置版本；以后单独运行也能进行基础导入/动态库检查。
for package, key in (('torch', 'TORCH_VERSION'), ('torch-npu', 'TORCH_NPU_VERSION'),
                     ('transformers', 'TRANSFORMERS_VERSION')):
    if os.environ.get(key):
        assert version(package) == os.environ[key], (package, version(package), os.environ[key])
assert torch.npu.is_available()
root = pathlib.Path(os.environ['SRC_DIR']).resolve()
for name in ('vllm', 'vllm_ascend', 'vllm_omni', 'verl', 'verl_omni'):
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    print(name, version(name.replace('_', '-')), path)
    expected_source = (pathlib.Path(os.environ['VERL_OMNI_DIR']).resolve()
                       if name == 'verl_omni' else root / name.replace('_', '-'))
    assert path.is_relative_to(expected_source), (name, 'loaded outside configured checkout', path, expected_source)
for name in ('ray', 'transformers', 'accelerate', 'peft', 'TransferQueue',
             'qwen-omni-utils', 'qwen-vl-utils', 'av', 'torchaudio'):
    print(name, version(name))
# 仅检查 NPU 基础运行；不是完整训练验证。
assert torch.ones(4, device='npu').sum().cpu().item() == 4
print('Basic import/NPU checks passed')
PY

# 清除父进程中保存的这些变量，再通过 conda run 激活并运行新的 Python 进程，
# 验证导入依赖的是 Conda 持久配置，而不是安装终端临时 source 的结果。
python - <<'PY' 2>&1 | tee "$LOG_DIR/import-check.txt"
import json
import os
import pathlib
import subprocess
import sys
root = pathlib.Path(os.environ['ENV_ROOT'])
settings = json.loads((root / 'conda-env-vars.json').read_text())
clean_env = dict(os.environ)
for name in settings:
    clean_env.pop(name, None)
# 避免同一 prefix 被 Conda 当作 reactivate，部分版本不会重新读取 config vars。
for name in list(clean_env):
    if name.startswith(('CONDA_PREFIX', '__CONDA_SHLVL_')) or name in ('CONDA_DEFAULT_ENV', 'CONDA_PROMPT_MODIFIER'):
        clean_env.pop(name)
clean_env['CONDA_SHLVL'] = '0'
result = subprocess.run([os.environ['CONDA_EXE'], 'run', '-p', os.environ['ENV_PREFIX'],
                         '--no-capture-output', 'python', '-u', str(root / 'check_environment.py')],
                        env=clean_env, cwd=root)
sys.exit(result.returncode)
PY
# 检查输出与退出状态；出现 import/ABI/API 错误时先解决安装问题。
{
  for repo in vllm vllm-ascend vllm-omni verl; do
    printf '%s ' "$repo"
    git -C "$SRC_DIR/$repo" rev-parse HEAD
  done
  printf 'verl-omni path: %s\n' "$VERL_OMNI_DIR"
  git -C "$VERL_OMNI_DIR" rev-parse HEAD
  # 同时记录自有源码是否含本地修改，SHA 本身不能代表未提交内容。
  git -C "$VERL_OMNI_DIR" status --short
} | tee "$LOG_DIR/source-shas.txt"
