#!/usr/bin/env bash
# Qwen3-Omni / Ascend A3：Conda 环境手动安装与验证清单。
# 在 Linux Bash 中逐节复制执行；任一步失败时先解决问题，再继续下一节。
# 适用：Linux / Ascend A3，已安装 CANN 9.1.0、ATB 和 Conda。
# 工具、编译器和用户态库优先安装到 Conda；驱动、固件、CANN/ATB 使用主机安装。
# 驱动、NPU 设备、CANN 和 ATB 必须已就绪；本文件不安装驱动、CANN 或 Conda。
# 使用全新的独立环境执行安装。

# ===== 1. 集中配置（换机器、目录或环境时先修改本节） =====
set -o pipefail

# 环境与源码目录；Conda 使用绝对路径 prefix，不使用环境名称。
export ENV_ROOT=/workspace/a3-vllm030
export SRC_DIR="$ENV_ROOT/src"
export ENV_PREFIX="$ENV_ROOT/conda"
export LOG_DIR="$ENV_ROOT/logs"
# 使用当前 shell 中的 conda；运行 conda info --base 可查看它的安装根目录。
export PYTHON_VERSION=3.12
export CONSTRAINTS_FILE="$ENV_ROOT/npu-constraints.txt"
export REQUIREMENTS_FILE="$ENV_ROOT/environment-requirements.txt"

# Conda 软件源和包列表；先安装全部 Conda 包，再安装下面的 pip 包。
export CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
export GCC_VERSION=12
export CLANG_VERSION=15
CONDA_PACKAGES=(
  pip setuptools wheel packaging
  git wget curl ca-certificates cmake make ninja pkg-config unzip
  "gcc=$GCC_VERSION" "gxx=$GCC_VERSION" "clang=$CLANG_VERSION" "clangxx=$CLANG_VERSION"
  "libgcc-ng>=12" "libstdcxx-ng>=12"
  numactl libnuma rdma-core jemalloc libhiredis
  libxcb libgl libglib xorg-libsm xorg-libxext xorg-libxrender
  ffmpeg libsndfile sox
)
# 若已有匹配主机 RDMA/OFED 的用户态库，可从列表移除 rdma-core，使用主机版本。
# LD_LIBRARY_PATH 默认优先查找环境的 lib；仅在检查仍加载旧 C++ 库时启用预加载。
export PRELOAD_CONDA_LIBSTDCXX=0

# Ascend 只配置安装基础路径，从其下的官方 set_env.sh 加载环境。
# 标准布局：ascend-toolkit/set_env.sh 和 nnal/atb/set_env.sh。
export ASCEND_ROOT=/usr/local/Ascend
# 留空时，固定版本的 vLLM-Ascend setup.py 通过 npu-smi 自动识别芯片。
# 无可见 NPU 或识别失败时，可手动填入目标型号，例如 ascend910_9391。
export SOC_VERSION=""
export COMPILE_CUSTOM_KERNELS=1

# 固定上游源码版本；更换版本前重新核对上游兼容性。
export VERL_OMNI_REF=14049614953cb92edacc3809dd2a60179d44c394
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

# 下载源。
export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
export ASCEND_PIP_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi
export PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu/
export MOONCAKE_INDEX_URL=https://mirrors.aliyun.com/pypi/web/simple

# 安装与构建设置。
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# ===== 固定版本说明 =====
#
# 查询时官方 main：14049614953cb92edacc3809dd2a60179d44c394
# 为了可复现，固定这次查询到的 SHA，不在安装过程中追随浮动 main。
# 依据：
# https://github.com/verl-project/verl-omni/blob/14049614953cb92edacc3809dd2a60179d44c394/pyproject.toml
#
# 已确认的上游依赖声明冲突（不是已经通过验证的全量 lockfile）：
# - verl 固定版本要求 transformers<5.11；verl-omni 要求 >=5.13,<=5.14.1；
#   vllm-ascend 固定版本要求 transformers==5.14.1。最终按后两者对齐。
# - vllm-omni 固定版本要求 accelerate==1.12.0；verl-omni 要求 >=1.14.0。
#   最终按 verl-omni 的最低版本 1.14.0 对齐；保留 pip check 的冲突报告。
# - vllm 0.30 要求 fastapi>=0.133,<0.137，vllm-ascend 固定版本要求 <0.124。
#   本清单保留原有安装顺序，安装后检查实际结果。
# 上述冲突意味着无法承诺 pip check 全绿或首次训练成功；本机仅做静态检查。
# 不要通过反复 pip install -U 整个栈来消除提示，也不要把 pip check 失败当作通过。

# ===== 2. 用 Conda 按绝对路径创建独立环境（不复用旧训练环境） =====
# 使用当前 shell 中的 conda，自动查询其安装根目录并加载同一套初始化脚本。
# conda info --base 查看 Conda 本体目录；echo "$CONDA_PREFIX" 查看当前激活环境目录。
# 如当前在旧训练环境，先 conda deactivate，再执行本节。
# 目标 prefix 已存在时不创建也不激活；修改第 1 节的路径后重新执行本节。
# 创建成功才激活，避免失败后把包误装到 base 或旧环境。
# 仅对创建命令优先采用 shell 代理；代理地址不保存到环境或配置文件。
conda_with_shell_proxy() {
  "$active_conda_base/bin/python" - "$@" <<'PY'
import os
import sys

proxy_names = ('http_proxy', 'https_proxy', 'all_proxy', 'no_proxy')
# 小写优先；显式空值也代表 shell 的选择。统一大小写以兼容不同下载后端。
use_shell_proxy = any(name in os.environ or name.upper() in os.environ for name in proxy_names)
if use_shell_proxy:
    for name in proxy_names:
        value = os.environ.get(name, os.environ.get(name.upper()))
        if value is not None:
            os.environ[name] = os.environ[name.upper()] = value
    from conda.base.context import context
    # Conda CLI 没有单次清空 proxy_servers 的参数，且空 condarc 会合并旧代理。
    # 仅在这个独立进程内覆盖配置，让下载后端读取 shell 的代理及 NO_PROXY。
    context.proxy_servers = {}

from conda.cli.main import main
sys.exit(main(*sys.argv[1:]))
PY
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
  # --insecure 仅对本次创建跳过 TLS 证书验证。
  conda_with_shell_proxy create -y --insecure -p "$ENV_PREFIX" --override-channels -c "$CONDA_CHANNEL" \
    --strict-channel-priority "python=$PYTHON_VERSION" "${CONDA_PACKAGES[@]}" &&
  conda activate "$ENV_PREFIX" &&
# 确认 Python 确实来自指定 prefix 且版本正确，通过后才执行 pip 安装。
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
# 在子 shell 中 source，避免把安装时的临时变量留在外层 shell。
# 使用 conda env config vars 管理持久变量；无需 activate.d/deactivate.d 钩子。
# CANN 安装路径发生变化后，重新执行本节并重新激活环境。
configure_conda_environment() (
  set -e
  if [[ "${CONDA_PREFIX:-}" != "$ENV_PREFIX" ]]; then
    printf '请先 conda activate "%s"，当前环境不匹配。\n' "$ENV_PREFIX" >&2
    return 1
  fi
  # 清除旧源码路径，再由官方脚本设置 CANN 自身需要的 Python 路径。
  unset PYTHONPATH
  # CANN 环境脚本可能读取未定义变量，此处不要启用 set -u。
  source "$ASCEND_ROOT/ascend-toolkit/set_env.sh" || return 1
  source "$ASCEND_ROOT/nnal/atb/set_env.sh" || return 1
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
    'ENV_ROOT': os.environ['ENV_ROOT'],
    'SRC_DIR': os.environ['SRC_DIR'],
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

# PATH 由 Conda 管理，不将安装时的完整 PATH 固定在环境配置中。
# 编译需要 CANN 工具时，加载 PATH 后再次应用 Conda 变量以维持 libstdc++ 的优先级。
source "$ASCEND_ROOT/ascend-toolkit/set_env.sh"
source "$ASCEND_ROOT/nnal/atb/set_env.sh"
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

# ===== 4. 安装固定版本的上游 verl-omni 源码 =====
git clone https://github.com/verl-project/verl-omni.git "$SRC_DIR/verl-omni"
cd "$SRC_DIR/verl-omni"
git checkout --detach "$VERL_OMNI_REF"
# 核对上游声明的配套版本。
test "$(tr -d '[:space:]' < .github/vllm_ascend_pin.txt)" = "$VLLM_ASCEND_REF"
test "$(tr -d '[:space:]' < .github/vllm_omni_pin.txt)" = "$VLLM_OMNI_REF"
test "$(tr -d '[:space:]' < .github/verl_pin.txt)" = "$VERL_REF"

# ===== 5. 基础 Python / PyTorch / 通信依赖 =====
python -m pip install modelscope "ray[default]==$RAY_VERSION" 'protobuf>3.20.0'
python -m pip install "mooncake-transfer-engine-npu==$MOONCAKE_VERSION" \
  --extra-index-url "$MOONCAKE_INDEX_URL"
python -m pip install --force-reinstall \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "$PYTORCH_CPU_INDEX_URL"
python -m pip install "torch-npu==$TORCH_NPU_VERSION" \
  --extra-index-url "$ASCEND_PIP_INDEX_URL" --extra-index-url "$PYTORCH_CPU_INDEX_URL"

# ===== 6. vLLM 0.30：NPU 路径使用 empty target，不安装 CUDA vLLM wheel =====
git clone --depth 1 --branch "$VLLM_REF" https://github.com/vllm-project/vllm.git "$SRC_DIR/vllm"
cd "$SRC_DIR/vllm"
VLLM_TARGET_DEVICE=empty python -m pip install -e '.[audio]' \
  --extra-index-url "$PYTORCH_CPU_INDEX_URL"
python -m pip uninstall -y triton

# ===== 7. vLLM-Ascend：自动识别芯片并构建自定义算子 =====
git clone https://github.com/vllm-project/vllm-ascend.git "$SRC_DIR/vllm-ascend"
cd "$SRC_DIR/vllm-ascend"
git checkout --detach "$VLLM_ASCEND_REF"
git submodule update --init --recursive
# SOC_VERSION 为空时由此版本 setup.py 调用 npu-smi 识别，不传入字面值 auto。
# 识别失败则安装报错；检查 NPU/驱动可见性，或在第 1 节手动指定型号后重试。
# 仅为这一条构建命令设置，避免影响后续训练环境。
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

cd "$SRC_DIR/verl-omni"
python -m pip install --no-deps -e .
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
src = pathlib.Path(os.environ['SRC_DIR'])
project = tomllib.loads((src / 'verl-omni/pyproject.toml').read_text())['project']
engine_packages = {'vllm', 'vllm-omni', 'verl'}
deps = [r for r in project['dependencies']
        if canonicalize_name(Requirement(r).name) not in engine_packages]
deps += project['optional-dependencies']['audio']
# torchvision 0.25 自带视频读取路径；不额外安装与 torch ABI 耦合的 TorchCodec。
deps += ['librosa>=0.10.2', 'soundfile>=0.13.1', 'av>=14',
         'qwen-vl-utils>=0.0.14', 'einops>=0.8.1', 'sox>=1.5.0']
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
  --extra-index-url "$ASCEND_PIP_INDEX_URL"
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
    assert path.is_relative_to(root), (name, 'loaded outside new checkout', path)
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
for repo in vllm vllm-ascend vllm-omni verl verl-omni; do
  printf '%s ' "$repo"
  git -C "$SRC_DIR/$repo" rev-parse HEAD
done | tee "$LOG_DIR/source-shas.txt"
