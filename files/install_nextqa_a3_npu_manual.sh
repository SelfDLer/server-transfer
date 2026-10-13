#!/usr/bin/env bash
# Qwen3-Omni / Ascend A3：Conda 环境手动安装与验证清单。
# 在 Linux Bash 中逐节复制执行；任一步失败时先解决问题，再继续下一节。
# 适用：Linux / Ascend A3，已安装 CANN 9.1.0、ATB 和 Conda。
# 系统依赖支持 apt-get、dnf、yum；具体软件包可用性取决于发行版和已配置的仓库。
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

# 系统依赖安装：auto 自动选择，也可指定 apt/dnf/yum/skip。
# skip 适用于依赖已由管理员装好的机器；跳过安装后仍需检查工具和 NPU。
export SYSTEM_PACKAGE_MANAGER=auto
export SYSTEM_USE_SUDO=auto  # auto：root 直接执行，否则使用 sudo；也可设 yes/no。
# 包名可按发行版仓库修改。RPM 系统的 ffmpeg 可能需要管理员配置额外软件源；
# 本清单不自动添加仓库。如果仓库提供 ffmpeg-free，可修改 RPM_MEDIA_PACKAGES。
APT_SYSTEM_PACKAGES=(
  git wget curl ca-certificates net-tools gcc g++ cmake make build-essential
  numactl libnuma-dev libibverbs-dev libjemalloc2 libhiredis-dev
  libxcb1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
  clang-15 ninja-build ffmpeg libsndfile1 sox unzip
)
RPM_SYSTEM_PACKAGES=(
  git wget curl ca-certificates net-tools gcc gcc-c++ cmake make
  numactl numactl-devel libibverbs-devel jemalloc hiredis-devel
  libxcb mesa-libGL glib2 libSM libXext libXrender
  clang ninja-build libsndfile sox unzip
)
RPM_MEDIA_PACKAGES=(ffmpeg)
# 如需指定编译器，可在这里配置，而不修改系统的全局 alternatives：
# export CC=clang-15
# export CXX=clang++-15

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
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
# 新开终端执行，清除指向旧 editable checkout 的手工 PYTHONPATH。
unset PYTHONPATH

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

# ===== 2. 系统依赖（自动适配包管理器和 sudo；可单独复制本节执行） =====
# 请先执行第 1 节。打印系统信息，便于包安装失败时核对发行版。
if [[ -r /etc/os-release ]]; then cat /etc/os-release; fi
install_system_dependencies() {
  local manager="$SYSTEM_PACKAGE_MANAGER" executable
  local -a privilege=()
  if [[ "$manager" == skip ]]; then
    printf '按配置跳过系统依赖安装。\n'
    return 0
  fi
  if [[ "$manager" == auto ]]; then
    if command -v apt-get >/dev/null 2>&1; then manager=apt
    elif command -v dnf >/dev/null 2>&1; then manager=dnf
    elif command -v yum >/dev/null 2>&1; then manager=yum
    else
      printf '未找到 apt-get/dnf/yum；请手动安装依赖，再设 SYSTEM_PACKAGE_MANAGER=skip。\n' >&2
      return 1
    fi
  fi
  case "$manager" in
    apt|apt-get) executable=apt-get ;;
    dnf|yum) executable="$manager" ;;
    *) printf '不支持的 SYSTEM_PACKAGE_MANAGER：%s\n' "$manager" >&2; return 1 ;;
  esac
  if ! command -v "$executable" >/dev/null 2>&1; then
    printf '指定的包管理器不可用：%s\n' "$executable" >&2
    return 1
  fi
  case "$SYSTEM_USE_SUDO" in
    auto) if (( EUID != 0 )); then privilege=(sudo); fi ;;
    yes) privilege=(sudo) ;;
    no) ;;
    *) printf 'SYSTEM_USE_SUDO 只能是 auto/yes/no。\n' >&2; return 1 ;;
  esac
  if (( ${#privilege[@]} )) && ! command -v sudo >/dev/null 2>&1; then
    printf '需要 sudo，但未找到 sudo 命令；请管理员安装依赖或使用 skip。\n' >&2
    return 1
  fi
  printf '系统包管理器：%s；权限方式：%s\n' "$executable" "${privilege[*]:-直接执行}"
  if [[ "$executable" == apt-get ]]; then
    "${privilege[@]}" apt-get update &&
      "${privilege[@]}" apt-get install -y --no-install-recommends "${APT_SYSTEM_PACKAGES[@]}" || return 1
  else
    "${privilege[@]}" "$executable" install -y "${RPM_SYSTEM_PACKAGES[@]}" || return 1
    if (( ${#RPM_MEDIA_PACKAGES[@]} )); then
      if ! "${privilege[@]}" "$executable" install -y "${RPM_MEDIA_PACKAGES[@]}"; then
        printf '音视频包安装失败：请检查软件源/包名，修改 RPM_MEDIA_PACKAGES 后重试。\n' >&2
        return 1
      fi
    fi
  fi
}
install_system_dependencies
# 上一条失败时停止本节，先调整包名、软件源或权限；不要继续安装后面的 Python 包。
# 已有依赖可以设 skip；不要把“跳过安装”当作“依赖已满足”。

# CANN 的环境脚本可能读取未定义变量，source 前不要 set -u。
source "$ASCEND_ROOT/ascend-toolkit/set_env.sh"
source "$ASCEND_ROOT/nnal/atb/set_env.sh"
check_system_tools() {
  local tool missing=0
  for tool in git gcc g++ cmake make ninja ffmpeg ffprobe; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      printf '缺少工具：%s，请先安装。\n' "$tool" >&2
      missing=1
    fi
  done
  if ! command -v clang-15 >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    printf '缺少 Clang：请按系统软件源安装，或调整编译器配置。\n' >&2
    missing=1
  fi
  return "$missing"
}
check_system_tools
# 任一检查失败时先补齐。安装成功也不能代替下面的 NPU/CANN 检查。
npu-smi info
ffmpeg -version
ffprobe -version
# 在这里确认 CANN 为 9.1.0，且 NPU/ATB 可用，再往下执行；Python 由 Conda 安装。

# ===== 3. 用 Conda 按绝对路径创建独立环境（不复用旧训练环境） =====
# 使用当前 shell 中的 conda，自动查询其安装根目录并加载同一套初始化脚本。
# conda info --base 查看 Conda 本体目录；echo "$CONDA_PREFIX" 查看当前激活环境目录。
# 如当前在旧训练环境，先 conda deactivate，再执行本节。
# 目标 prefix 已存在时不创建也不激活；修改第 1 节的路径后重新执行本节。
# 创建成功才激活，避免失败后把包误装到 base 或旧环境。
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
  conda create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" pip &&
  conda activate "$ENV_PREFIX" &&
# 确认 Python 确实来自指定 prefix 且版本正确，通过后才执行 pip 安装。
  python - <<'PY' && python -m pip install --upgrade pip setuptools wheel packaging
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
python -m pip freeze > "$LOG_DIR/pip-freeze.txt"
python -m pip check > "$LOG_DIR/pip-check.txt" 2>&1
# 上一条的退出码非零表示检查失败；文件开头列出的冲突可能出现在这里。
cat "$LOG_DIR/pip-check.txt"
python - <<'PY' | tee "$LOG_DIR/import-check.txt"
import importlib
import os
import pathlib
import sys
from importlib.metadata import version
import torch
import torch_npu
print('python:', sys.executable)
print('torch:', torch.__version__, torch.__file__, 'git:', torch.version.git_version)
print('NPU available:', torch.npu.is_available(), 'count:', torch.npu.device_count())
assert torch.__version__ == os.environ['TORCH_VERSION']
assert version('torch-npu') == os.environ['TORCH_NPU_VERSION']
assert version('transformers') == os.environ['TRANSFORMERS_VERSION']
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
# 检查输出与退出状态；出现 import/ABI/API 错误时先解决安装问题。
for repo in vllm vllm-ascend vllm-omni verl verl-omni; do
  printf '%s ' "$repo"
  git -C "$SRC_DIR/$repo" rev-parse HEAD
done | tee "$LOG_DIR/source-shas.txt"
