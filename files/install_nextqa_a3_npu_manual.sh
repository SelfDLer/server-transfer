#!/usr/bin/env bash
# Qwen3-Omni / Ascend A3：通用 Conda 环境手动安装清单，附 NExT-QA 示例。
# 在 Linux Bash 中逐节复制执行；不要整体 bash/source 本文件（末尾有可选训练命令）。
# 适用：Linux / Ascend A3，已安装 CANN 9.1.0、ATB 和 Conda。
# 系统依赖支持 apt-get、dnf、yum；具体软件包可用性取决于发行版和已配置的仓库。
# 驱动、NPU 设备、CANN 和 ATB 必须已就绪；本文件不安装驱动、CANN 或 Conda。
# 两台机器使用相同路径、代码和依赖；不要在当前运行训练的旧环境中执行安装。

# ===== 1. 集中配置（换机器、目录或环境时先修改本节） =====
set -o pipefail

# 环境与源码目录；Conda 使用绝对路径 prefix，不使用环境名称。
export ENV_ROOT=/workspace/a3-vllm030
export SRC_DIR="$ENV_ROOT/src"
export ENV_PREFIX="$ENV_ROOT/conda"
export LOG_DIR="$ENV_ROOT/logs"
export CONDA_ROOT="$HOME/miniconda3"
export PYTHON_VERSION=3.12
export CONSTRAINTS_FILE="$ENV_ROOT/npu-constraints.txt"
export REQUIREMENTS_FILE="$ENV_ROOT/training-requirements.txt"

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
export JEMALLOC_LIB=""  # 留空自动查找；也可指定完整 .so 路径，或设 none 禁用预加载。
export SOC_VERSION=ascend910_9391
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

# 可选数据/训练配置；DATA_DIR 指向实际数据集，不绑定环境名称。
export MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct
export DATA_DIR=/datasets/video_qa
export TRAIN_FILE="$DATA_DIR/train.parquet"
export VAL_FILE="$DATA_DIR/validation.parquet"
export EXPERIMENT_NAME=a3_vllm030
export OUTPUT_DIR="$ENV_ROOT/outputs/$EXPERIMENT_NAME"
export N_GPUS_PER_NODE=16
export NNODES=2
export ROLLOUT_TP=4
export TOTAL_TRAINING_STEPS=500
export TRAIN_BATCH_SIZE=$((N_GPUS_PER_NODE * NNODES * 2))
export HEAD_IP=192.168.1.10
export WORKER_IP=192.168.1.11
export RAY_PORT=6379
export BUILD_OMP_NUM_THREADS=1
export TRAIN_OMP_NUM_THREADS=8

# 运行时设置。
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TASK_QUEUE_ENABLE=1
export OMP_NUM_THREADS="$BUILD_OMP_NUM_THREADS"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VLLM_ASCEND_ENABLE_NZ=0
# SOC_VERSION 必须与实际硬件一致；不要照搬其他型号的配置。
# 新开终端执行，清除指向旧 editable checkout 的手工 PYTHONPATH。
unset PYTHONPATH

# 运行时辅助函数；新终端执行第 1 节时同时定义，供第 2/12 节使用。
configure_jemalloc_preload() {
  local library="$JEMALLOC_LIB" loader candidate
  [[ "$library" != none ]] || return 0
  if [[ -z "$library" ]]; then
    loader="$(command -v ldconfig || true)"
    if [[ -z "$loader" ]]; then
      for candidate in /sbin/ldconfig /usr/sbin/ldconfig; do
        if [[ -x "$candidate" ]]; then loader="$candidate"; break; fi
      done
    fi
    if [[ -n "$loader" ]]; then
      library="$("$loader" -p 2>/dev/null | awk '/libjemalloc[.]so[.]2 .*=>/ {print $NF; exit}')"
    fi
    if [[ -z "$library" ]]; then
      for candidate in /usr/lib64/libjemalloc.so.2 /lib64/libjemalloc.so.2 \
        /usr/lib/libjemalloc.so.2 "/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so.2"; do
        if [[ -r "$candidate" ]]; then library="$candidate"; break; fi
      done
    fi
    if [[ -z "$library" ]]; then
      printf '未找到 jemalloc，保留现有 LD_PRELOAD；可安装后重试或指定 JEMALLOC_LIB。\n' >&2
      return 0
    fi
  fi
  if [[ ! -r "$library" ]]; then
    printf 'JEMALLOC_LIB 不可读取：%s\n' "$library" >&2
    return 1
  fi
  case ":${LD_PRELOAD:-}:" in
    *":$library:"*) ;;
    *) export LD_PRELOAD="$library${LD_PRELOAD:+:$LD_PRELOAD}" ;;
  esac
}

# ===== 固定版本说明 =====
#
# 查询时官方 main：14049614953cb92edacc3809dd2a60179d44c394
# 为了可复现，固定这次查询到的 SHA，不在安装过程中追随浮动 main。
# 依据：
# https://github.com/verl-project/verl-omni/blob/14049614953cb92edacc3809dd2a60179d44c394/pyproject.toml
# https://github.com/verl-project/verl-omni/blob/14049614953cb92edacc3809dd2a60179d44c394/examples/gspo_trainer/README.md
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
configure_jemalloc_preload
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
# Conda 已安装；CONDA_ROOT 指向 Miniconda/Anaconda 安装根目录。
# 如当前在旧训练环境，先 conda deactivate，再执行本节。
# 目标 prefix 已存在时不创建也不激活；修改第 1 节的路径后重新执行本节。
# 创建成功才激活，避免失败后把包误装到 base 或旧环境。
if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
  printf '找不到 Conda 初始化脚本，请检查 CONDA_ROOT：%s\n' "$CONDA_ROOT" >&2
elif [[ "$ENV_PREFIX" != /* ]]; then
  printf 'ENV_PREFIX 必须是绝对路径：%s\n' "$ENV_PREFIX" >&2
elif [[ -e "$ENV_PREFIX" ]]; then
  printf '目标环境已存在，请改用新目录：%s\n' "$ENV_PREFIX" >&2
else
  source "$CONDA_ROOT/etc/profile.d/conda.sh" &&
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

# ===== 7. vLLM-Ascend：构建自定义算子 =====
git clone https://github.com/vllm-project/vllm-ascend.git "$SRC_DIR/vllm-ascend"
cd "$SRC_DIR/vllm-ascend"
git checkout --detach "$VLLM_ASCEND_REF"
git submodule update --init --recursive
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

# ===== 9. 补全训练与音视频依赖，最终对齐版本 =====
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
# 安装日志如出现其他依赖冲突，先记录，不要开始长训练。

# ===== 10. 安装检查（两台机器各执行一次） =====
python -m pip freeze > "$LOG_DIR/pip-freeze.txt"
python -m pip check > "$LOG_DIR/pip-check.txt" 2>&1
# 上一条的退出码非零表示检查失败；文件开头列出的冲突可能出现在这里。
cat "$LOG_DIR/pip-check.txt"
python - <<'PY' | tee "$LOG_DIR/import-check.txt"
import importlib
import inspect
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
from vllm_omni.entrypoints import AsyncOmni
assert hasattr(AsyncOmni, '_clear_frontend_mm_cache')
print(inspect.getsource(AsyncOmni._clear_frontend_mm_cache))
from verl_omni.utils.dataset.nextqa_rl_dataset import NextQARLHFDataset
from qwen_omni_utils import process_mm_info
print('NExT-QA dataset:', inspect.getfile(NextQARLHFDataset))
print('media loader:', inspect.getfile(process_mm_info))
# 仅检查 NPU 基础运行；不是完整训练验证。
assert torch.ones(4, device='npu').sum().cpu().item() == 4
print('Basic import/NPU checks passed')
PY
# 检查输出与退出状态；出现 import/ABI/API 错误时先停止，不启动训练。
for repo in vllm vllm-ascend vllm-omni verl verl-omni; do
  printf '%s ' "$repo"
  git -C "$SRC_DIR/$repo" rev-parse HEAD
done | tee "$LOG_DIR/source-shas.txt"

# ===== 11. 数据：已有相同路径的 parquet 和视频时跳过转换 =====
# MODEL_PATH、DATA_DIR、TRAIN_FILE 和 VAL_FILE 已在第 1 节配置。
# 模型使用原始完整 checkpoint。两台机器需要相同绝对路径的模型、视频、parquet。
# 新准备数据时，按官方文档取得 NExTVideo.zip，目录结构应为：
# $DATA_DIR/repo/{train.csv,val.csv,map_vid_vidorID.json} 和 $DATA_DIR/NExTVideo/...
# 若尚未准备标注（已有 repo 不要重复 clone）：
# git clone https://github.com/doc-doc/NExT-QA.git "$DATA_DIR/repo"
# cp "$DATA_DIR"/repo/dataset/nextqa/{train.csv,val.csv,map_vid_vidorID.json} "$DATA_DIR/repo/"
# 在 DATA_DIR 目录执行 unzip NExTVideo.zip；不要再加 -d NExTVideo。
# cd "$SRC_DIR/verl-omni"
# python examples/gspo_trainer/data_process/nextqa.py \
#   --input_dir "$DATA_DIR" --output_dir "$DATA_DIR"

# ===== 12. 运行环境（以后新终端，两台机器都执行第 1 节，再执行本节） =====
# 这里只激活已有 Conda 环境，不重复执行第 3 节的创建命令。
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
source "$ASCEND_ROOT/ascend-toolkit/set_env.sh"
source "$ASCEND_ROOT/nnal/atb/set_env.sh"
configure_jemalloc_preload
export OMP_NUM_THREADS="$TRAIN_OMP_NUM_THREADS"
python -c 'import os, pathlib, sys; assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(os.environ["ENV_PREFIX"]).resolve(); print(sys.executable)'
# 不设置 CUDA_LAUNCH_BLOCKING / ASCEND_LAUNCH_BLOCKING 或新的缓存开关。
# 解码器先用安装后实际选择的后端；查看 qwen-vl-utils 的启动日志，不盲目覆盖。
# 使用专用 Ray 集群；在安装前已启动的 Ray worker 不会自动切换 Python 环境。

# ===== 13A. 可选：单机启动（与 13B 二选一；安装/检查完成后再执行） =====
# 第 1 节设置 NNODES=1，并重新计算 TRAIN_BATCH_SIZE；然后执行第 12 节。
# cd "$SRC_DIR/verl-omni"
# bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh \
#   actor_rollout_ref.model.attn_backend=_native_npu \
#   trainer.resume_mode=disable \
#   ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES="$VERL_USE_EXTERNAL_MODULES"

# ===== 13B. 可选：双机启动（第 1 节配置地址与 NNODES=2；两节点先执行第 12 节） =====
# 如原集群仍在跑任务，不要 ray stop。等待结束后，在专用节点停旧 Ray 再启动新环境。
# HEAD 节点：
# ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --resources="{\"NPU\": $N_GPUS_PER_NODE}"
# WORKER 节点：
# ray start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$WORKER_IP" --resources="{\"NPU\": $N_GPUS_PER_NODE}"
# NPU 平台申请的是 Ray 的 NPU 资源；不要只用 --num-gpus 注册 GPU 资源。
# HEAD 节点检查，确认两节点和预期 NPU 资源注册完成：
# ray status --address="$HEAD_IP:$RAY_PORT"
# cd "$SRC_DIR/verl-omni"
# bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh \
#   actor_rollout_ref.model.attn_backend=_native_npu \
#   trainer.resume_mode=disable \
#   ++ray_kwargs.ray_init.address="$HEAD_IP:$RAY_PORT" \
#   ++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES="$VERL_USE_EXTERNAL_MODULES" \
#   ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_ASCEND_ENABLE_NZ="\"$VLLM_ASCEND_ENABLE_NZ\"" \
#   ++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
#   ++ray_kwargs.ray_init.runtime_env.env_vars.RAY_DEDUP_LOGS="\"$RAY_DEDUP_LOGS\""
# ++ray...env_vars 只传播环境变量，不分发源码；两台机器都必须已安装相同代码。
# 第一次验证用 resume_mode=disable，避免误读旧实验 checkpoint；需要续训时再调整。
