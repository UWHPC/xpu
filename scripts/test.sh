#!/usr/bin/env bash
#
#   ./scripts/test.sh              every available path: CPU, then CUDA, then HIP
#   ./scripts/test.sh --cpp        CPU path only
#   ./scripts/test.sh --cu         CUDA path only
#   ./scripts/test.sh --hip        HIP path only
#   ./scripts/test.sh --sanitize   also run compute-sanitizer on the CUDA path
#   ./scripts/test.sh --clean      wipe the build dirs first

set -euo pipefail

cd "$(dirname "$0")/.."

paths=()
sanitize=0
clean=0

for arg in "$@"; do
  case $arg in
    --cpp)      paths+=(cpp) ;;
    --cu)       paths+=(cu) ;;
    --hip)      paths+=(hip) ;;
    --sanitize) sanitize=1 ;;
    --clean)    clean=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

explicit=1
if [[ ${#paths[@]} == 0 ]]; then
  explicit=0
  paths=(cpp cu hip)
fi

wants() {
  local want=$1 p
  for p in "${paths[@]}"; do
    [[ $p == "$want" ]] && return 0
  done
  return 1
}

skip() {
  local unwanted=$1 p kept=()
  for p in "${paths[@]}"; do
    [[ $p == "$unwanted" ]] || kept+=("$p")
  done
  paths=("${kept[@]}")
}

nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
if ! [[ -x $nvcc ]]; then
  nvcc=$(command -v nvcc || true)
fi

hipcxx=${HIPCXX:-${ROCM_PATH:-/opt/rocm}/llvm/bin/clang++}
if [[ $hipcxx != /* ]]; then
  hipcxx=$(command -v "$hipcxx" || true)
fi

cxx=${CXX:-}
if [[ -z $cxx ]]; then
  cxx=$(command -v g++-15 || command -v g++ || true)
elif [[ $cxx != /* ]]; then
  cxx=$(command -v "$cxx" || true)
fi

if [[ -z $cxx || ! -x $cxx ]]; then
  echo "xpu: no C++ compiler found; set CXX to a C++23 compiler" >&2
  exit 2
fi

# a configure without the GPU compiler silently falls back to the CPU tests,
# which would look like the GPU path passing when it never got compiled
if wants cu && [[ -z $nvcc ]]; then
  if [[ $explicit == 1 ]]; then
    echo "--cu needs nvcc on PATH or at \$CUDA_HOME/bin/nvcc" >&2
    exit 2
  fi
  echo "xpu: no nvcc found, skipping the CUDA path"
  skip cu
fi

if wants hip && [[ -z $hipcxx || ! -x $hipcxx ]]; then
  if [[ $explicit == 1 ]]; then
    echo "--hip needs ROCm's clang++ at \$HIPCXX or \$ROCM_PATH/llvm/bin/clang++" >&2
    exit 2
  fi
  echo "xpu: no ROCm clang++ found, skipping the HIP path"
  skip hip
fi

if [[ $sanitize == 1 ]] && ! wants cu; then
  echo "--sanitize needs the CUDA path" >&2
  exit 2
fi

run() {
  local label=$1 build=$2
  shift 2

  echo
  echo "=== $label path ($build) ==="

  if [[ $clean == 1 ]]; then
    rm -rf "$build"
  fi

  local args=(
    -S . -B "$build" -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_COMPILER="$cxx"
    -DXPU_ENABLE_LINALG=ON
    "$@"
  )

  cmake "${args[@]}"
  cmake --build "$build"
  ctest --test-dir "$build" --output-on-failure --no-tests=ignore
}

if wants cpp; then
  run CPU build-cpp \
    -DXPU_ENABLE_CUDA=OFF \
    -DXPU_ENABLE_HIP=OFF
fi

if wants cu; then
  run CUDA build-cuda \
    -DXPU_ENABLE_CUDA=ON \
    -DXPU_ENABLE_HIP=OFF \
    -DCMAKE_CUDA_COMPILER="$nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$cxx"

  if [[ $sanitize == 1 ]]; then
    echo
    echo "=== compute-sanitizer ==="
    cuda_tests=(build-cuda/tests/xpu_test_*)
    for test_binary in "${cuda_tests[@]}"; do
      echo
      echo "--- $test_binary ---"
      "$(dirname "$nvcc")/compute-sanitizer" --tool memcheck \
        --error-exitcode 1 "$test_binary"
    done
  fi
fi

if wants hip; then
  run HIP build-hip \
    -DXPU_ENABLE_CUDA=OFF \
    -DXPU_ENABLE_HIP=ON \
    -DCMAKE_HIP_COMPILER="$hipcxx"
fi
