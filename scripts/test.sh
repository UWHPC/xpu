#!/usr/bin/env bash
#
#   ./scripts/test.sh              both paths: tests/smoke.cpp then tests/smoke.cu
#   ./scripts/test.sh --cpp        CPU path only
#   ./scripts/test.sh --cu         CUDA path only
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
    --sanitize) sanitize=1 ;;
    --clean)    clean=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

explicit=1
if [[ ${#paths[@]} == 0 ]]; then
  explicit=0
  paths=(cpp cu)
fi

wants() {
  local want=$1 p
  for p in "${paths[@]}"; do
    [[ $p == "$want" ]] && return 0
  done
  return 1
}

nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
if ! [[ -x $nvcc ]]; then
  nvcc=$(command -v nvcc || true)
fi

# a CUDA-less configure silently falls back to smoke.cpp, which would look like
# the .cu path passing when it never got compiled
if wants cu && [[ -z $nvcc ]]; then
  if [[ $explicit == 1 ]]; then
    echo "--cu needs nvcc on PATH or at \$CUDA_HOME/bin/nvcc" >&2
    exit 2
  fi
  echo "xpu: no nvcc found, skipping the CUDA path"
  paths=(cpp)
fi

if [[ $sanitize == 1 ]] && ! wants cu; then
  echo "--sanitize needs the CUDA path" >&2
  exit 2
fi

run() {
  local label=$1 cuda=$2 build=$3

  echo
  echo "=== $label path ($build) ==="

  if [[ $clean == 1 ]]; then
    rm -rf "$build"
  fi

  local args=(
    -S . -B "$build" -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_COMPILER=g++-14
    -DXPU_ENABLE_CUDA=$cuda
  )

  if [[ $cuda == ON ]]; then
    args+=(-DCMAKE_CUDA_HOST_COMPILER=g++-14)
  fi

  cmake "${args[@]}"
  cmake --build "$build"
  ctest --test-dir "$build" --output-on-failure --no-tests=ignore
}

if wants cpp; then
  run CPU OFF build-cpp
fi

if wants cu; then
  run CUDA ON build-cuda

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
