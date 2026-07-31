#!/usr/bin/env bash
#
#   ./scripts/test.sh              CUDA build
#   ./scripts/test.sh --cpu        CPU-only build
#   ./scripts/test.sh --sanitize   also run compute-sanitizer
#   ./scripts/test.sh --clean      wipe the build dir first

set -euo pipefail

cd "$(dirname "$0")/.."

build=build
cuda=ON
sanitize=0
clean=0

for arg in "$@"; do
  case $arg in
    --cpu)      cuda=OFF; build=build-cpu ;;
    --sanitize) sanitize=1 ;;
    --clean)    clean=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

[[ $clean == 1 ]] && rm -rf "$build"

cmake -S . -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 \
  -DXPU_ENABLE_CUDA=$cuda

cmake --build "$build"

ctest --test-dir "$build" --output-on-failure --no-tests=ignore

if [[ $sanitize == 1 ]]; then
  if [[ $cuda == OFF ]]; then
    echo "--sanitize needs the CUDA build" >&2
    exit 2
  fi
  echo
  echo "=== compute-sanitizer ==="
  "${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer" --tool memcheck \
    --error-exitcode 1 "$build/tests/xpu_smoke"
fi
