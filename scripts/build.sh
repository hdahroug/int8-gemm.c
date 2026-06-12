#!/bin/bash
set -euo pipefail

echo "=== Running Autotuner ==="
gcc autotuner/autotune.c -I. -O3 -march=core-avx2 -fopenmp -o autotune -lm -lpthread
./autotune | tee autotune_output.txt

read BEST_MC BEST_NC BEST_KC BEST_PA1 BEST_PB1 BEST_PA2 BEST_PB2 < /tmp/autotune_result.txt
echo "Autotune results: MC=$BEST_MC NC=$BEST_NC KC=$BEST_KC"

DEFINES="-DMC=$BEST_MC -DNC=$BEST_NC -DKC=$BEST_KC \
         -DPREFETCH_A_L1=$BEST_PA1 -DPREFETCH_B_L1=$BEST_PB1 \
         -DPREFETCH_A_L2=$BEST_PA2 -DPREFETCH_B_L2=$BEST_PB2"

MKL_INC_FLAGS=""

# 1. Check if Intel oneAPI setvars exists (For GitHub Actions or custom local setup)
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    echo "Intel oneAPI detected, sourcing environment..."
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
fi

# 2. Dynamically detect where MKL headers are located
if [ -n "${MKLROOT:-}" ]; then
    MKL_INC_FLAGS="-I$MKLROOT/include"
elif [ -d "/usr/include/mkl" ]; then
    MKL_INC_FLAGS="-I/usr/include/mkl"
elif [ -d "/usr/local/include/mkl" ]; then
    MKL_INC_FLAGS="-I/usr/local/include/mkl"
fi

# 3. Export Flags globally so CMake can catch them on ANY system
export CFLAGS="-g -O3 -march=core-avx2 $DEFINES $MKL_INC_FLAGS"
export CXXFLAGS="-g -O3 -march=core-avx2 $DEFINES $MKL_INC_FLAGS"
export LDFLAGS="-lmkl_rt -ldnnl -lpthread -lm -ldl"

echo ""
echo "=== Building INT8 GEMM kernel with CMake ==="
# Clean old build cache to prevent dirty path conflicts between devices
rm -rf build 

cmake -B build -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build build -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "Run: bash scripts/test.sh  (For Correctness)"
echo "Run: bash scripts/bench.sh (For Performance)"