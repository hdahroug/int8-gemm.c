#!/bin/bash
set -euo pipefail

if [ ! -f build/matmul ]; then
    echo "matmul binary not found. Run scripts/build.sh first."
    exit 1
fi

if [ -z "${MKLROOT:-}" ] && [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +eu
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
    set -euo pipefail
fi

NCORES=$(nproc)
export OMP_NUM_THREADS=$NCORES

echo "=== Correctness: Kernel and oneDNN vs Naive reference ==="
echo "Using Threads: $NCORES"
echo ""

build/matmul | tee standard_output.txt