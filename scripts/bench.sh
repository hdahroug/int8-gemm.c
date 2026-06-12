#!/bin/bash
set -euo pipefail

if [ ! -f build/google_bench ]; then
    echo "google_bench not found. Run scripts/build.sh first."
    exit 1
fi

if [ -z "${MKLROOT:-}" ] && [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +eu
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1 || true
    set -euo pipefail
fi

CPU_MODEL=$(lscpu | grep "Model name" | cut -d':' -f2 | xargs || echo "Generic CPU")
HAS_AVX512=false
if lscpu | grep -iq "avx512"; then HAS_AVX512=true; fi

NCORES=$(nproc)

echo "=== Benchmark: My Kernel vs Intel oneDNN ==="
echo "CPU: $CPU_MODEL"
echo "Threads: $NCORES"
echo "AVX-512 Support Detected: $HAS_AVX512"
echo ""

export OMP_NUM_THREADS=$NCORES
build/google_bench \
    --benchmark_out=google_results.json \
    --benchmark_out_format=json \
    --benchmark_filter="BM_MyKernel|BM_OneDNN" | tee google_output.txt

echo ""
echo "=== Plotting ==="
export CPU_MODEL="$CPU_MODEL"
export HAS_AVX512="$HAS_AVX512"
python3 scripts/plot_google.py

echo "Plot saved to google_bench_plot.png"