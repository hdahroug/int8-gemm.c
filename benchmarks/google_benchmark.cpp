/*
 * Benchmarking performance
 * using fixed small range [-16,15] to make sure onednn gets correct answers
 */

#include <benchmark/benchmark.h>
#include <immintrin.h>
#include <omp.h>
#include "dnnl.h"
#include <vector>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifndef MC
extern __thread int MC, NC, KC;
#endif

namespace std_kernel {
#include "../src/kernel.h"
}

void init_matrix(int8_t* A, int size) {
    for (int i = 0; i < size; ++i) {
        A[i] = (int8_t)((rand() % 32) - 16);
    }
}

void flush_cache() {
    const size_t flush_size = 32 * 1024 * 1024; 
    volatile char* buf = (char*)malloc(flush_size);
    for (size_t i = 0; i < flush_size; i += 64) {
        buf[i] = (char)i;
    }
    free((void*)buf);
}

bool VerifyResults(int32_t *test, int32_t *ref, int n) {
    for (int i = 0; i < n; i++) {
        if (std::abs(test[i] - ref[i]) > 0) {
            return false;
        }
    }
    return true;
}

static void BM_MyKernel(benchmark::State& state) {
    int N = state.range(0);
    int m = N, n = N, k = N;
    
    int8_t* A = (int8_t*) _mm_malloc(m * k * sizeof(int8_t), 64);
    int8_t* B = (int8_t*) _mm_malloc(k * n * sizeof(int8_t), 64);
    int32_t* C = (int32_t*)_mm_malloc(m * n * sizeof(int32_t), 64);
    int32_t* C_ref = (int32_t*)_mm_malloc(m * n * sizeof(int32_t), 64);
    
    srand(time(NULL));
    init_matrix(A, m * k);
    init_matrix(B, k * n);
    memset(C, 0, m * n * sizeof(int32_t));
    memset(C_ref, 0, m * n * sizeof(int32_t));

    int8_t ao = 0, bo = 0; int32_t oc = 0;
    
    dnnl_gemm_s8s8s32('N', 'N', 'F', m, n, k, 1.0f, A, m, ao, B, k, bo, 0.0f, C_ref, m, &oc);
    
    std_kernel::kernel(m, n, k, B, m, A, k, C, m);
    
    if (!VerifyResults(C, C_ref, m * n)) {
        printf("[WARNING] Size %d Failed Validation!\n", N);
    }

    flush_cache(); 

    for (auto _ : state) {
        memset(C, 0, m * n * sizeof(int32_t));
        std_kernel::kernel(m, n, k, B, m, A, k, C, m);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory(); 
    }
    
    state.counters["GFLOPS"] = benchmark::Counter(
        (2.0 * m * n * k) / 1e9, benchmark::Counter::kIsIterationInvariantRate);
        
    _mm_free(A); _mm_free(B); _mm_free(C); _mm_free(C_ref);
}

static void BM_OneDNN(benchmark::State& state) {
    int N = state.range(0);
    int m = N, n = N, k = N;
    int8_t* A = (int8_t*) _mm_malloc(m * k * sizeof(int8_t), 64);
    int8_t* B = (int8_t*) _mm_malloc(k * n * sizeof(int8_t), 64);
    int32_t* C = (int32_t*) _mm_malloc(m * n * sizeof(int32_t), 64);
    
    int8_t ao = 0, bo = 0; int32_t oc = 0;
    srand(time(NULL));
    init_matrix(A, m * k);
    init_matrix(B, k * n);

    flush_cache();
    dnnl_gemm_s8s8s32('N', 'N', 'F', m, n, k, 1.0f, A, m, ao, B, k, bo, 0.0f, C, m, &oc);

    for (auto _ : state) {
        memset(C, 0, m * n * sizeof(int32_t));
        dnnl_gemm_s8s8s32('N', 'N', 'F', m, n, k, 1.0f, A, m, ao, B, k, bo, 0.0f, C, m, &oc);
        benchmark::DoNotOptimize(C);
        benchmark::ClobberMemory();
    }
    
    state.counters["GFLOPS"] = benchmark::Counter(
        (2.0 * m * n * k) / 1e9, benchmark::Counter::kIsIterationInvariantRate);
        
    _mm_free(A); _mm_free(B); _mm_free(C);
}

BENCHMARK(BM_MyKernel)->DenseRange(100, 6000, 100)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_OneDNN)->DenseRange(100, 6000, 100)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();