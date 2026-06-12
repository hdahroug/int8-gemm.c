cmake_minimum_required(VERSION 3.16)
project(int8_gemm C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-O3 -march=core-avx2)

# ── OpenMP ────────────────────────────────────────────────────────────────────
find_package(OpenMP REQUIRED)

# ── oneDNN ────────────────────────────────────────────────────────────────────
find_library(DNNL_LIB dnnl)
find_path(DNNL_INCLUDE dnnl.h)
if(NOT DNNL_LIB OR NOT DNNL_INCLUDE)
    message(FATAL_ERROR
        "\n[MISSING] oneDNN\n"
        "  sudo apt-get install libdnnl-dev\n")
endif()
message(STATUS "oneDNN   : ${DNNL_LIB}")

# ── Google Benchmark ──────────────────────────────────────────────────────────
find_library(BENCHMARK_LIB benchmark)
find_path(BENCHMARK_INCLUDE benchmark/benchmark.h)
if(NOT BENCHMARK_LIB OR NOT BENCHMARK_INCLUDE)
    message(FATAL_ERROR
        "\n[MISSING] Google Benchmark\n"
        "  sudo apt-get install libbenchmark-dev\n")
endif()
message(STATUS "Benchmark: ${BENCHMARK_LIB}")

# ── MKL ───────────────────────────────────────────────────────────────────────
# Try system MKL first (apt install libmkl-dev or intel-mkl)
# then fall back to oneAPI install via MKLROOT or -DMKL_ROOT=
find_path(MKL_INCLUDE mkl.h
    PATHS
        /usr/include/mkl
        /usr/include
        /usr/local/include/mkl
        ${MKLROOT}/include
        $ENV{MKLROOT}/include)

find_library(MKL_RT_LIB mkl_rt
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        ${MKLROOT}/lib
        ${MKLROOT}/lib/intel64
        $ENV{MKLROOT}/lib
        $ENV{MKLROOT}/lib/intel64)

if(MKL_INCLUDE AND MKL_RT_LIB)
    # System MKL — link shared libmkl_rt
    set(MKL_LIBS ${MKL_RT_LIB})
    set(MKL_LINK_TYPE "shared (libmkl_rt)")
else()
    # Try oneAPI static MKL
    if(DEFINED MKL_ROOT)
        set(_mkl_search ${MKL_ROOT})
    elseif(DEFINED ENV{MKLROOT})
        set(_mkl_search $ENV{MKLROOT})
    else()
        foreach(candidate
            /opt/intel/oneapi/mkl/latest
            /opt/intel/oneapi/mkl/2025.0
            /opt/intel/oneapi/mkl/2024.2
            /opt/intel/mkl)
            if(EXISTS "${candidate}/include/mkl.h")
                set(_mkl_search ${candidate})
                break()
            endif()
        endforeach()
    endif()

    if(NOT _mkl_search)
        message(FATAL_ERROR
            "\n[MISSING] Intel MKL\n"
            "Option 1 — install system MKL (recommended):\n"
            "  sudo apt-get install libmkl-dev\n\n"
            "Option 2 — install Intel oneAPI MKL:\n"
            "  sudo apt-get install intel-oneapi-mkl-devel\n"
            "  source /opt/intel/oneapi/setvars.sh\n\n"
            "Option 3 — point to existing install:\n"
            "  cmake -B build -DMKL_ROOT=/path/to/mkl\n")
    endif()

    set(MKL_INCLUDE "${_mkl_search}/include")
    find_library(MKL_INTEL libmkl_intel_lp64.a
        PATHS "${_mkl_search}/lib" "${_mkl_search}/lib/intel64" NO_DEFAULT_PATH)
    find_library(MKL_GNU   libmkl_gnu_thread.a
        PATHS "${_mkl_search}/lib" "${_mkl_search}/lib/intel64" NO_DEFAULT_PATH)
    find_library(MKL_CORE  libmkl_core.a
        PATHS "${_mkl_search}/lib" "${_mkl_search}/lib/intel64" NO_DEFAULT_PATH)

    if(NOT MKL_INTEL OR NOT MKL_GNU OR NOT MKL_CORE)
        message(FATAL_ERROR
            "\n[MISSING] MKL static libs under ${_mkl_search}\n"
            "Try: sudo apt-get install libmkl-dev\n")
    endif()

    set(MKL_LIBS
        -Wl,--start-group
        ${MKL_INTEL} ${MKL_GNU} ${MKL_CORE}
        -Wl,--end-group)
    set(MKL_LINK_TYPE "static (oneAPI)")
endif()

message(STATUS "MKL      : ${MKL_INCLUDE} [${MKL_LINK_TYPE}]")

# ── Python packages ───────────────────────────────────────────────────────────
find_package(Python3 COMPONENTS Interpreter REQUIRED)
foreach(pkg matplotlib numpy pandas)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import ${pkg}"
        RESULT_VARIABLE _res OUTPUT_QUIET ERROR_QUIET)
    if(NOT _res EQUAL 0)
        message(STATUS "Installing Python package: ${pkg}")
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pip install ${pkg}
            RESULT_VARIABLE _pip)
        if(NOT _pip EQUAL 0)
            message(FATAL_ERROR "\n[MISSING] pip install ${pkg} failed\n")
        endif()
    endif()
endforeach()

# ── AVX-512 detection ─────────────────────────────────────────────────────────
execute_process(
    COMMAND grep -c avx512 /proc/cpuinfo
    OUTPUT_VARIABLE AVX512_COUNT
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(AVX512_COUNT GREATER 0)
    message(STATUS "AVX-512  : detected — oneDNN will use 512-bit instructions")
else()
    message(STATUS "AVX-512  : not detected — pure AVX2 path")
endif()

# ── Autotuner ────────────────────────────────────────────────────────────────
set(AUTOTUNE_RESULT /tmp/autotune_result.txt)

if(NOT EXISTS ${AUTOTUNE_RESULT})
    message(STATUS "")
    message(STATUS "Running autotuner — this takes 1-2 minutes, cached after first run...")
    message(STATUS "")

    execute_process(
        COMMAND gcc autotuner/autotune.c
            -I${CMAKE_SOURCE_DIR}
            -O3 -march=core-avx2 -fopenmp
            -o ${CMAKE_BINARY_DIR}/autotuner_tmp
            -lm -lpthread
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE _compile_res
        ERROR_VARIABLE  _compile_err)

    if(NOT _compile_res EQUAL 0)
        message(FATAL_ERROR "Failed to compile autotuner:\n${_compile_err}")
    endif()

    execute_process(
        COMMAND ${CMAKE_BINARY_DIR}/autotuner_tmp
        RESULT_VARIABLE _tune_res)

    if(NOT _tune_res EQUAL 0)
        message(FATAL_ERROR "Autotuner failed.")
    endif()
else()
    message(STATUS "Autotuner: using cached result (delete /tmp/autotune_result.txt to re-run)")
endif()

file(READ ${AUTOTUNE_RESULT} _raw)
string(STRIP "${_raw}" _raw)
separate_arguments(_vals UNIX_COMMAND "${_raw}")
list(GET _vals 0 BEST_MC)
list(GET _vals 1 BEST_NC)
list(GET _vals 2 BEST_KC)
list(GET _vals 3 BEST_PA1)
list(GET _vals 4 BEST_PB1)
list(GET _vals 5 BEST_PA2)
list(GET _vals 6 BEST_PB2)

message(STATUS "Tiles    : MC=${BEST_MC} NC=${BEST_NC} KC=${BEST_KC}")
message(STATUS "Prefetch : A_L1=${BEST_PA1} B_L1=${BEST_PB1} A_L2=${BEST_PA2} B_L2=${BEST_PB2}")

set(TUNE_DEFS
    MC=${BEST_MC} NC=${BEST_NC} KC=${BEST_KC}
    PREFETCH_A_L1=${BEST_PA1} PREFETCH_B_L1=${BEST_PB1}
    PREFETCH_A_L2=${BEST_PA2} PREFETCH_B_L2=${BEST_PB2})

# ── matmul binary ─────────────────────────────────────────────────────────────
add_executable(matmul src/matmul.c)
target_include_directories(matmul PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${MKL_INCLUDE} ${DNNL_INCLUDE})
target_compile_definitions(matmul PRIVATE ${TUNE_DEFS})
target_link_libraries(matmul PRIVATE
    OpenMP::OpenMP_C ${MKL_LIBS} ${DNNL_LIB} m pthread dl)

# ── google_bench binary ───────────────────────────────────────────────────────
add_executable(google_bench benchmarks/google_benchmark.cpp)
target_include_directories(google_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${MKL_INCLUDE} ${DNNL_INCLUDE} ${BENCHMARK_INCLUDE})
target_compile_definitions(google_bench PRIVATE ${TUNE_DEFS})
target_link_libraries(google_bench PRIVATE
    OpenMP::OpenMP_CXX ${MKL_LIBS} ${DNNL_LIB} ${BENCHMARK_LIB} m pthread dl)

message(STATUS "")
message(STATUS "Configure done. Run:")
message(STATUS "  cmake --build build -j\$(nproc)")
message(STATUS "  bash scripts/bench.sh")
message(STATUS "")