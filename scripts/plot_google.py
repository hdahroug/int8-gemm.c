import json
import os
import matplotlib.pyplot as plt
import pandas as pd

try:
    with open("google_results.json") as f:
        res = json.load(f)

    benchmarks = res["benchmarks"]
    data_list = []

    for b in benchmarks:
        if "aggregate_name" in b:
            continue

        name = b["name"]

        try:
            size = int(name.split("/")[-1])
        except ValueError:
            continue

        if "BM_MyKernel" in name:
            kernel_type = "ME"
        else:
            kernel_type = "DNNL"

        gflops = b.get("GFLOPS")
        if gflops is None:
            continue

        data_list.append({"Size": size, "GFLOPS": gflops, "Type": kernel_type})

    df = pd.DataFrame(data_list)
    me_df = df[df["Type"] == "ME"].sort_values("Size")
    dnnl_df = df[df["Type"] == "DNNL"].sort_values("Size")

    cpu_model = os.getenv("CPU_MODEL", "AVX2 CPU")
    has_avx512 = os.getenv("HAS_AVX512", "false") == "true"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    if not dnnl_df.empty:
        ax.plot(
            dnnl_df["Size"],
            dnnl_df["GFLOPS"],
            marker="o",
            markersize=4,
            linewidth=1.8,
            color="#0072B2",
            label="Intel oneDNN",
        )

    if not me_df.empty:
        ax.plot(
            me_df["Size"],
            me_df["GFLOPS"],
            marker="s",
            markersize=4,
            linewidth=1.8,
            color="#D55E00",
            label="My Custom Kernel",
        )

    ax.set_xlabel("Matrix Size (N)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title(f"INT8 GEMM Performance Comparison\nHardware: {cpu_model}")

    ax.grid(True, linewidth=0.4, alpha=0.3)

    ax.legend(frameon=False)


    if has_avx512:
        ax.text(
            0.98,
            0.05,
            f"Note:\nAVX-512 available\non this machine",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                linewidth=0.8,
            ),
        )

    plt.tight_layout()

    output_path = "google_bench_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Google Benchmark plot saved successfully to {output_path}!")

except Exception as e:
    print(f"Error in Google Plot: {e}")