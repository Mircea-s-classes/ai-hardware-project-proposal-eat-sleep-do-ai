import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_cfg_from_csv(csv_path: Path) -> pd.DataFrame:
    """从单个 results CSV 读取配置 + KV 显存信息，并去掉重复 run。"""
    df = pd.read_csv(csv_path)

    key_cols = [
        "model",
        "device",
        "context_window_size",
        "prefill_chunk_size",
        "max_batch_size",
        "sliding_window_size",
        "attention_sink_size",
        "num_runs",
        "max_tokens",
    ]
    df_cfg = df.drop_duplicates(subset=key_cols)

    # 用文件名里的 tag 标记量化方法（例如 Llama-3-8B-Instruct-q0f16-MLC）
    model_tag = csv_path.stem.replace("kv_cache_dse_results_", "")
    df_cfg["model_tag"] = model_tag
    return df_cfg


def combine_results(csv_paths: List[Path]) -> pd.DataFrame:
    dfs = [load_cfg_from_csv(p) for p in csv_paths]
    df = pd.concat(dfs, ignore_index=True)
    return df


def plot_kvcache_vs_context_across_quant(df: pd.DataFrame, out_path: Path) -> None:
    """在相同 design space 条件下，对比不同量化的 KV 显存 vs context_window_size。"""
    sns.set_theme(style="whitegrid")
    # 图整体放大，字体和标记略大
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    # 只挑一类典型配置，方便横向对比：
    # - max_batch_size = 1
    # - sliding_window_size = -1（不开 sliding）
    # - attention_sink_size = 0
    mask = (
        (df["max_batch_size"] == 1)
        & (df["sliding_window_size"] <= 0)
        & (df["attention_sink_size"] == 0)
    )
    df_plot = df[mask].copy()

    if df_plot.empty:
        raise ValueError("筛选后的数据为空，请确认 CSV 里有 batch=1, sliding=-1, sink=0 的配置。")

    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    # 加大画布，避免坐标轴和图例挤在一起
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df_plot,
        x="context_window_size",
        y="gpu_mem_kvcache_mb",
        hue="model_tag",
        marker="o",
        linewidth=1.5,
    )
    ax.set_xlabel("context_window_size")
    ax.set_ylabel("gpu_mem_kvcache_mb (MB)")
    ax.set_title("KV cache memory vs context window (different quantization)")
    # 图例放到右上角外侧，避免遮挡曲线/坐标
    ax.legend(
        title="model / quantization",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.5,
    )
    # 适当留白，避免横纵坐标文字贴边
    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_total_vs_context_across_quant(df: pd.DataFrame, out_path: Path) -> None:
    """在相同 design space 条件下，对比不同量化的 total GPU memory vs context_window_size。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    mask = (
        (df["max_batch_size"] == 1)
        & (df["sliding_window_size"] <= 0)
        & (df["attention_sink_size"] == 0)
    )
    df_plot = df[mask].copy()
    if df_plot.empty:
        raise ValueError("筛选后的数据为空，请确认 CSV 里有 batch=1, sliding=-1, sink=0 的配置。")

    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df_plot,
        x="context_window_size",
        y="gpu_mem_total_mb",
        hue="model_tag",
        marker="o",
        linewidth=1.5,
    )
    ax.set_xlabel("context_window_size")
    ax.set_ylabel("gpu_mem_total_mb (MB)")
    ax.set_title("Total GPU memory vs context window (different quantization)")
    ax.legend(
        title="model / quantization",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.5,
    )
    plt.tight_layout(pad=1.5)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="对比不同量化方法下 KV cache 显存随 context_window_size 的变化。"
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        required=True,
        help="一个或多个 kv_cache_dse_results_*.csv 文件路径，用空格分隔。",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="输出图片路径（例如 ./kv_cache_quant_compare.png）",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]
    for p in csv_paths:
        if not p.is_file():
            raise FileNotFoundError(f"CSV 文件不存在: {p}")

    df_all = combine_results(csv_paths)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # KV cache memory
    plot_kvcache_vs_context_across_quant(df_all, out_path)

    # Total GPU memory（文件名加后缀）
    total_out = out_path.with_name(out_path.stem + "_total" + out_path.suffix)
    plot_total_vs_context_across_quant(df_all, total_out)

    print(f"Done. Plots saved to: {out_path} and {total_out}")


if __name__ == "__main__":
    main()


