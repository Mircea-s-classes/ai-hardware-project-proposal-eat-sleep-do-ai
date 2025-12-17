import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    """读 results CSV，并按参数配置去重（同一配置多次 run 显存相同，只保留一条）。"""
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

    # 同一配置的 gpu_mem_* 在脚本中是初始化时解析出来的，理论上完全相同，直接去重即可
    df_cfg = df.drop_duplicates(subset=key_cols)
    return df_cfg


def plot_memory_vs_context(df: pd.DataFrame, out_dir: Path, model_tag: str) -> None:
    """显存 vs context_window_size，分 batch / sliding window / sink 看趋势。"""
    sns.set_theme(style="whitegrid")
    # 放大散点标记，便于在 design space 上看清每个配置
    sns.set_context("talk", rc={"lines.markersize": 8})

    # 只要需要的列，做个拷贝避免 SettingWithCopyWarning
    cols = [
        "context_window_size",
        "max_batch_size",
        "sliding_window_size",
        "attention_sink_size",
        "gpu_mem_kvcache_mb",
        "gpu_mem_total_mb",
    ]
    df_plot = df[cols].copy()

    # 方便图例展示
    df_plot["sliding_window_size"] = df_plot["sliding_window_size"].astype(int)
    df_plot["attention_sink_size"] = df_plot["attention_sink_size"].astype(int)
    df_plot["max_batch_size"] = df_plot["max_batch_size"].astype(int)

    # KV Cache 显存 vs context window
    g = sns.relplot(
        data=df_plot,
        x="context_window_size",
        y="gpu_mem_kvcache_mb",
        hue="sliding_window_size",
        style="attention_sink_size",
        col="max_batch_size",
        kind="line",
        marker="o",
        linewidth=1.0,
    )
    g.set_axis_labels("context_window_size", "gpu_mem_kvcache_mb (MB)")
    g.fig.suptitle(f"KV cache memory vs context window ({model_tag})", y=1.02)
    # 把图例移到右侧，避免和图像重叠
    if g._legend is not None:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_loc("center left")
    g.fig.tight_layout()
    out_path = out_dir / f"{model_tag}_mem_kvcache_vs_context.png"
    g.savefig(out_path, dpi=200)

    # 总显存 vs context window（同样的 facet）
    g2 = sns.relplot(
        data=df_plot,
        x="context_window_size",
        y="gpu_mem_total_mb",
        hue="sliding_window_size",
        style="attention_sink_size",
        col="max_batch_size",
        kind="line",
        marker="o",
        linewidth=1.0,
    )
    g2.set_axis_labels("context_window_size", "gpu_mem_total_mb (MB)")
    g2.fig.suptitle(f"Total GPU memory vs context window ({model_tag})", y=1.02)
    if g2._legend is not None:
        g2._legend.set_bbox_to_anchor((1.02, 0.5))
        g2._legend.set_loc("center left")
    g2.fig.tight_layout()
    out_path2 = out_dir / f"{model_tag}_mem_total_vs_context.png"
    g2.savefig(out_path2, dpi=200)


def plot_memory_vs_sliding_window(df: pd.DataFrame, out_dir: Path, model_tag: str) -> None:
    """显存 vs sliding_window_size，按 context_window_size 分列，看 sliding window / sink 的影响。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8})

    cols = [
        "context_window_size",
        "max_batch_size",
        "sliding_window_size",
        "attention_sink_size",
        "gpu_mem_kvcache_mb",
    ]
    df_plot = df[cols].copy()

    df_plot["sliding_window_size"] = df_plot["sliding_window_size"].astype(int)
    df_plot["attention_sink_size"] = df_plot["attention_sink_size"].astype(int)
    df_plot["max_batch_size"] = df_plot["max_batch_size"].astype(int)
    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    g = sns.relplot(
        data=df_plot,
        x="sliding_window_size",
        y="gpu_mem_kvcache_mb",
        hue="attention_sink_size",
        style="max_batch_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.0,
    )
    g.set_axis_labels("sliding_window_size", "gpu_mem_kvcache_mb (MB)")
    g.fig.suptitle(f"KV cache memory vs sliding window ({model_tag})", y=1.02)
    if g._legend is not None:
        g._legend.set_bbox_to_anchor((1.02, 0.5))
        g._legend.set_loc("center left")
    g.fig.tight_layout()
    out_path = out_dir / f"{model_tag}_mem_kvcache_vs_sliding.png"
    g.savefig(out_path, dpi=200)


def main():
    parser = argparse.ArgumentParser(
        description="从 kv_cache_dse_results_*.csv 画出显存 vs 设计空间参数的图。"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="kv_cache_dse_results_*.csv 的路径",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="输出图片目录（默认当前目录）",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_cfg = load_and_aggregate(csv_path)

    # 从文件名推一个 tag，类似 bench 脚本里的逻辑
    model_tag = csv_path.stem.replace("kv_cache_dse_results_", "")

    plot_memory_vs_context(df_cfg, out_dir, model_tag)
    plot_memory_vs_sliding_window(df_cfg, out_dir, model_tag)

    print(f"Done. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()


