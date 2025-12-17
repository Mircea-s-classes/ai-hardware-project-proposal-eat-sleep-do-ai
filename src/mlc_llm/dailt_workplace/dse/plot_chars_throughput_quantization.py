import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(csv_path: Path) -> pd.DataFrame:
    """读取单个 results CSV，并计算 per-run 的 chars/s。"""
    df = pd.read_csv(csv_path)
    model_tag = csv_path.stem.replace("kv_cache_dse_results_", "")
    df["model_tag"] = model_tag

    # 计算每次 run 的 approx_chars/s
    df["chars_per_s"] = df["approx_chars"] / df["time_s"].replace(0, pd.NA)
    return df


def combine_results(csv_paths: List[Path]) -> pd.DataFrame:
    dfs = [load_results(p) for p in csv_paths]
    return pd.concat(dfs, ignore_index=True)


def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """按配置平均 chars/s，得到每个 design point 的稳定吞吐。"""
    key_cols = [
        "model_tag",
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

    agg = (
        df.groupby(key_cols, as_index=False)["chars_per_s"]
        .mean()
        .rename(columns={"chars_per_s": "chars_per_s_mean"})
    )
    return agg


def plot_chars_vs_context(df_cfg: pd.DataFrame, out_dir: Path) -> None:
    """固定 batch=1, sliding=-1, sink=0，画 chars/s vs context_window_size（不同量化）。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    mask = (
        (df_cfg["max_batch_size"] == 1)
        & (df_cfg["sliding_window_size"] <= 0)
        & (df_cfg["attention_sink_size"] == 0)
    )
    df_plot = df_cfg[mask].copy()
    if df_plot.empty:
        raise ValueError("筛选后的数据为空，请确认 results CSV 里有 batch=1, sliding=-1, sink=0 的配置。")

    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        data=df_plot,
        x="context_window_size",
        y="chars_per_s_mean",
        hue="model_tag",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    ax.set_xlabel("context_window_size")
    ax.set_ylabel("approx chars/s")
    ax.set_title("Approx chars/s vs context window (different quantization)")
    ax.legend(
        title="model / quantization",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.5,
    )
    plt.tight_layout(pad=1.5)

    out_path = out_dir / "chars_throughput_vs_context.png"
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_chars_vs_batch(df_cfg: pd.DataFrame, out_dir: Path) -> None:
    """approx chars/s vs max_batch_size，在固定 sliding/sink 下看 batch 对整体吞吐的影响。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    mask = df_cfg["sliding_window_size"] <= 0
    df_plot = df_cfg[mask].copy()
    if df_plot.empty:
        return

    df_plot["max_batch_size"] = df_plot["max_batch_size"].astype(int)
    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    g = sns.relplot(
        data=df_plot,
        x="max_batch_size",
        y="chars_per_s_mean",
        hue="model_tag",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g.set_axis_labels("max_batch_size", "approx chars/s")
    g.fig.suptitle("Approx chars/s vs batch size (different quantization)", y=1.02)
    if g._legend is not None:
        g._legend.set_bbox_to_anchor((1.02, 1.0))
        g._legend.set_loc("upper left")
    g.fig.tight_layout()

    out_path = out_dir / "chars_throughput_vs_batch.png"
    g.savefig(out_path, dpi=200)


def plot_chars_vs_sliding_and_sink(df_cfg: pd.DataFrame, out_dir: Path) -> None:
    """approx chars/s vs sliding_window_size / attention_sink_size。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    mask = df_cfg["sliding_window_size"] > 0
    df_plot = df_cfg[mask].copy()
    if df_plot.empty:
        return

    df_plot["sliding_window_size"] = df_plot["sliding_window_size"].astype(int)
    df_plot["attention_sink_size"] = df_plot["attention_sink_size"].astype(int)
    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    g1 = sns.relplot(
        data=df_plot,
        x="sliding_window_size",
        y="chars_per_s_mean",
        hue="model_tag",
        style="attention_sink_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g1.set_axis_labels("sliding_window_size", "approx chars/s")
    g1.fig.suptitle(
        "Approx chars/s vs sliding_window_size (different quantization, varying sink)",
        y=1.02,
    )
    if g1._legend is not None:
        g1._legend.set_bbox_to_anchor((1.02, 1.0))
        g1._legend.set_loc("upper left")
    g1.fig.tight_layout()
    out_path1 = out_dir / "chars_throughput_vs_sliding.png"
    g1.savefig(out_path1, dpi=200)

    g2 = sns.relplot(
        data=df_plot,
        x="attention_sink_size",
        y="chars_per_s_mean",
        hue="model_tag",
        style="sliding_window_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g2.set_axis_labels("attention_sink_size", "approx chars/s")
    g2.fig.suptitle(
        "Approx chars/s vs attention_sink_size (different quantization, varying sliding)",
        y=1.02,
    )
    if g2._legend is not None:
        g2._legend.set_bbox_to_anchor((1.02, 1.0))
        g2._legend.set_loc("upper left")
    g2.fig.tight_layout()
    out_path2 = out_dir / "chars_throughput_vs_sink.png"
    g2.savefig(out_path2, dpi=200)


def main():
    parser = argparse.ArgumentParser(
        "根据 kv_cache_dse_results_*.csv 画 approx chars/s vs context_window_size（不同量化）"
    )
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        required=True,
        help="一个或多个 kv_cache_dse_results_*.csv 路径（不同量化、同一模型）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="输出图片目录",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]
    for p in csv_paths:
        if not p.is_file():
            raise FileNotFoundError(f"CSV 文件不存在: {p}")

    df_all = combine_results(csv_paths)
    df_cfg = aggregate_by_config(df_all)

    out_dir = Path(args.out_dir)
    plot_chars_vs_context(df_cfg, out_dir)
    plot_chars_vs_batch(df_cfg, out_dir)
    plot_chars_vs_sliding_and_sink(df_cfg, out_dir)

    print(f"Done. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()


