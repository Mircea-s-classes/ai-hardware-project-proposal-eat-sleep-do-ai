import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_metrics(csv_path: Path) -> pd.DataFrame:
    """读取单个 metrics CSV，并打上 model_tag。"""
    df = pd.read_csv(csv_path)
    model_tag = csv_path.stem.replace("kv_cache_dse_metrics_", "")
    df["model_tag"] = model_tag
    return df


def combine_metrics(csv_paths: List[Path]) -> pd.DataFrame:
    dfs = [load_metrics(p) for p in csv_paths]
    return pd.concat(dfs, ignore_index=True)


def build_throughput_table(df: pd.DataFrame) -> pd.DataFrame:
    """从 metrics 文本里抽出 prefill/decode TPS 和 latency，整理成宽表。"""
    target_metrics = [
        "prefill_tokens_per_s",
        "decode_tokens_per_s",
        "last_finished_request_end_to_end_latency_s",
        "last_finished_request_ttft_s",
    ]
    df_sel = df[df["metric_name"].isin(target_metrics)].copy()

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

    pivot = df_sel.pivot_table(
        index=key_cols,
        columns="metric_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()

    # 列名拍平
    pivot.columns.name = None

    return pivot


def plot_throughput_vs_context(pivot: pd.DataFrame, out_dir: Path) -> None:
    """在固定 design space（batch=1, 不开 sliding, sink=0）下，对比不同量化的 TPS vs context_window_size。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    mask = (
        (pivot["max_batch_size"] == 1)
        & (pivot["sliding_window_size"] <= 0)
        & (pivot["attention_sink_size"] == 0)
    )
    df_plot = pivot[mask].copy()
    if df_plot.empty:
        raise ValueError("筛选后的数据为空，请确认 metrics CSV 里有 batch=1, sliding=-1, sink=0 的配置。")

    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    # prefill_tokens_per_s
    plt.figure(figsize=(12, 6))
    ax1 = sns.lineplot(
        data=df_plot,
        x="context_window_size",
        y="prefill_tokens_per_s",
        hue="model_tag",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    ax1.set_xlabel("context_window_size")
    ax1.set_ylabel("prefill_tokens_per_s")
    ax1.set_title("Prefill tokens/s vs context window (different quantization)")
    ax1.legend(
        title="model / quantization",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.5,
    )
    plt.tight_layout(pad=1.5)
    out_path1 = out_dir / "throughput_prefill_vs_context.png"
    plt.savefig(out_path1, dpi=200)
    plt.close()

    # decode_tokens_per_s
    plt.figure(figsize=(12, 6))
    ax2 = sns.lineplot(
        data=df_plot,
        x="context_window_size",
        y="decode_tokens_per_s",
        hue="model_tag",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    ax2.set_xlabel("context_window_size")
    ax2.set_ylabel("decode_tokens_per_s")
    ax2.set_title("Decode tokens/s vs context window (different quantization)")
    ax2.legend(
        title="model / quantization",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.5,
    )
    plt.tight_layout(pad=1.5)
    out_path2 = out_dir / "throughput_decode_vs_context.png"
    plt.savefig(out_path2, dpi=200)
    plt.close()


def plot_decode_vs_batch(pivot: pd.DataFrame, out_dir: Path) -> None:
    """decode_tokens_per_s vs max_batch_size，在固定 sliding/sink 下看 batch 影响。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    mask = pivot["sliding_window_size"] <= 0
    df_plot = pivot[mask].copy()
    if df_plot.empty:
        return

    df_plot["max_batch_size"] = df_plot["max_batch_size"].astype(int)
    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    g = sns.relplot(
        data=df_plot,
        x="max_batch_size",
        y="decode_tokens_per_s",
        hue="model_tag",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g.set_axis_labels("max_batch_size", "decode_tokens_per_s")
    g.fig.suptitle("Decode tokens/s vs batch size (different quantization)", y=1.02)
    if g._legend is not None:
        g._legend.set_bbox_to_anchor((1.02, 1.0))
        g._legend.set_loc("upper left")
    g.fig.tight_layout()

    out_path = out_dir / "throughput_decode_vs_batch.png"
    g.savefig(out_path, dpi=200)


def plot_decode_vs_sliding_and_sink(pivot: pd.DataFrame, out_dir: Path) -> None:
    """decode_tokens_per_s vs sliding_window_size / attention_sink_size。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    # 只看开启 sliding 的配置
    mask = pivot["sliding_window_size"] > 0
    df_plot = pivot[mask].copy()
    if df_plot.empty:
        return

    df_plot["sliding_window_size"] = df_plot["sliding_window_size"].astype(int)
    df_plot["attention_sink_size"] = df_plot["attention_sink_size"].astype(int)
    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) sliding_window_size 维度，按 context facet，sink 作为 style
    g1 = sns.relplot(
        data=df_plot,
        x="sliding_window_size",
        y="decode_tokens_per_s",
        hue="model_tag",
        style="attention_sink_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g1.set_axis_labels("sliding_window_size", "decode_tokens_per_s")
    g1.fig.suptitle(
        "Decode tokens/s vs sliding_window_size (different quantization, varying sink)",
        y=1.02,
    )
    if g1._legend is not None:
        g1._legend.set_bbox_to_anchor((1.02, 1.0))
        g1._legend.set_loc("upper left")
    g1.fig.tight_layout()
    out_path1 = out_dir / "throughput_decode_vs_sliding.png"
    g1.savefig(out_path1, dpi=200)

    # 2) attention_sink_size 维度，按 context facet，sliding 作为 style
    g2 = sns.relplot(
        data=df_plot,
        x="attention_sink_size",
        y="decode_tokens_per_s",
        hue="model_tag",
        style="sliding_window_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g2.set_axis_labels("attention_sink_size", "decode_tokens_per_s")
    g2.fig.suptitle(
        "Decode tokens/s vs attention_sink_size (different quantization, varying sliding)",
        y=1.02,
    )
    if g2._legend is not None:
        g2._legend.set_bbox_to_anchor((1.02, 1.0))
        g2._legend.set_loc("upper left")
    g2.fig.tight_layout()
    out_path2 = out_dir / "throughput_decode_vs_sink.png"
    g2.savefig(out_path2, dpi=200)


def plot_prefill_vs_batch(pivot: pd.DataFrame, out_dir: Path) -> None:
    """prefill_tokens_per_s vs max_batch_size，在固定 sliding/sink 下看 batch 对 prefill 的影响。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    mask = pivot["sliding_window_size"] <= 0
    df_plot = pivot[mask].copy()
    if df_plot.empty:
        return

    df_plot["max_batch_size"] = df_plot["max_batch_size"].astype(int)
    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    g = sns.relplot(
        data=df_plot,
        x="max_batch_size",
        y="prefill_tokens_per_s",
        hue="model_tag",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g.set_axis_labels("max_batch_size", "prefill_tokens_per_s")
    g.fig.suptitle("Prefill tokens/s vs batch size (different quantization)", y=1.02)
    if g._legend is not None:
        g._legend.set_bbox_to_anchor((1.02, 1.0))
        g._legend.set_loc("upper left")
    g.fig.tight_layout()

    out_path = out_dir / "throughput_prefill_vs_batch.png"
    g.savefig(out_path, dpi=200)


def plot_prefill_vs_sliding_and_sink(pivot: pd.DataFrame, out_dir: Path) -> None:
    """prefill_tokens_per_s vs sliding_window_size / attention_sink_size。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    mask = pivot["sliding_window_size"] > 0
    df_plot = pivot[mask].copy()
    if df_plot.empty:
        return

    df_plot["sliding_window_size"] = df_plot["sliding_window_size"].astype(int)
    df_plot["attention_sink_size"] = df_plot["attention_sink_size"].astype(int)
    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    g1 = sns.relplot(
        data=df_plot,
        x="sliding_window_size",
        y="prefill_tokens_per_s",
        hue="model_tag",
        style="attention_sink_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g1.set_axis_labels("sliding_window_size", "prefill_tokens_per_s")
    g1.fig.suptitle(
        "Prefill tokens/s vs sliding_window_size (different quantization, varying sink)",
        y=1.02,
    )
    if g1._legend is not None:
        g1._legend.set_bbox_to_anchor((1.02, 1.0))
        g1._legend.set_loc("upper left")
    g1.fig.tight_layout()
    out_path1 = out_dir / "throughput_prefill_vs_sliding.png"
    g1.savefig(out_path1, dpi=200)

    g2 = sns.relplot(
        data=df_plot,
        x="attention_sink_size",
        y="prefill_tokens_per_s",
        hue="model_tag",
        style="sliding_window_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g2.set_axis_labels("attention_sink_size", "prefill_tokens_per_s")
    g2.fig.suptitle(
        "Prefill tokens/s vs attention_sink_size (different quantization, varying sliding)",
        y=1.02,
    )
    if g2._legend is not None:
        g2._legend.set_bbox_to_anchor((1.02, 1.0))
        g2._legend.set_loc("upper left")
    g2.fig.tight_layout()
    out_path2 = out_dir / "throughput_prefill_vs_sink.png"
    g2.savefig(out_path2, dpi=200)


def plot_latency_vs_context(pivot: pd.DataFrame, out_dir: Path) -> None:
    """end_to_end_latency / ttft vs context_window_size（batch=1, sliding=-1, sink=0）。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    cols_needed = [
        "last_finished_request_end_to_end_latency_s",
        "last_finished_request_ttft_s",
    ]
    for c in cols_needed:
        if c not in pivot.columns:
            return

    mask = (
        (pivot["max_batch_size"] == 1)
        & (pivot["sliding_window_size"] <= 0)
        & (pivot["attention_sink_size"] == 0)
    )
    df_plot = pivot[mask].copy()
    if df_plot.empty:
        return

    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    # end-to-end latency
    plt.figure(figsize=(12, 6))
    ax1 = sns.lineplot(
        data=df_plot,
        x="context_window_size",
        y="last_finished_request_end_to_end_latency_s",
        hue="model_tag",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    ax1.set_xlabel("context_window_size")
    ax1.set_ylabel("end_to_end_latency_s")
    ax1.set_title("End-to-end latency vs context window (different quantization)")
    ax1.legend(
        title="model / quantization",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.5,
    )
    plt.tight_layout(pad=1.5)
    out_path1 = out_dir / "latency_e2e_vs_context.png"
    plt.savefig(out_path1, dpi=200)
    plt.close()

    # TTFT
    plt.figure(figsize=(12, 6))
    ax2 = sns.lineplot(
        data=df_plot,
        x="context_window_size",
        y="last_finished_request_ttft_s",
        hue="model_tag",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    ax2.set_xlabel("context_window_size")
    ax2.set_ylabel("ttft_s")
    ax2.set_title("TTFT vs context window (different quantization)")
    ax2.legend(
        title="model / quantization",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.5,
    )
    plt.tight_layout(pad=1.5)
    out_path2 = out_dir / "latency_ttft_vs_context.png"
    plt.savefig(out_path2, dpi=200)
    plt.close()


def plot_latency_vs_batch(pivot: pd.DataFrame, out_dir: Path) -> None:
    """end_to_end_latency / ttft vs batch size，在固定 sliding/sink 下看 batch 影响。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    cols_needed = [
        "last_finished_request_end_to_end_latency_s",
        "last_finished_request_ttft_s",
    ]
    for c in cols_needed:
        if c not in pivot.columns:
            return

    mask = pivot["sliding_window_size"] <= 0
    df_plot = pivot[mask].copy()
    if df_plot.empty:
        return

    df_plot["max_batch_size"] = df_plot["max_batch_size"].astype(int)
    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    # end-to-end latency vs batch
    g1 = sns.relplot(
        data=df_plot,
        x="max_batch_size",
        y="last_finished_request_end_to_end_latency_s",
        hue="model_tag",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g1.set_axis_labels("max_batch_size", "end_to_end_latency_s")
    g1.fig.suptitle(
        "End-to-end latency vs batch size (different quantization)", y=1.02
    )
    if g1._legend is not None:
        g1._legend.set_bbox_to_anchor((1.02, 1.0))
        g1._legend.set_loc("upper left")
    g1.fig.tight_layout()
    out_path1 = out_dir / "latency_e2e_vs_batch.png"
    g1.savefig(out_path1, dpi=200)

    # TTFT vs batch
    g2 = sns.relplot(
        data=df_plot,
        x="max_batch_size",
        y="last_finished_request_ttft_s",
        hue="model_tag",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g2.set_axis_labels("max_batch_size", "ttft_s")
    g2.fig.suptitle("TTFT vs batch size (different quantization)", y=1.02)
    if g2._legend is not None:
        g2._legend.set_bbox_to_anchor((1.02, 1.0))
        g2._legend.set_loc("upper left")
    g2.fig.tight_layout()
    out_path2 = out_dir / "latency_ttft_vs_batch.png"
    g2.savefig(out_path2, dpi=200)


def plot_latency_vs_sliding_and_sink(pivot: pd.DataFrame, out_dir: Path) -> None:
    """end_to_end_latency / ttft vs sliding_window_size / attention_sink_size。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"lines.markersize": 8, "font.size": 12})

    cols_needed = [
        "last_finished_request_end_to_end_latency_s",
        "last_finished_request_ttft_s",
    ]
    for c in cols_needed:
        if c not in pivot.columns:
            return

    mask = pivot["sliding_window_size"] > 0
    df_plot = pivot[mask].copy()
    if df_plot.empty:
        return

    df_plot["sliding_window_size"] = df_plot["sliding_window_size"].astype(int)
    df_plot["attention_sink_size"] = df_plot["attention_sink_size"].astype(int)
    df_plot["context_window_size"] = df_plot["context_window_size"].astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    # end-to-end latency vs sliding
    g1 = sns.relplot(
        data=df_plot,
        x="sliding_window_size",
        y="last_finished_request_end_to_end_latency_s",
        hue="model_tag",
        style="attention_sink_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g1.set_axis_labels("sliding_window_size", "end_to_end_latency_s")
    g1.fig.suptitle(
        "End-to-end latency vs sliding_window_size (different quantization, varying sink)",
        y=1.02,
    )
    if g1._legend is not None:
        g1._legend.set_bbox_to_anchor((1.02, 1.0))
        g1._legend.set_loc("upper left")
    g1.fig.tight_layout()
    out_path1 = out_dir / "latency_e2e_vs_sliding.png"
    g1.savefig(out_path1, dpi=200)

    # end-to-end latency vs sink
    g2 = sns.relplot(
        data=df_plot,
        x="attention_sink_size",
        y="last_finished_request_end_to_end_latency_s",
        hue="model_tag",
        style="sliding_window_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g2.set_axis_labels("attention_sink_size", "end_to_end_latency_s")
    g2.fig.suptitle(
        "End-to-end latency vs attention_sink_size (different quantization, varying sliding)",
        y=1.02,
    )
    if g2._legend is not None:
        g2._legend.set_bbox_to_anchor((1.02, 1.0))
        g2._legend.set_loc("upper left")
    g2.fig.tight_layout()
    out_path2 = out_dir / "latency_e2e_vs_sink.png"
    g2.savefig(out_path2, dpi=200)

    # TTFT vs sliding
    g3 = sns.relplot(
        data=df_plot,
        x="sliding_window_size",
        y="last_finished_request_ttft_s",
        hue="model_tag",
        style="attention_sink_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g3.set_axis_labels("sliding_window_size", "ttft_s")
    g3.fig.suptitle(
        "TTFT vs sliding_window_size (different quantization, varying sink)",
        y=1.02,
    )
    if g3._legend is not None:
        g3._legend.set_bbox_to_anchor((1.02, 1.0))
        g3._legend.set_loc("upper left")
    g3.fig.tight_layout()
    out_path3 = out_dir / "latency_ttft_vs_sliding.png"
    g3.savefig(out_path3, dpi=200)

    # TTFT vs sink
    g4 = sns.relplot(
        data=df_plot,
        x="attention_sink_size",
        y="last_finished_request_ttft_s",
        hue="model_tag",
        style="sliding_window_size",
        col="context_window_size",
        kind="line",
        marker="o",
        linewidth=1.5,
        errorbar=None,
    )
    g4.set_axis_labels("attention_sink_size", "ttft_s")
    g4.fig.suptitle(
        "TTFT vs attention_sink_size (different quantization, varying sliding)",
        y=1.02,
    )
    if g4._legend is not None:
        g4._legend.set_bbox_to_anchor((1.02, 1.0))
        g4._legend.set_loc("upper left")
    g4.fig.tight_layout()
    out_path4 = out_dir / "latency_ttft_vs_sink.png"
    g4.savefig(out_path4, dpi=200)


def main():
    parser = argparse.ArgumentParser("根据 kv_cache_dse_metrics_*.csv 画 prefill/decode_tokens_per_s vs context_window_size 图")
    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        required=True,
        help="一个或多个 kv_cache_dse_metrics_*.csv 路径（不同量化、不同模型）",
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

    df_all = combine_metrics(csv_paths)
    pivot = build_throughput_table(df_all)

    out_dir = Path(args.out_dir)
    plot_throughput_vs_context(pivot, out_dir)
    plot_decode_vs_batch(pivot, out_dir)
    plot_decode_vs_sliding_and_sink(pivot, out_dir)
    plot_prefill_vs_batch(pivot, out_dir)
    plot_prefill_vs_sliding_and_sink(pivot, out_dir)
    plot_latency_vs_context(pivot, out_dir)
    plot_latency_vs_batch(pivot, out_dir)
    plot_latency_vs_sliding_and_sink(pivot, out_dir)

    print(f"Done. Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()


