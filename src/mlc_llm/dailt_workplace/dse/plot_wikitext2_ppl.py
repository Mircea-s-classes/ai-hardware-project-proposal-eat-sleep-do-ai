import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_ppl(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # 给每一行加一个简单的 quant 标签，便于画图
    def _quant_tag(row):
        if row["fake_q3bit"]:
            return "fake_q3"
        if row["load_in_4bit"]:
            return "4bit"
        if row["load_in_8bit"]:
            return "8bit"
        return "fp16"

    df["quant"] = df.apply(_quant_tag, axis=1)
    return df


def plot_ppl_by_quant(df: pd.DataFrame, out_path: Path, dataset_name: str) -> None:
    """同一 HF 模型下，不同量化方式的 PPL 对比。"""
    sns.set_theme(style="whitegrid")
    sns.set_context("talk", rc={"font.size": 12})

    # 按 hf_model 分组画多个子图，当前你只有 8B 一个模型，也兼容之后加 1B
    g = sns.catplot(
        data=df,
        x="quant",
        y="ppl",
        hue="quant",
        col="hf_model",
        kind="bar",
        dodge=False,
        height=4,
        aspect=1.1,
    )
    g.set_axis_labels("quantization", f"PPL ({dataset_name})")
    g.fig.suptitle(f"{dataset_name} PPL vs quantization", y=1.05)
    for ax in g.axes.flat:
        # 旋转 x 轴标签
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
        # 在柱子顶部标注数值
        for p in ax.patches:
            height = p.get_height()
            if pd.isna(height):
                continue
            ax.text(
                p.get_x() + p.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(out_path, dpi=200)


def main():
    parser = argparse.ArgumentParser("根据 wikitext2_ppl_results.csv 画不同量化下的 PPL 对比图")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(Path(__file__).with_name("wikitext2_ppl_results.csv")),
        help="wikitext2_ppl_results.csv 路径（默认同目录下）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="wikitext2_ppl_vs_quant.png",
        help="输出图片路径",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = load_ppl(csv_path)

    out_path = Path(args.out)
    # 从文件名中猜一个数据集名字，例如 wikitext2_ppl_results.csv / wikitext103_ppl_results.csv
    stem = csv_path.stem  # e.g. wikitext2_ppl_results
    dataset_name = stem.split("_ppl_results")[0]

    plot_ppl_by_quant(df, out_path, dataset_name=dataset_name)
    print(f"Done. Plot saved to: {out_path}")


if __name__ == "__main__":
    main()


