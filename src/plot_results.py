"""Render the accuracy intervals and answer outcomes for the frozen holdout."""

import argparse
import json
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


CONDITIONS = {
    "tfidf_frozen": "TF-IDF with frozen KB",
    "tfidf_updated": "TF-IDF with updated KB",
    "closed_book": "Qwen without retrieval",
    "bm25_updated": "BM25 with updated KB",
}
INK = "#15283B"
MUTED = "#475C6E"
GRID = "#DCE4EA"
BLUE = "#1E5985"
COLORS = ("#19776B", "#E2AC54", "#725699")


def create_figure(title, subtitle, right, bottom):
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    fig.subplots_adjust(left=0.295, right=right, top=0.72, bottom=bottom)
    fig.text(0.05, 0.918, title, ha="left", fontsize=19, fontweight="bold", color=INK)
    fig.text(0.05, 0.849, subtitle, ha="left", fontsize=11, color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(axis="both", length=0, pad=8, labelsize=10.5, colors=MUTED)
    ax.set_yticks(range(len(CONDITIONS)), list(CONDITIONS.values()), color=INK)
    ax.set_ylim(len(CONDITIONS) - 0.5, -0.5)
    return fig, ax


def accuracy_figure(summary, output):
    fig, ax = create_figure(
        "Release-tag accuracy on the final test set",
        "Dots show accuracy after tag normalization. Whiskers show 95% Wilson intervals.",
        right=0.70, bottom=0.20,
    )
    ax.set_xlim(-3, 103)
    ax.set_xticks(range(0, 101, 25))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Accuracy", color=MUTED, labelpad=12)
    transform = ax.get_yaxis_transform()
    for x, heading in ((1.09, "Correct / N"), (1.31, "Accuracy [95% CI]")):
        ax.text(x, -0.62, heading, transform=transform, color=MUTED,
                fontsize=9.3, fontweight="bold", clip_on=False)
    for i, condition in enumerate(CONDITIONS):
        result = summary["conditions"][condition]
        accuracy = 100 * result["accuracy"]
        low, high = (100 * value for value in result["wilson_95_ci"])
        ax.errorbar(
            accuracy, i, xerr=[[max(0.0, accuracy - low)], [max(0.0, high - accuracy)]],
            fmt="o", color=BLUE, markersize=7.5, elinewidth=2.1, capsize=4.5, capthick=1.7,
            zorder=3,
        )
        for x, label in ((1.09, f"{result['correct']}/{result['n']}"),
                         (1.31, f"{accuracy:.1f}% [{low:.1f}, {high:.1f}]")):
            ax.text(x, i, label, transform=transform, fontsize=10.3, color=INK,
                    va="center", clip_on=False)
    fig.savefig(output / "Assignment4_Fig1_Accuracy_Wilson95.png", dpi=300, facecolor="white")
    plt.close(fig)


def outcomes_figure(summary, output):
    n = summary["n_questions"]
    fig, ax = create_figure(
        "Answer outcomes by condition",
        f"Each condition answered the same {n} test questions.",
        right=0.93, bottom=0.25,
    )
    ax.set_xlim(0, n)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, n])
    ax.set_xlabel("Number of answers", color=MUTED, labelpad=10)
    keys = ("correct", "outdated", "abstention")
    labels = ("Correct", "Outdated release tag", "Abstained (UNKNOWN)")
    left = [0] * len(CONDITIONS)
    for key, label, color in zip(keys, labels, COLORS, strict=True):
        values = []
        for condition in CONDITIONS:
            result = summary["conditions"][condition]
            assert set(result["error_categories"]) <= {"outdated", "abstention"}
            values.append(result["correct"] if key == "correct" else result["error_categories"].get(key, 0))
        ax.barh(range(len(CONDITIONS)), values, left=left, height=0.54, color=color, label=label)
        for i, value in enumerate(values):
            if value:
                ax.text(left[i] + value / 2, i, str(value), ha="center", va="center",
                        color=INK if key == "outdated" else "white", fontsize=11, fontweight="bold")
            left[i] += value
    assert left == [n] * len(CONDITIONS)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=3,
                       frameon=False, fontsize=10, handlelength=1.25, columnspacing=1.65)
    plt.setp(legend.get_texts(), color=INK)
    fig.savefig(output / "Assignment4_Fig2_Answer_Outcomes.png", dpi=300, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_bytes())
    args.output.mkdir(parents=True, exist_ok=True)
    accuracy_figure(summary, args.output)
    outcomes_figure(summary, args.output)


if __name__ == "__main__":
    main()
