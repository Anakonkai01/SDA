import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"


def load_json(path):
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def first_model(result):
    return next(iter(result.values()))


def plot_dpo_tradeoff():
    sft = first_model(load_json("results_b2_sft_strat1000.json"))
    dpo = first_model(load_json("reports/dpo_checkpoint_eval_balanced_v1/checkpoint_500pairs_results.json"))

    qtypes = [
        "attribute",
        "color",
        "count",
        "location",
        "negative",
        "shape",
        "sign_type",
        "yes_no",
    ]
    labels = [
        "Attribute",
        "Color",
        "Count",
        "Location",
        "Negative",
        "Shape",
        "Sign type",
        "Yes/No",
    ]
    sft_acc = [sft["by_question_type"][q]["vqa_accuracy"] for q in qtypes]
    dpo_acc = [dpo["by_question_type"][q]["vqa_accuracy"] for q in qtypes]

    x = np.arange(len(qtypes))
    width = 0.38

    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, sft_acc, width, label="B2-SFT", color="#4C78A8")
    plt.bar(x + width / 2, dpo_acc, width, label="B2-DPO 500 pairs", color="#F58518")
    plt.axhline(sft["vqa_accuracy"], color="#4C78A8", linestyle="--", linewidth=1, alpha=0.7)
    plt.axhline(dpo["vqa_accuracy"], color="#F58518", linestyle="--", linewidth=1, alpha=0.7)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylim(0, 1.08)
    plt.ylabel("VQA accuracy")
    plt.title("Trade-off của DPO theo loại câu hỏi (stratified 1000)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dpo_sft_tradeoff_vi.png", dpi=180)
    plt.close()


def plot_checkpoint_curve():
    summary = load_json("reports/dpo_checkpoint_eval_balanced_v1/summary.json")
    pair_points = []
    special = []
    for row in summary:
        checkpoint = row["checkpoint"]
        acc = row["accuracy"]
        match = re.search(r"checkpoint_(\d+)pairs", checkpoint)
        if match:
            pair_points.append((int(match.group(1)), acc))
        else:
            special.append((Path(checkpoint).name, acc))

    pair_points.sort()
    xs = [p for p, _ in pair_points]
    ys = [a for _, a in pair_points]

    plt.figure(figsize=(10, 5.6))
    plt.plot(xs, ys, marker="o", linewidth=2.4, color="#54A24B", label="Checkpoint theo số pairs")
    best_idx = int(np.argmax(ys))
    plt.scatter([xs[best_idx]], [ys[best_idx]], s=120, color="#E45756", zorder=3, label="Best by VQA eval")
    plt.annotate(
        f"{xs[best_idx]} pairs\nacc={ys[best_idx]:.3f}",
        xy=(xs[best_idx], ys[best_idx]),
        xytext=(xs[best_idx] + 45, ys[best_idx] + 0.0015),
        arrowprops={"arrowstyle": "->", "color": "#E45756"},
    )

    if special:
        offset_x = max(xs) + 70
        for i, (name, acc) in enumerate(special):
            plt.scatter([offset_x + i * 70], [acc], marker="X", s=100, color="#B279A2")
            plt.text(offset_x + i * 70, acc - 0.0014, name, ha="center", va="top", fontsize=9)

    plt.ylim(min(ys + [a for _, a in special]) - 0.003, max(ys) + 0.004)
    plt.xlabel("Số preference pairs đã train")
    plt.ylabel("VQA accuracy trên stratified 1000")
    plt.title("Đường cong checkpoint DPO cân bằng: metric không đi cùng DPO loss")
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dpo_balanced_checkpoint_curve_vi.png", dpi=180)
    plt.close()


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_dpo_tradeoff()
    plot_checkpoint_curve()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
