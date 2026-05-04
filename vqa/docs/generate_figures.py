"""Generate all figures for VQA report."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi":     140,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

MODEL_COLORS = {
    "A1":  "#1976D2",
    "A2":  "#E91E63",
    "B1":  "#FF8F00",
    "B2":  "#388E3C",
    "DPO": "#7B1FA2",
}
SPLIT_COLORS = {"train": "#42A5F5", "val": "#FFA726", "test": "#66BB6A"}


# ── 1. Dataset split overview ────────────────────────────────────────────────

def fig_data_split():
    data = {
        "Train": {"Images": 2193, "QA Pairs": 104146},
        "Val":   {"Images": 272,  "QA Pairs": 12944},
        "Test":  {"Images": 271,  "QA Pairs": 12966},
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Images
    ax = axes[0]
    splits = list(data.keys())
    vals   = [data[s]["Images"] for s in splits]
    bars   = ax.bar(splits, vals, color=list(SPLIT_COLORS.values()), width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Number of Images per Split", fontweight="bold")
    ax.set_ylabel("Images")
    ax.set_ylim(0, max(vals) * 1.2)

    # QA pairs
    ax = axes[1]
    vals = [data[s]["QA Pairs"] for s in splits]
    bars = ax.bar(splits, vals, color=list(SPLIT_COLORS.values()), width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
                f"{v:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Number of QA Pairs per Split", fontweight="bold")
    ax.set_ylabel("QA Pairs")
    ax.set_ylim(0, max(vals) * 1.2)

    fig.suptitle("Dataset Split Overview", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_split_overview.png", bbox_inches="tight")
    plt.close()
    print("✓ data_split_overview.png")


# ── 2. Question type distribution ───────────────────────────────────────────

def fig_question_type_dist():
    qtype_test = {
        "yes_no": 1635, "count": 1635, "sign_type": 1635, "color": 1635,
        "shape": 1635, "negative": 1635, "location": 1632, "attribute": 1524,
    }
    labels = list(qtype_test.keys())
    vals   = list(qtype_test.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6, edgecolor="white")
    for bar, v in zip(bars, vals[::-1]):
        ax.text(v + 10, bar.get_y() + bar.get_height()/2,
                f"{v:,}", va="center", fontsize=10)
    ax.set_xlabel("Number of QA pairs")
    ax.set_title("Question Type Distribution — Test Set", fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.15)

    # Add total annotation
    ax.text(0.98, 0.02, f"Total: {sum(vals):,} QA pairs\n8 question types",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgray"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_qtype_dist.png", bbox_inches="tight")
    plt.close()
    print("✓ data_qtype_dist.png")


# ── 3. Answer type distribution ─────────────────────────────────────────────

def fig_answer_type():
    sizes  = [8061, 3270, 1635]
    labels = ["Other\n(names, colors,\nshapes, locations)", "Yes/No\n(Có/Không)", "Number\n(counts)"]
    colors = ["#42A5F5", "#EF5350", "#66BB6A"]
    explode = (0.03, 0.03, 0.03)

    fig, ax = plt.subplots(figsize=(7, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", pctdistance=0.75, startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=10),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")

    ax.set_title("Answer Type Distribution — Test Set\n(Total: 12,966 QA pairs)",
                 fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_answer_type.png", bbox_inches="tight")
    plt.close()
    print("✓ data_answer_type.png")


# ── 4. Split balance heatmap ─────────────────────────────────────────────────

def fig_split_balance():
    data = {
        "train": {"yes_no":13131,"count":13131,"sign_type":13131,"color":13131,
                  "shape":13131,"negative":13131,"location":13112,"attribute":12248},
        "val":   {"yes_no":1632,"count":1632,"sign_type":1632,"color":1632,
                  "shape":1632,"negative":1632,"location":1629,"attribute":1523},
        "test":  {"yes_no":1635,"count":1635,"sign_type":1635,"color":1635,
                  "shape":1635,"negative":1635,"location":1632,"attribute":1524},
    }
    qtypes = ["yes_no","count","sign_type","color","shape","negative","location","attribute"]
    splits = ["train","val","test"]

    # Normalize each split to percentage
    matrix = np.array([[data[s][q]/sum(data[s].values())*100 for q in qtypes] for s in splits])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    im = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0, vmax=15)
    ax.set_xticks(range(len(qtypes)))
    ax.set_xticklabels(qtypes, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(splits)))
    ax.set_yticklabels([s.capitalize() for s in splits])
    for i in range(len(splits)):
        for j in range(len(qtypes)):
            ax.text(j, i, f"{matrix[i,j]:.1f}%", ha="center", va="center",
                    fontsize=9, color="black" if matrix[i,j] < 10 else "white")
    plt.colorbar(im, ax=ax, label="% of split total", shrink=0.8)
    ax.set_title("Question Type Balance Across Splits (% of each split)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_split_balance.png", bbox_inches="tight")
    plt.close()
    print("✓ data_split_balance.png")


# ── 5. Per-qtype accuracy heatmap ────────────────────────────────────────────

def fig_qtype_heatmap():
    # Question types shared across all 4 models
    COMMON = ["yes_no","count","sign_type","color","shape","negative","location","attribute"]
    EXTRA  = ["context","count_total","multi_object","spatial_rel"]  # A1/A2/B2 only

    accs = {
        "A1":  {"yes_no":1.000,"count":0.985,"sign_type":0.939,"color":0.976,
                "shape":0.974,"negative":1.000,"location":0.796,"attribute":0.998,
                "context":1.000,"count_total":0.846,"multi_object":0.942,"spatial_rel":0.863},
        "A2":  {"yes_no":0.998,"count":0.982,"sign_type":0.918,"color":0.982,
                "shape":0.983,"negative":1.000,"location":0.749,"attribute":0.997,
                "context":0.998,"count_total":0.810,"multi_object":0.925,"spatial_rel":0.816},
        "B1":  {"yes_no":0.326,"count":0.597,"sign_type":0.007,"color":0.017,
                "shape":0.001,"negative":0.608,"location":0.000,"attribute":0.000,
                "context":None,"count_total":None,"multi_object":None,"spatial_rel":None},
        "B2":  {"yes_no":0.987,"count":0.992,"sign_type":0.902,"color":0.971,
                "shape":0.970,"negative":1.000,"location":0.772,"attribute":1.000,
                "context":0.993,"count_total":0.780,"multi_object":0.944,"spatial_rel":0.905},
    }

    all_types = COMMON + EXTRA
    models    = ["A1", "A2", "B1", "B2"]
    matrix    = np.full((len(models), len(all_types)), np.nan)
    for i, m in enumerate(models):
        for j, qt in enumerate(all_types):
            v = accs[m].get(qt)
            if v is not None:
                matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(13, 4))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#E0E0E0")

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(all_types)))
    ax.set_xticklabels(all_types, rotation=35, ha="right", fontsize=9.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([f"{m} ({'LSTM' if m=='A1' else 'Transf.' if m=='A2' else 'Zero-shot' if m=='B1' else 'LoRA'})" for m in models], fontsize=10)

    for i in range(len(models)):
        for j in range(len(all_types)):
            if not np.isnan(matrix[i, j]):
                val = matrix[i, j]
                color = "white" if val < 0.5 or val > 0.85 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold")
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=8, color="#9E9E9E")

    # Vertical separator between common and extra
    ax.axvline(len(COMMON) - 0.5, color="gray", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.text(len(COMMON) - 0.5, -0.7, "A1/A2/B2 only →",
            ha="center", va="top", fontsize=8, color="gray", style="italic")

    plt.colorbar(im, ax=ax, label="VQA Accuracy", shrink=0.85)
    ax.set_title("Per-Question-Type VQA Accuracy — All Models (Full Test Set)", fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qtype_accuracy_heatmap.png", bbox_inches="tight")
    plt.close()
    print("✓ qtype_accuracy_heatmap.png")


# ── 6. Radar chart — model comparison ────────────────────────────────────────

def fig_radar():
    metrics = ["VQA Acc", "BLEU-4", "ROUGE-L", "METEOR", "BERTScore"]
    model_data = {
        "A1": [0.9484, 0.9602, 0.9584, 0.9558, 0.9631],
        "A2": [0.9377, 0.9476, 0.9486, 0.9454, 0.9713],
        "B1": [0.1962, 0.0350, 0.2899, 0.3017, 0.4753],
        "B2": [0.9379, 0.9494, 0.9508, 0.9478, 0.9111],
    }

    N    = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for model, values in model_data.items():
        vals = values + values[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=model, color=MODEL_COLORS[model], markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="gray")
    ax.grid(color="lightgray", linestyle="--", linewidth=0.8)
    ax.set_title("Model Comparison — All Metrics\n(Full Test Set)", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_radar.png", bbox_inches="tight")
    plt.close()
    print("✓ model_radar.png")


# ── 7. DPO per-qtype tradeoff (replace table) ────────────────────────────────

def fig_dpo_tradeoff():
    qtypes = ["negative","location","color","yes_no","count","shape","attribute","sign_type"]
    sft_acc = [0.387, 0.840, 0.848, 1.000, 0.824, 0.936, 0.464, 0.672]  # wait, negative for SFT fulltest
    # From results_b2_sft_strat1000.json and results_b2_dpo_strat1000.json
    sft_acc = [0.336, 0.840, 0.848, 1.000, 0.824, 0.936, 0.464, 0.672]
    dpo_acc = [0.536, 0.888, 0.856, 0.992, 0.816, 0.928, 0.424, 0.600]
    delta   = [d - s for d, s in zip(dpo_acc, sft_acc)]

    x = np.arange(len(qtypes))
    width = 0.35
    colors_delta = ["#388E3C" if d >= 0 else "#D32F2F" for d in delta]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: grouped bar SFT vs DPO
    ax = axes[0]
    ax.bar(x - width/2, sft_acc, width, label="B2-SFT", color="#4CAF50", alpha=0.85, edgecolor="white")
    ax.bar(x + width/2, dpo_acc, width, label="B2-DPO (100p)", color="#7B1FA2", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(qtypes, rotation=30, ha="right", fontsize=9.5)
    ax.set_ylabel("VQA Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("B2-SFT vs B2-DPO by Question Type\n(Stratified-1000)", fontweight="bold")
    ax.legend()
    for xi, (sv, dv) in enumerate(zip(sft_acc, dpo_acc)):
        ax.text(xi - width/2, sv + 0.01, f"{sv:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.text(xi + width/2, dv + 0.01, f"{dv:.2f}", ha="center", va="bottom", fontsize=7.5)

    # Right: delta bar
    ax = axes[1]
    bars = ax.bar(x, delta, color=colors_delta, alpha=0.85, edgecolor="white", width=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(qtypes, rotation=30, ha="right", fontsize=9.5)
    ax.set_ylabel("Δ Accuracy (DPO − SFT)")
    ax.set_title("DPO Improvement/Regression by Question Type", fontweight="bold")
    for bar, d in zip(bars, delta):
        ypos = bar.get_height() + 0.005 if d >= 0 else bar.get_height() - 0.018
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{d:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "dpo_qtype_tradeoff.png", bbox_inches="tight")
    plt.close()
    print("✓ dpo_qtype_tradeoff.png")


# ── 8. Latency comparison (improved) ─────────────────────────────────────────

def fig_latency():
    models   = ["A1\n(LSTM)", "A2\n(Transformer)", "B1\n(Zero-shot)", "B2\n(SFT)", "B2-DPO\n(100p)"]
    latency  = [11.0, 12.5, 179, 484, 216]
    colors   = [MODEL_COLORS["A1"], MODEL_COLORS["A2"], MODEL_COLORS["B1"],
                MODEL_COLORS["B2"], MODEL_COLORS["DPO"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(models, latency, color=colors, width=0.5, edgecolor="white")
    for bar, v in zip(bars, latency):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{v:.0f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Inference latency (ms / sample)")
    ax.set_title("Inference Latency Comparison", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(5, 1000)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}ms"))
    ax.axhline(12.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Speed annotations
    base = latency[3]  # B2 SFT as baseline
    for i, (bar, v) in enumerate(zip(bars, latency)):
        if i < 2:
            ratio = base / v
            ax.text(bar.get_x() + bar.get_width()/2, v * 1.8,
                    f"{ratio:.0f}× faster", ha="center", fontsize=8.5, color="gray")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "latency_improved.png", bbox_inches="tight")
    plt.close()
    print("✓ latency_improved.png")


# ── 9. Detailed answer type breakdown ───────────────────────────────────────

def fig_answer_type_detailed():
    labels = ["yes_no", "negative", "count", "sign_type", "color", "shape", "location", "attribute"]
    counts = [1635, 1635, 1635, 1635, 1635, 1635, 1632, 1524]
    groups = ["Binary", "Binary", "Numeric", "Open", "Open", "Open", "Open", "Open"]
    G_COLORS = {"Binary": "#EF5350", "Numeric": "#42A5F5", "Open": "#66BB6A"}
    bar_colors = [G_COLORS[g] for g in groups]
    total = sum(counts)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})

    ax = axes[0]
    bars = ax.barh(labels[::-1], counts[::-1], color=bar_colors[::-1], height=0.6, edgecolor="white")
    for bar, cnt in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}  ({cnt / total * 100:.1f}%)", va="center", fontsize=9.5)
    ax.set_xlim(0, 2300)
    ax.set_xlabel("QA pairs in test set")
    ax.set_title("Answer Counts by Question Type", fontweight="bold")
    ax.axhline(5.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(4.5, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.text(-50, 6.5, "Binary\n(Có/Không)", fontsize=8.5, color=G_COLORS["Binary"],
            fontweight="bold", ha="left", va="center")
    ax.text(-50, 5.0, "Numeric", fontsize=8.5, color=G_COLORS["Numeric"],
            fontweight="bold", ha="left", va="center")
    ax.text(-50, 2.0, "Open-ended\n(names, colors,\nshapes, locations\n& attributes)",
            fontsize=8.5, color=G_COLORS["Open"], fontweight="bold", ha="left", va="center")

    ax = axes[1]
    g_sums = {
        "Binary\n(Có / Không)":   sum(c for c, g in zip(counts, groups) if g == "Binary"),
        "Numeric\n(counts)":      sum(c for c, g in zip(counts, groups) if g == "Numeric"),
        "Open-ended\n(names,\ncolors, shapes,\nlocations…)":
                                  sum(c for c, g in zip(counts, groups) if g == "Open"),
    }
    sizes  = list(g_sums.values())
    glabels = list(g_sums.keys())
    gcolors = [G_COLORS["Binary"], G_COLORS["Numeric"], G_COLORS["Open"]]
    _, _, autotexts = ax.pie(
        sizes, labels=glabels, colors=gcolors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.70,
        wedgeprops=dict(edgecolor="white", linewidth=2.5, width=0.55),
        textprops=dict(fontsize=9),
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")
    ax.set_title("Semantic Groups\n(donut chart)", fontweight="bold")

    fig.suptitle(f"Answer Distribution — Test Set ({total:,} QA pairs, 8 question types)",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_answer_type_detailed.png", bbox_inches="tight")
    plt.close()
    print("✓ data_answer_type_detailed.png")


# ── 10. Route A architecture with tensor shapes ───────────────────────────────

def fig_arch_route_a():
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(15, 5.2))
    ax.set_xlim(0, 15); ax.set_ylim(0, 5.2); ax.axis("off")

    def bx(cx, cy, w, h, txt, fc, ec, fs=9):
        ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                     boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=2, zorder=3))
        ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
                multialignment="center", zorder=4)

    def ar(x1, y1, x2, y2, lbl=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#37474F", lw=2), zorder=5)
        if lbl:
            ax.text((x1+x2)/2, max(y1,y2)+0.18, lbl, ha="center", va="bottom",
                    fontsize=7.5, color="#C62828", fontweight="bold",
                    bbox=dict(fc="white", ec="none", alpha=0.9))

    Yi, Yt = 3.9, 1.3
    Yco = (Yi + Yt) / 2

    bx(1.3, Yi, 2.1, 0.9, "Image I\n[B, 3, 224, 224]", "#FFFDE7", "#F9A825")
    bx(1.3, Yt, 2.1, 0.9, "Question q\ntoken IDs [B, N]", "#FFFDE7", "#F9A825")

    ar(2.35, Yi, 3.15, Yi)
    ar(2.35, Yt, 3.15, Yt)
    bx(4.0, Yi, 1.65, 0.9, "CLIP ViT-B/16\nfrozen (86M)", "#E8F5E9", "#2E7D32")
    bx(4.0, Yt, 1.65, 0.9, "PhoBERT-base\nfrozen (135M)", "#E8F5E9", "#2E7D32")

    ax.text(5.5, Yi+0.22, "[B, 197, 768]", ha="center", va="bottom",
            fontsize=7.5, color="#C62828", fontweight="bold")
    ax.text(5.5, Yt-0.22, "[B, N, 768]", ha="center", va="top",
            fontsize=7.5, color="#C62828", fontweight="bold")

    ax.plot([4.83, 5.9, 5.9], [Yi, Yi, Yco], color="#37474F", lw=2, zorder=2)
    ax.plot([4.83, 5.9, 5.9], [Yt, Yt, Yco], color="#37474F", lw=2, zorder=2)
    ax.annotate("", xy=(6.2, Yco), xytext=(5.9, Yco),
                arrowprops=dict(arrowstyle="->", color="#37474F", lw=2), zorder=5)

    bx(7.8, Yco, 3.0, 3.4,
       "Co-Attention Fusion\n\nText→Image:\n  Attn(T, V, V)\n\n"
       "Image→Text:\n  Attn(V, T, T)\n\n→ context [B, 768]",
       "#E3F2FD", "#1565C0", fs=8.5)

    ax.plot([9.3, 9.8, 9.8], [Yco, Yco, Yi], color="#37474F", lw=2, zorder=2)
    ax.annotate("", xy=(10.2, Yi), xytext=(9.8, Yi),
                arrowprops=dict(arrowstyle="->", color="#37474F", lw=2), zorder=5)
    ax.plot([9.3, 9.8, 9.8], [Yco, Yco, Yt], color="#37474F", lw=2, zorder=2)
    ax.annotate("", xy=(10.2, Yt), xytext=(9.8, Yt),
                arrowprops=dict(arrowstyle="->", color="#37474F", lw=2), zorder=5)

    bx(11.4, Yi, 2.3, 1.0, "A1: LSTM Decoder\n[B, T, vocab_size]", "#FCE4EC", "#C62828")
    bx(11.4, Yt, 2.3, 1.0, "A2: Transformer Dec.\n[B, T, vocab_size]", "#EDE7F6", "#4527A0")

    ar(12.55, Yi, 13.3, Yi, "argmax")
    ar(12.55, Yt, 13.3, Yt, "argmax")
    bx(13.9, Yi, 1.1, 0.7, "Answer\n(A1)", "#FFF3E0", "#E65100")
    bx(13.9, Yt, 1.1, 0.7, "Answer\n(A2)", "#FFF3E0", "#E65100")

    ax.text(7.5, 5.08, "Route A: CLIP + PhoBERT + Co-Attention (with tensor shapes)",
            ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.text(0.1, 0.05,
            "B=batch  N=question tokens  197=CLIP patches (14×14 grid + CLS)  "
            "768=hidden dim  T=answer length  vocab_size≈40k",
            ha="left", va="bottom", fontsize=7.5, color="#607D8B", style="italic")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "arch_route_a.png", bbox_inches="tight", dpi=140)
    plt.close()
    print("✓ arch_route_a.png")


# ── 11. LoRA vs QLoRA ─────────────────────────────────────────────────────────

def fig_lora_qlora():
    from matplotlib.patches import FancyBboxPatch

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("white")

    specs = [
        dict(ax=axes[0], title="Standard LoRA  (fp16/bf16 base)",
             W_txt="W₀  [d × k]\nfp16 / bf16   (frozen)",
             W_fc="#E3F2FD", W_ec="#1565C0",
             extra="",
             notes=["W₀: fp16 → 2 bytes/param",
                    "A, B adapters: fp16/bf16 (trainable)",
                    "Forward:  h = W₀x + (α/r)BAx",
                    "Gradient flows only through A and B"]),
        dict(ax=axes[1], title="QLoRA  (NF4 4-bit base + bf16 adapters)",
             W_txt="W₀  [d × k]\nNF4 4-bit   (frozen, dequant on fwd)",
             W_fc="#FFF9C4", W_ec="#F9A825",
             extra="≈ 0.5 bytes/param  →  ~3.5× VRAM savings vs fp16",
             notes=["W₀: NF4 4-bit → ~0.5 bytes/param",
                    "A, B adapters: bf16 (trainable)",
                    "Forward:  h = dequant(W₀)x + (α/r)BAx",
                    "~3.5× VRAM reduction vs fp16 W₀"]),
    ]

    for s in specs:
        ax = s["ax"]
        ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
        ax.set_facecolor("#FAFAFA")

        def bx(cx, cy, w, h, txt, fc, ec, fs=9.5, _ax=ax):
            _ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                          boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=2, zorder=3))
            _ax.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
                     multialignment="center", zorder=4)

        ax.text(5, 6.75, s["title"], ha="center", va="center",
                fontsize=10.5, fontweight="bold")

        bx(2.8, 5.0, 4.5, 1.4, s["W_txt"], s["W_fc"], s["W_ec"], fs=9)
        ax.text(2.8, 4.15, "frozen — no gradient", ha="center", va="center",
                fontsize=8, color="#888", style="italic")
        if s["extra"]:
            ax.text(2.8, 3.75, s["extra"], ha="center", va="center",
                    fontsize=8, color="#C62828", fontweight="bold")

        bx(7.8, 5.6, 2.2, 0.9, "A  [r × k]\n(init: Gaussian)", "#FCE4EC", "#C62828", fs=8.5)
        bx(7.8, 4.3, 2.2, 0.9, "B  [d × r]\n(init: zeros)", "#FCE4EC", "#C62828", fs=8.5)
        ax.text(7.8, 3.6, "r = 8  ≪  min(d, k)", ha="center", va="center",
                fontsize=8, color="#555", style="italic")

        ax.text(5.5, 4.9, "+", fontsize=28, ha="center", va="center",
                color="#1B5E20", fontweight="bold")

        ax.text(5.0, 3.05, r"$\Delta W = \dfrac{\alpha}{r}\,B\,A$",
                ha="center", va="center", fontsize=14, color="#1B5E20")
        ax.text(5.0, 2.45, r"$W_{\!eff} = W_0 + \Delta W$",
                ha="center", va="center", fontsize=14, color="#1B5E20")

        r, d, k = 8, 4096, 4096
        trainable = d*r + r*k
        total_w   = d*k
        ax.text(5.0, 1.85,
                f"Trainable: {trainable:,} / {total_w:,} params  ({trainable/total_w*100:.2f}%)",
                ha="center", va="center", fontsize=8.5, color="#37474F",
                bbox=dict(fc="#F5F5F5", ec="#CFD8DC", boxstyle="round,pad=0.3"))

        for i, note in enumerate(s["notes"]):
            c = "#C62828" if any(k2 in note for k2 in ["NF4","3.5×","dequant"]) else "#37474F"
            ax.text(0.3, 1.25 - i*0.28, f"• {note}", ha="left", va="center",
                    fontsize=8.5, color=c)

    fig.suptitle(
        "LoRA vs QLoRA — Low-Rank Adaptation for B2-SFT (Qwen2.5-VL-3B-Instruct, r=8, α=8)",
        fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lora_qlora.png", bbox_inches="tight", dpi=140)
    plt.close()
    print("✓ lora_qlora.png")


# ── 12. Human evaluation — overall + per question type ───────────────────────

def fig_human_eval():
    models = ["A1", "A2", "B1", "B2_SFT", "B2_DPO"]
    labels = ["A1\n(LSTM)", "A2\n(Transf.)", "B1\n(Zero-shot)", "B2-SFT\n(LoRA)", "B2-DPO\n(DPO)"]
    colors = [MODEL_COLORS["A1"], MODEL_COLORS["A2"], MODEL_COLORS["B1"],
              MODEL_COLORS["B2"], MODEL_COLORS["DPO"]]

    # --- data ---
    human_acc = [61.0, 57.0, 38.0, 70.0, 73.0]
    auto_acc  = [94.84, 93.77, 19.62, 93.79, None]  # None = not comparable (stratified only)

    # Per-type data from 100-question human eval (100 records, 21 images)
    # Types with very low N (yes_no=2Q, shape=1Q) are excluded from per-type chart
    qtypes  = ["permission\n(38Q)", "count\n(23Q)", "location\n(23Q)",
               "sign_type\n(9Q)", "color\n(4Q)"]
    # rows: A1, A2, B1, B2_SFT, B2_DPO  (values in %)
    qtype_data = {
        "A1":     [45, 78, 65, 78, 50],
        "A2":     [42, 61, 74, 67, 50],
        "B1":     [42, 26, 35, 44, 50],
        "B2_SFT": [63, 74, 70, 78, 75],
        "B2_DPO": [76, 74, 65, 78, 50],
    }

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 1, hspace=0.48)

    # ── Top panel: overall comparison (human vs auto) ──
    ax1 = fig.add_subplot(gs[0])
    x   = np.arange(len(models))
    w   = 0.35

    bars_h = ax1.bar(x - w/2, human_acc, w, color=colors, alpha=0.95,
                     label="Human Eval (free-form)", edgecolor="white", linewidth=0.5)
    for i, (bar, v) in enumerate(zip(bars_h, human_acc)):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 1.2,
                 f"{v:.0f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                 color=colors[i])

    # Auto eval bars (None → skip B2_DPO)
    auto_vals = [v if v is not None else 0 for v in auto_acc]
    bars_a = ax1.bar(x + w/2, auto_vals, w, color=colors, alpha=0.40,
                     label="Auto Eval / template exact-match", edgecolor="white",
                     linewidth=0.5, hatch="///")
    for i, (bar, v) in enumerate(zip(bars_a, auto_acc)):
        if v is None:
            ax1.text(bar.get_x() + bar.get_width()/2, 2,
                     "n/a*", ha="center", va="bottom", fontsize=8, color="#9E9E9E")
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, v + 1.2,
                     f"{v:.0f}%", ha="center", va="bottom", fontsize=9.5,
                     color=colors[i], alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Human Evaluation vs Automatic Evaluation — Overall Accuracy\n"
                  "(Human: 100 free-form QA pairs across 21 images;  Auto: full test 12 966 template QA)",
                  fontweight="bold", fontsize=10.5)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.text(0.99, 0.04, "* B2-DPO auto eval is on stratified-1000 subset only (0.759) — not plotted",
             transform=ax1.transAxes, ha="right", fontsize=7.5, color="#9E9E9E", style="italic")

    # Ranking annotation
    rank_h = sorted(range(len(models)), key=lambda i: -human_acc[i])
    rank_a = sorted(range(len(models)), key=lambda i: -(auto_acc[i] or 0))
    ax1.text(0.01, 0.96,
             f"Human rank: {' > '.join(models[i] for i in rank_h)}",
             transform=ax1.transAxes, fontsize=8.5, va="top", color="#1B5E20", fontweight="bold")
    ax1.text(0.01, 0.88,
             f"Auto rank:  {' > '.join(models[i] for i in rank_a)}",
             transform=ax1.transAxes, fontsize=8.5, va="top", color="#B71C1C", fontweight="bold")

    # ── Bottom panel: per question type ──
    ax2 = fig.add_subplot(gs[1])
    nq  = len(qtypes)
    nm  = len(models)
    w2  = 0.14
    offsets = np.linspace(-(nm-1)/2 * w2, (nm-1)/2 * w2, nm)
    x2  = np.arange(nq)

    for mi, (m, c) in enumerate(zip(models, colors)):
        vals = qtype_data[m]
        bars = ax2.bar(x2 + offsets[mi], vals, w2, color=c, alpha=0.88,
                       label=labels[mi].replace("\n", " "), edgecolor="white", linewidth=0.3)
        for bar, v in zip(bars, vals):
            if v >= 85:
                ax2.text(bar.get_x() + bar.get_width()/2, v + 1,
                         f"{v}", ha="center", va="bottom", fontsize=6.5,
                         color=c, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(qtypes, fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Human Evaluation — Per Question Type Breakdown",
                  fontweight="bold", fontsize=10.5)
    ax2.axhline(50, color="#BDBDBD", lw=0.8, ls="--", zorder=0)
    ax2.legend(loc="upper right", fontsize=8.5, ncol=5,
               framealpha=0.85, borderpad=0.4, columnspacing=0.8)

    fig.suptitle("Human Evaluation Results: 100 Free-Form QA Pairs, 21 Images",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(FIG_DIR / "human_eval_results.png", bbox_inches="tight", dpi=140)
    plt.close()
    print("✓ human_eval_results.png")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig_data_split()
    fig_question_type_dist()
    fig_answer_type()
    fig_answer_type_detailed()
    fig_split_balance()
    fig_qtype_heatmap()
    fig_radar()
    fig_dpo_tradeoff()
    fig_latency()
    fig_arch_route_a()
    fig_lora_qlora()
    fig_human_eval()
    print("\nAll figures saved to", FIG_DIR)
