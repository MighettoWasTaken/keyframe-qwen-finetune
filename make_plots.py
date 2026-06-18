"""
Two output figures:
  outputs/plots/training_curves.png  — train loss + eval loss + LR over steps
  outputs/plots/metrics_comparison.png — before/after bar chart (key eval metrics)

Requires: matplotlib  (uv add matplotlib  or  pip install matplotlib)

Usage:
  uv run python make_plots.py
  uv run python make_plots.py --show   # also open windows interactively
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({
    "figure.facecolor": "#0f0f13",
    "axes.facecolor":   "#16161e",
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  "#aaa",
    "xtick.color":      "#888",
    "ytick.color":      "#888",
    "text.color":       "#ddd",
    "grid.color":       "#fff",
    "grid.alpha":       0.12,
    "legend.framealpha": 0.25,
    "legend.edgecolor": "#444",
})

BLUE   = "#5b9bd5"
ORANGE = "#ed7d31"
GREEN  = "#a9d18e"
GREY   = "#888"


def load_jsonl(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_metrics(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_log(rows: list[dict]):
    train, eval_, lr = [], [], []
    for r in rows:
        s = r["step"]
        if "loss" in r:
            train.append((s, r["loss"]))
        if "eval_loss" in r:
            eval_.append((s, r["eval_loss"]))
        if "learning_rate" in r:
            lr.append((s, r["learning_rate"]))
    return train, eval_, lr


def plot_training(log_path: str, out: str, show: bool, dpi: int):
    rows = load_jsonl(log_path)
    if not rows:
        print(f"  {log_path} is empty — skipping training curves")
        return

    train, eval_, lr = parse_log(rows)
    has_eval = bool(eval_)
    has_lr   = bool(lr)

    ncols = 2 if has_lr else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols + 2, 4.5), squeeze=False)
    fig.suptitle("Training Curves", fontsize=12, color="#fff", y=1.01)

    # --- loss subplot ---
    ax = axes[0][0]
    if train:
        xs, ys = zip(*train)
        ax.plot(xs, ys, color=BLUE, linewidth=1.5, label="train loss", alpha=0.9)
    if has_eval:
        xs, ys = zip(*eval_)
        ax.plot(xs, ys, color=ORANGE, linewidth=1.8, marker="o", markersize=4,
                label="eval loss", zorder=3)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss", color="#ddd")
    ax.legend(labelcolor="#ccc")
    ax.grid(True)
    if train and has_eval:
        all_y = [y for _, y in train] + [y for _, y in eval_]
        ax.set_ylim(max(0, min(all_y) * 0.9), max(all_y) * 1.05)

    # --- LR subplot ---
    if has_lr:
        ax = axes[0][1]
        xs, ys = zip(*lr)
        scale = 1e4
        ax.plot(xs, [y * scale for y in ys], color=GREEN, linewidth=1.5)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("LR (×1e-4)")
        ax.set_title("Learning Rate Schedule", color="#ddd")
        ax.grid(True)

    fig.tight_layout(pad=2)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  -> {out}")
    if show:
        plt.show()
    plt.close(fig)


def plot_metrics(base_path: str, ft_path: str, out: str, show: bool, dpi: int):
    base = load_metrics(base_path)
    ft   = load_metrics(ft_path)

    # Pass rate = 1 - knockout_rate  (% of all val that both rendered AND matched)
    def pass_rate(m, variant):
        return (1 - m[variant]["knockout_rate"]) * 100

    pct_rows = [
        # (label,               base_val,                              ft_val,                             note)
        ("Format-valid\nrate",  base["format_valid_rate"] * 100,       ft["format_valid_rate"] * 100,       "↑"),
        ("Render pass\n(ext.)", base["extracted"]["render_rate"] * 100, ft["extracted"]["render_rate"] * 100, "↑"),
        ("Anim pass\n(ext.)",   pass_rate(base, "extracted"),           pass_rate(ft, "extracted"),          "↑"),
        ("Render pass\n(raw)",  base["raw"]["render_rate"] * 100,       ft["raw"]["render_rate"] * 100,      "↑"),
        ("Anim pass\n(raw)",    pass_rate(base, "raw"),                 pass_rate(ft, "raw"),                "↑"),
    ]
    ppl_rows = [
        ("Perplexity", base["perplexity"], ft["perplexity"]),
    ]

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle("Base vs Fine-tuned: Eval Metrics", fontsize=12, color="#fff", y=1.02)

    # left panel: perplexity
    ax_ppl = fig.add_axes([0.04, 0.12, 0.14, 0.76])
    x = np.array([0])
    w = 0.32
    b_val = ppl_rows[0][1]
    f_val = ppl_rows[0][2]
    ax_ppl.bar(x - w / 2, [b_val], w, color=BLUE,   alpha=0.85, label="Base")
    ax_ppl.bar(x + w / 2, [f_val], w, color=ORANGE, alpha=0.85, label="Fine-tuned")
    for xi, v, col in [(x[0] - w / 2, b_val, "#aaa"), (x[0] + w / 2, f_val, "#ddd")]:
        ax_ppl.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=col)
    ax_ppl.set_xticks([])
    ax_ppl.set_title("Perplexity\n(↓ better)", fontsize=9, color="#ddd")
    ax_ppl.set_ylim(0, max(b_val, f_val) * 1.35)
    ax_ppl.legend(fontsize=8, labelcolor="#ccc", loc="upper right")
    ax_ppl.grid(True, axis="y")

    # right panel: % metrics
    ax_pct = fig.add_axes([0.24, 0.12, 0.74, 0.76])
    labels  = [r[0] for r in pct_rows]
    b_vals  = [r[1] for r in pct_rows]
    f_vals  = [r[2] for r in pct_rows]
    x = np.arange(len(labels))
    w = 0.34
    ax_pct.bar(x - w / 2, b_vals, w, color=BLUE,   alpha=0.85, label="Base")
    ax_pct.bar(x + w / 2, f_vals, w, color=ORANGE, alpha=0.85, label="Fine-tuned")

    for xi, v in zip(x - w / 2, b_vals):
        ax_pct.text(xi, v + 0.8, f"{v:.1f}%", ha="center", va="bottom",
                    fontsize=7.5, color="#aaa")
    for xi, v, bv in zip(x + w / 2, f_vals, b_vals):
        delta = v - bv
        sign  = "+" if delta >= 0 else ""
        ax_pct.text(xi, v + 0.8, f"{v:.1f}%\n({sign}{delta:.1f})",
                    ha="center", va="bottom", fontsize=7.5, color="#ddd", linespacing=1.3)

    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(labels, fontsize=9)
    ax_pct.set_ylabel("% of val set")
    ax_pct.set_title("% Metrics (↑ better)", color="#ddd", fontsize=10)
    ax_pct.set_ylim(0, 118)
    ax_pct.axhline(100, color="#444", linewidth=0.6, linestyle="--", zorder=0)
    ax_pct.legend(fontsize=9, labelcolor="#ccc")
    ax_pct.grid(True, axis="y")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  -> {out}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log",          default="outputs/qwen3b-keyframes-lora/train_log.jsonl")
    p.add_argument("--base-metrics", default="outputs/eval/base_metrics.json")
    p.add_argument("--ft-metrics",   default="outputs/eval/ft_metrics.json")
    p.add_argument("--out-dir",      default="outputs/plots")
    p.add_argument("--dpi", type=int, default=300, help="Output resolution (PPI)")
    p.add_argument("--show", action="store_true", help="Open plot windows interactively")
    args = p.parse_args()

    print("Training curves...")
    plot_training(args.log, f"{args.out_dir}/training_curves.png", args.show, args.dpi)

    print("Metrics comparison...")
    plot_metrics(args.base_metrics, args.ft_metrics,
                 f"{args.out_dir}/metrics_comparison.png", args.show, args.dpi)

    print("Done.")


if __name__ == "__main__":
    main()
