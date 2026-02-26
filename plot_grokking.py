"""Generate grokking comparison plot: 274p (+ fine-tuning) vs 262p (single run)."""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 274p: base + 2 rounds of fine-tuning
BASE_274_CSV = "../10-digit-addition/digit-addition-311p/results/runs/posrank2_274p_s42/metrics.csv"
FT1_274_CSV = "../10-digit-addition/digit-addition-311p/results/finetune/posrank2_274p_ft1/metrics.csv"
FT2_274_CSV = "../10-digit-addition/digit-addition-311p/results/finetune/posrank2_274p_ft2/metrics.csv"

# 262p: single run, no fine-tuning
BASE_262_CSV = "../10-digit-addition/digit-addition-311p/results/runs/dff6_262p_s256_500k/metrics.csv"


def read_csv(path):
    steps, loss, exact, tok_acc, lr = [], [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            loss.append(float(row["train_loss"]))
            exact.append(float(row["val_exact"]))
            tok_acc.append(float(row["val_token_acc"]))
            lr.append(float(row["lr"]))
    return {
        "step": np.array(steps),
        "loss": np.array(loss),
        "exact": np.array(exact),
        "tok_acc": np.array(tok_acc),
        "lr": np.array(lr),
    }


def main():
    # --- Load 274p data ---
    base274 = read_csv(BASE_274_CSV)
    ft1 = read_csv(FT1_274_CSV)
    ft2 = read_csv(FT2_274_CSV)

    base274_end = int(base274["step"][-1])
    ft1_offset = base274_end
    ft2_offset = ft1_offset + int(ft1["step"][-1])

    # Concatenate 274p steps/accuracy/loss for a single series
    steps_274 = np.concatenate([
        base274["step"],
        ft1["step"] + ft1_offset,
        ft2["step"] + ft2_offset,
    ])
    exact_274 = np.concatenate([base274["exact"], ft1["exact"], ft2["exact"]])
    loss_274 = np.concatenate([base274["loss"], ft1["loss"], ft2["loss"]])

    # --- Load 262p data ---
    base262 = read_csv(BASE_262_CSV)

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    # Top: Accuracy
    ax1.plot(steps_274 / 1000, exact_274 * 100,
             color="#6b7280", linewidth=1.2, alpha=0.7, label="274p (seed=42 + fine-tuning)")
    ax1.plot(base262["step"] / 1000, base262["exact"] * 100,
             color="#2563eb", linewidth=2, label="262p (seed=256, no fine-tuning)")

    # Fine-tune boundaries for 274p
    for boundary, label in [(ft1_offset / 1000, "FT1"), (ft2_offset / 1000, "FT2")]:
        ax1.axvline(boundary, color="#d1d5db", linestyle="--", alpha=0.6, linewidth=0.8)
        ax1.text(boundary + 1, 8, label, fontsize=7, color="#9ca3af")

    ax1.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
    ax1.set_ylim(-2, 105)
    ax1.legend(loc="center left", fontsize=9)
    ax1.set_title("262p vs 274p: Grokking Comparison", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Annotate 262p grokking
    grok_idx = np.argmax(base262["exact"] > 0.5)
    if grok_idx > 0:
        grok_step = base262["step"][grok_idx]
        ax1.annotate(f"262p grokking\n~{grok_step // 1000}K steps",
                     xy=(grok_step / 1000, base262["exact"][grok_idx] * 100),
                     xytext=(grok_step / 1000 - 120, 60),
                     arrowprops=dict(arrowstyle="->", color="#2563eb"),
                     fontsize=9, color="#2563eb")

    # Annotate 262p best (not last — eval points may fluctuate)
    best_262_idx = np.argmax(base262["exact"])
    best_262 = base262["exact"][best_262_idx] * 100
    best_262_step = base262["step"][best_262_idx]
    ax1.annotate(f"262p: {best_262:.0f}%\n(step {best_262_step // 1000}K)",
                 xy=(best_262_step / 1000, best_262),
                 xytext=(best_262_step / 1000 - 130, 85),
                 arrowprops=dict(arrowstyle="->", color="#2563eb"),
                 fontsize=10, fontweight="bold", color="#2563eb")

    # Annotate 274p final
    final_274 = exact_274[-1] * 100
    ax1.annotate(f"274p: {final_274:.1f}%",
                 xy=(steps_274[-1] / 1000, final_274),
                 xytext=(steps_274[-1] / 1000 - 80, 70),
                 arrowprops=dict(arrowstyle="->", color="#6b7280"),
                 fontsize=9, color="#6b7280")

    # Bottom: Loss
    ax2.plot(steps_274 / 1000, loss_274,
             color="#6b7280", linewidth=1, alpha=0.6)
    ax2.plot(base262["step"] / 1000, base262["loss"],
             color="#2563eb", linewidth=1.5)

    for boundary in [ft1_offset / 1000, ft2_offset / 1000]:
        ax2.axvline(boundary, color="#d1d5db", linestyle="--", alpha=0.6, linewidth=0.8)

    ax2.set_ylabel("Train Loss", fontsize=11)
    ax2.set_xlabel("Training Steps (×1000)", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("grokking_plot.png", dpi=150, bbox_inches="tight")
    print("Saved grokking_plot.png")


if __name__ == "__main__":
    main()
