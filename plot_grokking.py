"""Generate grokking + fine-tuning plot for the 274p model."""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_CSV = "../10-digit-addition/digit-addition-311p/results/runs/posrank2_274p_s42/metrics.csv"
FT1_CSV = "../10-digit-addition/digit-addition-311p/results/finetune/posrank2_274p_ft1/metrics.csv"
FT2_CSV = "../10-digit-addition/digit-addition-311p/results/finetune/posrank2_274p_ft2/metrics.csv"


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
    base = read_csv(BASE_CSV)
    ft1 = read_csv(FT1_CSV)
    ft2 = read_csv(FT2_CSV)

    # Offset finetune steps to be cumulative
    base_end = int(base["step"][-1])
    ft1_offset = base_end
    ft2_offset = ft1_offset + int(ft1["step"][-1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})

    # --- Top: Accuracy ---
    ax1.plot(base["step"] / 1000, base["exact"] * 100,
             color="#2563eb", linewidth=1.5, label="Base training")
    ax1.plot((ft1["step"] + ft1_offset) / 1000, ft1["exact"] * 100,
             color="#16a34a", linewidth=1.5, label="Fine-tune 1 (lr=0.001)")
    ax1.plot((ft2["step"] + ft2_offset) / 1000, ft2["exact"] * 100,
             color="#dc2626", linewidth=1.5, label="Fine-tune 2 (lr=0.0003)")

    # Phase boundaries
    for boundary, label in [(ft1_offset / 1000, "FT1"), (ft2_offset / 1000, "FT2")]:
        ax1.axvline(boundary, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax1.text(boundary + 1, 5, label, fontsize=8, color="gray")

    ax1.set_ylabel("Exact Match Accuracy (%)", fontsize=11)
    ax1.set_ylim(-2, 105)
    ax1.legend(loc="center right", fontsize=9)
    ax1.set_title("274-Parameter Transformer: Grokking + Fine-Tuning", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Annotate key points
    # Grokking onset
    grok_idx = np.argmax(base["exact"] > 0.01)
    if grok_idx > 0:
        grok_step = base["step"][grok_idx]
        ax1.annotate(f"Grokking onset\n~{grok_step // 1000}K steps",
                     xy=(grok_step / 1000, base["exact"][grok_idx] * 100),
                     xytext=(grok_step / 1000 - 30, 50),
                     arrowprops=dict(arrowstyle="->", color="gray"),
                     fontsize=8, color="gray")

    # Final accuracy
    final_acc = ft2["exact"][-1] * 100
    final_step = (ft2["step"][-1] + ft2_offset) / 1000
    ax1.annotate(f"{final_acc:.1f}%",
                 xy=(final_step, final_acc),
                 xytext=(final_step - 20, 85),
                 arrowprops=dict(arrowstyle="->", color="#dc2626"),
                 fontsize=9, fontweight="bold", color="#dc2626")

    # --- Bottom: Loss ---
    ax2.plot(base["step"] / 1000, base["loss"],
             color="#2563eb", linewidth=1, alpha=0.8)
    ax2.plot((ft1["step"] + ft1_offset) / 1000, ft1["loss"],
             color="#16a34a", linewidth=1, alpha=0.8)
    ax2.plot((ft2["step"] + ft2_offset) / 1000, ft2["loss"],
             color="#dc2626", linewidth=1, alpha=0.8)

    for boundary in [ft1_offset / 1000, ft2_offset / 1000]:
        ax2.axvline(boundary, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    ax2.set_ylabel("Train Loss", fontsize=11)
    ax2.set_xlabel("Training Steps (×1000)", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("grokking_plot.png", dpi=150, bbox_inches="tight")
    print("Saved grokking_plot.png")


if __name__ == "__main__":
    main()
