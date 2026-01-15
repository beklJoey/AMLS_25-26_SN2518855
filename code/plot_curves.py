# plot_curves.py
import os
import json
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def moving_average(x: List[float], w: int = 5) -> List[float]:
    """moving average that tolerates None by skipping them locally."""
    if w <= 1 or len(x) < 2:
        return x

    out = []
    for i in range(len(x)):
        l = max(0, i - w + 1)
        window = [v for v in x[l:i+1] if v is not None]
        if len(window) == 0:
            out.append(None)
        else:
            out.append(sum(window) / len(window))
    return out


def plot_one(json_path: str, out_prefix: str, smooth_w: int = 5):
    d: Dict[str, Any] = json.load(open(json_path, "r", encoding="utf-8"))
    hist = d.get("history", [])
    if not hist:
        print(f"Skip {json_path}: no history")
        return

    epochs = [h.get("epoch") for h in hist]

    # -------- ACC --------
    train_acc = [h.get("train_acc") for h in hist]
    val_acc   = [h.get("val_acc") for h in hist]

    train_acc_s = moving_average(train_acc, smooth_w)
    val_acc_s   = moving_average(val_acc, smooth_w)

    plt.figure()
    plt.plot(epochs, train_acc_s, label="train_acc (smoothed)")
    plt.plot(epochs, val_acc_s, label="val_acc (smoothed)")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"{d.get('task', out_prefix)} Accuracy")
    plt.legend()
    os.makedirs("results/figures", exist_ok=True)
    acc_path = f"results/figures/{out_prefix}_acc.png"
    plt.savefig(acc_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {acc_path}")

    # -------- LOSS (only if numbers exist & not all None) --------
    train_loss = [h.get("train_loss") for h in hist]
    val_loss   = [h.get("val_loss") for h in hist]

    has_any_loss = any(v is not None for v in train_loss) and any(v is not None for v in val_loss)
    if not has_any_loss:
        print(f"Skip loss for {json_path} (loss missing/None).")
        return

    train_loss_s = moving_average(train_loss, smooth_w)
    val_loss_s   = moving_average(val_loss, smooth_w)

    plt.figure()
    plt.plot(epochs, train_loss_s, label="train_loss (smoothed)")
    plt.plot(epochs, val_loss_s, label="val_loss (smoothed)")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{d.get('task', out_prefix)} Loss")
    plt.legend()
    loss_path = f"results/figures/{out_prefix}_loss.png"
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {loss_path}")


def main():
    if os.path.exists("results/taskA.json"):
        plot_one("results/taskA.json", "taskA", smooth_w=5)
    if os.path.exists("results/taskB.json"):
        plot_one("results/taskB.json", "taskB", smooth_w=5)


if __name__ == "__main__":
    main()
