import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def heatmap_grid(rows, x_key, y_key, z_key, title, outpath, x_order=None, y_order=None):
    """
    rows: list of dict
    x_key,y_key,z_key: keys in dict
    Produces heatmap with annotated z values.
    """
    xs = sorted(set(r[x_key] for r in rows), key=lambda v: (x_order.index(v) if x_order and v in x_order else v))
    ys = sorted(set(r[y_key] for r in rows), key=lambda v: (y_order.index(v) if y_order and v in y_order else v))

    x_to_i = {x:i for i,x in enumerate(xs)}
    y_to_i = {y:i for i,y in enumerate(ys)}
    Z = np.full((len(ys), len(xs)), np.nan, dtype=float)

    for r in rows:
        xi = x_to_i[r[x_key]]
        yi = y_to_i[r[y_key]]
        Z[yi, xi] = float(r[z_key])

    plt.figure()
    im = plt.imshow(Z, aspect="auto")
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.xticks(range(len(xs)), xs, rotation=0)
    plt.yticks(range(len(ys)), ys)

    # annotate
    for yi in range(len(ys)):
        for xi in range(len(xs)):
            if not np.isnan(Z[yi, xi]):
                plt.text(xi, yi, f"{Z[yi, xi]:.3f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def line_plot(xs, ys_list, labels, title, xlabel, ylabel, outpath):
    plt.figure()
    for ys, lab in zip(ys_list, labels):
        plt.plot(xs, ys, marker="o", label=lab)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    ensure_dir("figs")

    # ---------------- Task A ----------------
    A = load_json("results/taskA.json")
    A_runs = A["all_runs"]

    # Normalize keys for plotting
    # We will plot best_val_acc vs (pca_dim capacity) and epochs, split by augment, fix train_fraction=1.0
    A_rows = []
    for r in A_runs:
        cfg = r["config"]
        A_rows.append({
            "capacity": int(cfg["pca_dim"]),          # 784/100/50
            "augment": bool(cfg["augment"]),
            "epochs": int(cfg["epochs"]),
            "train_fraction": float(cfg["train_fraction"]),
            "best_val_acc": float(r["best_val_acc"]),
            "test_acc": float(r["test_acc"]),
        })

    cap_order_A = sorted(set(x["capacity"] for x in A_rows), reverse=True)  # show 784 first
    epoch_order_A = sorted(set(x["epochs"] for x in A_rows))
    frac_order_A = sorted(set(x["train_fraction"] for x in A_rows))

    for aug in [False, True]:
        rows = [x for x in A_rows if x["augment"] == aug and abs(x["train_fraction"] - 1.0) < 1e-9]
        heatmap_grid(
            rows,
            x_key="capacity",
            y_key="epochs",
            z_key="best_val_acc",
            title=f"Task A (LogReg) best_val_acc heatmap | augment={aug} | train_fraction=1.0",
            outpath=f"figs/taskA_heatmap_aug_{aug}.png",
            x_order=cap_order_A,
            y_order=epoch_order_A
        )

    # Budget-by-samples curve at fixed: epochs=max, capacity=max (784)
    max_epochs_A = max(epoch_order_A)
    max_cap_A = max(cap_order_A)
    for_cap = max_cap_A
    for_epochs = max_epochs_A
    ys_off = []
    ys_on = []
    for frac in frac_order_A:
        off = [x for x in A_rows if x["capacity"]==for_cap and x["epochs"]==for_epochs and x["augment"]==False and abs(x["train_fraction"]-frac)<1e-9]
        on  = [x for x in A_rows if x["capacity"]==for_cap and x["epochs"]==for_epochs and x["augment"]==True  and abs(x["train_fraction"]-frac)<1e-9]
        ys_off.append(off[0]["best_val_acc"] if off else np.nan)
        ys_on.append(on[0]["best_val_acc"] if on else np.nan)

    line_plot(
        xs=frac_order_A,
        ys_list=[ys_off, ys_on],
        labels=["augment=False", "augment=True"],
        title=f"Task A budget (train_fraction) effect | capacity={for_cap} epochs={for_epochs}",
        xlabel="train_fraction",
        ylabel="best_val_acc",
        outpath="figs/taskA_budget_train_fraction.png"
    )

    # ---------------- Task B ----------------
    B = load_json("results/taskB.json")
    B_runs = B["all_runs"]

    B_rows = []
    for r in B_runs:
        B_rows.append({
            "capacity": r["capacity"],                 # small/base/large
            "augment": bool(r["augment"]),
            "epochs": int(r["epochs_budget"]),
            "train_fraction": float(r["train_fraction"]),
            "best_val_acc": float(r["best_val_acc"]),
            "test_acc": float(r["test_acc"]),
        })

    cap_order_B = ["small", "base", "large"]
    epoch_order_B = sorted(set(x["epochs"] for x in B_rows))
    frac_order_B = sorted(set(x["train_fraction"] for x in B_rows))

    for aug in [False, True]:
        rows = [x for x in B_rows if x["augment"] == aug and abs(x["train_fraction"] - 1.0) < 1e-9]
        heatmap_grid(
            rows,
            x_key="capacity",
            y_key="epochs",
            z_key="best_val_acc",
            title=f"Task B (CNN) best_val_acc heatmap | augment={aug} | train_fraction=1.0",
            outpath=f"figs/taskB_heatmap_aug_{aug}.png",
            x_order=cap_order_B,
            y_order=epoch_order_B
        )

    # Budget-by-samples curve at fixed: epochs=max, capacity=base
    max_epochs_B = max(epoch_order_B)
    for_cap = "base"
    for_epochs = max_epochs_B
    ys_off = []
    ys_on = []
    for frac in frac_order_B:
        off = [x for x in B_rows if x["capacity"]==for_cap and x["epochs"]==for_epochs and x["augment"]==False and abs(x["train_fraction"]-frac)<1e-9]
        on  = [x for x in B_rows if x["capacity"]==for_cap and x["epochs"]==for_epochs and x["augment"]==True  and abs(x["train_fraction"]-frac)<1e-9]
        ys_off.append(off[0]["best_val_acc"] if off else np.nan)
        ys_on.append(on[0]["best_val_acc"] if on else np.nan)

    line_plot(
        xs=frac_order_B,
        ys_list=[ys_off, ys_on],
        labels=["augment=False", "augment=True"],
        title=f"Task B budget (train_fraction) effect | capacity={for_cap} epochs={for_epochs}",
        xlabel="train_fraction",
        ylabel="best_val_acc",
        outpath="figs/taskB_budget_train_fraction.png"
    )

    print("Done. Figures saved under ./figs/")
    print("Key figures proving 3-axis requirement:")
    print(" - figs/taskA_heatmap_aug_False.png + figs/taskA_heatmap_aug_True.png")
    print(" - figs/taskB_heatmap_aug_False.png + figs/taskB_heatmap_aug_True.png")

if __name__ == "__main__":
    main()
