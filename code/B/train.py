# B/train.py
import numpy as np
import torch
import torch.nn as nn

from B.dataset import get_dataloaders
from B.model import SimpleCNN_B
from utils import set_seed, save_json


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    num_classes = None
    cm = None

    for x, y in loader:
        x = x.to(device)
        if y.ndim > 1:
            y = y.squeeze(1)
        y = y.long().to(device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * x.size(0)

        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

        if num_classes is None:
            num_classes = int(logits.shape[1])
            cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y.detach().cpu().numpy(), pred.detach().cpu().numpy()):
            cm[int(t), int(p)] += 1

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, cm


def run_task_B(
    seed: int = 42,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    # budgets
    epochs_set=(10, 30),
    fracs=(1.0, 0.5),
    capacities=("small", "base", "large"),
    augments=(False, True),
    # prints
    verbose_epoch: bool = False,  # <- 默认不打印每一轮
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_runs = []
    best_run = None

    for cap in capacities:
        for aug in augments:
            for epochs in epochs_set:
                for frac in fracs:
                    tag = f"[B cap={cap} aug={aug} ep={epochs} frac={frac}]"

                    try:
                        train_loader, val_loader, test_loader, meta = get_dataloaders(
                            batch_size=batch_size,
                            num_workers=num_workers,
                            augment=aug,
                            train_fraction=frac,
                        )

                        model = SimpleCNN_B(
                            num_classes=meta["num_classes"],
                            in_channels=meta["in_channels"],
                            capacity=cap,
                        ).to(device)

                        criterion = nn.CrossEntropyLoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                        best_val_acc = -1.0
                        best_epoch = None
                        best_state = None
                        history = []

                        # ---- train epochs ----
                        for epoch in range(1, int(epochs) + 1):
                            model.train()
                            for x, y in train_loader:
                                x = x.to(device)
                                if y.ndim > 1:
                                    y = y.squeeze(1)
                                y = y.long().to(device)

                                optimizer.zero_grad(set_to_none=True)
                                loss = criterion(model(x), y)
                                loss.backward()
                                optimizer.step()

                            # val
                            _, val_acc, _ = evaluate(model, val_loader, criterion, device)
                            history.append({"epoch": epoch, "val_acc": float(val_acc)})

                            if verbose_epoch:
                                print(f"{tag} epoch {epoch} val_acc={val_acc:.4f}")

                            if float(val_acc) > best_val_acc:
                                best_val_acc = float(val_acc)
                                best_epoch = epoch
                                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                        # ---- test using best ----
                        if best_state is not None:
                            model.load_state_dict(best_state)

                        test_loss, test_acc, test_cm = evaluate(model, test_loader, criterion, device)

                        # ✅ EXACTLY like Task A: one-line summary per run
                        print(f"{tag} best_val_acc={best_val_acc:.4f} test_acc={float(test_acc):.4f}")

                        run = {
                            "capacity": cap,
                            "augment": bool(aug),
                            "epochs_budget": int(epochs),
                            "train_fraction": float(frac),
                            "best_val_acc": float(best_val_acc),
                            "best_epoch": int(best_epoch) if best_epoch is not None else None,
                            "test_loss": float(test_loss),
                            "test_acc": float(test_acc),
                            "test_confusion_matrix": test_cm.tolist(),
                            "history": history,
                        }

                    except Exception as e:
                        print(f"{tag} ERROR: {repr(e)}")
                        run = {
                            "capacity": cap,
                            "augment": bool(aug),
                            "epochs_budget": int(epochs),
                            "train_fraction": float(frac),
                            "error": repr(e),
                        }

                    all_runs.append(run)

                    if "best_val_acc" in run:
                        if best_run is None or run["best_val_acc"] > best_run["best_val_acc"]:
                            best_run = run

    out = {
        "task": "B",
        "dataset": "BreastMNIST",
        "best_run": best_run,
        "all_runs": all_runs,
        "hyperparams": {
            "seed": seed,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    }

    save_json("results/taskB.json", out)
    return out


if __name__ == "__main__":
    run_task_B()
