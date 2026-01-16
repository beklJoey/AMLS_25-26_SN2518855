# B/train.py
# -----------------------------------------------------------------------------
# Task B (Deep Learning): CNN training + evaluation for BreastMNIST.
#
# This script runs a controlled experiment grid to satisfy the coursework
# requirement of analysing three axes:
#   1) Capacity axis      -> CNN width controlled by 'capacity' ("small/base/large")
#   2) Augmentation axis  -> training-time augmentation toggled via dataloader
#   3) Budget axis        -> epochs_set (training epochs) and fracs (train_fraction)
#
# Key evaluation principles implemented here:
#   - Use official MedMNIST train/val/test splits via get_dataloaders()
#   - Validation is computed in eval() mode under torch.no_grad()
#   - Select the best checkpoint by validation accuracy
#   - Evaluate the test set ONCE using that best-validation checkpoint
#     (to avoid test leakage / optimistic bias)
#   - Print a single-line summary per configuration to match Task A style:
#       best_val_acc and test_acc
#
# Outputs:
#   - results/taskB.json containing all runs and the best run (by val accuracy)
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn

from B.dataset import get_dataloaders
from B.model import SimpleCNN_B
from utils import set_seed, save_json


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Evaluate a model on a given dataloader.

    IMPORTANT:
    - Decorated with @torch.no_grad() to disable gradient tracking (faster, safer).
    - Calls model.eval() to ensure deterministic behaviour for modules such as:
        * Dropout (disabled in eval mode)
        * BatchNorm (uses running stats instead of per-batch stats)
      This avoids misleading validation curves caused by incorrect evaluation mode.

    Parameters
    ----------
    model : torch.nn.Module
        The CNN model to evaluate.
    loader : torch.utils.data.DataLoader
        DataLoader for validation or test split (or train split if needed).
    criterion : torch.nn.Module
        Loss function (CrossEntropyLoss).
    device : str
        "cuda" or "cpu".

    Returns
    -------
    avg_loss : float
        Average loss over all samples in loader.
    acc : float
        Accuracy over all samples in loader.
    cm : np.ndarray
        Confusion matrix of shape (num_classes, num_classes).
        Row index is true label, column index is predicted label.
    """
    # Switch to evaluation mode:
    # - disables dropout
    # - makes BN use running statistics
    model.eval()

    total_loss, correct, total = 0.0, 0, 0
    num_classes = None
    cm = None

    for x, y in loader:
        # Move images to device
        x = x.to(device)

        # MedMNIST labels are sometimes shaped as [B,1], so squeeze to [B]
        if y.ndim > 1:
            y = y.squeeze(1)
        y = y.long().to(device)

        # Forward pass -> logits (unnormalised class scores)
        logits = model(x)

        # Cross-entropy loss expects logits shape [B, C] and targets [B]
        loss = criterion(logits, y)

        # Accumulate weighted loss by batch size
        total_loss += float(loss.item()) * x.size(0)

        # Predictions: argmax over class dimension
        pred = torch.argmax(logits, dim=1)

        # Count correct predictions
        correct += int((pred == y).sum().item())
        total += int(y.numel())

        # Initialise confusion matrix lazily on first batch
        if num_classes is None:
            num_classes = int(logits.shape[1])
            cm = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Update confusion matrix entry-by-entry
        # (true label t, predicted label p)
        for t, p in zip(y.detach().cpu().numpy(), pred.detach().cpu().numpy()):
            cm[int(t), int(p)] += 1

    # Safe division (avoid division by zero)
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
    verbose_epoch: bool = False,  # default: do NOT print per-epoch logs
):
    """
    Run the Task B experiment grid across capacity/augmentation/budget axes.

    Axes mapping:
    - Capacity axis:
        capacities=("small","base","large") controls CNN width in SimpleCNN_B
    - Augmentation axis:
        augments=(False, True) toggles augmentation in get_dataloaders()
        (augmentation is applied ONLY on the training split inside the dataset/dataloader)
    - Budget axis:
        epochs_set controls epoch budget; fracs controls training sample fraction

    Evaluation protocol (to avoid leakage):
    - Track validation accuracy every epoch
    - Save the model state dict corresponding to the best validation accuracy
    - After training, load that best state and evaluate the test set ONCE

    Returns
    -------
    out : dict
        JSON-friendly structure containing:
        - best_run selected by validation accuracy
        - all_runs for heatmaps/curves
        - hyperparameters for reproducibility
    """
    # Fix random seeds for reproducibility (numpy, torch, etc.)
    set_seed(seed)

    # Choose device automatically
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_runs = []
    best_run = None

    # Grid search across the three axes
    for cap in capacities:
        for aug in augments:
            for epochs in epochs_set:
                for frac in fracs:
                    # Tag used in logs for easy run identification
                    tag = f"[B cap={cap} aug={aug} ep={epochs} frac={frac}]"

                    try:
                        # -----------------------------------------------------------------
                        # Data loading with controlled settings
                        # - augment toggles training-time augmentation only
                        # - train_fraction subsamples training data (budget axis)
                        # - val/test remain deterministic to ensure fair evaluation
                        # -----------------------------------------------------------------
                        train_loader, val_loader, test_loader, meta = get_dataloaders(
                            batch_size=batch_size,
                            num_workers=num_workers,
                            augment=aug,
                            train_fraction=frac,
                        )

                        # -----------------------------------------------------------------
                        # Model construction (capacity axis)
                        # - capacity controls width (channels) inside SimpleCNN_B
                        # - meta provides dataset-specific settings:
                        #     num_classes (BreastMNIST -> 2)
                        #     in_channels (BreastMNIST -> 1)
                        # -----------------------------------------------------------------
                        model = SimpleCNN_B(
                            num_classes=meta["num_classes"],
                            in_channels=meta["in_channels"],
                            capacity=cap,
                        ).to(device)

                        # Loss for multi-class classification (works for binary with C=2)
                        criterion = nn.CrossEntropyLoss()

                        # Optimiser:
                        # - Adam is robust for small CNNs
                        # - weight_decay provides L2 regularisation (helps generalisation)
                        optimizer = torch.optim.Adam(
                            model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay
                        )

                        # Track best validation performance for checkpoint selection
                        best_val_acc = -1.0
                        best_epoch = None
                        best_state = None  # stores state_dict for best validation epoch
                        history = []       # stores val_acc per epoch for curves

                        # -----------------------------------------------------------------
                        # Training loop (budget axis: epochs)
                        # -----------------------------------------------------------------
                        for epoch in range(1, int(epochs) + 1):
                            # Switch to training mode (enables BN updates, dropout if any)
                            model.train()

                            # Iterate through training mini-batches
                            for x, y in train_loader:
                                x = x.to(device)

                                # Ensure target shape is [B] and dtype long for CE loss
                                if y.ndim > 1:
                                    y = y.squeeze(1)
                                y = y.long().to(device)

                                # Standard training step
                                optimizer.zero_grad(set_to_none=True)
                                loss = criterion(model(x), y)
                                loss.backward()
                                optimizer.step()

                            # -------------------------------------------------------------
                            # Validation evaluation (NO gradients; eval mode inside evaluate)
                            # -------------------------------------------------------------
                            _, val_acc, _ = evaluate(model, val_loader, criterion, device)
                            history.append({"epoch": epoch, "val_acc": float(val_acc)})

                            # Optional per-epoch print (off by default to keep logs clean)
                            if verbose_epoch:
                                print(f"{tag} epoch {epoch} val_acc={val_acc:.4f}")

                            # -------------------------------------------------------------
                            # Checkpoint selection by BEST validation accuracy
                            # - This implements a strict "select-by-val" protocol
                            # - Prevents test set influence (no test leakage)
                            # -------------------------------------------------------------
                            if float(val_acc) > best_val_acc:
                                best_val_acc = float(val_acc)
                                best_epoch = epoch

                                # Save a CPU copy of the model parameters:
                                # - detach() avoids autograd references
                                # - clone() ensures tensor storage is independent
                                best_state = {
                                    k: v.detach().cpu().clone()
                                    for k, v in model.state_dict().items()
                                }

                        # -----------------------------------------------------------------
                        # Final test evaluation using BEST checkpoint (no leakage)
                        # -----------------------------------------------------------------
                        if best_state is not None:
                            model.load_state_dict(best_state)

                        test_loss, test_acc, test_cm = evaluate(model, test_loader, criterion, device)

                        # One-line summary per run (exactly matches Task A style)
                        print(f"{tag} best_val_acc={best_val_acc:.4f} test_acc={float(test_acc):.4f}")

                        # JSON-friendly run record
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
                        # Robustness: if any configuration fails (e.g., OOM, shape error),
                        # record the error so the grid run can continue.
                        print(f"{tag} ERROR: {repr(e)}")
                        run = {
                            "capacity": cap,
                            "augment": bool(aug),
                            "epochs_budget": int(epochs),
                            "train_fraction": float(frac),
                            "error": repr(e),
                        }

                    # Store run (successful or failed) for later inspection
                    all_runs.append(run)

                    # Track best run by validation accuracy among successful runs
                    if "best_val_acc" in run:
                        if best_run is None or run["best_val_acc"] > best_run["best_val_acc"]:
                            best_run = run

    # Final output structure for saving / reporting
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

    # Save results for plotting heatmaps / curves in the report
    save_json("results/taskB.json", out)
    return out


if __name__ == "__main__":
    # Allow running this file directly for Task B only
    run_task_B()
