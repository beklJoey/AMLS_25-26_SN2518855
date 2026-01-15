# A/train.py
import copy
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

import torch
from torchvision import transforms

from A.dataset import get_dataloaders
from utils import set_seed, save_json


def _collect_images_labels(loader):
    """Return images as numpy (N, H, W) and labels as numpy (N,)."""
    imgs, labels = [], []
    for x, y in loader:
        # x: [B, C, H, W] -> (B,H,W) (BreastMNIST is grayscale)
        x = x[:, 0].cpu().numpy().astype(np.float32)
        y = y.squeeze().cpu().numpy().astype(np.int64)
        imgs.append(x)
        labels.append(y)
    X = np.concatenate(imgs, axis=0)   # (N,H,W)
    y = np.concatenate(labels, axis=0) # (N,)
    return X, y


def _apply_aug_numpy_images(X_hw: np.ndarray, seed: int) -> np.ndarray:
    """
    Apply light augmentation (flip + small rotation) to numpy images (N,H,W).
    Only used for training data in classical ML.
    """
    rng = np.random.RandomState(seed)
    tfm = transforms.Compose([
        transforms.ToTensor(),  # (H,W) -> (1,H,W)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ])

    out = []
    for i in range(X_hw.shape[0]):
        img = X_hw[i]
        # make augmentation deterministic-ish per image by reseeding
        torch.manual_seed(int(rng.randint(0, 10_000_000)))
        aug = tfm(img)          # (1,H,W)
        out.append(aug[0].numpy())
    return np.stack(out, axis=0).astype(np.float32)


def _flatten(X_hw: np.ndarray) -> np.ndarray:
    """(N,H,W) -> (N, H*W)"""
    return X_hw.reshape(X_hw.shape[0], -1).astype(np.float32)


def _acc(y_true, y_pred):
    return float((y_true == y_pred).mean())


def _build_clf(seed: int, lr: float, weight_decay: float):
    """
    Logistic Regression via SGDClassifier (log_loss) in a Pipeline with StandardScaler.
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=weight_decay,
            learning_rate="constant",
            eta0=lr,
            max_iter=1,      # we iterate via partial_fit
            tol=None,
            random_state=seed,
        ))
    ])


def _train_eval_one_setting(
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    batch_size: int,
    augment: bool,
    train_fraction: float,
    pca_dim: int,   # capacity axis: 784 (no PCA) / 100 / 50 etc.
):
    set_seed(seed)

    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=batch_size)
    Xtr_hw, ytr = _collect_images_labels(train_loader)
    Xva_hw, yva = _collect_images_labels(val_loader)
    Xte_hw, yte = _collect_images_labels(test_loader)

    # budget axis: subset training samples
    train_fraction = float(train_fraction)
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError("train_fraction must be in (0,1].")
    if train_fraction < 1.0:
        n = Xtr_hw.shape[0]
        k = max(1, int(round(n * train_fraction)))
        Xtr_hw = Xtr_hw[:k]
        ytr = ytr[:k]

    # augmentation axis (train only)
    if augment:
        Xtr_hw = _apply_aug_numpy_images(Xtr_hw, seed=seed)

    # flatten
    Xtr = _flatten(Xtr_hw)
    Xva = _flatten(Xva_hw)
    Xte = _flatten(Xte_hw)

    # capacity axis: PCA dimension
    pca = None
    if pca_dim != 784:
        pca = PCA(n_components=pca_dim, random_state=seed)
        Xtr = pca.fit_transform(Xtr)
        Xva = pca.transform(Xva)
        Xte = pca.transform(Xte)

    clf = _build_clf(seed=seed, lr=lr, weight_decay=weight_decay)
    classes = np.array([0, 1], dtype=np.int64)

    history = []
    best_state = None
    best_val_acc = -1.0
    best_epoch = None
    stopped_epoch = None
    bad = 0

    for epoch in range(1, epochs + 1):
        lr_model = clf.named_steps["lr"]

        if epoch == 1:
            Xtr_s = clf.named_steps["scaler"].fit_transform(Xtr)
            lr_model.partial_fit(Xtr_s, ytr, classes=classes)
        else:
            Xtr_s = clf.named_steps["scaler"].transform(Xtr)
            lr_model.partial_fit(Xtr_s, ytr)

        # eval
        Xtr_s = clf.named_steps["scaler"].transform(Xtr)
        Xva_s = clf.named_steps["scaler"].transform(Xva)

        train_pred = lr_model.predict(Xtr_s)
        val_pred = lr_model.predict(Xva_s)
        train_acc = _acc(ytr, train_pred)
        val_acc = _acc(yva, val_pred)

        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(clf)
            bad = 0
        else:
            bad += 1

        if bad >= patience:
            stopped_epoch = epoch
            break

    if best_state is not None:
        clf = best_state

    # test
    Xte_s = clf.named_steps["scaler"].transform(Xte)
    test_pred = clf.named_steps["lr"].predict(Xte_s)
    test_acc = _acc(yte, test_pred)
    test_cm = confusion_matrix(yte, test_pred).tolist()

    return {
        "config": {
            "augment": augment,
            "train_fraction": train_fraction,
            "epochs": epochs,
            "pca_dim": pca_dim,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "patience": patience,
        },
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "stopped_epoch": int(stopped_epoch) if stopped_epoch is not None else None,
        "test_acc": float(test_acc),
        "test_confusion_matrix": test_cm,
        "history": history,
    }


def run_task_A(
    seed=42,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    patience=8,
):
    """
    Task A grid: performance vs
      - model capacity: PCA dim (784 vs 100 vs 50)
      - data augmentation: off/on (flip+rotation on train)
      - training budget: epochs (10 vs 50) and train_fraction (1.0 vs 0.5)
    Saves results/taskA.json
    """
    set_seed(seed)

    capacities = [784, 100, 50]       # capacity axis (feature dimension)
    augments = [False, True]          # augmentation axis
    epochs_set = [10, 50]             # budget axis (epochs)
    fracs = [1.0, 0.5]                # budget axis (samples)

    all_runs = []
    best = None

    for pca_dim in capacities:
        for aug in augments:
            for epochs in epochs_set:
                for frac in fracs:
                    out = _train_eval_one_setting(
                        seed=seed,
                        epochs=epochs,
                        lr=lr,
                        weight_decay=weight_decay,
                        patience=patience,
                        batch_size=batch_size,
                        augment=aug,
                        train_fraction=frac,
                        pca_dim=pca_dim,
                    )
                    out["task"] = "A"
                    out["dataset"] = "BreastMNIST"
                    out["model"] = "Logistic Regression (SGDClassifier log_loss)"
                    all_runs.append(out)

                    if best is None or out["best_val_acc"] > best["best_val_acc"]:
                        best = out

                    print(f"[A pca={pca_dim} aug={aug} ep={epochs} frac={frac}] best_val_acc={out['best_val_acc']:.4f} test_acc={out['test_acc']:.4f}")

    final = {
        "task": "A",
        "dataset": "BreastMNIST",
        "best_run": best,
        "all_runs": all_runs,
    }

    save_json("results/taskA.json", final)
    return final
