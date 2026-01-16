# A/train.py
# -----------------------------------------------------------------------------
# Task A (Classical ML): Logistic Regression baseline for BreastMNIST.
#
# This script implements a controlled experiment grid for Task A:
#   1) Capacity axis      -> PCA dimensionality (784 vs 100 vs 50)
#   2) Augmentation axis  -> light image transforms applied ONLY to training set
#   3) Budget axis        -> epochs (10 vs 50) and train_fraction (1.0 vs 0.5)
#
# It uses:
#   - MedMNIST BreastMNIST official train/val/test splits via get_dataloaders()
#   - A linear classifier (SGDClassifier with log_loss, i.e., logistic regression)
#   - StandardScaler to normalise features for stable optimisation
#   - Early stopping based on validation accuracy (patience)
#
# Outputs:
#   - results/taskA.json with best_run and all_runs for downstream plotting/report
# -----------------------------------------------------------------------------

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
    """
    Collect ALL samples from a PyTorch DataLoader into numpy arrays.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Expected to yield batches (x, y) where:
        - x has shape [B, C, H, W]
        - y has shape [B, 1] or [B] depending on dataset wrapper

    Returns
    -------
    X : np.ndarray, shape (N, H, W)
        Images as float32 numpy array. For BreastMNIST, C=1 (grayscale),
        so we drop the channel dimension and keep (H, W).
    y : np.ndarray, shape (N,)
        Integer labels as int64 numpy array.
    """
    imgs, labels = [], []
    for x, y in loader:
        # x: [B, C, H, W]
        # BreastMNIST is single-channel (grayscale), so take channel 0:
        #   [B, 1, H, W] -> [B, H, W]
        x = x[:, 0].cpu().numpy().astype(np.float32)

        # y may be [B,1] (common in MedMNIST wrappers), so squeeze to [B]
        y = y.squeeze().cpu().numpy().astype(np.int64)

        # Collect this batch
        imgs.append(x)
        labels.append(y)

    # Concatenate across all batches:
    # imgs list contains arrays of shape (B,H,W); concat -> (N,H,W)
    # labels list contains arrays of shape (B,); concat -> (N,)
    X = np.concatenate(imgs, axis=0)   # (N, H, W)
    y = np.concatenate(labels, axis=0) # (N,)
    return X, y


def _apply_aug_numpy_images(X_hw: np.ndarray, seed: int) -> np.ndarray:
    """
    Apply light augmentation to numpy images of shape (N, H, W).

    IMPORTANT:
    - This augmentation is used ONLY for training data in Task A (classical ML),
      to satisfy the "augmentation axis" requirement.
    - Validation and test sets remain deterministic / unaugmented.

    Augmentation policy:
    - Random horizontal flip (p=0.5)
    - Random rotation within +/- 15 degrees

    Parameters
    ----------
    X_hw : np.ndarray, shape (N, H, W)
        Input images as float32. Values already normalised by dataset pipeline.
    seed : int
        Seed for reproducibility. We use a numpy RNG to generate per-image seeds.

    Returns
    -------
    X_aug : np.ndarray, shape (N, H, W)
        Augmented images as float32 numpy array.
    """
    # Use numpy RNG to generate deterministic pseudo-random seeds
    rng = np.random.RandomState(seed)

    # Torchvision transforms expect PIL image or tensor.
    # We convert each (H,W) numpy image to a tensor (1,H,W),
    # apply random transforms, then convert back to numpy (H,W).
    tfm = transforms.Compose([
        transforms.ToTensor(),  # (H,W) -> (1,H,W) and scales appropriately
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
    ])

    out = []
    for i in range(X_hw.shape[0]):
        img = X_hw[i]  # (H, W)

        # Make randomness "deterministic-ish" per image:
        # - draw a seed from rng
        # - set torch manual seed so torchvision randomness changes per sample
        torch.manual_seed(int(rng.randint(0, 10_000_000)))

        # Apply transforms: output tensor (1,H,W)
        aug = tfm(img)

        # Store back as numpy (H,W) by removing channel dimension
        out.append(aug[0].numpy())

    # Stack back into (N,H,W)
    return np.stack(out, axis=0).astype(np.float32)


def _flatten(X_hw: np.ndarray) -> np.ndarray:
    """
    Flatten images from (N, H, W) into feature vectors (N, H*W).

    For BreastMNIST:
      H*W = 28*28 = 784 features per image.

    Returns float32 for efficiency and compatibility.
    """
    return X_hw.reshape(X_hw.shape[0], -1).astype(np.float32)


def _acc(y_true, y_pred):
    """
    Simple accuracy function:
      accuracy = mean(y_true == y_pred)

    Returns python float for JSON serialisation / printing.
    """
    return float((y_true == y_pred).mean())


def _build_clf(seed: int, lr: float, weight_decay: float):
    """
    Build a scikit-learn Pipeline for logistic regression-like training.

    We implement logistic regression via SGDClassifier with:
      - loss="log_loss" (logistic regression objective)
      - penalty="l2" (L2 regularisation)
      - alpha=weight_decay (L2 strength; scikit-learn uses alpha)
      - learning_rate="constant" and eta0=lr (fixed step size)

    The pipeline includes StandardScaler to normalise input features:
      - stabilises SGD optimisation
      - ensures each feature has comparable scale

    NOTE:
    - max_iter is set to 1 because we run multiple "epochs" manually via
      partial_fit() in the outer loop.
    """
    return Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=weight_decay,
            learning_rate="constant",
            eta0=lr,
            max_iter=1,      # single pass per call; we iterate via partial_fit
            tol=None,        # disable convergence stopping; we control epochs
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
    """
    Train + validate + test for ONE configuration (one point in the grid).

    Axes mapping:
    - Capacity axis:
        pca_dim controls representation dimension after PCA.
        pca_dim=784 means "no PCA" (use all pixels).
    - Augmentation axis:
        augment=True applies random flip/rotation ONLY to training split.
    - Budget axis:
        epochs controls training duration + early stopping.
        train_fraction subsamples training data to simulate limited data budget.

    Returns a dictionary with:
    - config: hyperparameters for this run
    - best_val_acc: best observed validation accuracy
    - test_acc: test accuracy evaluated using the best validation checkpoint
    - history: per-epoch train/val accuracy (for curves)
    - test_confusion_matrix: confusion matrix on test set
    """
    # Ensure reproducibility for numpy / torch / python random where applicable
    set_seed(seed)

    # Load official splits through dataloaders
    train_loader, val_loader, test_loader, meta = get_dataloaders(batch_size=batch_size)

    # Convert all splits to numpy arrays:
    # X*_hw: (N,H,W); y*: (N,)
    Xtr_hw, ytr = _collect_images_labels(train_loader)
    Xva_hw, yva = _collect_images_labels(val_loader)
    Xte_hw, yte = _collect_images_labels(test_loader)

    # ----------------------------
    # Budget axis: train_fraction
    # ----------------------------
    # Restrict the number of training samples to a fraction of the training split.
    # This simulates limited-data scenarios and tests sample efficiency.
    train_fraction = float(train_fraction)
    if not (0.0 < train_fraction <= 1.0):
        raise ValueError("train_fraction must be in (0,1].")

    if train_fraction < 1.0:
        n = Xtr_hw.shape[0]
        k = max(1, int(round(n * train_fraction)))
        # NOTE: Taking the first k samples (deterministic). If you wanted random
        # subsampling you could shuffle indices, but we keep it simple/reproducible.
        Xtr_hw = Xtr_hw[:k]
        ytr = ytr[:k]

    # --------------------------------
    # Augmentation axis (train only)
    # --------------------------------
    # Apply augmentation ONLY to the training images.
    # Validation and test sets are left untouched to ensure deterministic evaluation.
    if augment:
        Xtr_hw = _apply_aug_numpy_images(Xtr_hw, seed=seed)

    # ----------------------------
    # Feature extraction: flatten
    # ----------------------------
    # Convert each image into a 784-dim vector (28x28)
    Xtr = _flatten(Xtr_hw)
    Xva = _flatten(Xva_hw)
    Xte = _flatten(Xte_hw)

    # ----------------------------
    # Capacity axis: PCA dimension
    # ----------------------------
    # PCA reduces dimensionality and can act as capacity control + regularisation.
    # pca_dim=784 means no PCA (identity).
    pca = None
    if pca_dim != 784:
        pca = PCA(n_components=pca_dim, random_state=seed)
        # Fit PCA ONLY on training data to avoid leakage from val/test
        Xtr = pca.fit_transform(Xtr)
        Xva = pca.transform(Xva)
        Xte = pca.transform(Xte)

    # Build model pipeline (StandardScaler + SGD logistic regression)
    clf = _build_clf(seed=seed, lr=lr, weight_decay=weight_decay)

    # Binary classification classes for partial_fit
    classes = np.array([0, 1], dtype=np.int64)

    # Training bookkeeping
    history = []             # list of per-epoch metrics for plotting
    best_state = None        # deep copy of pipeline at best validation epoch
    best_val_acc = -1.0
    best_epoch = None
    stopped_epoch = None
    bad = 0                  # number of consecutive epochs without improvement

    # ----------------------------
    # Budget axis: epochs + early stopping
    # ----------------------------
    for epoch in range(1, epochs + 1):
        # Access the SGDClassifier inside the pipeline
        lr_model = clf.named_steps["lr"]

        # StandardScaler must be fit before transform:
        # - for epoch 1: fit scaler on Xtr, then partial_fit with classes
        # - later epochs: reuse fitted scaler and continue SGD via partial_fit
        if epoch == 1:
            Xtr_s = clf.named_steps["scaler"].fit_transform(Xtr)
            lr_model.partial_fit(Xtr_s, ytr, classes=classes)
        else:
            Xtr_s = clf.named_steps["scaler"].transform(Xtr)
            lr_model.partial_fit(Xtr_s, ytr)

        # ----------------------------
        # Evaluation on train + validation
        # ----------------------------
        # We compute accuracy for monitoring (train) and model selection (val).
        Xtr_s = clf.named_steps["scaler"].transform(Xtr)
        Xva_s = clf.named_steps["scaler"].transform(Xva)

        train_pred = lr_model.predict(Xtr_s)
        val_pred = lr_model.predict(Xva_s)
        train_acc = _acc(ytr, train_pred)
        val_acc = _acc(yva, val_pred)

        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        # ----------------------------
        # Model selection by best validation accuracy
        # ----------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # Copy the full pipeline (scaler + classifier) for later test evaluation
            best_state = copy.deepcopy(clf)
            bad = 0
        else:
            bad += 1

        # ----------------------------
        # Early stopping (patience)
        # ----------------------------
        # Stop if validation has not improved for 'patience' consecutive epochs.
        if bad >= patience:
            stopped_epoch = epoch
            break

    # Restore best checkpoint (best validation performance)
    if best_state is not None:
        clf = best_state

    # ----------------------------
    # Final test evaluation (no leakage)
    # ----------------------------
    # Evaluate test set ONCE, using the model checkpoint selected by validation.
    Xte_s = clf.named_steps["scaler"].transform(Xte)
    test_pred = clf.named_steps["lr"].predict(Xte_s)
    test_acc = _acc(yte, test_pred)

    # Confusion matrix (for analysis/reporting)
    test_cm = confusion_matrix(yte, test_pred).tolist()

    # Return a JSON-friendly result record
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
    Run the Task A experiment grid.

    Grid definition (three axes):
      - Capacity axis:
          PCA feature dimensionality: [784, 100, 50]
      - Augmentation axis:
          Apply flip+rotation to training images: [False, True]
      - Budget axis:
          Epoch budget: [10, 50]
          Training data budget (fraction of training set): [1.0, 0.5]

    For each configuration:
      - Train with early stopping (patience)
      - Select checkpoint by BEST validation accuracy
      - Evaluate test set ONCE using that best-validation checkpoint
      - Print a single-line summary for easy debugging/comparison

    Saves:
      - results/taskA.json containing best_run and all_runs
    """
    # Set global seed (also set again inside _train_eval_one_setting)
    set_seed(seed)

    # Capacity axis: feature dimension after PCA (784 means "no PCA")
    capacities = [784, 100, 50]

    # Augmentation axis: whether to augment training images
    augments = [False, True]

    # Budget axis: training epochs
    epochs_set = [10, 50]

    # Budget axis: fraction of training samples to use
    fracs = [1.0, 0.5]

    all_runs = []  # collect all run outputs for later visualisation/analysis
    best = None    # keep track of best configuration by validation accuracy

    # Full grid search across axes
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

                    # Add metadata for downstream reporting
                    out["task"] = "A"
                    out["dataset"] = "BreastMNIST"
                    out["model"] = "Logistic Regression (SGDClassifier log_loss)"

                    all_runs.append(out)

                    # Track best run by validation accuracy
                    if best is None or out["best_val_acc"] > best["best_val_acc"]:
                        best = out

                    # Consistent log line format for each run (easy to grep/compare)
                    print(
                        f"[A pca={pca_dim} aug={aug} ep={epochs} frac={frac}] "
                        f"best_val_acc={out['best_val_acc']:.4f} "
                        f"test_acc={out['test_acc']:.4f}"
                    )

    # Final JSON structure
    final = {
        "task": "A",
        "dataset": "BreastMNIST",
        "best_run": best,
        "all_runs": all_runs,
    }

    # Persist results for plotting and report tables
    save_json("results/taskA.json", final)
    return final
