# AMLS Coursework (2025–26): Medical Image Classification on BreastMNIST

This repository contains my coursework implementation for medical image classification on **BreastMNIST (binary)**.
Two complementary paradigms are implemented:

- **Task A (Classical ML):** engineered vector representations + a linear classifier.
- **Task B (Deep Learning):** a lightweight CNN with configurable capacity.

Experiments systematically vary:
1) **Model capacity** (e.g., feature dimension / CNN width),
2) **Data augmentation** (train-time on/off),
3) **Training budget** (epochs and training-set fraction).

---

## Repository Structure

- `main.py` — entry point to run Task A / Task B experiments.

- `A/`
  - `dataset.py` — data loading for Task A
  - `model.py` — classical model definition
  - `train.py` — training / evaluation for Task A

- `B/`
  - `dataset.py` — data loading for Task B (optional train-time augmentation)
  - `model.py` — CNN model (`SimpleCNN_B`) with capacity control (small/base/large)
  - `train.py` — training / evaluation for Task B

- `Datasets/`
  - `breastmnist.npz` — dataset file (required)

- `figs/` — generated figures (created automatically)
- `results/` — saved experiment results/logs (created automatically)

- `plot_curves.py` / `plot_results.py` — plotting scripts
- `requirements.txt` — Python dependencies

---

## Environment Setup

### Option 1: use the provided virtual environment (if available)

Activate the existing venv in the repository root (Windows PowerShell):

    .\venv\Scripts\Activate.ps1

### Option 2: create a fresh environment

Create and activate a new venv, then install dependencies:

    python -m venv venv
    .\venv\Scripts\Activate.ps1
    pip install -r requirements.txt

---

## Dataset Setup

Place the dataset file at:

    Datasets/breastmnist.npz

The loader expects NPZ keys for splits:

    train_images, train_labels
    val_images, val_labels  (or valid_images, valid_labels)
    test_images, test_labels

---

## Preprocessing and Augmentation (Task B)

All splits:
- images are converted to tensors and normalised (mean = 0.5, std = 0.5) for the grayscale channel.

If augmentation is enabled, it is applied **only to the training split**:
- random horizontal flip with probability **p = 0.5**
- random rotation uniformly sampled within **±15°**

Validation and test splits are not augmented.

---

## Model Capacity (Task B)

The CNN width is controlled by a capacity setting:
- `small`: base channels 16
- `base`: base channels 32
- `large`: base channels 64

---

## Running Experiments

From the repository root:

    python main.py

The scripts run a grid of experiments over capacity / augmentation / budget settings and write outputs to:
- `results/` (metrics/logs)
- `figs/` (plots)

---

## Plotting

After experiments finish:

    python plot_curves.py
    python plot_results.py

Figures will be saved to `figs/`.

---

## Reproducibility Notes

- Model selection is performed using **validation accuracy**.
- Final performance is reported on the **test** split using the selected checkpoint.
- Accuracy is used as the primary reported metric in this coursework.

---

## Author

Joey
