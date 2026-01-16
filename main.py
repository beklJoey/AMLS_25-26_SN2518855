# main.py
# -----------------------------------------------------------------------------
# Entry point for running both Task A and Task B experiments.
#
# This script is intentionally minimal:
#   - Task A: classical machine learning pipeline (logistic regression baseline)
#   - Task B: deep learning pipeline (CNN)
#
# Each task is responsible for:
#   - Loading BreastMNIST using the official train/val/test split
#   - Running a controlled experiment grid (capacity / augmentation / budget)
#   - Selecting best checkpoint by validation accuracy (to avoid test leakage)
#   - Saving results (typically JSON files under results/) for plotting/reporting
#
# Usage:
#   python main.py
#
# Notes:
#   - Ensure your Python environment has all dependencies installed.
#   - Results will be printed to the console and saved to disk by each task.
# -----------------------------------------------------------------------------

# Import the top-level runners for both tasks.
# These functions encapsulate the full experiment loops and result saving.
from A.train import run_task_A
from B.train import run_task_B


def main():
    """
    Run Task A and Task B sequentially.

    The order is Task A -> Task B because:
      - Task A is faster and provides a stable baseline first.
      - Task B is typically slower (CNN training) and may depend on GPU/CPU speed.

    Both tasks print per-run summaries and write structured result files for
    later analysis and visualisation in the report.
    """
    # -----------------------------
    # Run classical ML baseline
    # -----------------------------
    print("Running Task A (BreastMNIST - Classical ML)...")
    run_task_A()

    # -----------------------------
    # Run deep learning CNN pipeline
    # -----------------------------
    # Newline in the message improves readability in console logs.
    print("\nRunning Task B (BreastMNIST - Deep Learning CNN)...")
    run_task_B()


# Standard Python idiom:
# - If the file is executed directly: run main()
# - If the file is imported as a module: do not auto-run experiments
if __name__ == "__main__":
    main()
