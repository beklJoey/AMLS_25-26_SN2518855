# main.py
from A.train import run_task_A
from B.train import run_task_B


def main():
    print("Running Task A (BreastMNIST - Classical ML)...")
    run_task_A()

    print("\nRunning Task B (BreastMNIST - Deep Learning CNN)...")
    run_task_B()


if __name__ == "__main__":
    main()
