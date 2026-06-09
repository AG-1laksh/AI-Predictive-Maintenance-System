from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score


def main():
    base = Path("outputs/predictions")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    dataset_ids = ["FD001", "FD002", "FD003", "FD004", "all_fd"]
    labels = []
    train_acc = []
    test_acc = []

    for did in dataset_ids:
        train_file = base / f"train_{did}_predictions.csv"
        test_file = base / f"test_{did}_predictions.csv"

        if not train_file.exists() or not test_file.exists():
            continue

        tr = pd.read_csv(train_file)
        te = pd.read_csv(test_file)

        ta = accuracy_score(tr["actual_failure"], tr["predicted_failure"])
        va = accuracy_score(te["actual_failure"], te["predicted_failure"])

        labels.append(did.upper())
        train_acc.append(ta)
        test_acc.append(va)

    x = range(len(labels))
    width = 0.38

    plt.figure(figsize=(10, 5.5))
    plt.bar([i - width / 2 for i in x], train_acc, width=width, label="Train Accuracy")
    plt.bar([i + width / 2 for i in x], test_acc, width=width, label="Test Accuracy")

    for i, (ta, va) in enumerate(zip(train_acc, test_acc)):
        gap = ta - va
        plt.text(i, max(ta, va) + 0.0015, f"gap={gap:.4f}", ha="center", fontsize=9)

    plt.ylim(0.95, 1.0)
    plt.xticks(list(x), labels)
    plt.ylabel("Accuracy")
    plt.title("Overfitting Check: Train vs Test Accuracy")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()

    image_path = out_dir / "overfitting_graph.png"
    csv_path = out_dir / "overfitting_metrics.csv"

    pd.DataFrame(
        {
            "dataset": labels,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "gap": [ta - va for ta, va in zip(train_acc, test_acc)],
        }
    ).to_csv(csv_path, index=False)

    plt.savefig(image_path, dpi=180)

    print(f"Saved graph: {image_path}")
    print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
