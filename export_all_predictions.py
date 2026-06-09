from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


def export_predictions(input_csv: str, model_path: str, output_csv: str):
    df = pd.read_csv(input_csv)
    bundle = joblib.load(model_path)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    X = df[feature_columns]
    y = df["failure"] if "failure" in df.columns else None

    pred = model.predict(X)

    result = df.copy()
    result["predicted_failure"] = pred

    if y is not None:
        result["actual_failure"] = y
        accuracy = accuracy_score(y, pred)
    else:
        accuracy = None

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    return result, accuracy


def main():
    Path("outputs/predictions").mkdir(parents=True, exist_ok=True)

    dataset_ids = ["FD001", "FD002", "FD003", "FD004"]

    for dataset_id in dataset_ids:
        train_csv = f"outputs/train_{dataset_id}_clean.csv"
        test_csv = f"outputs/test_{dataset_id}_clean.csv"
        model_path = f"models/rf_{dataset_id}_model.pkl"

        train_output = f"outputs/predictions/train_{dataset_id}_predictions.csv"
        test_output = f"outputs/predictions/test_{dataset_id}_predictions.csv"

        train_result, train_acc = export_predictions(train_csv, model_path, train_output)
        test_result, test_acc = export_predictions(test_csv, model_path, test_output)

        print("=" * 60)
        print(dataset_id)
        print(f"Train predictions saved: {train_output} | rows={len(train_result)} | accuracy={train_acc:.4f}")
        print(f"Test predictions saved : {test_output} | rows={len(test_result)} | accuracy={test_acc:.4f}")
        print("Train head:")
        print(train_result.head())
        print("Test head:")
        print(test_result.head())

    # Combined model outputs
    combined_train_csv = "outputs/train_all_fd_clean.csv"
    combined_test_csv = "outputs/test_all_fd_clean.csv"
    combined_model_path = "models/rf_all_fd_model.pkl"

    combined_train_output = "outputs/predictions/train_all_fd_predictions.csv"
    combined_test_output = "outputs/predictions/test_all_fd_predictions.csv"

    combined_train_result, combined_train_acc = export_predictions(
        combined_train_csv, combined_model_path, combined_train_output
    )
    combined_test_result, combined_test_acc = export_predictions(
        combined_test_csv, combined_model_path, combined_test_output
    )

    print("=" * 60)
    print("COMBINED FD001-FD004")
    print(
        f"Train predictions saved: {combined_train_output} | rows={len(combined_train_result)} | accuracy={combined_train_acc:.4f}"
    )
    print(
        f"Test predictions saved : {combined_test_output} | rows={len(combined_test_result)} | accuracy={combined_test_acc:.4f}"
    )
    print("Combined train head:")
    print(combined_train_result.head())
    print("Combined test head:")
    print(combined_test_result.head())


if __name__ == "__main__":
    main()
