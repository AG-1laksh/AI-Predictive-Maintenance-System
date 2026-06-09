import joblib
from pathlib import Path

import pandas as pd

from src.data_pipeline import prepare_dataset, prepare_test_dataset_with_rul
from src.predict_sample import show_sample_predictions
from src.train_model import evaluate_binary_classification, train_random_forest


def main():
    threshold = 30
    dataset_ids = ["FD001", "FD002", "FD003", "FD004"]

    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    train_frames = []
    test_frames = []

    for dataset_id in dataset_ids:
        print("\n" + "=" * 60)
        print(f"Preparing {dataset_id}")

        train_txt = f"train_{dataset_id}.txt"
        test_txt = f"test_{dataset_id}.txt"
        rul_txt = f"RUL_{dataset_id}.txt"

        train_csv = f"outputs/train_{dataset_id}_clean.csv"
        test_csv = f"outputs/test_{dataset_id}_clean.csv"

        train_df = prepare_dataset(train_txt, train_csv, threshold=threshold)
        test_df = prepare_test_dataset_with_rul(
            input_test_txt=test_txt,
            input_rul_txt=rul_txt,
            output_csv=test_csv,
            threshold=threshold,
        )

        train_df["dataset_id"] = dataset_id
        test_df["dataset_id"] = dataset_id

        print("Train head:")
        print(train_df.head())
        print("Test head:")
        print(test_df.head())

        train_frames.append(train_df)
        test_frames.append(test_df)

    combined_train = pd.concat(train_frames, ignore_index=True)
    combined_test = pd.concat(test_frames, ignore_index=True)

    combined_train_csv = "outputs/train_all_fd_clean.csv"
    combined_test_csv = "outputs/test_all_fd_clean.csv"

    combined_train.to_csv(combined_train_csv, index=False)
    combined_test.to_csv(combined_test_csv, index=False)

    print("\nCombined TRAIN head:")
    print(combined_train.head())
    print("Combined TEST head:")
    print(combined_test.head())

    model_path = "models/rf_all_fd_model.pkl"

    print("\nTraining one combined model for FD001-FD004...")
    train_result = train_random_forest(
        input_csv=combined_train_csv,
        model_output_path=model_path,
    )
    print(f"Split accuracy (combined): {train_result['accuracy']:.4f}")
    print(
        "Split metrics "
        f"(P={train_result['split_metrics']['precision']:.4f}, "
        f"R={train_result['split_metrics']['recall']:.4f}, "
        f"F1={train_result['split_metrics']['f1']:.4f})"
    )
    print("Split confusion matrix:")
    print(train_result["split_metrics"]["confusion_matrix"])
    print("Training dataframe head:")
    print(train_result["dataframe_head"])

    print("\nEvaluating combined model on combined TEST set...")
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    X_test_real = combined_test[feature_columns]
    y_test_real = combined_test["failure"]
    y_test_pred = model.predict(X_test_real)
    test_metrics = evaluate_binary_classification(y_test_real, y_test_pred)

    print(f"Test accuracy (combined): {test_metrics['accuracy']:.4f}")
    print(
        "Test metrics "
        f"(P={test_metrics['precision']:.4f}, "
        f"R={test_metrics['recall']:.4f}, "
        f"F1={test_metrics['f1']:.4f})"
    )
    print("Test confusion matrix:")
    print(test_metrics["confusion_matrix"])

    print("\nSample predictions from combined model:")
    prediction_result = show_sample_predictions(
        input_csv=combined_train_csv,
        model_path=model_path,
        sample_count=10,
    )
    print(prediction_result["predictions"])
    print("DataFrame head:")
    print(prediction_result["dataframe_head"])


if __name__ == "__main__":
    main()
