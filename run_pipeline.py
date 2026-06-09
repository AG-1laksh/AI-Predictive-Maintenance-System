import joblib
from pathlib import Path

from src.data_pipeline import (
    load_single_column_values,
    prepare_dataset,
    prepare_test_dataset_with_rul,
)
from src.predict_sample import show_sample_predictions
from src.train_model import evaluate_binary_classification, train_random_forest


def main():
    threshold = 30
    dataset_ids = ["FD001", "FD002", "FD003", "FD004"]

    Path("outputs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    print("Preparing x.txt as an additional one-column file...")
    x_df = load_single_column_values("x.txt", column_name="x_value")
    x_df.to_csv("outputs/x_clean.csv", index=False)
    print("x.txt head:")
    print(x_df.head())

    for dataset_id in dataset_ids:
        print("\n" + "=" * 60)
        print(f"Processing {dataset_id}")

        train_txt = f"train_{dataset_id}.txt"
        test_txt = f"test_{dataset_id}.txt"
        rul_txt = f"RUL_{dataset_id}.txt"

        train_csv = f"outputs/train_{dataset_id}_clean.csv"
        test_csv = f"outputs/test_{dataset_id}_clean.csv"
        model_path = f"models/rf_{dataset_id}_model.pkl"

        print("\nStep 1-6: Preparing TRAIN dataset...")
        train_df = prepare_dataset(train_txt, train_csv, threshold=threshold)
        print("Train head:")
        print(train_df.head())

        print("\nPreparing TEST dataset using true RUL file...")
        test_df = prepare_test_dataset_with_rul(
            input_test_txt=test_txt,
            input_rul_txt=rul_txt,
            output_csv=test_csv,
            threshold=threshold,
        )
        print("Test head:")
        print(test_df.head())

        print("\nStep 7-8: Training and saving model...")
        train_result = train_random_forest(
            input_csv=train_csv,
            model_output_path=model_path,
        )
        print(f"Split accuracy ({dataset_id}): {train_result['accuracy']:.4f}")
        print(
            "Split metrics "
            f"(P={train_result['split_metrics']['precision']:.4f}, "
            f"R={train_result['split_metrics']['recall']:.4f}, "
            f"F1={train_result['split_metrics']['f1']:.4f})"
        )
        print("Split confusion matrix:")
        print(train_result["split_metrics"]["confusion_matrix"])
        print("Train DataFrame head:")
        print(train_result["dataframe_head"])

        print("\nEvaluating on TEST dataset...")
        model_bundle = joblib.load(model_path)
        model = model_bundle["model"]
        feature_columns = model_bundle["feature_columns"]

        X_test_real = test_df[feature_columns]
        y_test_real = test_df["failure"]
        y_test_pred = model.predict(X_test_real)
        real_test_metrics = evaluate_binary_classification(y_test_real, y_test_pred)
        print(f"Test accuracy ({dataset_id}): {real_test_metrics['accuracy']:.4f}")
        print(
            "Test metrics "
            f"(P={real_test_metrics['precision']:.4f}, "
            f"R={real_test_metrics['recall']:.4f}, "
            f"F1={real_test_metrics['f1']:.4f})"
        )
        print("Test confusion matrix:")
        print(real_test_metrics["confusion_matrix"])

        print("\nStep 9: Sample predictions on TRAIN dataset...")
        prediction_result = show_sample_predictions(
            input_csv=train_csv,
            model_path=model_path,
            sample_count=10,
        )
        print(prediction_result["predictions"])
        print("DataFrame head:")
        print(prediction_result["dataframe_head"])


if __name__ == "__main__":
    main()
