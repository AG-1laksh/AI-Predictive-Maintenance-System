from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_binary_classification(y_true, y_pred) -> dict:
    """Return a compact evaluation summary for binary classification."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
    }


def train_random_forest(
    input_csv: str,
    model_output_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Train RandomForestClassifier and save model bundle."""
    df = pd.read_csv(input_csv)

    X = df.drop(columns=["failure", "RUL", "unit"], errors="ignore")
    X = X.select_dtypes(include=["number"])  # keep model input strictly numeric
    y = df["failure"]

    print("Class distribution:")
    print(y.value_counts(normalize=True))

    # Unit-based split to avoid leakage from same unit appearing in both train and test rows
    units = df["unit"].unique()
    split_index = int((1 - test_size) * len(units))
    train_units = units[:split_index]
    test_units = units[split_index:]

    train_df = df[df["unit"].isin(train_units)]
    test_df = df[df["unit"].isin(test_units)]

    X_train = train_df.drop(columns=["failure", "RUL", "unit"], errors="ignore")
    X_train = X_train.select_dtypes(include=["number"])
    y_train = train_df["failure"]

    X_test = test_df.drop(columns=["failure", "RUL", "unit"], errors="ignore")
    X_test = X_test.select_dtypes(include=["number"])
    y_test = test_df["failure"]

    # Ensure same feature order for train and test
    feature_columns = X_train.columns.tolist()
    X_test = X_test[feature_columns]

    # random_state ensures reproducible results
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    split_metrics = evaluate_binary_classification(y_test, y_pred)

    importances = model.feature_importances_
    feature_importance = dict(zip(feature_columns, importances))
    print("Top features:")
    print(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])

    model_bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "metrics": split_metrics,
    }

    output_path = Path(model_output_path)
    joblib.dump(model_bundle, output_path)

    return {
        "accuracy": split_metrics["accuracy"],
        "split_metrics": split_metrics,
        "feature_importance": feature_importance,
        "X_test": X_test,
        "y_test": y_test,
        "dataframe_head": df.head(),
    }


if __name__ == "__main__":
    result = train_random_forest(
        input_csv="train_FD001_clean.csv",
        model_output_path="rf_predictive_maintenance_model.pkl",
    )

    print("Model trained and saved!")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Classification report:")
    print(result["split_metrics"]["classification_report"])
    print("Confusion matrix:")
    print(result["split_metrics"]["confusion_matrix"])
    print(result["dataframe_head"])
