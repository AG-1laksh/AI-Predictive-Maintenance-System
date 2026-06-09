import joblib
import pandas as pd


def show_sample_predictions(
    input_csv: str,
    model_path: str,
    sample_count: int = 10,
):
    """Load saved model and print sample predictions."""
    df = pd.read_csv(input_csv)

    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]

    X = df[feature_columns]
    y = df["failure"]

    sample_features = X.head(sample_count).copy()
    sample_actual = y.head(sample_count).reset_index(drop=True)
    sample_pred = model.predict(sample_features)

    result_df = pd.DataFrame(
        {
            "actual_failure": sample_actual,
            "predicted_failure": sample_pred,
        }
    )

    return {
        "predictions": result_df,
        "dataframe_head": df.head(),
    }


if __name__ == "__main__":
    result = show_sample_predictions(
        input_csv="train_FD001_clean.csv",
        model_path="rf_predictive_maintenance_model.pkl",
        sample_count=10,
    )

    print("Sample predictions:")
    print(result["predictions"])
    print("\nDataFrame head:")
    print(result["dataframe_head"])
