from pathlib import Path

import pandas as pd


def load_raw_dataset(file_path: str) -> pd.DataFrame:
    """Load NASA turbofan txt file (space-separated) into a DataFrame."""
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    return df


def load_single_column_values(file_path: str, column_name: str = "value") -> pd.DataFrame:
    """Load single-column txt file like RUL vectors or custom one-column files."""
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    df = remove_empty_columns(df)
    df.columns = [column_name]
    return df


def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are fully empty (all NaN)."""
    return df.dropna(axis=1, how="all")


def assign_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Assign columns: unit, cycle, sensor_1 ... sensor_n."""
    sensor_count = df.shape[1] - 2
    column_names = ["unit", "cycle"] + [f"sensor_{i}" for i in range(1, sensor_count + 1)]
    df.columns = column_names
    return df


def create_rul_and_label(df: pd.DataFrame, threshold: int = 30) -> pd.DataFrame:
    """Create RUL and binary failure label."""
    max_cycle_per_unit = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle_per_unit - df["cycle"]
    df["failure"] = (df["RUL"] < threshold).astype(int)
    return df


def prepare_dataset(input_txt: str, output_csv: str, threshold: int = 30) -> pd.DataFrame:
    """Complete data pipeline: load -> clean -> name cols -> RUL -> label -> save CSV."""
    df = load_raw_dataset(input_txt)
    df = remove_empty_columns(df)
    df = assign_column_names(df)
    df = create_rul_and_label(df, threshold=threshold)

    output_path = Path(output_csv)
    df.to_csv(output_path, index=False)
    return df


def prepare_test_dataset_with_rul(
    input_test_txt: str,
    input_rul_txt: str,
    output_csv: str,
    threshold: int = 30,
) -> pd.DataFrame:
    """Prepare test dataset and compute row-level RUL using provided per-unit final RUL."""
    test_df = load_raw_dataset(input_test_txt)
    test_df = remove_empty_columns(test_df)
    test_df = assign_column_names(test_df)

    # Observed max cycle in truncated test trajectories
    test_df["max_cycle_observed"] = test_df.groupby("unit")["cycle"].transform("max")

    # True final RUL for each unit (one value per unit in unit order)
    rul_df = load_single_column_values(input_rul_txt, column_name="final_rul")
    rul_df["unit"] = range(1, len(rul_df) + 1)

    test_df = test_df.merge(rul_df, on="unit", how="left")

    # Row-level true RUL = remaining cycles to end of test + final RUL provided
    test_df["RUL"] = (test_df["max_cycle_observed"] - test_df["cycle"]) + test_df["final_rul"]
    test_df["failure"] = (test_df["RUL"] < threshold).astype(int)

    test_df = test_df.drop(columns=["max_cycle_observed", "final_rul"])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(output_csv, index=False)
    return test_df


if __name__ == "__main__":
    input_file = "train_FD001.txt"
    output_file = "train_FD001_clean.csv"

    data = prepare_dataset(input_file, output_file, threshold=30)
    print("Dataset prepared and saved!")
    print("Shape:", data.shape)
    print(data.head())
