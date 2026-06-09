from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


def _load_pickle(path: Path) -> Any:
    """Load a pickle file using pickle first, then joblib fallback."""
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        try:
            import joblib

            return joblib.load(path)
        except Exception as exc:
            raise RuntimeError(f"Could not load {path}: {exc}") from exc


def _safe_get(params: dict[str, Any], key: str) -> Any:
    return params.get(key, "") if isinstance(params, dict) else ""


def unlock_pickles(root: Path, output_dir: Path) -> tuple[Path, Path, int]:
    """Unlock all project pickle files and export text + CSV summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "pickle_unlock_report.txt"
    csv_path = output_dir / "pickle_summary.csv"

    skip_parts = {".venv", "site-packages", "__pycache__"}
    pkl_files = sorted(
        p
        for p in root.rglob("*.pkl")
        if not any(part in skip_parts for part in p.parts)
    )

    text_lines: list[str] = [f"Found {len(pkl_files)} project pickle file(s)\n"]
    csv_rows: list[dict[str, Any]] = []

    for pkl_file in pkl_files:
        rel = pkl_file.relative_to(root)
        text_lines.append(f"=== {rel} ===")

        try:
            obj = _load_pickle(pkl_file)
        except Exception as exc:
            text_lines.append("Status: FAILED to load")
            text_lines.append(f"Error: {exc!r}")
            text_lines.append("")
            csv_rows.append(
                {
                    "file": str(rel),
                    "status": "FAILED",
                    "object_type": "",
                    "dict_keys": "",
                    "feature_count": "",
                    "model_type": "",
                    "n_estimators": "",
                    "max_depth": "",
                    "random_state": "",
                    "accuracy": "",
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "top_features": "",
                    "error": str(exc),
                }
            )
            continue

        text_lines.append("Status: LOADED")
        text_lines.append(f"Type: {type(obj)}")

        row: dict[str, Any] = {
            "file": str(rel),
            "status": "LOADED",
            "object_type": type(obj).__name__,
            "dict_keys": "",
            "feature_count": "",
            "model_type": "",
            "n_estimators": "",
            "max_depth": "",
            "random_state": "",
            "accuracy": "",
            "precision": "",
            "recall": "",
            "f1": "",
            "top_features": "",
            "error": "",
        }

        if isinstance(obj, dict):
            keys = list(obj.keys())
            row["dict_keys"] = ";".join(map(str, keys))
            text_lines.append(f"dict keys: {keys}")

            feature_columns = obj.get("feature_columns")
            if feature_columns is not None:
                fc = list(feature_columns)
                row["feature_count"] = len(fc)
                text_lines.append(f"feature_columns count: {len(fc)}")
                text_lines.append(f"feature_columns first 15: {fc[:15]}")
            else:
                fc = []

            metrics = obj.get("metrics", {})
            if isinstance(metrics, dict):
                row["accuracy"] = metrics.get("accuracy", "")
                row["precision"] = metrics.get("precision", "")
                row["recall"] = metrics.get("recall", "")
                row["f1"] = metrics.get("f1", "")
                if metrics:
                    text_lines.append("metrics:")
                    for k, v in metrics.items():
                        text_lines.append(f"  - {k}: {v}")

            model = obj.get("model")
            if model is not None:
                row["model_type"] = type(model).__name__
                text_lines.append(f"model type: {type(model)}")
                try:
                    params = model.get_params()
                except Exception:
                    params = {}

                row["n_estimators"] = _safe_get(params, "n_estimators")
                row["max_depth"] = _safe_get(params, "max_depth")
                row["random_state"] = _safe_get(params, "random_state")
                text_lines.append(
                    "model params (selected): "
                    f"{{'n_estimators': {row['n_estimators']}, "
                    f"'max_depth': {row['max_depth']}, "
                    f"'random_state': {row['random_state']}}}"
                )

                if hasattr(model, "feature_importances_"):
                    try:
                        fi = model.feature_importances_
                        top_idx = sorted(range(len(fi)), key=lambda i: fi[i], reverse=True)[:10]
                        if fc:
                            top = [f"{fc[i]}:{float(fi[i]):.6f}" for i in top_idx if i < len(fc)]
                        else:
                            top = [f"idx_{i}:{float(fi[i]):.6f}" for i in top_idx]
                        row["top_features"] = "|".join(top)
                        text_lines.append("top 10 feature importances:")
                        for item in top:
                            name, val = item.split(":", 1)
                            text_lines.append(f"  - {name}: {val}")
                    except Exception as exc:
                        text_lines.append(f"feature importances read error: {exc!r}")

        else:
            text_lines.append(f"repr preview: {repr(obj)[:500]}")

        text_lines.append("")
        csv_rows.append(row)

    report_path.write_text("\n".join(text_lines), encoding="utf-8")

    try:
        import pandas as pd

        df = pd.DataFrame(csv_rows)
        df.to_csv(csv_path, index=False)
    except Exception:
        import csv

        fieldnames = [
            "file",
            "status",
            "object_type",
            "dict_keys",
            "feature_count",
            "model_type",
            "n_estimators",
            "max_depth",
            "random_state",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "top_features",
            "error",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    return report_path, csv_path, len(pkl_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unlock and summarize all project pickle files.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root to scan (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "outputs",
        help="Output directory for reports (default: ./outputs)",
    )
    args = parser.parse_args()

    report_path, csv_path, count = unlock_pickles(args.root.resolve(), args.output_dir.resolve())
    print(f"Processed {count} pickle file(s)")
    print(f"Text report: {report_path}")
    print(f"CSV summary: {csv_path}")


if __name__ == "__main__":
    main()
