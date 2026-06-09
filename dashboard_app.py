from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Predictive Maintenance Dashboard", page_icon="📊", layout="wide")

PREDICTIONS_DIR = Path("outputs/predictions")
OVERFITTING_FILE = Path("outputs/overfitting_metrics.csv")
MODELS_DIR = Path("models")

DATASET_EXPLANATIONS = {
    "FD001": "Single operating condition + one fault mode (simpler case).",
    "FD002": "Six operating conditions + one fault mode.",
    "FD003": "Single operating condition + two fault modes.",
    "FD004": "Six operating conditions + two fault modes (most complex case).",
    "all_fd": "Combined data from FD001, FD002, FD003, FD004.",
}


def _find_prediction_file(dataset_id: str, split: str) -> Path:
    target = PREDICTIONS_DIR / f"{split}_{dataset_id}_predictions.csv"
    if not target.exists():
        raise FileNotFoundError(f"Missing file: {target}")
    return target


def _load_predictions(dataset_id: str, split: str) -> pd.DataFrame:
    file_path = _find_prediction_file(dataset_id=dataset_id, split=split)
    return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def load_overfitting_metrics() -> pd.DataFrame:
    if not OVERFITTING_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(OVERFITTING_FILE)


@st.cache_data(show_spinner=False)
def list_available_prediction_keys() -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if not PREDICTIONS_DIR.exists():
        return keys

    for file_path in PREDICTIONS_DIR.glob("*_predictions.csv"):
        stem = file_path.stem.replace("_predictions", "")
        if "_" not in stem:
            continue
        split, dataset_id = stem.split("_", 1)
        if split in {"train", "test"}:
            keys.append((dataset_id, split))

    return sorted(set(keys))


def summarize_binary_metrics(df: pd.DataFrame) -> dict:
    if "actual_failure" not in df.columns or "predicted_failure" not in df.columns:
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "total": len(df),
        }

    y_true = df["actual_failure"].astype(int)
    y_pred = df["predicted_failure"].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = len(df)
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def build_overview_table() -> pd.DataFrame:
    rows: list[dict] = []
    for dataset_id, split in list_available_prediction_keys():
        df = _load_predictions(dataset_id=dataset_id, split=split)
        metrics = summarize_binary_metrics(df)
        rows.append(
            {
                "dataset_id": dataset_id,
                "split": split,
                "rows": metrics["total"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "positives(pred)": int(df["predicted_failure"].sum()) if "predicted_failure" in df else 0,
                "positives(actual)": int(df["actual_failure"].sum()) if "actual_failure" in df else 0,
            }
        )
    return pd.DataFrame(rows)


def build_confusion_df(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type": [
                "True Positive (TP)",
                "True Negative (TN)",
                "False Positive (FP)",
                "False Negative (FN)",
            ],
            "count": [metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]],
        }
    )


@st.cache_data(show_spinner=False)
def load_model_feature_importance(dataset_id: str) -> pd.DataFrame:
    model_name = "rf_all_fd_model.pkl" if dataset_id == "all_fd" else f"rf_{dataset_id.upper()}_model.pkl"
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        return pd.DataFrame()

    bundle = joblib.load(model_path)
    model = bundle.get("model")
    feature_cols = bundle.get("feature_columns", [])

    if model is None or not hasattr(model, "feature_importances_"):
        return pd.DataFrame()

    fi_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return fi_df


def add_graph_note(depicts: str, explanation: str | None = None):
    extra = f"<br><span>{explanation}</span>" if explanation else ""
    st.markdown(
        f"<div class='pm-card'><b>This graph depicts:</b> {depicts}{extra}</div>",
        unsafe_allow_html=True,
    )


st.title("📊 Predictive Maintenance Dashboard")
st.caption("Predictive maintenance results dashboard with metric and chart explanations")

st.markdown(
    """
    <style>
    .pm-card {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 10px;
        background: rgba(250, 250, 250, 0.6);
    }
    .pm-good {color: #1b8a3d; font-weight: 700;}
    .pm-warn {color: #c27a00; font-weight: 700;}
    .pm-bad {color: #b00020; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

available_keys = list_available_prediction_keys()
if not available_keys:
    st.error("No prediction files found in `outputs/predictions`. Run your pipeline first.")
    st.stop()

available_datasets = sorted({dataset_id for dataset_id, _ in available_keys})

with st.sidebar:
    st.header("Controls")
    view_mode = st.radio("Dashboard mode", options=["Presentation", "Analysis"], index=0)

    selected_dataset = st.selectbox("Dataset", options=available_datasets, index=0)

    split_options = [split for dataset_id, split in available_keys if dataset_id == selected_dataset]
    split_options = sorted(set(split_options))
    selected_split = st.selectbox("Split", options=split_options, index=0)

    selected_sensor = st.selectbox(
        "Sensor column",
        options=[f"sensor_{i}" for i in range(1, 25)],
        index=0,
    )

    st.markdown("---")
    st.subheader("Help for options")
    st.markdown(
        """
        - **Dataset**: which NASA subset you want to inspect (FD001…FD004 or all_fd).
        - **Split**:
          - `train` = model learning data
          - `test` = model evaluation data
        - **Sensor column**: picks which sensor trend to show in unit timeline.
        """
    )


pred_df = _load_predictions(dataset_id=selected_dataset, split=selected_split)
metrics = summarize_binary_metrics(pred_df)
overview_df = build_overview_table()
feature_df = load_model_feature_importance(selected_dataset)

st.info(
    f"You are viewing **{selected_dataset.upper()}** ({DATASET_EXPLANATIONS.get(selected_dataset, 'NASA subset')}) | "
    f"**{selected_split.upper()} split**"
)

with st.expander("Metric glossary (plain language)"):
    st.markdown(
        """
        - **Accuracy**: overall percentage of correct predictions.
        - **Precision**: out of all predicted failures, how many were actually failures.
        - **Recall**: out of all actual failures, how many the model correctly detected.
        - **F1 Score**: balance between precision and recall (higher is better balance).
        - **True Positive (TP)**: correctly predicted failure.
        - **True Negative (TN)**: correctly predicted non-failure.
        - **False Positive (FP)**: false alarm (predicted failure, but actually normal).
        - **False Negative (FN)**: missed failure (predicted normal, but actually failure).
        """
    )

st.subheader("Dataset snapshot")
snap_col1, snap_col2, snap_col3, snap_col4 = st.columns(4)

units_count = int(pred_df["unit"].nunique()) if "unit" in pred_df.columns else 0
max_cycle = int(pred_df["cycle"].max()) if "cycle" in pred_df.columns else 0
avg_rul = float(pred_df["RUL"].mean()) if "RUL" in pred_df.columns else 0.0
near_failure_pct = float(pred_df["actual_failure"].mean() * 100) if "actual_failure" in pred_df.columns else 0.0

snap_col1.metric("Engine units", f"{units_count:,}")
snap_col2.metric("Max cycle", f"{max_cycle:,}")
snap_col3.metric("Average RUL", f"{avg_rul:.1f}")
snap_col4.metric("Near-failure rate", f"{near_failure_pct:.2f}%")

st.caption(
    "This snapshot gives quick context about data scale (units/cycles), remaining life (RUL), and how frequent near-failure samples are."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{metrics['total']:,}")
col2.metric("Accuracy", f"{metrics['accuracy']:.4f}" if metrics["accuracy"] is not None else "N/A")
col3.metric("Precision", f"{metrics['precision']:.4f}" if metrics["precision"] is not None else "N/A")
col4.metric("Recall", f"{metrics['recall']:.4f}" if metrics["recall"] is not None else "N/A")

if metrics["total"] > 0:
    false_alarm_rate = metrics["fp"] / (metrics["fp"] + metrics["tn"]) if (metrics["fp"] + metrics["tn"]) else 0
    miss_rate = metrics["fn"] / (metrics["fn"] + metrics["tp"]) if (metrics["fn"] + metrics["tp"]) else 0
    # Quick qualitative label for presentation
    if miss_rate <= 0.10:
        risk_label = "<span class='pm-good'>Strong missed-failure control</span>"
    elif miss_rate <= 0.20:
        risk_label = "<span class='pm-warn'>Moderate missed-failure risk</span>"
    else:
        risk_label = "<span class='pm-bad'>High missed-failure risk</span>"

    st.markdown(
        f"<div class='pm-card'><b>Quick takeaway:</b> false alarm rate = <b>{false_alarm_rate:.2%}</b>, "
        f"missed-failure rate = <b>{miss_rate:.2%}</b>. {risk_label}</div>",
        unsafe_allow_html=True,
    )

overfit_df = load_overfitting_metrics()

if view_mode == "Presentation":
    slide1, slide2, slide3, slide4 = st.tabs(
        ["Slide 1: Problem", "Slide 2: Results", "Slide 3: Overfitting", "Slide 4: Unit Story"]
    )

    with slide1:
        st.subheader("What this project does")
        st.markdown(
            """
            - Predicts if an engine is near failure using sensor data.
            - **RUL** = Remaining Useful Life (how many cycles left before failure).
            - Label rule used here: **failure = 1 when RUL < 30**, else 0.
            - **Train split** teaches the model; **test split** checks real-world performance.
            """
        )
        st.markdown(
            f"<div class='pm-card'><b>{selected_dataset.upper()}</b>: "
            f"{DATASET_EXPLANATIONS.get(selected_dataset, 'NASA subset')}</div>",
            unsafe_allow_html=True,
        )

    with slide2:
        st.subheader("How well is the model performing?")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics['accuracy']:.3f}" if metrics["accuracy"] is not None else "N/A")
        c2.metric("Precision", f"{metrics['precision']:.3f}" if metrics["precision"] is not None else "N/A")
        c3.metric("Recall", f"{metrics['recall']:.3f}" if metrics["recall"] is not None else "N/A")

        st.caption("In predictive maintenance, recall is very important because missed failures are risky.")

        if not overview_df.empty:
            fig_acc = px.bar(
                overview_df,
                x="dataset_id",
                y="accuracy",
                color="split",
                barmode="group",
                text_auto=".3f",
            )
            fig_acc.update_layout(height=380)
            st.plotly_chart(fig_acc, use_container_width=True)
            add_graph_note(
                "Accuracy for each FD dataset, shown separately for train and test.",
                "This bar chart compares how accurate the model is on seen data (train) versus unseen data (test) across FD datasets.",
            )

        cm_df = build_confusion_df(metrics)
        fig_cm = px.bar(cm_df, x="type", y="count", color="type", text_auto=True)
        fig_cm.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_cm, use_container_width=True)
        add_graph_note(
            "Confusion categories with full terms: True Positive, True Negative, False Positive, and False Negative for the selected dataset and split.",
            "This chart shows correct predictions and errors; reducing FN is especially important because missed failures are risky.",
        )

    with slide3:
        st.subheader("Is the model overfitting?")
        st.caption("Overfitting means model performs much better on train data than unseen test data.")
        if not overfit_df.empty:
            overfit_df = overfit_df.copy()
            overfit_df["dataset"] = overfit_df["dataset"].astype(str)
            fig_gap = px.line(overfit_df, x="dataset", y=["train_accuracy", "test_accuracy"], markers=True)
            fig_gap.update_layout(height=340)
            st.plotly_chart(fig_gap, use_container_width=True)
            add_graph_note(
                "Two lines: train accuracy and test accuracy for each dataset.",
                "If train and test lines stay close, the model generalizes well and is not heavily overfitting.",
            )

            fig_gap_bar = px.bar(overfit_df, x="dataset", y="gap", color="gap", text_auto=".3f")
            fig_gap_bar.update_layout(height=320)
            st.plotly_chart(fig_gap_bar, use_container_width=True)
            add_graph_note(
                "The train-test accuracy gap (overfitting gap) per dataset.",
                "Smaller gap means better generalization; larger gap means possible overfitting.",
            )

            avg_gap = float(overfit_df["gap"].mean()) if "gap" in overfit_df else 0.0
            if avg_gap < 0.03:
                msg = "<span class='pm-good'>Low average train-test gap (good generalization).</span>"
            elif avg_gap < 0.06:
                msg = "<span class='pm-warn'>Moderate gap (watch for mild overfitting).</span>"
            else:
                msg = "<span class='pm-bad'>High gap (possible overfitting).</span>"
            st.markdown(f"<div class='pm-card'><b>Average gap:</b> {avg_gap:.3f}. {msg}</div>", unsafe_allow_html=True)
        else:
            st.info("`outputs/overfitting_metrics.csv` not found. Run `overfitting_graph.py` first if needed.")

    with slide4:
        st.subheader("One engine unit story")
        st.caption("Use this to explain sensor behaviour and RUL trend over time.")
        if {"unit", "cycle"}.issubset(pred_df.columns):
            unit_ids = sorted(pred_df["unit"].dropna().unique().tolist())
            selected_unit = st.selectbox("Unit", options=unit_ids, index=0)
            unit_df = pred_df[pred_df["unit"] == selected_unit].sort_values("cycle")

            c_left, c_right = st.columns(2)
            with c_left:
                if "RUL" in unit_df.columns:
                    fig_rul = px.line(unit_df, x="cycle", y="RUL", markers=True, title="RUL decreases over cycle")
                    st.plotly_chart(fig_rul, use_container_width=True)
                    add_graph_note(
                        "Remaining Useful Life (RUL) of one engine unit as cycles increase.",
                        "RUL naturally decreases with cycle count, showing the engine approaching end-of-life.",
                    )
            with c_right:
                if selected_sensor in unit_df.columns:
                    fig_sensor = px.line(unit_df, x="cycle", y=selected_sensor, markers=True, title=f"{selected_sensor} trend")
                    st.plotly_chart(fig_sensor, use_container_width=True)
                    add_graph_note(
                        f"Sensor trend for {selected_sensor} over cycles for one unit.",
                        "This helps show how sensor behavior changes as the engine ages.",
                    )

            if "predicted_failure" in unit_df.columns:
                fig_pred = px.scatter(unit_df, x="cycle", y="predicted_failure", color="predicted_failure", title="Predicted failure signal")
                st.plotly_chart(fig_pred, use_container_width=True)
                add_graph_note(
                    "Model failure prediction over time for one unit (0 = normal, 1 = near failure).",
                    "Towards end-of-life, more points become 1, indicating rising failure risk.",
                )

else:
    left, right = st.columns((3, 2))

    with left:
        st.subheader("Accuracy by dataset/split")
        st.caption("Compares model correctness across datasets and between training vs unseen testing data.")
        if not overview_df.empty:
            fig_acc = px.bar(
                overview_df,
                x="dataset_id",
                y="accuracy",
                color="split",
                barmode="group",
                hover_data=["rows", "precision", "recall"],
            )
            fig_acc.update_layout(height=360)
            st.plotly_chart(fig_acc, use_container_width=True)
            add_graph_note(
                "Accuracy per FD dataset for train and test.",
                "This compares learning performance vs real evaluation performance across datasets.",
            )

    with right:
        st.subheader("Confusion components")
        st.caption("Shows where the model is right/wrong: TP, TN, FP (false alarm), FN (missed failure).")
        cm_df = build_confusion_df(metrics)
        fig_cm = px.bar(cm_df, x="type", y="count", color="type")
        fig_cm.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_cm, use_container_width=True)
        add_graph_note(
            "Counts of True Positive, True Negative, False Positive, and False Negative for the selected data.",
            "This shows where the model is correct and where it makes mistakes.",
        )

    st.subheader("Overfitting view (train vs test gap)")
    st.caption("If train accuracy is much higher than test accuracy, the model may be overfitting.")
    if not overfit_df.empty:
        overfit_df = overfit_df.copy()
        overfit_df["dataset"] = overfit_df["dataset"].astype(str)
        fig_gap = px.line(
            overfit_df,
            x="dataset",
            y=["train_accuracy", "test_accuracy"],
            markers=True,
        )
        fig_gap.update_layout(height=320)
        st.plotly_chart(fig_gap, use_container_width=True)
        add_graph_note(
            "Train and test accuracy lines across datasets.",
            "Closer train and test values indicate better real-world generalization.",
        )

        fig_gap_bar = px.bar(overfit_df, x="dataset", y="gap", color="gap", color_continuous_scale="Reds")
        fig_gap_bar.update_layout(height=280)
        st.plotly_chart(fig_gap_bar, use_container_width=True)
        add_graph_note(
            "Overfitting gap bar chart by dataset.",
            "Lower gap is better; higher gap means the model may be memorizing training patterns.",
        )
    else:
        st.info("`outputs/overfitting_metrics.csv` not found. Run `overfitting_graph.py` first if needed.")

    if {"unit", "cycle"}.issubset(pred_df.columns):
        st.subheader("Unit-level timeline")
        st.caption(
            "Pick one engine unit and inspect how RUL and sensors change over cycles. "
            "As cycle increases, RUL typically decreases."
        )
        unit_ids = sorted(pred_df["unit"].dropna().unique().tolist())
        selected_unit = st.selectbox("Unit", options=unit_ids, index=0)

        unit_df = pred_df[pred_df["unit"] == selected_unit].sort_values("cycle")

        chart_cols = st.columns(2)

        with chart_cols[0]:
            if "RUL" in unit_df.columns:
                fig_rul = px.line(unit_df, x="cycle", y="RUL", markers=True, title="RUL over cycle")
                st.plotly_chart(fig_rul, use_container_width=True)
                add_graph_note(
                    "RUL trend for one unit over cycles.",
                    "This should decline as the machine gets closer to failure.",
                )

        with chart_cols[1]:
            if selected_sensor in unit_df.columns:
                fig_sensor = px.line(
                    unit_df,
                    x="cycle",
                    y=selected_sensor,
                    markers=True,
                    title=f"{selected_sensor} over cycle",
                )
                st.plotly_chart(fig_sensor, use_container_width=True)
                add_graph_note(
                    f"Trend of {selected_sensor} for one engine unit.",
                    "Sensor drift/pattern change can indicate health degradation.",
                )

        if "predicted_failure" in unit_df.columns:
            fig_pred = px.scatter(
                unit_df,
                x="cycle",
                y="predicted_failure",
                color="predicted_failure",
                title="Predicted failure signal",
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            add_graph_note(
                "Predicted failure status for each cycle of one unit.",
                "More 1s near later cycles suggest the model is flagging approaching failure.",
            )

    st.subheader("Sample data")
    st.caption("Raw rows used in the selected view. Useful for validating column values and model output.")
    st.dataframe(pred_df.head(50), use_container_width=True)

st.subheader("Additional insights")
insight_tab1, insight_tab2, insight_tab3 = st.tabs(
    ["Failure balance", "Feature importance", "Dataset performance table"]
)

with insight_tab1:
    if {"actual_failure", "predicted_failure"}.issubset(pred_df.columns):
        balance_df = pd.DataFrame(
            {
                "Category": ["Actual near-failure", "Predicted near-failure"],
                "Percentage": [
                    float(pred_df["actual_failure"].mean() * 100),
                    float(pred_df["predicted_failure"].mean() * 100),
                ],
            }
        )
        fig_balance = px.bar(balance_df, x="Category", y="Percentage", text_auto=".2f", color="Category")
        fig_balance.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_balance, use_container_width=True)
        add_graph_note(
            "Comparison of actual near-failure rate vs model-predicted near-failure rate for the selected dataset/split.",
            "Large mismatch may indicate under-prediction or over-prediction of risk.",
        )
    else:
        st.info("Failure balance chart needs `actual_failure` and `predicted_failure` columns.")

with insight_tab2:
    if not feature_df.empty:
        top_n = st.slider("Top important features to show", min_value=5, max_value=20, value=10, step=1)
        top_fi = feature_df.head(top_n).sort_values("importance", ascending=True)
        fig_fi = px.bar(top_fi, x="importance", y="feature", orientation="h", text_auto=".3f")
        fig_fi.update_layout(height=420)
        st.plotly_chart(fig_fi, use_container_width=True)
        add_graph_note(
            "Top model feature importances from the trained Random Forest model.",
            "Higher importance means that feature contributes more to the model's decision-making.",
        )
    else:
        st.info("Feature importance is unavailable because a matching trained model file was not found.")

with insight_tab3:
    if not overview_df.empty:
        perf_df = overview_df.copy()
        perf_df["accuracy"] = perf_df["accuracy"].round(4)
        perf_df["precision"] = perf_df["precision"].round(4)
        perf_df["recall"] = perf_df["recall"].round(4)
        st.dataframe(perf_df.sort_values(["split", "accuracy"], ascending=[True, False]), use_container_width=True)

        if "test" in perf_df["split"].values:
            test_perf = perf_df[perf_df["split"] == "test"].sort_values("accuracy", ascending=False)
            if not test_perf.empty:
                best_row = test_perf.iloc[0]
                st.success(
                    f"Best test accuracy: **{best_row['dataset_id']}** with **{best_row['accuracy']:.4f}** accuracy."
                )
        st.caption("This table compares all datasets together, helpful for final evaluation summary.")
    else:
        st.info("Performance summary table is unavailable because overview data is empty.")
