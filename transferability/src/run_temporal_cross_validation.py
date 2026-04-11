# coding: utf-8

"""
Transferability project
Temporal validation for the retained XGBoost model
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

from data_loading import load_raw_datasets
from feature_engineering import build_modeling_dataset
from modeling import align_feature_columns
from modeling import build_model_matrix
from modeling import split_train_test_by_year
from run_final_models import add_snapshot_features
from run_final_models import build_prediction_table
from run_final_models import compute_metrics
from run_final_models import select_clean_features


XGBOOST_RETAINED_PARAMETERS = {
    "n_estimators": 450,
    "max_depth": 5,
    "learning_rate": 0.04,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 7,
    "reg_lambda": 2.5,
    "reg_alpha": 0.2,
    "gamma": 0.2,
}


def get_project_paths():
    """
    Resolve project paths for either an api/src or a simple src layout.

    Returns
    -------
    tuple
        Project root, raw data directory and output directory.
    """
    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "transferability" / "data" / "raw"
    output_dir = project_root / "transferability" / "outputs" / "temporal_validation"

    if not raw_data_dir.exists():
        project_root = Path(__file__).resolve().parents[1]
        raw_data_dir = project_root / "data" / "raw"
        output_dir = project_root / "outputs" / "temporal_validation"

    return project_root, raw_data_dir, output_dir


def train_xgboost_retained(train_features, train_target):
    """
    Train the retained XGBoost model on one temporal fold.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature matrix.
    train_target : pd.Series
        Training target.

    Returns
    -------
    XGBClassifier
        Fitted retained model.
    """
    positive_count = max(int(train_target.sum()), 1)
    negative_count = max(int((1 - train_target).sum()), 1)
    scale_pos_weight = negative_count / positive_count

    model_parameters = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }
    model_parameters.update(XGBOOST_RETAINED_PARAMETERS)

    model = XGBClassifier(**model_parameters)
    model.fit(train_features, train_target)
    return model


def build_confusion_row(train_years, test_year, y_true, y_proba, threshold=0.5):
    """
    Build a confusion matrix row for one temporal fold.

    Parameters
    ----------
    train_years : list of int
        Years used for training.
    test_year : int
        Test year.
    y_true : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities.
    threshold : float, optional
        Classification threshold.

    Returns
    -------
    dict
        Confusion matrix summary.
    """
    y_pred = (y_proba >= threshold).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    return {
        "model_name": "xgboost_retained",
        "train_years": ",".join(str(year) for year in train_years),
        "test_year": int(test_year),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
        "predicted_positive_rate": float(y_pred.mean()),
    }


def main():
    """
    Run expanding-window temporal validation for xgboost_retained.

    Returns
    -------
    None
    """
    _, raw_data_dir, output_dir = get_project_paths()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_datasets = load_raw_datasets(raw_data_dir)
    modeling_dataset = build_modeling_dataset(raw_datasets)
    modeling_dataset = modeling_dataset.loc[
        modeling_dataset["summer_year"].isin([2022, 2023, 2024, 2025])
    ].copy()
    modeling_dataset = add_snapshot_features(modeling_dataset, raw_datasets["players"])
    modeling_dataset = select_clean_features(modeling_dataset)

    metric_rows = []
    confusion_rows = []

    for test_year in [2023, 2024, 2025]:
        train_years = [
            year for year in sorted(modeling_dataset["summer_year"].unique()) if year < test_year
        ]
        train_dataset, test_dataset = split_train_test_by_year(
            modeling_dataset,
            train_years=train_years,
            test_years=[test_year],
        )

        train_features, train_target = build_model_matrix(train_dataset)
        test_features, test_target = build_model_matrix(test_dataset)
        train_features, test_features = align_feature_columns(train_features, test_features)

        model = train_xgboost_retained(train_features, train_target)
        test_proba = model.predict_proba(test_features)[:, 1]
        metrics = compute_metrics(test_target, test_proba, threshold=0.5)

        metrics["model_name"] = "xgboost_retained"
        metrics["train_years"] = ",".join(str(year) for year in train_years)
        metrics["test_year"] = int(test_year)
        metrics["n_train_rows"] = int(train_dataset.shape[0])
        metrics["n_test_rows"] = int(test_dataset.shape[0])
        metrics["train_positive_rate"] = float(train_target.mean())
        metrics["test_positive_rate"] = float(test_target.mean())
        metric_rows.append(metrics)

        confusion_rows.append(
            build_confusion_row(train_years, test_year, test_target, test_proba, threshold=0.5)
        )

        prediction_table = build_prediction_table(test_dataset, test_proba, threshold=0.5)
        prediction_table.to_csv(
            output_dir / "xgboost_retained_predictions_{0}.csv".format(test_year),
            index=False,
        )

    metrics_table = pd.DataFrame(metric_rows)
    metrics_table = metrics_table[
        [
            "model_name",
            "train_years",
            "test_year",
            "n_train_rows",
            "n_test_rows",
            "train_positive_rate",
            "test_positive_rate",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]
    ]
    confusion_table = pd.DataFrame(confusion_rows)

    metrics_table.to_csv(output_dir / "xgboost_retained_temporal_metrics.csv", index=False)
    confusion_table.to_csv(output_dir / "xgboost_retained_temporal_confusion.csv", index=False)

    print("TEMPORAL VALIDATION - xgboost_retained")
    print(metrics_table.to_string(index=False))
    print()
    print("CONFUSION MATRICES")
    print(confusion_table.to_string(index=False))
    print()
    print("Saved outputs to:", output_dir)


if __name__ == "__main__":
    main()
