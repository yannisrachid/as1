# coding: utf-8

"""
Transferability project
Run the main modeling experiments before selecting the retained model
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from data_loading import load_raw_datasets
from feature_engineering import build_modeling_dataset
from modeling import align_feature_columns
from modeling import build_model_matrix
from modeling import compute_classification_metrics
from modeling import split_train_test_by_year
from run_retained_models import add_snapshot_features
from run_retained_models import select_clean_features


def train_logistic_regression(train_features, train_target):
    """
    Train a simple linear baseline.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature matrix.
    train_target : pd.Series
        Training target.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    model = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        solver="liblinear",
        random_state=42,
    )
    model.fit(train_features, train_target)
    return model


def train_random_forest(train_features, train_target, max_depth=None):
    """
    Train a Random Forest experiment.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature matrix.
    train_target : pd.Series
        Training target.
    max_depth : int, optional
        Maximum tree depth.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=max_depth,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_features, train_target)
    return model


def train_xgboost(train_features, train_target, parameters):
    """
    Train one XGBoost experiment.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature matrix.
    train_target : pd.Series
        Training target.
    parameters : dict
        XGBoost parameters.

    Returns
    -------
    XGBClassifier
        Fitted model.
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
    model_parameters.update(parameters)

    model = XGBClassifier(**model_parameters)
    model.fit(train_features, train_target)
    return model


def extract_top_feature_importance(model, feature_names, top_n=40):
    """
    Extract feature importances when the model exposes them.

    Parameters
    ----------
    model : estimator
        Fitted model.
    feature_names : list-like
        Encoded feature names.
    top_n : int, optional
        Number of rows to keep.

    Returns
    -------
    pd.DataFrame
        Feature importance table.
    """
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature_name", "importance"])

    feature_importance = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": model.feature_importances_,
        }
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False).head(top_n)
    return feature_importance.reset_index(drop=True)


def build_prediction_table(test_dataset, target_proba, threshold=0.5):
    """
    Build a compact prediction table for one experiment.

    Parameters
    ----------
    test_dataset : pd.DataFrame
        Test dataset with identifiers.
    target_proba : array-like
        Predicted probabilities.
    threshold : float, optional
        Classification threshold.

    Returns
    -------
    pd.DataFrame
        Prediction table.
    """
    prediction_table = test_dataset[
        ["player_id", "player_name", "season_name", "summer_year", "target"]
    ].copy()
    prediction_table["transfer_probability"] = target_proba
    prediction_table["predicted_label"] = (prediction_table["transfer_probability"] >= threshold).astype(int)
    prediction_table["is_error"] = (
        prediction_table["predicted_label"] != prediction_table["target"]
    ).astype(int)
    prediction_table = prediction_table.sort_values("transfer_probability", ascending=False)
    return prediction_table.reset_index(drop=True)


def main():
    """
    Run the main model experiments and save their 2025 holdout results.

    Returns
    -------
    None
    """
    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "transferability" / "data" / "raw"
    output_dir = project_root / "transferability" / "outputs" / "model_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_datasets = load_raw_datasets(raw_data_dir)
    modeling_dataset = build_modeling_dataset(raw_datasets)
    modeling_dataset = modeling_dataset.loc[
        modeling_dataset["summer_year"].isin([2022, 2023, 2024, 2025])
    ].copy()
    modeling_dataset = add_snapshot_features(modeling_dataset, raw_datasets["players"])
    modeling_dataset = select_clean_features(modeling_dataset)

    train_dataset, test_dataset = split_train_test_by_year(
        modeling_dataset,
        train_years=[2022, 2023, 2024],
        test_years=[2025],
    )
    train_features, train_target = build_model_matrix(train_dataset)
    test_features, test_target = build_model_matrix(test_dataset)
    train_features, test_features = align_feature_columns(train_features, test_features)

    experiments = [
        {
            "model_name": "logistic_regression_baseline",
            "model": train_logistic_regression(train_features, train_target),
            "comment": "Linear baseline, useful as a sanity check.",
        },
        {
            "model_name": "random_forest_shallow",
            "model": train_random_forest(train_features, train_target, max_depth=6),
            "comment": "Conservative RF, less variance but less expressive.",
        },
        {
            "model_name": "random_forest_benchmark",
            "model": train_random_forest(train_features, train_target, max_depth=10),
            "comment": "Main RF benchmark.",
        },
        {
            "model_name": "xgboost_default",
            "model": train_xgboost(
                train_features,
                train_target,
                {
                    "n_estimators": 300,
                    "max_depth": 6,
                    "learning_rate": 0.10,
                    "subsample": 0.85,
                    "colsample_bytree": 0.85,
                    "min_child_weight": 5,
                    "reg_lambda": 1.0,
                },
            ),
            "comment": "Untuned XGBoost starting point.",
        },
        {
            "model_name": "xgboost_more_regularized",
            "model": train_xgboost(
                train_features,
                train_target,
                {
                    "n_estimators": 400,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.90,
                    "colsample_bytree": 0.80,
                    "min_child_weight": 8,
                    "reg_lambda": 2.0,
                    "reg_alpha": 0.1,
                    "gamma": 0.1,
                },
            ),
            "comment": "Regularized XGBoost to reduce noisy splits.",
        },
        {
            "model_name": "xgboost_retained",
            "model": train_xgboost(
                train_features,
                train_target,
                {
                    "n_estimators": 450,
                    "max_depth": 5,
                    "learning_rate": 0.04,
                    "subsample": 0.90,
                    "colsample_bytree": 0.80,
                    "min_child_weight": 7,
                    "reg_lambda": 2.5,
                    "reg_alpha": 0.2,
                    "gamma": 0.2,
                },
            ),
            "comment": "Retained model after manual tuning.",
        },
    ]

    metric_rows = []

    for experiment in experiments:
        model_name = experiment["model_name"]
        model = experiment["model"]
        target_proba = model.predict_proba(test_features)[:, 1]
        metrics = compute_classification_metrics(test_target, target_proba, threshold=0.5)
        metrics["model_name"] = model_name
        metrics["comment"] = experiment["comment"]
        metrics["n_features_after_encoding"] = int(train_features.shape[1])
        metric_rows.append(metrics)

        prediction_table = build_prediction_table(test_dataset, target_proba, threshold=0.5)
        prediction_table.to_csv(output_dir / "{0}_predictions_2025.csv".format(model_name), index=False)

        feature_importance = extract_top_feature_importance(model, train_features.columns, top_n=60)
        if not feature_importance.empty:
            feature_importance.to_csv(
                output_dir / "{0}_feature_importance.csv".format(model_name),
                index=False,
            )

    comparison_table = pd.DataFrame(metric_rows)
    comparison_table = comparison_table[
        [
            "model_name",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
            "positive_rate",
            "n_features_after_encoding",
            "comment",
        ]
    ].sort_values(["average_precision", "roc_auc"], ascending=False)
    comparison_table.to_csv(output_dir / "model_experiments_comparison_2025.csv", index=False)

    print(comparison_table.to_string(index=False))
    print()
    print("Saved experiment outputs to:", output_dir)


if __name__ == "__main__":
    main()
