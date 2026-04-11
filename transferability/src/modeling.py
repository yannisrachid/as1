# coding: utf-8

"""
Transferability project
Baseline modeling helpers
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


IDENTIFIER_COLUMNS = [
    "player_id",
    "player_name",
    "season_name",
    "summer_year",
    "reference_date",
    "target",
]


def split_train_test_by_year(modeling_dataset, train_years, test_years):
    """
    Split the modeling dataset with an explicit time-based rule.

    Parameters
    ----------
    modeling_dataset : pd.DataFrame
        Final modeling dataset.
    train_years : list of int
        Summer years used for training.
    test_years : list of int
        Summer years used for testing.

    Returns
    -------
    tuple
        Train DataFrame, test DataFrame.
    """
    train_dataset = modeling_dataset.loc[modeling_dataset["summer_year"].isin(train_years)].copy()
    test_dataset = modeling_dataset.loc[modeling_dataset["summer_year"].isin(test_years)].copy()
    return train_dataset, test_dataset


def build_model_matrix(dataframe):
    """
    Convert the modeling dataset into a machine-learning matrix.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Modeling dataset.

    Returns
    -------
    tuple
        Feature matrix and target vector.
    """
    feature_dataframe = dataframe.drop(columns=IDENTIFIER_COLUMNS, errors="ignore").copy()
    feature_dataframe = pd.get_dummies(feature_dataframe, drop_first=False)
    target_vector = dataframe["target"].astype(int)
    return feature_dataframe, target_vector


def align_feature_columns(train_features, test_features):
    """
    Align train and test feature matrices after one-hot encoding.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature matrix.
    test_features : pd.DataFrame
        Test feature matrix.

    Returns
    -------
    tuple
        Aligned train and test feature matrices.
    """
    aligned_train_features, aligned_test_features = train_features.align(
        test_features, join="left", axis=1, fill_value=0
    )
    return aligned_train_features, aligned_test_features


def train_random_forest(train_features, train_target):
    """
    Train the baseline Random Forest model.

    Parameters
    ----------
    train_features : pd.DataFrame
        Training feature matrix.
    train_target : pd.Series
        Training target vector.

    Returns
    -------
    RandomForestClassifier
        Fitted baseline model.
    """
    baseline_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    baseline_model.fit(train_features, train_target)
    return baseline_model


def compute_classification_metrics(target_true, target_proba, threshold=0.5):
    """
    Compute the key classification metrics used in the first test.

    Parameters
    ----------
    target_true : array-like
        Observed target values.
    target_proba : array-like
        Predicted positive class probabilities.
    threshold : float, optional
        Probability threshold used for the hard predictions.

    Returns
    -------
    dict
        Metric dictionary.
    """
    target_pred = (target_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(target_true, target_pred)),
        "precision": float(precision_score(target_true, target_pred, zero_division=0)),
        "recall": float(recall_score(target_true, target_pred, zero_division=0)),
        "f1": float(f1_score(target_true, target_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(target_true, target_proba)),
        "average_precision": float(average_precision_score(target_true, target_proba)),
        "threshold": float(threshold),
        "positive_rate": float(pd.Series(target_true).mean()),
    }
    return metrics


def build_prediction_table(test_dataset, target_proba):
    """
    Build a readable prediction table for the test period.

    Parameters
    ----------
    test_dataset : pd.DataFrame
        Test modeling dataset.
    target_proba : array-like
        Predicted probabilities.

    Returns
    -------
    pd.DataFrame
        Test prediction table.
    """
    prediction_table = test_dataset[
        ["player_id", "player_name", "season_name", "summer_year", "target"]
    ].copy()
    prediction_table["transfer_probability"] = target_proba
    prediction_table = prediction_table.sort_values(
        ["transfer_probability", "player_name"], ascending=[False, True]
    )
    return prediction_table


def extract_feature_importance(model, feature_names, top_n=30):
    """
    Extract the most important baseline features.

    Parameters
    ----------
    model : RandomForestClassifier
        Fitted baseline model.
    feature_names : list-like
        Feature matrix column names.
    top_n : int, optional
        Number of features to return.

    Returns
    -------
    pd.DataFrame
        Feature importance table.
    """
    feature_importance = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": model.feature_importances_,
        }
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False).head(top_n)
    return feature_importance.reset_index(drop=True)


def save_metrics(metrics, output_path):
    """
    Save the model metrics as JSON.

    Parameters
    ----------
    metrics : dict
        Metric dictionary.
    output_path : str or Path
        Output JSON path.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(metrics, output_file, indent=4)


def save_prediction_table(prediction_table, output_path):
    """
    Save the test prediction table as CSV.

    Parameters
    ----------
    prediction_table : pd.DataFrame
        Prediction table.
    output_path : str or Path
        Output CSV path.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(output_path, index=False)


def save_feature_importance(feature_importance, output_path):
    """
    Save the feature importance table as CSV.

    Parameters
    ----------
    feature_importance : pd.DataFrame
        Feature importance table.
    output_path : str or Path
        Output CSV path.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.to_csv(output_path, index=False)


def save_model(model, output_path):
    """
    Save the fitted model with joblib.

    Parameters
    ----------
    model : sklearn estimator
        Fitted model.
    output_path : str or Path
        Output model path.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
