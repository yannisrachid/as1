# coding: utf-8

"""
Transferability project
Train and compare the retained models on the 2025 holdout
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from data_loading import load_raw_datasets
from feature_engineering import build_modeling_dataset
from modeling import align_feature_columns
from modeling import build_model_matrix
from modeling import split_train_test_by_year


IDENTIFIER_COLUMNS = [
    "player_id",
    "player_name",
    "season_name",
    "summer_year",
    "reference_date",
    "target",
]


CLEAN_FEATURE_COLUMNS = [
    "age_at_window",
    "age_band",
    "main_position",
    "is_eu",
    "citizenship_focus_group",
    "is_french_spanish_portuguese",
    "matches_in_squad",
    "matches_played",
    "minutes_played",
    "previous_minutes_played",
    "minutes_trend",
    "goals_per_90",
    "assists_per_90",
    "bench_appearance_share",
    "nb_market_moves_before_window",
    "nb_transfers_before_window",
    "nb_loans_before_window",
    "nb_destination_teams_before_window",
    "days_since_last_move",
    "moved_in_last_365_days",
    "was_last_move_loan",
    "was_last_move_transfer",
    "nb_injuries_last_365_days",
    "total_injury_days_last_365_days",
    "total_games_missed_last_365_days",
    "had_severe_injury_last_365_days",
    "nb_national_records_before_window",
    "total_national_caps_before_window",
    "national_caps_last_365_days",
    "has_senior_national_team_record",
    "primary_team_name",
    "primary_competition_name",
    "competition_tier_proxy",
    "primary_competition_is_lower_division",
    "team_season_player_count",
    "team_season_competition_count",
    "team_previous_market_move_rate",
    "team_two_year_market_move_rate",
    "team_previous_market_move_count",
    "competition_previous_market_move_rate",
    "competition_previous_market_move_count",
    "competition_previous_intra_league_rate",
    "tier_previous_market_move_rate",
    "team_previous_top5_export_rate",
    "team_previous_top5_export_count",
    "competition_previous_top5_export_rate",
    "competition_previous_top5_export_count",
    "team_is_high_churn_market",
    "team_is_top5_exporter_market",
    "competition_is_top5_export_market",
    "competition_is_intra_league_market",
    "performance_boost_score",
    "market_profile_score",
    "young_export_talent_profile",
    "english_domestic_market_profile",
    "stalled_star_profile",
    "national_team_showcase_profile",
    "loan_relaunch_profile",
    "top5_ready_profile",
    "has_contract_expiry_latest",
    "days_to_contract_expiry_latest",
    "contract_expiry_bucket_latest",
    "contract_expiring_within_1_year_latest",
    "contract_expiring_within_2_years_latest",
    "contract_trade_window_latest",
    "prime_age_contract_profile",
    "young_contract_growth_profile",
    "elite_contract_window_profile",
    "loan_contract_relaunch_profile",
]


def map_citizenship_focus_group(citizenship):
    """
    Map raw citizenship into a compact football-market group.

    Parameters
    ----------
    citizenship : str
        Raw citizenship.

    Returns
    -------
    str
        Citizenship group.
    """
    if pd.isna(citizenship) or citizenship == "Unknown":
        return "Unknown"
    if citizenship in ["Brazil", "Argentina"]:
        return "Brazil_Argentina"
    if citizenship == "England":
        return "England"
    if citizenship in ["France", "Spain", "Portugal"]:
        return "France_Spain_Portugal"
    return "Other"


def add_snapshot_features(modeling_dataset, players):
    """
    Add the few snapshot fields kept for the final simple model.

    Parameters
    ----------
    modeling_dataset : pd.DataFrame
        Modeling dataset built from historical sources.
    players : pd.DataFrame
        Raw players table.

    Returns
    -------
    pd.DataFrame
        Dataset with contract and demographic helper fields.
    """
    dataset = modeling_dataset.copy()
    players = players.copy()
    players["contract_expires"] = pd.to_datetime(players["contract_expires"], errors="coerce")

    dataset = dataset.merge(
        players[["player_id", "contract_expires"]],
        on="player_id",
        how="left",
    )
    dataset["reference_date"] = pd.to_datetime(dataset["reference_date"], errors="coerce")
    dataset["days_to_contract_expiry_latest"] = (
        dataset["contract_expires"] - dataset["reference_date"]
    ).dt.days
    dataset["has_contract_expiry_latest"] = dataset["contract_expires"].notna().astype(int)
    dataset["contract_expiring_within_1_year_latest"] = (
        dataset["days_to_contract_expiry_latest"].between(0, 365, inclusive="both")
    ).astype(int)
    dataset["contract_expiring_within_2_years_latest"] = (
        dataset["days_to_contract_expiry_latest"].between(0, 730, inclusive="both")
    ).astype(int)
    dataset["contract_trade_window_latest"] = (
        dataset["days_to_contract_expiry_latest"].between(180, 730, inclusive="both")
    ).astype(int)

    dataset["contract_expiry_bucket_latest"] = "Unknown"
    dataset.loc[dataset["days_to_contract_expiry_latest"] < 0, "contract_expiry_bucket_latest"] = (
        "Expired_or_before_window"
    )
    dataset.loc[
        dataset["days_to_contract_expiry_latest"].between(0, 365, inclusive="both"),
        "contract_expiry_bucket_latest",
    ] = "0_to_1_year"
    dataset.loc[
        dataset["days_to_contract_expiry_latest"].between(366, 730, inclusive="both"),
        "contract_expiry_bucket_latest",
    ] = "1_to_2_years"
    dataset.loc[dataset["days_to_contract_expiry_latest"] > 730, "contract_expiry_bucket_latest"] = (
        "2_plus_years"
    )
    dataset["days_to_contract_expiry_latest"] = (
        dataset["days_to_contract_expiry_latest"].clip(lower=-365, upper=3650).fillna(0)
    )

    dataset["age_band"] = pd.cut(
        dataset["age_at_window"],
        bins=[0, 21, 24, 28, 50],
        labels=["u21", "21_24", "25_28", "29_plus"],
        right=True,
    ).astype(str)
    dataset["citizenship_focus_group"] = dataset["citizenship"].apply(map_citizenship_focus_group)
    dataset["is_french_spanish_portuguese"] = dataset["citizenship"].isin(
        ["France", "Spain", "Portugal"]
    ).astype(int)

    dataset["prime_age_contract_profile"] = (
        dataset["age_at_window"].between(23, 29, inclusive="both")
        & (dataset["contract_trade_window_latest"] == 1)
        & ((dataset["minutes_played"] >= 1200) | (dataset["national_caps_last_365_days"] >= 3))
    ).astype(int)
    dataset["young_contract_growth_profile"] = (
        (dataset["age_at_window"] <= 24)
        & (dataset["contract_trade_window_latest"] == 1)
        & ((dataset["previous_minutes_played"] >= 900) | (dataset["performance_boost_score"] >= 1))
    ).astype(int)
    dataset["elite_contract_window_profile"] = (
        (dataset["contract_trade_window_latest"] == 1)
        & (
            (dataset["is_goal_contribution_star"] == 1)
            | (dataset["elite_national_output_boost"] == 1)
            | (dataset["strong_previous_season_then_minutes_drop_no_injury"] == 1)
        )
    ).astype(int)
    dataset["loan_contract_relaunch_profile"] = (
        (dataset["contract_trade_window_latest"] == 1)
        & (dataset["was_last_move_loan"] == 1)
        & (dataset["minutes_played"] >= 1200)
    ).astype(int)

    return dataset.drop(columns=["contract_expires"], errors="ignore")


def select_clean_features(modeling_dataset):
    """
    Keep identifiers, target and the clean feature list available in the dataset.

    Parameters
    ----------
    modeling_dataset : pd.DataFrame
        Full modeling dataset.

    Returns
    -------
    pd.DataFrame
        Reduced modeling dataset.
    """
    available_features = [
        column for column in CLEAN_FEATURE_COLUMNS if column in modeling_dataset.columns
    ]
    return modeling_dataset[IDENTIFIER_COLUMNS + available_features].copy()


def compute_metrics(y_true, y_proba, threshold=0.5):
    """
    Compute the metrics used in the model comparison.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities.
    threshold : float, optional
        Classification threshold.

    Returns
    -------
    dict
        Metric dictionary.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "positive_rate": float(pd.Series(y_true).mean()),
    }


def build_prediction_table(test_dataset, y_proba, threshold=0.5):
    """
    Build a readable prediction table for notebook review.

    Parameters
    ----------
    test_dataset : pd.DataFrame
        Test dataset with identifiers.
    y_proba : array-like
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
    prediction_table["transfer_probability"] = y_proba
    prediction_table["predicted_label"] = (prediction_table["transfer_probability"] >= threshold).astype(int)
    prediction_table["error_type"] = "correct"
    prediction_table.loc[
        (prediction_table["predicted_label"] == 1) & (prediction_table["target"] == 0),
        "error_type",
    ] = "false_positive"
    prediction_table.loc[
        (prediction_table["predicted_label"] == 0) & (prediction_table["target"] == 1),
        "error_type",
    ] = "false_negative"
    return prediction_table.sort_values("transfer_probability", ascending=False).reset_index(drop=True)


def save_feature_importance(model, feature_names, output_path, top_n=50):
    """
    Save tree feature importances.

    Parameters
    ----------
    model : estimator
        Fitted tree model.
    feature_names : list-like
        Encoded feature names.
    output_path : str or Path
        Output CSV path.
    top_n : int, optional
        Number of rows to save.

    Returns
    -------
    None
    """
    feature_importance = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance": model.feature_importances_,
        }
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False).head(top_n)
    feature_importance.to_csv(output_path, index=False)


def main():
    """
    Train the retained models and save a compact set of review artifacts.

    Returns
    -------
    None
    """
    project_root = Path(__file__).resolve().parents[2]
    raw_data_dir = project_root / "transferability" / "data" / "raw"
    output_dir = project_root / "transferability" / "outputs" / "final_models"
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

    positive_count = max(int(train_target.sum()), 1)
    negative_count = max(int((1 - train_target).sum()), 1)
    scale_pos_weight = negative_count / positive_count

    models = {
        "random_forest_benchmark": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost_retained": XGBClassifier(
            n_estimators=450,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=7,
            reg_lambda=2.5,
            reg_alpha=0.2,
            gamma=0.2,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        ),
    }

    metric_rows = []

    for model_name, model in models.items():
        model.fit(train_features, train_target)
        test_proba = model.predict_proba(test_features)[:, 1]
        metrics = compute_metrics(test_target, test_proba, threshold=0.5)
        metrics["model_name"] = model_name
        metrics["n_features_after_encoding"] = int(train_features.shape[1])
        metric_rows.append(metrics)

        prediction_table = build_prediction_table(test_dataset, test_proba, threshold=0.5)
        prediction_table.to_csv(output_dir / "{0}_predictions_2025.csv".format(model_name), index=False)

        save_feature_importance(
            model,
            train_features.columns,
            output_dir / "{0}_feature_importance.csv".format(model_name),
            top_n=60,
        )
        joblib.dump(model, output_dir / "{0}.joblib".format(model_name))

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
        ]
    ]
    comparison_table.to_csv(output_dir / "model_comparison_2025.csv", index=False)

    print(comparison_table.to_string(index=False))
    print()
    print("Saved outputs to:", output_dir)


if __name__ == "__main__":
    main()
