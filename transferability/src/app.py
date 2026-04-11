# coding: utf-8

"""
Transferability project
Simple Streamlit app for model comparison and player transfer probabilities
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def get_project_paths():
    """
    Resolve project paths for both api/src and simple src repository layouts.

    Returns
    -------
    dict
        Useful project paths.
    """
    current_file = Path(__file__).resolve()

    api_project_root = current_file.parents[2]
    api_outputs_dir = api_project_root / "transferability" / "outputs"
    api_raw_data_dir = api_project_root / "transferability" / "data" / "raw"

    if api_outputs_dir.exists():
        return {
            "project_root": api_project_root,
            "outputs_dir": api_outputs_dir,
            "raw_data_dir": api_raw_data_dir,
        }

    simple_project_root = current_file.parents[1]
    return {
        "project_root": simple_project_root,
        "outputs_dir": simple_project_root / "outputs",
        "raw_data_dir": simple_project_root / "data" / "raw",
    }


def load_csv_if_exists(file_path):
    """
    Load a CSV file if it exists.

    Parameters
    ----------
    file_path : str or Path
        CSV path.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame, or an empty DataFrame when the file is missing.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def load_app_data(paths):
    """
    Load model outputs and raw source tables used by the app.

    Parameters
    ----------
    paths : dict
        Project paths.

    Returns
    -------
    dict
        Loaded app data.
    """
    retained_dir = paths["outputs_dir"] / "final_models"
    raw_data_dir = paths["raw_data_dir"]

    data = {
        "comparison": load_csv_if_exists(retained_dir / "model_comparison_2025.csv"),
        "rf_predictions": load_csv_if_exists(retained_dir / "random_forest_benchmark_predictions_2025.csv"),
        "xgb_predictions": load_csv_if_exists(retained_dir / "xgboost_retained_predictions_2025.csv"),
        "rf_importance": load_csv_if_exists(retained_dir / "random_forest_benchmark_feature_importance.csv"),
        "xgb_importance": load_csv_if_exists(retained_dir / "xgboost_retained_feature_importance.csv"),
        "players": load_csv_if_exists(raw_data_dir / "players.csv"),
        "performances": load_csv_if_exists(raw_data_dir / "performances.csv"),
        "transfers": load_csv_if_exists(raw_data_dir / "transfers.csv"),
        "injuries": load_csv_if_exists(raw_data_dir / "injuries.csv"),
        "national_performances": load_csv_if_exists(raw_data_dir / "national_performances.csv"),
    }
    return data


def format_probability(probability):
    """
    Format a probability as a percentage.

    Parameters
    ----------
    probability : float
        Probability value.

    Returns
    -------
    str
        Formatted percentage.
    """
    return "{0:.1%}".format(float(probability))


def build_confusion_figure(prediction_table, title):
    """
    Build a confusion matrix figure.

    Parameters
    ----------
    prediction_table : pd.DataFrame
        Prediction table with target and predicted_label.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        Confusion matrix figure.
    """
    matrix = confusion_matrix(prediction_table["target"], prediction_table["predicted_label"])
    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Pred stay", "Pred move"])
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Actual stay", "Actual move"])

    for row_index in range(2):
        for column_index in range(2):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    figure.tight_layout()
    return figure


def build_curves_figure(rf_predictions, xgb_predictions):
    """
    Build ROC and Precision-Recall curves for both retained models.

    Parameters
    ----------
    rf_predictions : pd.DataFrame
        Random Forest predictions.
    xgb_predictions : pd.DataFrame
        XGBoost predictions.

    Returns
    -------
    matplotlib.figure.Figure
        Curve comparison figure.
    """
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    for model_name, prediction_table, color in [
        ("Random Forest", rf_predictions, "#6A994E"),
        ("XGBoost", xgb_predictions, "#1D3557"),
    ]:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            prediction_table["target"],
            prediction_table["transfer_probability"],
        )
        precision, recall, _ = precision_recall_curve(
            prediction_table["target"],
            prediction_table["transfer_probability"],
        )
        axes[0].plot(false_positive_rate, true_positive_rate, label=model_name, color=color, linewidth=2)
        axes[1].plot(recall, precision, label=model_name, color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()

    axes[1].set_title("Precision-Recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()

    figure.tight_layout()
    return figure


def build_feature_importance_figure(rf_importance, xgb_importance):
    """
    Build a side-by-side feature importance figure.

    Parameters
    ----------
    rf_importance : pd.DataFrame
        Random Forest feature importances.
    xgb_importance : pd.DataFrame
        XGBoost feature importances.

    Returns
    -------
    matplotlib.figure.Figure
        Feature importance figure.
    """
    figure, axes = plt.subplots(1, 2, figsize=(13, 7))

    rf_plot = rf_importance.head(15).sort_values("importance")
    xgb_plot = xgb_importance.head(15).sort_values("importance")

    axes[0].barh(rf_plot["feature_name"], rf_plot["importance"], color="#6A994E")
    axes[0].set_title("Random Forest")
    axes[0].set_xlabel("Importance")

    axes[1].barh(xgb_plot["feature_name"], xgb_plot["importance"], color="#1D3557")
    axes[1].set_title("XGBoost retained")
    axes[1].set_xlabel("Importance")

    figure.tight_layout()
    return figure


def render_player_avatar(player_name):
    """
    Render a simple placeholder avatar for a player.

    Parameters
    ----------
    player_name : str
        Player name.

    Returns
    -------
    None
    """
    name_parts = str(player_name).split()
    initials = "".join(part[0].upper() for part in name_parts[:2] if len(part) > 0)
    if initials == "":
        initials = "?"

    st.markdown(
        """
        <div style="
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: linear-gradient(135deg, #1D3557, #457B9D);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 44px;
            font-weight: 700;
            margin-bottom: 12px;">
            {0}
        </div>
        """.format(initials),
        unsafe_allow_html=True,
    )


def get_player_profile(players, player_id):
    """
    Return one player profile row.

    Parameters
    ----------
    players : pd.DataFrame
        Raw players table.
    player_id : int
        Player id.

    Returns
    -------
    pd.Series
        Player profile.
    """
    if players.empty:
        return pd.Series(dtype=object)

    player_rows = players.loc[players["player_id"] == player_id]
    if player_rows.empty:
        return pd.Series(dtype=object)
    return player_rows.iloc[0]


def get_player_performances(performances, player_id, season_name):
    """
    Aggregate player performance by competition for one season.

    Parameters
    ----------
    performances : pd.DataFrame
        Raw performances table.
    player_id : int
        Player id.
    season_name : str
        Season name.

    Returns
    -------
    pd.DataFrame
        Performance table.
    """
    if performances.empty:
        return pd.DataFrame()

    player_performances = performances.loc[
        (performances["player_id"] == player_id) & (performances["season_name"] == season_name)
    ].copy()
    if player_performances.empty:
        return pd.DataFrame()

    performance_table = (
        player_performances.groupby(["competition_name", "team_name"], as_index=False)
        .agg(
            matches_in_squad=("nb_in_group", "sum"),
            matches_played=("nb_on_pitch", "sum"),
            minutes_played=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
        )
        .sort_values("minutes_played", ascending=False)
    )
    return performance_table


def render_model_comparison_page(data):
    """
    Render the model comparison page.

    Parameters
    ----------
    data : dict
        Loaded app data.

    Returns
    -------
    None
    """
    st.title("Model comparison")
    st.caption("2025 temporal holdout, same train/test split for both retained models.")

    comparison = data["comparison"]
    rf_predictions = data["rf_predictions"]
    xgb_predictions = data["xgb_predictions"]

    if comparison.empty or rf_predictions.empty or xgb_predictions.empty:
        st.error("Missing model outputs. Run `python src/run_retained_models.py` first.")
        return

    st.subheader("Main metrics")
    st.dataframe(comparison, use_container_width=True)

    xgb_row = comparison.loc[comparison["model_name"] == "xgboost_retained"].iloc[0]
    rf_row = comparison.loc[comparison["model_name"] == "random_forest_benchmark"].iloc[0]

    metric_columns = st.columns(4)
    metric_columns[0].metric("XGBoost ROC AUC", "{0:.3f}".format(xgb_row["roc_auc"]))
    metric_columns[1].metric("XGBoost AP", "{0:.3f}".format(xgb_row["average_precision"]))
    metric_columns[2].metric("XGBoost Precision", "{0:.3f}".format(xgb_row["precision"]))
    metric_columns[3].metric("XGBoost Recall", "{0:.3f}".format(xgb_row["recall"]))

    st.markdown(
        "The Random Forest is kept as a non-linear benchmark. "
        "The retained XGBoost is the model used for transfer probabilities because it gives the better ranking quality."
    )

    st.subheader("ROC and Precision-Recall curves")
    st.pyplot(build_curves_figure(rf_predictions, xgb_predictions))

    st.subheader("Confusion matrices at threshold 0.50")
    left_column, right_column = st.columns(2)
    left_column.pyplot(build_confusion_figure(rf_predictions, "Random Forest"))
    right_column.pyplot(build_confusion_figure(xgb_predictions, "XGBoost retained"))

    st.subheader("Feature importance")
    if data["rf_importance"].empty or data["xgb_importance"].empty:
        st.warning("Feature importance files are missing.")
    else:
        st.pyplot(build_feature_importance_figure(data["rf_importance"], data["xgb_importance"]))


def render_player_probability_page(data):
    """
    Render the player transfer probability page.

    Parameters
    ----------
    data : dict
        Loaded app data.

    Returns
    -------
    None
    """
    st.title("Player transfer probability")
    st.caption("Probability shown by the retained XGBoost model on the 2025 holdout.")

    xgb_predictions = data["xgb_predictions"]
    if xgb_predictions.empty:
        st.error("Missing XGBoost predictions. Run `python src/run_retained_models.py` first.")
        return

    search_text = st.text_input("Search player", "")
    filtered_predictions = xgb_predictions.copy()
    if search_text.strip() != "":
        filtered_predictions = filtered_predictions.loc[
            filtered_predictions["player_name"].astype(str).str.contains(search_text, case=False, na=False)
        ].copy()

    filtered_predictions = filtered_predictions.sort_values("transfer_probability", ascending=False)
    player_options = (
        filtered_predictions["player_name"]
        + " | id="
        + filtered_predictions["player_id"].astype(str)
        + " | p="
        + filtered_predictions["transfer_probability"].map(format_probability)
    ).tolist()

    if len(player_options) == 0:
        st.warning("No player found.")
        return

    selected_option = st.selectbox("Select a player", player_options)
    selected_player_id = int(selected_option.split("id=")[1].split(" | ")[0])
    selected_prediction = xgb_predictions.loc[xgb_predictions["player_id"] == selected_player_id].iloc[0]
    player_profile = get_player_profile(data["players"], selected_player_id)

    left_column, right_column = st.columns([1, 2])

    with left_column:
        render_player_avatar(selected_prediction["player_name"])
        st.caption("No player image URL is available in the raw source files, so the app uses a neutral placeholder.")
        st.metric("Transfer probability", format_probability(selected_prediction["transfer_probability"]))
        st.progress(float(selected_prediction["transfer_probability"]))
        st.write("Model decision at 0.50:", int(selected_prediction["predicted_label"]))
        st.write("Observed target:", int(selected_prediction["target"]))

    with right_column:
        st.subheader(str(selected_prediction["player_name"]))

        profile_rows = []
        for column_name in [
            "main_position",
            "position",
            "foot",
            "citizenship",
            "second_citizenship",
            "current_club_name",
            "current_league_name",
            "market_value",
            "contract_expires",
            "is_on_loan",
        ]:
            if column_name in player_profile.index:
                profile_rows.append({"field": column_name, "value": player_profile[column_name]})

        st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, hide_index=True)

    st.subheader("Season performance")
    performance_table = get_player_performances(
        data["performances"],
        selected_player_id,
        selected_prediction["season_name"],
    )
    if performance_table.empty:
        st.info("No raw performance rows found for this player and season.")
    else:
        st.dataframe(performance_table, use_container_width=True, hide_index=True)

        figure, axis = plt.subplots(figsize=(10, 4))
        plot_table = performance_table.head(8).sort_values("minutes_played")
        axis.barh(plot_table["competition_name"], plot_table["minutes_played"], color="#1D3557")
        axis.set_title("Minutes by competition")
        axis.set_xlabel("Minutes")
        figure.tight_layout()
        st.pyplot(figure)

    st.subheader("Transfer history")
    transfers = data["transfers"]
    if transfers.empty:
        st.info("No transfer source file found.")
    else:
        player_transfers = transfers.loc[transfers["player_id"] == selected_player_id].copy()
        if player_transfers.empty:
            st.info("No transfer history found for this player.")
        else:
            transfer_columns = [
                "season_name",
                "transfer_date",
                "from_team_name",
                "to_team_name",
                "transfer_type",
                "value_at_transfer",
                "transfer_fee",
            ]
            st.dataframe(
                player_transfers[transfer_columns].sort_values("transfer_date", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Probability context")
    figure, axis = plt.subplots(figsize=(10, 4))
    xgb_predictions["transfer_probability"].hist(bins=50, ax=axis, color="#A8DADC")
    axis.axvline(
        float(selected_prediction["transfer_probability"]),
        color="#E63946",
        linewidth=3,
        label="selected player",
    )
    axis.set_title("Where this player sits in the XGBoost probability distribution")
    axis.set_xlabel("Transfer probability")
    axis.set_ylabel("Players")
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure)


def main():
    """
    Run the Streamlit app.

    Returns
    -------
    None
    """
    st.set_page_config(page_title="Transferability", layout="wide")
    paths = get_project_paths()
    data = load_app_data(paths)

    st.sidebar.title("Transferability")
    st.sidebar.write("Outputs:", str(paths["outputs_dir"]))
    page_name = st.sidebar.radio(
        "Page",
        ["Model comparison", "Player probability"],
    )

    if page_name == "Model comparison":
        render_model_comparison_page(data)
    else:
        render_player_probability_page(data)


if __name__ == "__main__":
    main()
