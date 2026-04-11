# coding: utf-8

"""
Transferability project
Data loading helpers
"""

from pathlib import Path

import pandas as pd


RAW_FILE_NAMES = {
    "players": "players.csv",
    "transfers": "transfers.csv",
    "performances": "performances.csv",
    "injuries": "injuries.csv",
    "national_performances": "national_performances.csv",
}


DATE_COLUMNS_BY_TABLE = {
    "players": [
        "date_of_birth",
        "contract_expires",
        "date_of_last_contract_extension",
        "contract_there_expires",
        "date_of_death",
    ],
    "transfers": ["transfer_date"],
    "performances": [],
    "injuries": ["from_date", "end_date"],
    "national_performances": ["date_debut"],
}


def load_csv_table(table_path, date_columns=None):
    """
    Load one CSV table and parse its date columns.

    Parameters
    ----------
    table_path : str or Path
        Path to the CSV file.
    date_columns : list of str, optional
        Columns to parse as datetimes.

    Returns
    -------
    pd.DataFrame
        Loaded table.
    """
    table_path = Path(table_path)
    dataframe = pd.read_csv(table_path)

    if date_columns is None:
        date_columns = []

    for date_column in date_columns:
        if date_column in dataframe.columns:
            dataframe[date_column] = pd.to_datetime(dataframe[date_column], errors="coerce")

    return dataframe


def load_raw_datasets(raw_data_folder):
    """
    Load every extracted CSV dataset used by the first test pipeline.

    Parameters
    ----------
    raw_data_folder : str or Path
        Directory containing the extracted CSV files.

    Returns
    -------
    dict
        Dictionary of pandas DataFrames keyed by dataset name.
    """
    raw_data_folder = Path(raw_data_folder)
    datasets = {}

    for dataset_name, file_name in RAW_FILE_NAMES.items():
        file_path = raw_data_folder / file_name
        date_columns = DATE_COLUMNS_BY_TABLE.get(dataset_name, [])
        datasets[dataset_name] = load_csv_table(file_path, date_columns=date_columns)

    return datasets


def describe_dataset(dataframe, dataset_name):
    """
    Build a compact summary for one dataset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to describe.
    dataset_name : str
        Human-readable dataset name.

    Returns
    -------
    dict
        Summary with shape and missing value information.
    """
    missing_share = dataframe.isna().mean().sort_values(ascending=False)

    return {
        "dataset_name": dataset_name,
        "n_rows": int(dataframe.shape[0]),
        "n_columns": int(dataframe.shape[1]),
        "top_missing": missing_share.head(10).to_dict(),
    }


def save_dataset(dataframe, output_path):
    """
    Save a dataset as CSV, creating its parent directory if needed.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset to save.
    output_path : str or Path
        Output CSV path.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
