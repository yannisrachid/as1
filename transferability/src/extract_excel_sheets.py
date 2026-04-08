# coding: utf-8

"""
Transferability project
Excel extraction helpers
"""

from pathlib import Path
import re

import pandas as pd


def normalize_sheet_name(sheet_name):
    """
    Convert an Excel sheet name into a filesystem-friendly file name.

    Parameters
    ----------
    sheet_name : str
        Original sheet name from the Excel file.

    Returns
    -------
    str
        Lowercase file name with underscores.
    """
    normalized_name = sheet_name.strip().lower()
    normalized_name = re.sub(r"[^a-z0-9]+", "_", normalized_name)
    normalized_name = normalized_name.strip("_")
    return normalized_name


def export_excel_sheets(excel_path, output_folder):
    """
    Export every sheet of the source Excel file into a CSV file.

    Parameters
    ----------
    excel_path : str or Path
        Path to the source Excel workbook.
    output_folder : str or Path
        Directory where the extracted CSV files will be saved.

    Returns
    -------
    list of dict
        Export summary with sheet names, output paths and shapes.
    """
    excel_path = Path(excel_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    sheet_mapping = []
    workbook = pd.read_excel(excel_path, sheet_name=None)

    for sheet_name, sheet_data in workbook.items():
        file_name = normalize_sheet_name(sheet_name) + ".csv"
        output_path = output_folder / file_name
        sheet_data.to_csv(output_path, index=False)
        sheet_mapping.append(
            {
                "sheet_name": sheet_name,
                "output_path": str(output_path),
                "n_rows": int(sheet_data.shape[0]),
                "n_columns": int(sheet_data.shape[1]),
            }
        )

    return sheet_mapping


def print_export_summary(export_summary):
    """
    Print a short export summary for the extracted sheets.

    Parameters
    ----------
    export_summary : list of dict
        Output of ``export_excel_sheets``.

    Returns
    -------
    None
    """
    print("Excel extraction summary")

    for summary_line in export_summary:
        print(
            "- {sheet_name}: {n_rows} rows x {n_columns} columns -> {output_path}".format(
                **summary_line
            )
        )


def main():
    """
    Export the Transfermarkt workbook into individual CSV files.

    Returns
    -------
    None
    """
    project_root = Path(__file__).resolve().parents[1]
    excel_path = project_root / "data" / "transferability_tmkt.xlsx"
    output_folder = project_root / "data" / "raw"

    export_summary = export_excel_sheets(excel_path, output_folder)
    print_export_summary(export_summary)


if __name__ == "__main__":
    main()
