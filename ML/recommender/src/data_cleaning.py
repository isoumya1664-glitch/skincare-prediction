import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def clean_column_name(column):
    """
    Standardize dataframe column names.
    """
    column = str(column).strip().lower()
    column = re.sub(r"\s+", " ", column)
    return column


def clean_dataframe(df):
    """
    Clean a skincare dataframe.
    """

    # Clean column names
    df.columns = [
        clean_column_name(col)
        for col in df.columns
    ]

    # Remove unwanted index columns
    unwanted_columns = [
        "unnamed: 0",
        "index"
    ]

    df = df.drop(
        columns=[
            col for col in unwanted_columns
            if col in df.columns
        ],
        errors="ignore"
    )

    # Clean string columns
    for column in df.columns:

        if df[column].dtype == "object":

            df[column] = (
                df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

    # Remove completely empty rows
    df = df.dropna(
        how="all"
    )

    # Remove duplicate products
    if "name" in df.columns:
        df = df.drop_duplicates(
            subset=["name"],
            keep="first"
        )

    return df


def load_dataset(filename):
    """
    Load a CSV from the processed folder.
    """

    file_path = PROCESSED_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    return clean_dataframe(df)


def save_clean_dataset(df, filename):
    """
    Save cleaned dataframe to processed folder.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = PROCESSED_DIR / filename

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    return output_path


if __name__ == "__main__":

    input_file = RAW_DIR / "result.csv"

    if input_file.exists():

        print("Loading raw dataset...")

        df = pd.read_csv(input_file)

        print(
            f"Original rows: {len(df)}"
        )

        df = clean_dataframe(df)

        print(
            f"Clean rows: {len(df)}"
        )

        output_file = save_clean_dataset(
            df,
            "cleaned_skincare.csv"
        )

        print(
            f"Saved cleaned dataset to:\n{output_file}"
        )

    else:

        print(
            f"Raw dataset not found:\n{input_file}"
        )