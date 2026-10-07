import pandas as pd
import os
from config import BASE_DIR, CAT_COLS, CURRENT_YEAR, FILE_PATH, GENDER_FILL
from preprocessing import (
    calculate_age,
    convert_to_category,
    convert_to_int,
    fill_gender_na,
    fill_missing_values,
    read_files,
)

OUTPUT_DATA_PATH = BASE_DIR / "cleaned_bikeshare.csv"


def run_pipeline():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Raw dataset not found at '{FILE_PATH}'")

    print(f"1. Loading raw dataset from '{FILE_PATH}'...")
    df = read_files(str(FILE_PATH))
    print(f"   Initial records loaded: {len(df):,}")

    print("2. Imputing missing values...")
    df = fill_missing_values(df)
    df = fill_gender_na(df, fill_value=GENDER_FILL)

    print("3. Engineering features and casting types...")
    df = calculate_age(df, reference_year=CURRENT_YEAR)
    df = convert_to_int(df, "age")
    df = convert_to_category(df, [col for col in CAT_COLS if col in df.columns])

    # Standardize trip duration to minutes and filter extreme outliers
    df["duration_min"] = (df["duration_sec"] / 60).round(2)
    df = df[(df["age"] <= 80) & (df["duration_min"] <= 120)]

    print(f"4. Saving cleaned dataset to '{OUTPUT_DATA_PATH}'...")
    df.to_csv(OUTPUT_DATA_PATH, index=False)
    print(
        f"Pipeline executed successfully. Cleaned records: {len(df):,} ({df.shape[1]} columns)"
    )


if __name__ == "__main__":
    run_pipeline()