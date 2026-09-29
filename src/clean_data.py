import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import re

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "messy_sales_dataset.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "cleaned_sales_dataset.parquet"


def clean_dates(value):
    """Parses messy date strings into datetime. Returns NaT for invalid."""
    if pd.isnull(value):
        return pd.NaT

    value = str(value).strip().lower()

    if value in ('n/a', 'null', 'nan', 'date'):
        return pd.NaT

    # remove ordinal suffix (1st, 2nd, 3rd, 4th...)
    value = re.sub(r'(st|nd|rd|th)', '', value)

    # remove unwanted numeric / money values
    if '$' in value or re.match(r'^\d+(\.\d+)?$', value):
        return pd.NaT

    formats = [
        "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y",
        "%m-%d-%Y", "%m/%d/%Y", "%d-%m-%y", "%m/%d/%y",
        "%d-%b-%Y", "%d-%B-%Y", "%b-%d-%Y", "%d %b %Y",
        "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%Y%m%d"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return pd.NaT


def remove_iqr_outliers(series: pd.Series) -> pd.Series:
    """Replaces values outside IQR bounds with NaN."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return series.where((series >= lower) & (series <= upper), other=np.nan)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Master cleaning function. Returns cleaned DataFrame."""
    
    
    df = df.copy()
    df.columns = df.columns.str.lower()
    df = df.drop_duplicates()

    # Clean and standardize date column
    # Convert valid values to date, invalid entries to NaT
    df['date'] = df['date'].apply(clean_dates)

    # category column contains upper, lower, mix case
    # standardized the category column
    df['category'] = df['category'].fillna(df['category'].mode()[0])
    df['category'] = df['category'].str.title().str.strip()

    # region contains upper, lower and mix case
    # standardized it
    df['region'] = df['region'].fillna(df['region'].mode()[0])
    df['region'] = df['region'].str.title().str.strip()

    # standardized channel - contains upper, lower, mix case
    df['channel'] = df['channel'].fillna(df['channel'].mode()[0])
    df['channel'] = df['channel'].str.title().str.strip()

    # payment_method has upper, lower, mix case - standardized it
    # fill nulls with column own mode
    df['payment_method'] = df['payment_method'].fillna(df['payment_method'].mode()[0])
    df['payment_method'] = df['payment_method'].str.title().str.strip()
    # PayPal is a brand name - title case gives 'Paypal' which is wrong
    df['payment_method'] = df['payment_method'].str.replace(
        'Paypal', 'PayPal', regex=False
    )

    # units_sold was stored as mixed text (e.g. "15 units", "20pcs")
    # extracting only the leading integer; non-numeric entries become NaN
    df['units_sold'] = pd.to_numeric(
        df['units_sold'].str.extract(r'(\d+)')[0], errors='coerce'
    ).astype('float64')

    # Converting to float64 NOT Int64
    # Float64 handles NaN natively, Int64 causes issues with np.where
    df['units_sold'] = remove_iqr_outliers(df['units_sold'])

    # After fillna there are no NaN values left so converting to int
    df['units_sold'] = df['units_sold'].fillna(
        df['units_sold'].median()
    ).astype(int)

    # convert to numeric
    df['sales_amount'] = pd.to_numeric(
        df['sales_amount'], errors='coerce'
    ).astype('float64')
    df['sales_amount'] = remove_iqr_outliers(df['sales_amount'])
    df['sales_amount'] = df['sales_amount'].fillna(df['sales_amount'].median())

    # 54 values are missing in discount
    # missing means no discount was applied so filling with zero
    df['discount_pct'] = df['discount_pct'].fillna(0)

    # customer_rating values out of range > 5.0
    # masking to NaN - cannot fabricate an opinion
    df.loc[
        (df['customer_rating'] < 1.0) | (df['customer_rating'] > 5.0),
        'customer_rating'
    ] = np.nan

    # Fill with median after outlier removal
    df['customer_rating'] = df['customer_rating'].fillna(
        df['customer_rating'].median()
    )

    return df


if __name__ == "__main__":
    df_raw = pd.read_csv(RAW_PATH)
    initial_rows = len(df_raw)

    df = clean_dataframe(df_raw)

    # parquet saves the original dtypes
    df.to_parquet(PROCESSED_PATH)

    print("\n--- CLEANING VERIFICATION ---")
    print(f"Shape: {df.shape}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nSample:\n{df.head()}")
    print(f"\nRows before cleaning: {initial_rows}")
    print(f"Rows after cleaning:  {len(df)}")
    print(f"Rows removed:         {initial_rows - len(df)}")