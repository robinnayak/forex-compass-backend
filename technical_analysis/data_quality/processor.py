import pandas as pd
import numpy as np
from typing import List, Optional, Dict

class DataQualityProcessor:
    """
    Utility class for cleaning and improving the quality of financial data.
    """
    @staticmethod
    def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
        print("Initial DataFrame shape:", df.shape)
        print("Initial columns:", df.columns.tolist())
        # Remove leading/trailing spaces from column names
        df = df.rename(columns=lambda x: x.strip())
        print("Columns after strip:", df.columns.tolist())
        # Drop duplicate rows (optional)
        if df.duplicated().any():
            before = df.shape[0]
            df = df.drop_duplicates()
            after = df.shape[0]
            print(f"Dropped {before - after} duplicate rows.")
        # Drop rows with all NaN values
        before = df.shape[0]
        df = df.dropna(how='all')
        after = df.shape[0]
        print(f"Dropped {before - after} rows with all NaN values.")
        # Always return the cleaned DataFrame
        return df
    @staticmethod
    def refine_data(
        df: pd.DataFrame,
        columns_to_clean: Optional[List[str]] = None,
        missing_method: str = 'ffill',
        remove_duplicates: bool = True,
        outlier_method: Optional[str] = 'zscore',
        outlier_threshold: float = 3.0,
        outlier_columns: Optional[List[str]] = None,
        min_length: int = 10,
        fill_volume_with_zero: bool = True,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Dynamically refine financial data with configurable options.
        Args:
            df: Input DataFrame
            columns_to_clean: List of columns to clean (default: ['Open','High','Low','Close'])
            missing_method: How to fill missing values ('ffill', 'bfill', 'mean', 'median', 'drop')
            remove_duplicates: Whether to drop duplicate rows
            outlier_method: Method for outlier removal ('zscore' or None)
            outlier_threshold: Z-score threshold for outlier removal
            outlier_columns: Columns to check for outliers (default: ['Open','High','Low','Close'])
            min_length: Minimum length required for rolling/window ops
            fill_volume_with_zero: Fill missing volume with zero
            verbose: Print debug info
        Returns:
            Refined DataFrame
        """
        df = df.copy()
        if verbose:
            print("Initial shape:", df.shape)
        # Strip column names
        df = df.rename(columns=lambda x: x.strip())
        if columns_to_clean is None:
            columns_to_clean = ['Open', 'High', 'Low', 'Close']
        if outlier_columns is None:
            outlier_columns = columns_to_clean
        # Remove duplicates
        if remove_duplicates:
            before = df.shape[0]
            df = df.drop_duplicates()
            after = df.shape[0]
            if verbose:
                print(f"Dropped {before - after} duplicate rows.")
        # Drop rows with all NaN
        before = df.shape[0]
        df = df.dropna(how='all')
        after = df.shape[0]
        if verbose:
            print(f"Dropped {before - after} rows with all NaN values.")
        # Clean price columns
        for col in columns_to_clean:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                nan_count = df[col].isna().sum()
                if verbose:
                    print(f"Column '{col}' NaN before fill: {nan_count}")
                if missing_method == 'ffill':
                    df[col] = df[col].ffill().bfill()
                elif missing_method == 'bfill':
                    df[col] = df[col].bfill().ffill()
                elif missing_method == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif missing_method == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif missing_method == 'drop':
                    df = df.dropna(subset=[col])
                nan_count_after = df[col].isna().sum()
                if verbose:
                    print(f"Column '{col}' NaN after fill: {nan_count_after}")
        # Fill volume
        if fill_volume_with_zero and 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
            if verbose:
                print(f"Column 'volume' NaN after fill: {df['volume'].isna().sum()}")
        # Remove outliers
        if outlier_method == 'zscore':
            for col in outlier_columns:
                if col in df.columns:
                    if len(df) >= min_length:
                        z = np.abs((df[col] - df[col].mean()) / df[col].std())
                        before = df.shape[0]
                        df = df[z < outlier_threshold]
                        after = df.shape[0]
                        if verbose:
                            print(f"Column '{col}': Removed {before - after} outlier rows.")
                    else:
                        if verbose:
                            print(f"Column '{col}': Not enough data for outlier removal (min_length={min_length})")
        if verbose:
            print("Final shape:", df.shape)
        return df


def clean_indicator_data(indicator_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
    """
    Clean indicator data by applying common preprocessing steps.

    Args:
        indicator_data: Dictionary of indicator data series

    Returns:
        Dictionary of cleaned indicator data series
    """
    cleaned_data = {}
    for key, series in indicator_data.items():
        # Apply basic cleaning: fill NaNs, remove outliers, etc.
        series = series.ffill().bfill()
        z = np.abs((series - series.mean()) / series.std())
        series[z > 3] = np.nan  # Remove outliers
        cleaned_data[key] = series
    return cleaned_data