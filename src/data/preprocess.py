import pandas as pd
import numpy as np

def impute_missing_numeric(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    for col in columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def apply_log_transformation(df, target_col='SalePrice'):
    df['LogSalePrice'] = np.log1p(df[target_col])
    return df


def remove_large_outliers(df, column='GrLivArea', threshold=4500):
    return df[df[column] <= threshold].copy()