# tests/test_preprocess.py
import pandas as pd
import pytest
from src.data.preprocess import (
    impute_missing_numeric,
    apply_log_transformation,
    remove_large_outliers
)

@pytest.fixture
def sample_df():
    data = {
        'GrLivArea': [1000, 2000, 5000, 1500],
        'SalePrice': [200000, 300000, 800000, 250000],
        'OverallQual': [5, 7, 10, 6]
    }
    return pd.DataFrame(data)

def test_impute_missing_numeric(sample_house_data):
    df = impute_missing_numeric(sample_house_data.copy(), ['LotFrontage'])
    assert df['LotFrontage'].isna().sum() == 0


def test_apply_log_transformation(sample_house_data):
    df = apply_log_transformation(sample_house_data.copy())
    assert 'LogSalePrice' in df.columns
    assert df['LogSalePrice'].skew() < df['SalePrice'].skew()


def test_remove_large_outliers(sample_house_data):
    df_with_outlier = pd.concat([
        sample_house_data,
        pd.DataFrame({'GrLivArea': [5000], 'SalePrice': [1000000]})
    ], ignore_index=True)
    
    cleaned = remove_large_outliers(df_with_outlier)
    assert len(cleaned) == len(sample_house_data) + 0