# tests/test_preprocess.py
import pandas as pd
import pytest
from src.data.preprocess import fill_missing_values, remove_outliers  # adapte les imports

@pytest.fixture
def sample_df():
    data = {
        'GrLivArea': [1000, 2000, 5000, 1500],
        'SalePrice': [200000, 300000, 800000, 250000],
        'OverallQual': [5, 7, 10, 6]
    }
    return pd.DataFrame(data)

def test_remove_outliers(sample_df):
    cleaned = remove_outliers(sample_df)
    assert len(cleaned) == 3  # suppose que 5000 est supprimé
    assert cleaned['GrLivArea'].max() < 4000
