from src.config import FEATURES
from src.train import load_data


def test_data_loaded_and_encoded():
    df = load_data()
    assert df.shape[0] > 500
    assert set(FEATURES).issubset(df.columns)
    assert set(df["mainroad"].unique()) <= {0, 1}
    assert df.isna().sum().sum() == 0
