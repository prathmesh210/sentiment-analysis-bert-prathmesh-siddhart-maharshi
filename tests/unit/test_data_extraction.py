import pytest
import pandas as pd
import os
from src.data_extraction import load_data

def test_loads_sample_csv(tmp_path):
    # Create dummy CSV
    sample = tmp_path / "sample.csv"
    sample.write_text("text,label\nI love this,positive\nI hate this,negative")
    df = load_data(str(sample))
    assert isinstance(df, pd.DataFrame)
    assert "text" in df.columns and "label" in df.columns
    assert len(df) == 2

def test_wrong_path():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent.csv")

def test_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("review\nnice movie")
    with pytest.raises(ValueError):
        load_data(str(bad))
