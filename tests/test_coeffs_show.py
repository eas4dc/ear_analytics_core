import pytest
import os
import pandas as pd
import numpy as np

from ear_analytics_core.coeffs_show import cpu_coeffs

BINARY_FILE = "data/coeffs.basic_perc.amd_genoa"


@pytest.mark.skipif(not os.path.exists(BINARY_FILE), reason="Binary file not found")
def test_cpu_coeffs_loads_data():
    df = cpu_coeffs(BINARY_FILE)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert  df.shape == (36, 9)


@pytest.mark.skipif(not os.path.exists(BINARY_FILE), reason="Binary file not found")
def test_cpu_coeffs_structure():
    df = cpu_coeffs(BINARY_FILE)

    # Frequencies and Available columns must exit and be int64
    for col in ['FROM', 'TO', 'Available']:
        assert col in df.columns
        assert df[col].dtype == np.int64

    # Frequency values
    expected_from = [2400000, 2300000, 2200000, 2100000, 2000000, 1900000]
    assert sorted(df['FROM'].unique().tolist(), reverse=True) == expected_from
    assert sorted(df['TO'].unique().tolist(), reverse=True) == expected_from

    # Coefficients columns must exist and be float64
    for col in ['A', 'B', 'C', 'D', 'E', 'F']:
        assert col in df.columns
        assert df[col].dtype == np.float64


@pytest.mark.skipif(not os.path.exists(BINARY_FILE), reason="Binary file not found")
def test_cpu_coeffs_tocsv():

    df = cpu_coeffs(str(BINARY_FILE), save_csv=True)

    expected_csv = str(BINARY_FILE) + ".csv"
    assert os.path.exists(expected_csv)
