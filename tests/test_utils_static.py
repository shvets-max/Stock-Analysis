import unittest
from io import StringIO

import pandas as pd

from scoring.utils.static import load_and_normalize_percentages


class TestLoadAndNormalizePercentages(unittest.TestCase):
    def setUp(self):
        self.csv_data = StringIO(
            "Symbol,Column1,Column2,Column3\n"
            "AAPL,50%,30%,text\n"
            "MSFT,70%,40%,100\n"
            "GOOG,90%,60%,80%"
        )
        self.expected_normalized = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT", "GOOG"],
                "Column1": [0.5, 0.7, 0.9],
                "Column2": [0.3, 0.4, 0.6],
                "Column3": ["text", "100", "80%"],
            }
        ).set_index("Symbol")

    def test_normalize_percentage_columns(self):
        df = load_and_normalize_percentages(self.csv_data)
        pd.testing.assert_frame_equal(df, self.expected_normalized)


if __name__ == "__main__":
    unittest.main()
