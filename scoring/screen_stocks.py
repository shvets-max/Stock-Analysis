import os

import pandas as pd

from scoring.constants import SCREENER_DIR
from scoring.utils.static import load_and_normalize_percentages

data = load_and_normalize_percentages(
    os.path.join(SCREENER_DIR, "screener-stocks-2025-02-12.csv")
)

# Marcap
data = data[data["Market Cap"] > 5e9]

# Growth
data = data[(data["Rev. Growth 5Y"] > 0.02) | (pd.isna(data["Rev. Growth 5Y"]))]
data = data[(data["Rev. Growth 3Y"] > 0.02) | (pd.isna(data["Rev. Growth 3Y"]))]
data = data[(data["Rev. Growth"] > 0.02) | (pd.isna(data["Rev. Growth"]))]
# data = data[(data["Rev. Growth (Q)"] > 0) | (pd.isna(data["Rev. Growth (Q)"]))]

# data = data[(data["Rev Gr. Next 5Y"] > 0.02) | (pd.isna(data["Rev. Growth 5Y"]))]
# controversial because of long horizon.
data = data[(data["Rev Gr. Next Y"] > 0.02) | (pd.isna(data["Rev Gr. Next Y"]))]
# data = data[(data["Rev Gr. Next Q"] > -0.02) | (pd.isna(data["Rev. Growth"]))]

# data = data[(data["OpInc Growth (5Y)"] > 0.0) | (pd.isna(data["OpInc Growth (5Y)"]))]
# data = data[(data["OpInc Growth 3Y"] > 0.0) | (pd.isna(data["OpInc Growth 3Y"]))]
# data = data[(data["OpInc Growth"] > 0.0) | (pd.isna(data["OpInc Growth"]))]
# data = data[data["OpInc Growth (Q)"] > 0 | (pd.isna(data["OpInc Growth (Q)"]))]

# Note: Lacking OCF.

# Z-score
data = data[(data["Z-Score"] > 3) | (pd.isna(data["Z-Score"]))]

# ROA / ROI
data = data[(data["ROA"] > 0.03) | (pd.isna(data["ROA"]))]
data = data[(data["ROA (5Y)"] > 0.03) | (pd.isna(data["ROA (5Y)"]))]
data = data[(data["ROIC"] > 0.05) | (pd.isna(data["ROIC"]))]
data = data[(data["ROIC (5Y)"] > 0.05) | (pd.isna(data["ROIC (5Y)"]))]

data = data[(data["Oper. Margin"] > 0.08) | (pd.isna(data["Oper. Margin"]))]

# Analysts / Upside
data = data[(data["Analysts"] > 5) | (pd.isna(data["Analysts"]))]
data = data[(data["PT Upside (%)"] > 0.05) | (pd.isna(data["PT Upside (%)"]))]

print(data)
