import numpy as np
import pandas as pd
import os
from scoring.constants import SCREENER_DIR
from scoring.utils import load_and_normalize_percentages

data = pd.read_csv(os.path.join(SCREENER_DIR, "screener-stocks-fundamentals-hist-growth-2024-11-29.csv"))
percentage_cols = data.map(lambda x: "%" in x if isinstance(x, str) else False).any(axis=0)
percentage_colnames = data.columns[percentage_cols].tolist()

data = load_and_normalize_percentages(os.path.join(SCREENER_DIR, "screener-stocks-fundamentals-hist-growth-2024-11-29.csv"), norm_columns=percentage_colnames)

# Marcap
data = data[data["Market Cap"] > 5e9]

# Growth
data = data[(data["Rev. Growth 5Y"] > 0.02) | (pd.isna(data["Rev. Growth 5Y"]))]
data = data[(data["Rev. Growth 3Y"] > 0.02) | (pd.isna(data["Rev. Growth 3Y"]))]
data = data[(data["Rev. Growth"] > 0.02) | (pd.isna(data["Rev. Growth"]))]
data = data[(data["Rev. Growth (Q)"] > 0) | (pd.isna(data["Rev. Growth (Q)"]))]

# data = data[(data["Rev Gr. Next 5Y"] > 0.02) | (pd.isna(data["Rev. Growth 5Y"]))]  # controversial because of long horizon.
data = data[(data["Rev Gr. Next Y"] > 0.02) | (pd.isna(data["Rev. Growth"]))]
data = data[(data["Rev Gr. Next Q"] > -0.02) | (pd.isna(data["Rev. Growth"]))]

data = data[(data["OpInc Growth 5Y"] > 0.02) | (pd.isna(data["OpInc Growth 5Y"]))]
data = data[(data["OpInc Growth 3Y"] > 0.02) | (pd.isna(data["OpInc Growth 3Y"]))]
# data = data[data["OpInc Growth"] > 0.03 | (pd.isna(data["OpInc Growth 3Y"]))]
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
data = data[(data["Analyst Count"] > 5) | (pd.isna(data["Analyst Count"]))]
data = data[(data["Price Target Upside (%)"] > 0.05) | (pd.isna(data["Price Target Upside (%)"]))]

print("Done.")