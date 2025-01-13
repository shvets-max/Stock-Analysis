import os

# Directories
WORKDIR = "scoring"
METRICS_DIR = os.path.join(WORKDIR, "metrics_data")
SCREENER_DIR = os.path.join(WORKDIR, "screener_data")

ALLOWED_GROUPS = ("Sector", "Industry", "Country")

# Quantiles
Q_LOW = 0.05
Q_HIGH = 0.95

# Minimal number of peers in a group
MIN_GROUP = 10