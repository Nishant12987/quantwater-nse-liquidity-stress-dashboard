import os

# Check if running in Google Colab (Cloud) or Laptop (Local)
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Set Root Path based on environment
if IN_COLAB:
    # Path inside Google Colab after mounting drive
    ROOT_PATH = "/content/drive/MyDrive/nse_liquidity_dashboard/"
else:
    # Local path (one level up from src)
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Standardized directory mapping for Nishant & Harsh
DATA_RAW = os.path.join(ROOT_PATH, "data/raw")
DATA_PROCESSED = os.path.join(ROOT_PATH, "data/processed")
MODEL_CHECKPOINTS = os.path.join(ROOT_PATH, "models/checkpoints")
METADATA_DIR = os.path.join(ROOT_PATH, "data/metadata")

# Configuration for the Ensemble Engine
STRESS_THRESHOLD_SIGMA = 2.5  # Harsh's Week 3 logic
LOOKBACK_WINDOW = 10          # Nishant's Week 4 LSTM window