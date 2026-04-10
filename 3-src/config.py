import os

# Check if running in Google Colab (Cloud) or Laptop (Local)
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# =========================
# ROOT PATH FIX (IMPORTANT)
# =========================

if IN_COLAB:
    # Use current working directory (repo folder)
    ROOT_PATH = os.getcwd()
else:
    # Local path (one level up from src)
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# =========================
# STANDARD PATHS
# =========================

DATA_RAW = os.path.join(ROOT_PATH, "data/raw")
DATA_PROCESSED = os.path.join(ROOT_PATH, "data/processed")
MODEL_CHECKPOINTS = os.path.join(ROOT_PATH, "models/checkpoints")
METADATA_DIR = os.path.join(ROOT_PATH, "data/metadata")

# Ensure directories exist
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)
os.makedirs(MODEL_CHECKPOINTS, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

# =========================
# MODEL CONFIG
# =========================

STRESS_THRESHOLD_SIGMA = 2.5
LOOKBACK_WINDOW = 10
