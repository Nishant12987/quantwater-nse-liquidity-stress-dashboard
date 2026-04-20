import os
from pathlib import Path

print("Starting Quant Pipeline...")

# Run XGBoost
print("Running XGBoost...")
os.system("python 3-src/3-models/1-xgboost_engine.py")

# Run LSTM
print("Running LSTM...")
os.system("python 3-src/3-models/2-lstm_architecture.py")

print("Pipeline completed successfully.")
