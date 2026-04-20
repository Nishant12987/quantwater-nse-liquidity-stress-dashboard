import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report
from xgboost import XGBClassifier


# -------------------------------------------------
# PATHS
# -------------------------------------------------

BASE = Path(os.getcwd())

FEATURES_DIR = BASE / "data/processed"
MODEL_DIR = BASE / "models" / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

amihud = pd.read_csv(FEATURES_DIR / "amihud.csv")
roll = pd.read_csv(FEATURES_DIR / "roll_spread.csv")
commonality = pd.read_csv(FEATURES_DIR / "liquidity_commonality.csv")
market = pd.read_csv(FEATURES_DIR / "market_amihud.csv", parse_dates=["DATE"])
labels = pd.read_csv(FEATURES_DIR / "stress_labels.csv", parse_dates=["DATE"])


df = (
    amihud
    .merge(roll, on="SYMBOL", how="left")
    .merge(commonality, on="SYMBOL", how="left")
)

df["DATE"] = pd.to_datetime(df["DATE"])
df = df.merge(market, on="DATE", how="left")
df = df.merge(labels, on="DATE", how="inner")


# -------------------------------------------------
# FEATURE ENGINEERING (PIT SAFE)
# -------------------------------------------------

rolling_mean = df["MARKET_AMIHUD"].rolling(60).mean().shift(1)
rolling_std = df["MARKET_AMIHUD"].rolling(60).std().shift(1)

df["MARKET_AMIHUD_Z"] = (
    (df["MARKET_AMIHUD"] - rolling_mean) / rolling_std
)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()


FEATURES = [
    "AMIHUD",
    "ROLL_SPREAD",
    "LIQUIDITY_COMMONALITY",
    "MARKET_AMIHUD_Z"
]

X = df[FEATURES]
y = df["STRESS_LABEL"]


# -------------------------------------------------
# TRAIN / TEST SPLIT
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=False
)


# -------------------------------------------------
# HANDLE CLASS IMBALANCE
# -------------------------------------------------

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


# -------------------------------------------------
# MODEL
# -------------------------------------------------

model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)


# -------------------------------------------------
# EVALUATION
# -------------------------------------------------

y_prob = model.predict_proba(X_test)[:, 1]

threshold = np.quantile(y_prob, 0.97)
y_pred = (y_prob > threshold).astype(int)

print("Threshold:", threshold)
print("Final Precision:", precision_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

joblib.dump(model, MODEL_DIR / "xgboost_stress_gatekeeper.pkl")
joblib.dump(threshold, MODEL_DIR / "xgboost_threshold.pkl")


# -------------------------------------------------
# GENERATE FULL PREDICTIONS (FIXED)
# -------------------------------------------------

df["PRED_PROBA"] = model.predict_proba(X)[:, 1]
df["PRED_STRESS"] = (df["PRED_PROBA"] > threshold).astype(int)

# ✅ KEEP FEATURES + PREDICTIONS (CRITICAL FIX)
pred_df = df[[
    "DATE",
    "SYMBOL",
    "AMIHUD",
    "ROLL_SPREAD",
    "LIQUIDITY_COMMONALITY",
    "MARKET_AMIHUD_Z",
    "PRED_PROBA",
    "PRED_STRESS"
]].copy()

pred_df = pred_df.sort_values("DATE")

output_path = MODEL_DIR / "xgboost_full_predictions.csv.gz"

pred_df.to_csv(
    output_path,
    index=False,
    compression="gzip"
)

print("Model + threshold + compressed predictions saved at:", output_path)
