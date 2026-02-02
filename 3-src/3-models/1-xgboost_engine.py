import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report
from xgboost import XGBClassifier


BASE = Path("/content/data/NSE_Liquidity_Project")
FEATURES_DIR = BASE / "features"
MODEL_DIR = BASE / "models" / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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


df["MARKET_AMIHUD_Z"] = (
    (df["MARKET_AMIHUD"] - df["MARKET_AMIHUD"].rolling(60).mean())
    / df["MARKET_AMIHUD"].rolling(60).std()
)

df = df.dropna()


FEATURES = [
    "AMIHUD",
    "ROLL_SPREAD",
    "LIQUIDITY_COMMONALITY",
    "MARKET_AMIHUD_Z"
]

X = df[FEATURES]
y = df["STRESS_LABEL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=False
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    device="cuda",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

model.fit(X_train, y_train)


y_prob = model.predict_proba(X_test)[:, 1]
threshold = np.quantile(y_prob, 0.97)
y_pred = (y_prob > threshold).astype(int)


print("Threshold:", threshold)
print("Final Precision:", precision_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(
    model,
    MODEL_DIR / "xgboost_stress_gatekeeper.pkl"
)
