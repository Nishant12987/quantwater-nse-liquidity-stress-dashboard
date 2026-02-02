import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, classification_report
from xgboost import XGBClassifier


BASE = Path("/content/data/NSE_Liquidity_Project")

features_path = BASE / "features"
model_path = BASE / "models" / "checkpoints"
model_path.mkdir(parents=True, exist_ok=True)

amihud = pd.read_csv(features_path / "amihud.csv")
roll = pd.read_csv(features_path / "roll_spread.csv")
commonality = pd.read_csv(features_path / "liquidity_commonality.csv")
labels = pd.read_csv(features_path / "stress_labels.csv")

df = (
    amihud
    .merge(roll, on="SYMBOL", how="inner")
    .merge(commonality, on="SYMBOL", how="inner")
    .merge(labels, on="DATE", how="inner")
)

df = df.dropna()

X = df[["AMIHUD", "ROLL_SPREAD", "LIQUIDITY_COMMONALITY"]]
y = df["STRESS_LABEL"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, shuffle=False
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

param_grid = {
    "n_estimators": [100, 150, 200],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.01, 0.03, 0.05],
    "subsample": [0.6, 0.7, 0.8]
}

grid = GridSearchCV(
    base_model,
    param_grid,
    scoring="precision",
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

y_prob = best_model.predict_proba(X_test)[:, 1]
threshold = np.quantile(y_prob, 0.995)
y_pred = (y_prob > threshold).astype(int)

precision = precision_score(y_test, y_pred)

print("Best parameters:", grid.best_params_)
print("Threshold:", threshold)
print("Final Precision:", precision)
print(classification_report(y_test, y_pred))

joblib.dump(best_model, model_path / "xgboost_stress_gatekeeper.pkl")
