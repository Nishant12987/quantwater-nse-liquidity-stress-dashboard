# (Amihud, roll, Price Impact)
"""
Liquidity Metrics Module
Implements:
- Log Returns
- Amihud Illiquidity
- Roll Spread Proxy
- Price Impact

Formulas verified against academic literature:
Amihud (2002), Roll (1984)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os


# =========================
# CONFIG (FIXED)
# =========================

BASE = Path(os.getcwd())

BASE_DATA_PATH = BASE / "data/raw"
OUTPUT_PATH = BASE / "data/processed"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

ROLL_WINDOW_MIN = 30  # minimum observations for Roll spread


# =========================
# DATA LOADING
# =========================

def load_raw_data() -> pd.DataFrame:
    dfs = []

    if not BASE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data folder not found at {BASE_DATA_PATH}. Run ingestion first."
        )

    for year_dir in BASE_DATA_PATH.iterdir():
        if not year_dir.is_dir():
            continue

        for csv in year_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv)
                df["DATE"] = pd.to_datetime(df["TIMESTAMP"])
                dfs.append(df)
            except Exception as e:
                print(f"Skipping {csv}: {e}")

    if not dfs:
        raise ValueError("No CSV files found in raw data folder.")

    df_all = pd.concat(dfs, ignore_index=True)

    # Keep only EQ series
    df_all = df_all[df_all["SERIES"] == "EQ"]

    # Drop NSE trailing comma artifact
    df_all = df_all.loc[:, ~df_all.columns.str.contains("^Unnamed")]

    return df_all.sort_values(["SYMBOL", "DATE"])


# =========================
# RETURNS
# =========================

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["RET"] = np.log(df["CLOSE"] / df["PREVCLOSE"])
    return df


# =========================
# AMIHUD ILLIQUIDITY
# =========================

def compute_amihud(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AMIHUD"] = np.abs(df["RET"]) / df["TOTTRDVAL"]
    df.loc[df["TOTTRDVAL"] <= 0, "AMIHUD"] = np.nan
    return df[["DATE", "SYMBOL", "AMIHUD"]]


# =========================
# PRICE IMPACT
# =========================

def compute_price_impact(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PRICE_IMPACT"] = np.abs(df["RET"]) / df["TOTTRDQTY"]
    df.loc[df["TOTTRDQTY"] <= 0, "PRICE_IMPACT"] = np.nan
    return df[["DATE", "SYMBOL", "PRICE_IMPACT"]]


# =========================
# ROLL SPREAD
# =========================

def compute_roll_spread(df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for symbol, g in df.groupby("SYMBOL"):
        g = g.sort_values("DATE").dropna(subset=["RET"])
        r = g["RET"].values

        if len(r) < ROLL_WINDOW_MIN:
            continue

        cov = np.cov(r[1:], r[:-1])[0, 1]
        roll = 2 * np.sqrt(-cov) if cov < 0 else np.nan

        results.append({
            "SYMBOL": symbol,
            "ROLL_SPREAD": roll
        })

    return pd.DataFrame(results)


# =========================
# MAIN ORCHESTRATION
# =========================

def build_liquidity_metrics():
    print("Loading raw data...")
    df = load_raw_data()

    print("Computing returns...")
    df = compute_returns(df)

    print("Computing Amihud...")
    amihud = compute_amihud(df)
    amihud.to_csv(OUTPUT_PATH / "amihud.csv", index=False)

    print("Computing Price Impact...")
    price_impact = compute_price_impact(df)
    price_impact.to_csv(OUTPUT_PATH / "price_impact.csv", index=False)

    print("Computing Roll Spread...")
    roll = compute_roll_spread(df)
    roll.to_csv(OUTPUT_PATH / "roll_spread.csv", index=False)

    print("✅ Liquidity metrics saved successfully at:", OUTPUT_PATH)


if __name__ == "__main__":
    build_liquidity_metrics()
