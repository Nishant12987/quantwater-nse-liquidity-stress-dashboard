# (Liquidity Commonality logic)
"""
Market Context & Liquidity Commonality Module

Implements:
- Daily Market-Wide Liquidity (Average Amihud)
- Liquidity Commonality (Stock vs Market)

Uses outputs from:
- amihud.csv (from 1-liquidity_metrics.py)

No ML libraries. Lean, vectorized, PIT-safe.
"""

import pandas as pd
from pathlib import Path
import os


# =========================
# PATHS (FIXED)
# =========================

BASE = Path(os.getcwd())

FEATURES_PATH = BASE / "data/processed"
OUTPUT_PATH = FEATURES_PATH

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD DATA
# =========================

def load_amihud() -> pd.DataFrame:
    path = FEATURES_PATH / "amihud.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"amihud.csv not found at {path}. Run liquidity_metrics.py first."
        )

    return pd.read_csv(path, parse_dates=["DATE"])


# =========================
# MARKET-WIDE LIQUIDITY
# =========================

def compute_market_amihud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional mean Amihud per day
    """
    market = (
        df.groupby("DATE", as_index=False)["AMIHUD"]
        .mean()
        .rename(columns={"AMIHUD": "MARKET_AMIHUD"})
    )
    return market


# =========================
# LIQUIDITY COMMONALITY
# =========================

def compute_liquidity_commonality(
    df: pd.DataFrame,
    market: pd.DataFrame
) -> pd.DataFrame:
    """
    Correlation of stock Amihud with market Amihud
    Computed per stock over time
    """
    merged = df.merge(market, on="DATE", how="inner")

    results = []

    for symbol, g in merged.groupby("SYMBOL"):

        if g["AMIHUD"].notna().sum() < 30:
            continue  # insufficient observations

        corr = g["AMIHUD"].corr(g["MARKET_AMIHUD"])

        results.append({
            "SYMBOL": symbol,
            "LIQUIDITY_COMMONALITY": corr
        })

    return pd.DataFrame(results)


# =========================
# MAIN ORCHESTRATION
# =========================

def build_market_context():
    print("Loading Amihud data...")
    df = load_amihud()

    print("Computing market-wide liquidity...")
    market = compute_market_amihud(df)
    market.to_csv(OUTPUT_PATH / "market_amihud.csv", index=False)

    print("Computing liquidity commonality...")
    commonality = compute_liquidity_commonality(df, market)
    commonality.to_csv(OUTPUT_PATH / "liquidity_commonality.csv", index=False)

    print("✅ Market context features saved successfully at:", OUTPUT_PATH)


if __name__ == "__main__":
    build_market_context()
