import requests
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import os

# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------

BASE = Path(os.getcwd())
RAW_DIR = BASE / "data/raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DIR / "nse_live.csv"


# -------------------------------------------------
# NSE CONFIG
# -------------------------------------------------

NSE_URL = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}


# -------------------------------------------------
# FETCH FUNCTION
# -------------------------------------------------

def fetch_nse_data(retries=3):

    session = requests.Session()

    # Initial call to get cookies
    session.get("https://www.nseindia.com", headers=HEADERS)

    for attempt in range(retries):
        try:
            response = session.get(NSE_URL, headers=HEADERS, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data

        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)
            time.sleep(2)

    raise Exception("Failed to fetch NSE data after retries")


# -------------------------------------------------
# PROCESS DATA
# -------------------------------------------------

def process_data(data):

    records = []

    for item in data.get("data", []):
        records.append({
            "DATE": datetime.now(),
            "SYMBOL": item.get("symbol"),
            "PRICE": item.get("lastPrice"),
            "VOLUME": item.get("totalTradedVolume"),
            "VALUE": item.get("totalTradedValue"),
        })

    df = pd.DataFrame(records)

    return df


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    print("Fetching NSE live data...")

    data = fetch_nse_data()

    df = process_data(data)

    if df.empty:
        raise ValueError("No data fetched from NSE")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved live NSE data to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
