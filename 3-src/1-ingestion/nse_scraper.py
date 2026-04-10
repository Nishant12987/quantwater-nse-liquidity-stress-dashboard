import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# ✅ FIX: make config import work
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import config


ROOT_PATH = Path(config.ROOT_PATH)

# ✅ USE STANDARD RAW PATH
RAW_PATH = Path(config.DATA_RAW)

BASE_URL = "https://archives.nseindia.com/content/historical/EQUITIES"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def daterange(start, end):
    for n in range((end - start).days + 1):
        yield start + timedelta(n)


def clean_and_save(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["SERIES"] == "EQ"]
    df = df.drop(columns=["ISIN"], errors="ignore")
    df.to_csv(csv_path, index=False)


def download_bhavcopy(date):
    year = date.strftime("%Y")
    month = date.strftime("%b").upper()
    date_str = date.strftime("%d%b%Y").upper()

    # ✅ FIXED PATH
    year_dir = RAW_PATH / year
    year_dir.mkdir(parents=True, exist_ok=True)

    csv_path = year_dir / f"cm{date_str}bhav.csv"
    if csv_path.exists():
        return

    url = f"{BASE_URL}/{year}/{month}/cm{date_str}bhav.csv.zip"

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)

        if r.status_code != 200:
            return

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(year_dir)

        clean_and_save(csv_path)

    except Exception as e:
        print(f"Error on {date}: {e}")


def main():
    start = datetime(2021, 1, 1)

    # ⚠️ TEMP REDUCE FOR COLAB (important)
    end = datetime(2023, 12, 31)

    for date in tqdm(list(daterange(start, end))):
        if date.weekday() < 5:
            download_bhavcopy(date)


if __name__ == "__main__":
    main()
