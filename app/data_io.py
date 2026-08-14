# app/data_io.py — pure data loading helpers (no Streamlit; testable directly)

import datetime
import io
import re
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl


def read_tabular(uploaded) -> pl.DataFrame:
    """Read an uploaded CSV/Excel/Parquet file into a DataFrame.
    Raises ValueError on unsupported extensions."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(io.BytesIO(uploaded.getvalue()))
    if suffix == ".csv":
        return pl.read_csv(
            io.BytesIO(uploaded.getvalue()),
            infer_schema_length=10000,
            try_parse_dates=True,
        )
    if suffix in (".xlsx", ".xls"):
        return pl.read_excel(io.BytesIO(uploaded.getvalue()))
    raise ValueError("Please upload a CSV, Excel, or Parquet file.")


def sample_sales() -> pl.DataFrame:
    """Deterministic demo dataset: 18 months of shop sales with a gentle
    upward trend, so every page has something real to show."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 600
    start = datetime.date(2024, 1, 1)
    days = rng.integers(0, 540, n)
    products = rng.choice(["Espresso", "Latte", "Cold Brew", "Tea", "Pastry"], n)
    price = {"Espresso": 3.0, "Latte": 4.5, "Cold Brew": 5.0, "Tea": 2.5, "Pastry": 3.5}
    units = rng.poisson(8, n) + 1
    return pl.DataFrame(
        {
            "date": [start + datetime.timedelta(days=int(d)) for d in days],
            "region": rng.choice(["North", "South", "East", "West"], n),
            "product": products,
            "units": units.astype("int64"),
            "revenue": [
                round(u * price[p] * (1 + 0.4 * d / 540) * rng.uniform(0.85, 1.15), 2)
                for u, p, d in zip(units, products, days, strict=True)
            ],
        }
    )


def sheets_csv_url(url: str) -> str | None:
    """Rewrite a Google Sheets share link to its CSV export URL (or None)."""
    m = re.search(r"docs\.google\.com/spreadsheets/d/([\w-]+)", url)
    if not m:
        return None
    gid = re.search(r"[#?&]gid=(\d+)", url)
    return (
        f"https://docs.google.com/spreadsheets/d/{m.group(1)}"
        f"/export?format=csv&gid={gid.group(1) if gid else '0'}"
    )


def load_from_url(url: str) -> tuple[str, pl.DataFrame]:
    """Fetch a CSV/Parquet file — or any shared Google Sheet — by URL."""
    sheet = sheets_csv_url(url)
    if sheet:
        name = "google_sheet"
        url = sheet
    else:
        name = Path(urllib.parse.urlparse(url).path).stem or "web_data"
    req = urllib.request.Request(url, headers={"User-Agent": "neuroviz/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    if url.split("?")[0].lower().endswith(".parquet"):
        return name, pl.read_parquet(io.BytesIO(raw))
    return name, pl.read_csv(
        io.BytesIO(raw), infer_schema_length=10000, try_parse_dates=True
    )
