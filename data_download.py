"""
data_download.py

Downloads and aligns three daily time series for the volatility forecasting horse race:
  1. S&P 500 daily log returns         (Yahoo Finance)
  2. VIX daily close                   (Yahoo Finance)
  3. Realized volatility               (Parkinson estimator, computed from Yahoo Finance
                                        daily High/Low prices)

Output: a single aligned DataFrame saved to data/aligned_data.csv

Note on the Parkinson estimator
────────────────────────────────
The Parkinson (1980) range-based estimator uses the daily high-low spread as a
proxy for intraday volatility:

    RV_park,t = (1 / (4 · ln2)) · [ln(H_t / L_t)]²

It is approximately 5× more efficient than squared daily close-to-close returns
because it incorporates intraday price information without requiring tick data.
It assumes no drift and continuous trading, which are reasonable approximations
for liquid index data like the S&P 500.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf  # pip install yfinance

# ── Configuration ─────────────────────────────────────────────────────────────

START_DATE  = "2015-01-01"
END_DATE    = "2024-12-31"
OUTPUT_DIR  = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "aligned_data.csv")

# Parkinson constant: 1 / (4 · ln(2))
PARKINSON_CONST = 1.0 / (4.0 * np.log(2))

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_spx(start: str, end: str) -> pd.DataFrame:
    """
    Downloads S&P 500 daily OHLC from Yahoo Finance in a single call.
    Returns a DataFrame with columns: spx_log_return, spx_high, spx_low.

    We pull High and Low here (not just Close) because they feed directly
    into the Parkinson RV estimator.
    """
    print("Downloading S&P 500 OHLC from Yahoo Finance...")
    raw = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)

    close = raw["Close"].squeeze()
    high  = raw["High"].squeeze()
    low   = raw["Low"].squeeze()

    log_returns = np.log(close / close.shift(1))

    out = pd.DataFrame({
        "spx_log_return": log_returns,
        "spx_high":       high,
        "spx_low":        low,
    }).dropna()

    print(f"  S&P 500 OHLC: {len(out)} observations")
    return out


def fetch_vix(start: str, end: str) -> pd.Series:
    """
    Downloads VIX daily close from Yahoo Finance.
    VIX is in annualized percentage vol units (e.g., 20 = 20% annualized).
    Stored raw here; converted to daily variance units (VIX²/252/10000)
    in models.py to keep raw data clean.
    """
    print("Downloading VIX from Yahoo Finance...")
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    vix = raw["Close"].squeeze()
    vix.name = "vix"
    print(f"  VIX: {len(vix)} observations")
    return vix


def compute_parkinson_rv(spx: pd.DataFrame) -> pd.Series:
    """
    Computes the Parkinson (1980) range-based realized variance from daily
    High and Low prices:

        RV_park,t = (1 / (4·ln2)) · [ln(H_t / L_t)]²

    Returns a Series named 'realized_var' in squared log-return units,
    consistent with what GARCH and VIX² will produce after unit conversion.
    """
    log_hl = np.log(spx["spx_high"] / spx["spx_low"])
    rv = PARKINSON_CONST * (log_hl ** 2)
    rv.name = "realized_var"
    print(f"  Parkinson RV computed: {len(rv)} observations")
    return rv


# ── Main ──────────────────────────────────────────────────────────────────────

def download_all() -> pd.DataFrame:
    """
    Downloads all series, computes Parkinson RV, aligns on trading dates,
    and returns a single DataFrame with columns:
        spx_log_return  — daily log return of S&P 500
        vix             — VIX close (annualized %, raw)
        realized_var    — Parkinson range-based daily realized variance
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    spx = fetch_spx(START_DATE, END_DATE)
    vix = fetch_vix(START_DATE, END_DATE)
    rv  = compute_parkinson_rv(spx)

    # Drop the High/Low columns — only returns and RV go into the output file
    combined = pd.concat([spx["spx_log_return"], vix, rv], axis=1, join="inner")
    combined.index.name = "date"
    combined = combined.loc[START_DATE:END_DATE].dropna()

    print(f"\nAligned dataset: {len(combined)} trading days "
          f"({combined.index[0].date()} to {combined.index[-1].date()})")

    _sanity_check(combined)

    combined.to_csv(OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")
    return combined


def _sanity_check(df: pd.DataFrame):
    """Prints a quick summary to catch obvious data issues before moving on."""
    print("\n── Sanity Check ─────────────────────────────────")
    print(df.describe().round(6))

    missing = df.isnull().sum()
    if missing.any():
        print("\nWARNING: missing values detected:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values. ✓")

    # VIX should be in reasonable range (roughly 10–90 over this sample)
    vix_min, vix_max = df["vix"].min(), df["vix"].max()
    if vix_min < 5 or vix_max > 100:
        print(f"WARNING: VIX range looks odd ({vix_min:.1f} – {vix_max:.1f})")
    else:
        print(f"VIX range: {vix_min:.1f} – {vix_max:.1f}  ✓")

    # Realized variance should be strictly positive (Parkinson uses log ratio)
    if (df["realized_var"] <= 0).any():
        print("WARNING: non-positive Parkinson RV values detected — check High/Low data")
    else:
        rv_mean = df["realized_var"].mean()
        print(f"Parkinson RV: all positive, mean = {rv_mean:.6f}  ✓")

    # Quick implied-vol sanity: annualized vol from mean Parkinson RV
    ann_vol = np.sqrt(df["realized_var"].mean() * 252) * 100
    print(f"Implied annualized vol from Parkinson RV: {ann_vol:.1f}%  (expect ~15–20%)")
    print("─────────────────────────────────────────────────\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = download_all()
    print(data.head(10).to_string())
