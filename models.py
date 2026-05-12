"""
models.py

Generates out-of-sample volatility forecasts for each horse in the race.
Currently implemented: Horse 1 (VIX alone).

Rolling window approach: at each day t, the model is fit on the previous
WINDOW_SIZE days, then used to forecast realized variance for day t.
This ensures all forecasts are genuine out-of-sample.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

# ── Configuration ─────────────────────────────────────────────────────────────

WINDOW_SIZE = 1000   # ~4 years of trading days
DATA_FILE   = "data/aligned_data.csv"

# ── Data Loading ──────────────────────────────────────────────────────────────

def load_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    """
    Loads the aligned dataset and adds derived columns needed by the models.
    """
    data = pd.read_csv(filepath, index_col="date", parse_dates=True)

    # Convert VIX from annualized percentage vol to daily variance units
    # VIX = 20 means 20% annualized vol → daily variance = (0.20)² / 252
    # Yahoo Finance VIX is already in percentage points, so divide by 100 first
    data["vix_var"] = (data["vix"] / 100) ** 2 / 252

    return data

# ── Horse 1: VIX Alone ────────────────────────────────────────────────────────

def run_vix_horse(data: pd.DataFrame) -> pd.Series:
    """
    Horse 1: VIX² as the sole predictor of next-day realized variance.

    At each day t, fits:
        realized_var = a + β₁ · vix_var + ε
    on the previous WINDOW_SIZE days, then forecasts realized_var for day t
    using today's vix_var as input.

    Returns a Series of out-of-sample forecasts aligned to the same index
    as the input data (NaN for the first WINDOW_SIZE days where we have
    no training history yet).
    """
    forecasts = pd.Series(index=data.index, dtype=float)

    n = len(data)

    for t in range(WINDOW_SIZE, n):

        # ── Training slice ────────────────────────────────────────────────────
        train = data.iloc[t - WINDOW_SIZE : t]

        y = train["realized_var"]
        X = sm.add_constant(train["vix_var"])   # adds intercept column

        # ── Fit OLS on training window ────────────────────────────────────────
        model = sm.OLS(y, X).fit()

        # ── Forecast for day t using today's VIX ─────────────────────────────
        today_X = sm.add_constant(
            pd.DataFrame({"vix_var": [data["vix_var"].iloc[t]]}),
            has_constant="add"
        )
        forecasts.iloc[t] = model.predict(today_X).iloc[0]

        # Progress update every 250 steps
        if (t - WINDOW_SIZE) % 250 == 0:
            pct = (t - WINDOW_SIZE) / (n - WINDOW_SIZE) * 100
            print(f"  VIX horse: {pct:.0f}% complete (day {t}/{n})")

    print("  VIX horse: done.")
    return forecasts

# ── GARCH Pre-computation ─────────────────────────────────────────────────────

REFIT_FREQ       = 21    # re-estimate GARCH parameters monthly (~21 trading days)
MIN_GARCH_WINDOW = 252   # minimum days needed to fit a stable GARCH model (~1 year)

def compute_garch_var_series(data: pd.DataFrame) -> pd.Series:
    """
    Pre-computes an out-of-sample GARCH conditional variance series.

    At each day t, h_t is the one-step-ahead conditional variance forecast
    produced using only data available before day t — no future information.

    Window strategy:
      - Days 252–999: expanding window (all available history). Necessary
        because there isn't yet a full WINDOW_SIZE of history. Produces valid
        garch_var values from day 252 onward, so the OLS training windows
        in Horses 2 and 4 have real data to work with from the start.
      - Days 1000+: rolling window of WINDOW_SIZE days (standard).

    This series is computed once and shared by Horse 2 and Horse 4.
    NaN rows are dropped in OLS steps to handle the early partial window.
    """
    garch_var = pd.Series(index=data.index, dtype=float, name="garch_var")
    n = len(data)

    omega, alpha, gamma, beta = None, None, None, None
    h_prev     = None
    last_refit = None

    for t in range(WINDOW_SIZE, n):
        # Pure rolling window throughout — no expanding phase
        # This ensures all garch_var values are produced with the same
        # WINDOW_SIZE days of history, eliminating the advantage that
        # expanding-window estimates had from growing sample sizes.
        train_returns = data["spx_log_return"].iloc[t - WINDOW_SIZE : t] * 100

        should_refit = (last_refit is None) or ((t - last_refit) >= REFIT_FREQ)

        if should_refit:
            am  = arch_model(train_returns, mean="Zero", vol="GARCH", p=1, o=1, q=1)
            res = am.fit(disp="off")

            omega = res.params["omega"]
            alpha = res.params["alpha[1]"]
            gamma = res.params.get("gamma[1]", 0.0)
            beta  = res.params["beta[1]"]

            h_t        = res.forecast(horizon=1).variance.iloc[-1, 0]   # in %²
            last_refit = t
        else:
            r_prev    = data["spx_log_return"].iloc[t - 1] * 100
            indicator = 1.0 if r_prev < 0 else 0.0
            h_t = omega + (alpha + gamma * indicator) * (r_prev ** 2) + beta * h_prev

        garch_var.iloc[t] = h_t / 10000   # convert to decimal variance
        h_prev = h_t

        if t % 250 == 0:
            print(f"  GARCH pre-computation: day {t}/{n}")

    print("  GARCH pre-computation: done.")
    return garch_var


# ── Horse 2: GARCH(1,1) Alone ─────────────────────────────────────────────────

def run_garch_horse(data: pd.DataFrame, garch_var: pd.Series) -> pd.Series:
    """
    Horse 2: GJR-GARCH(1,1) calibrated via rolling OLS.

    Uses the pre-computed garch_var series as the sole predictor:
        RV_t = a + β · garch_var_t + ε

    Because garch_var[t] was itself produced using only data up to t-1,
    using it as a predictor in the OLS training window is leak-free —
    every value in the training slice is a genuine past forecast.
    """
    forecasts = pd.Series(index=data.index, dtype=float)
    n = len(data)

    # Start at WINDOW_SIZE. garch_var is NaN for the first WINDOW_SIZE days,
    # so the OLS dropna() will use fewer training observations early on (growing
    # from ~0 to 1000 over the second WINDOW_SIZE period). The len<50 guard
    # skips forecasts until there's enough training data to fit reliably.
    start_t = WINDOW_SIZE
    for t in range(start_t, n):
        train = pd.concat([
            data["realized_var"].iloc[t - WINDOW_SIZE : t],
            garch_var.iloc[t - WINDOW_SIZE : t],
        ], axis=1).dropna()

        if len(train) < 50:
            continue

        y = train["realized_var"]
        X = sm.add_constant(train["garch_var"])

        model = sm.OLS(y, X).fit()

        if pd.isna(garch_var.iloc[t]):
            continue

        today_X = sm.add_constant(
            pd.DataFrame({"garch_var": [garch_var.iloc[t]]}),
            has_constant="add"
        )
        forecasts.iloc[t] = model.predict(today_X).iloc[0]

        if (t - start_t) % 250 == 0:
            pct = (t - start_t) / (n - start_t) * 100
            print(f"  GARCH horse: {pct:.0f}% complete (day {t}/{n})")

    print("  GARCH horse: done.")
    return forecasts


# ── Horse 3: INTRA (Naive Persistence) ───────────────────────────────────────

def run_intra_horse(data: pd.DataFrame) -> pd.Series:
    """
    Horse 3: Yesterday's realized variance as the forecast for today's.

    This is the naive persistence benchmark — it assumes volatility tomorrow
    will look like volatility today. Despite its simplicity, it is surprisingly
    hard to beat because realized variance is highly autocorrelated (volatile
    days tend to cluster together).

    No rolling window or fitting needed: the forecast for day t is simply
    realized_var at day t-1. We still restrict to the same out-of-sample
    period as the other horses (starting at WINDOW_SIZE) for a fair comparison.
    """
    forecasts = data["realized_var"].shift(1)
    forecasts.name = "intra_forecast"

    # Restrict to the out-of-sample window so all horses are evaluated on the
    # same dates
    forecasts.iloc[:WINDOW_SIZE] = np.nan

    print("  INTRA horse: done.")
    return forecasts


# ── Horse 4: Constant-Weight Blend (Blair Replication) ───────────────────────

def run_blair_horse(data: pd.DataFrame, garch_var: pd.Series) -> pd.Series:
    """
    Horse 4: VIX + GARCH with constant OLS weights — the Blair et al. (2001)
    encompassing regression.

    At each day t, fits:
        RV_t = a + β₁·vix_var_t + β₂·garch_var_t + ε

    on the previous WINDOW_SIZE days. Both predictors are aligned series,
    so the OLS training window is always correctly matched — no misalignment.

    If β₂ ≈ 0, VIX subsumes GARCH (Blair's finding).
    If β₁ ≈ 0, GARCH subsumes VIX (Becker's finding).
    The disagreement between Blair and Becker across sample periods is the
    motivation for Horse 5: constant weights are wrong across regimes.
    """
    forecasts = pd.Series(index=data.index, dtype=float)
    n = len(data)

    start_t      = WINDOW_SIZE
    first_window = True   # print coefficients once to sanity-check signs

    for t in range(start_t, n):
        train = pd.concat([
            data["realized_var"].iloc[t - WINDOW_SIZE : t],
            data["vix_var"].iloc[t - WINDOW_SIZE : t],
            garch_var.iloc[t - WINDOW_SIZE : t],
        ], axis=1).dropna()

        if len(train) < 50:
            continue

        y = train["realized_var"]
        X = sm.add_constant(train[["vix_var", "garch_var"]])

        ols = sm.OLS(y, X).fit()

        # Print coefficients on first window to check for unexpected signs
        if first_window:
            print(f"  Blair first-window coefficients — "
                  f"intercept: {ols.params['const']:.6f}, "
                  f"VIX: {ols.params['vix_var']:.4f}, "
                  f"GARCH: {ols.params['garch_var']:.4f}")
            if ols.params["vix_var"] < 0 or ols.params["garch_var"] < 0:
                print("  WARNING: negative coefficient detected — "
                      "possible multicollinearity between VIX and GARCH.")
            first_window = False

        if pd.isna(garch_var.iloc[t]):
            continue

        today_X = sm.add_constant(
            pd.DataFrame({
                "vix_var":   [data["vix_var"].iloc[t]],
                "garch_var": [garch_var.iloc[t]],
            }),
            has_constant="add"
        )
        forecasts.iloc[t] = ols.predict(today_X).iloc[0]

        if (t - start_t) % 250 == 0:
            pct = (t - start_t) / (n - start_t) * 100
            print(f"  Blair horse: {pct:.0f}% complete (day {t}/{n})")

    print("  Blair horse: done.")
    return forecasts


# ── Shared Evaluation Helper ──────────────────────────────────────────────────

def evaluate(realized: pd.Series, forecast: pd.Series, label: str):
    """Prints MSE, correlation, and R² for a forecast series."""
    errors = realized - forecast
    mse    = (errors ** 2).mean()
    corr   = realized.corr(forecast)
    ss_res = (errors ** 2).sum()
    ss_tot = ((realized - realized.mean()) ** 2).sum()
    r2     = 1 - ss_res / ss_tot
    print(f"\n── {label} ──────────────────────────")
    print(f"  MSE:         {mse:.8f}")
    print(f"  Correlation: {corr:.4f}")
    print(f"  R²:          {r2:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    data = load_data()

    print(f"Loaded {len(data)} trading days "
          f"({data.index[0].date()} to {data.index[-1].date()})")

    print(f"\nPre-computing GARCH conditional variance series...")
    garch_var = compute_garch_var_series(data)

    print(f"\nRunning VIX horse...")
    vix_forecasts   = run_vix_horse(data)

    print(f"\nRunning GARCH horse...")
    garch_forecasts = run_garch_horse(data, garch_var)

    print(f"\nRunning INTRA horse...")
    intra_forecasts = run_intra_horse(data)

    print(f"\nRunning Blair horse...")
    blair_forecasts = run_blair_horse(data, garch_var)

    # Align everything on dates where all series are available
    results = pd.DataFrame({
        "realized_var":    data["realized_var"],
        "vix_forecast":    vix_forecasts,
        "garch_forecast":  garch_forecasts,
        "intra_forecast":  intra_forecasts,
        "blair_forecast":  blair_forecasts,
    }).dropna()

    total_possible = len(data) - WINDOW_SIZE
    print(f"\nEvaluation window: {len(results)} days "
          f"({results.index[0].date()} to {results.index[-1].date()})")
    print(f"  ({total_possible - len(results)} days excluded by NaN drops)")

    evaluate(results["realized_var"], results["vix_forecast"],   "Horse 1: VIX")
    evaluate(results["realized_var"], results["garch_forecast"], "Horse 2: GARCH")
    evaluate(results["realized_var"], results["intra_forecast"], "Horse 3: INTRA")
    evaluate(results["realized_var"], results["blair_forecast"], "Horse 4: Blair (constant blend)")
