import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_FILE = "data/forecast_results.csv"

def load_and_prep_results():
    df = pd.read_csv(RESULTS_FILE, index_col="date", parse_dates=True)
    return df

def calculate_metrics(y_true, y_pred):
    """Calculates standard forecasting metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return {"RMSE": rmse, "R2": r2, "Corr": corr}

def run_evaluation():
    # 1. Load data
    df = load_and_prep_results()
    horses = [col for col in df.columns if "_forecast" in col]
    
    # 2. Global Scorecard
    print("\n" + "="*40)
    print("      GLOBAL VOLATILITY SCORECARD")
    print("="*40)
    
    results_list = []
    for horse in horses:
        metrics = calculate_metrics(df["realized_var"], df[horse])
        results_list.append(pd.Series(metrics, name=horse.replace("_forecast", "")))
    
    scorecard = pd.concat(results_list, axis=1).T
    print(scorecard.sort_values("R2", ascending=False).round(5))

    # 3. Regime Analysis: High VIX vs Low VIX
    # We define 'Panic' as VIX being in the top 20% of the sample
    panic_threshold = df["vix_forecast"].quantile(0.80)
    panic_df = df[df["vix_forecast"] > panic_threshold]
    calm_df = df[df["vix_forecast"] <= panic_threshold]

    print(f"\n--- REGIME PERFORMANCE (R2 Score) ---")
    print(f"Panic Regime (N={len(panic_df)}) | Calm Regime (N={len(calm_df)})")
    
    regime_data = []
    for horse in horses:
        h_name = horse.replace("_forecast", "")
        r2_panic = r2_score(panic_df["realized_var"], panic_df[horse])
        r2_calm = r2_score(calm_df["realized_var"], calm_df[horse])
        regime_data.append({"Horse": h_name, "Panic_R2": r2_panic, "Calm_R2": r2_calm})
    
    print(pd.DataFrame(regime_data).set_index("Horse").round(4))

    # 4. Visualization: The Big Picture
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(df.index, df["realized_var"], color='black', label='Actual Volatility (RV)', alpha=0.2, lw=1)
    ax.plot(df.index, df["blair_forecast"], label='Horse 4: Blair (Baseline)', alpha=0.8)
    ax.plot(df.index, df["regime_forecast"], label='Horse 5: Your Model', alpha=0.8, linestyle='--')

    ax.set_title("Forecast Accuracy: Blair (2001) vs. Your Regime-Switching Model", fontsize=14)
    ax.set_ylabel("Daily Variance")
    ax.legend()

    # Save the chart
    plt.savefig("data/volatility_comparison.png")
    print("\nChart saved to data/volatility_comparison.png")
    plt.show()

if __name__ == "__main__":
    try:
        run_evaluation()
    except FileNotFoundError:
        print("Error: forecast_results.csv not found. Did you run models.py first?")