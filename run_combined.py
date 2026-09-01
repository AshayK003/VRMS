"""Run combined VIX + momentum strategy with parameter optimization."""
from __future__ import annotations

import logging
import sys

from src.strategies.combined_vix_momentum import (
    fetch_data,
    generate_signals,
    run_backtest,
    optimize_parameters,
    calculate_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_single_backtest(
    nifty_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    params: dict,
) -> dict:
    """Run a single backtest with given params."""
    signals = generate_signals(vix_df, nifty_df, params)
    result = run_backtest(signals, nifty_df, params)
    
    # Calculate annualized return
    days = (nifty_df.index.max() - nifty_df.index.min()).days
    annualized = (1 + result.metrics['total_return']) ** (365 / max(days, 1)) - 1
    
    return {
        'params': params,
        'metrics': result.metrics,
        'annualized': annualized,
        'trades': len(result.trades),
    }


def main():
    """Run optimization and show results."""
    print("Fetching data...")
    
    nifty_df = fetch_data("^NSEI", period="3y")
    vix_df = fetch_data("^INDIAVIX", period="3y")
    
    print(f"Nifty: {len(nifty_df)} days ({nifty_df.index.min().strftime('%Y-%m-%d')} to {nifty_df.index.max().strftime('%Y-%m-%d')})")
    print(f"VIX: {len(vix_df)} days")
    
    if nifty_df.empty or vix_df.empty:
        print("ERROR: Could not fetch required data")
        return
    
    # Define parameter grid
    param_grid = {
        'vix_high': [16, 17, 18],
        'vix_low': [12, 13, 14],
        'adx_threshold': [20, 25],
        'target_pct': [0.03, 0.05, 0.07],
        'stop_loss_pct': [0.03, 0.05, 0.07],
    }
    
    print(f"\nParameter grid: {len(param_grid)} parameters")
    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"Total combinations: {total}")
    print("\nRunning optimization (this may take a few minutes)...")
    
    results = optimize_parameters(nifty_df, vix_df, param_grid)
    
    if not results:
        print("ERROR: No valid results")
        return
    
    print(f"\n{'='*80}")
    print("COMBINED VIX + MOMENTUM STRATEGY — PARAMETER OPTIMIZATION RESULTS")
    print("="*80)
    
    # Show top 10 results
    print(f"\nTop 10 by Sharpe ratio:")
    print("-" * 80)
    print(f"{'Rank':<5} {'VIX Hi':<7} {'VIX Lo':<7} {'ADX':<5} {'Target':<8} {'Stop':<8} {'Trades':<8} {'Win%':<8} {'Return':<10} {'Sharpe':<8}")
    print("-" * 80)
    
    for i, r in enumerate(results[:10], 1):
        p = r['params']
        m = r['metrics']
        print(f"{i:<5} {p['vix_high']:<7.0f} {p['vix_low']:<7.0f} {p['adx_threshold']:<5.0f} "
              f"{p['target_pct']:<8.0%} {p['stop_loss_pct']:<8.0%} {r['trades']:<8} "
              f"{m['win_rate']:<8.0%} {m['total_return']:<10.1%} {m['sharpe']:<8.2f}")
    
    # Show best result
    best = results[0]
    p = best['params']
    m = best['metrics']
    days = (nifty_df.index.max() - nifty_df.index.min()).days
    annualized = (1 + m['total_return']) ** (365 / max(days, 1)) - 1
    
    print(f"\n{'='*80}")
    print("BEST PARAMETERS:")
    print("="*80)
    print(f"VIX High (fear):        {p['vix_high']}")
    print(f"VIX Low (complacency):  {p['vix_low']}")
    print(f"ADX Threshold:          {p['adx_threshold']}")
    print(f"Target:                 {p['target_pct']:.0%}")
    print(f"Stop Loss:              {p['stop_loss_pct']:.0%}")
    print(f"\nPerformance:")
    print(f"Total trades:           {best['trades']}")
    print(f"Win rate:               {m['win_rate']:.1%}")
    print(f"Total return:           {m['total_return']:.1%}")
    print(f"Annualized:             {annualized:.1%}")
    print(f"Sharpe ratio:           {m['sharpe']:.2f}")
    print(f"Max drawdown:           {m['max_drawdown']:.1%}")
    print(f"Avg win:                {m['avg_win']:.2%}")
    print(f"Avg loss:               {m['avg_loss']:.2%}")
    print(f"Profit factor:          {m['profit_factor']:.2f}")
    
    # Verdict
    if m['win_rate'] >= 0.55 and m['sharpe'] >= 0.8:
        print(f"\nVERDICT: ✅ PASS — Edge exists with these parameters")
    elif m['win_rate'] >= 0.50:
        print(f"\nVERDICT: ⚠️ MARGINAL — Edge exists but needs improvement")
    else:
        print(f"\nVERDICT: ❌ FAIL — No edge, kill it")
    
    print("="*80)


if __name__ == "__main__":
    main()
