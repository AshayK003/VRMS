"""Run Directional VIX strategy backtest."""
from __future__ import annotations

import logging

from src.strategies.directional_vix import (
    fetch_data,
    generate_directional_signals,
    run_directional_backtest,
    calculate_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the strategy backtest."""
    print("Fetching data...")
    
    nifty_df = fetch_data("^NSEI", period="5y")
    vix_df = fetch_data("^INDIAVIX", period="5y")
    
    print(f"Nifty: {len(nifty_df)} days ({nifty_df.index.min().strftime('%Y-%m-%d')} to {nifty_df.index.max().strftime('%Y-%m-%d')})")
    print(f"VIX: {len(vix_df)} days")
    
    if nifty_df.empty or vix_df.empty:
        print("ERROR: Could not fetch required data")
        return
    
    # Generate signals
    signals = generate_directional_signals(vix_df, nifty_df)
    
    buy_signals = [s for s in signals if s.direction == "BUY"]
    sell_signals = [s for s in signals if s.direction == "SELL"]
    
    print(f"\nSignals generated: {len(signals)}")
    print(f"BUY signals: {len(buy_signals)}")
    print(f"SELL signals: {len(sell_signals)}")
    
    # Run backtest
    trades, equity_curve = run_directional_backtest(signals, nifty_df)
    
    # Calculate metrics
    metrics = calculate_metrics(trades)
    
    print("\n" + "="*60)
    print("Directional VIX Strategy — Backtest Results")
    print("="*60)
    print(f"Period: {nifty_df.index.min().strftime('%Y-%m-%d')} to {nifty_df.index.max().strftime('%Y-%m-%d')}")
    print(f"Transaction cost: 0.3%")
    print()
    print(f"Total trades: {metrics['n_trades']}")
    print(f"Win rate: {metrics['win_rate']:.1%}")
    print(f"Total return: {metrics['total_return']:.1%}")
    print(f"Annualized: {(1 + metrics['total_return']) ** (365 / (nifty_df.index.max() - nifty_df.index.min()).days) - 1:.1%}")
    print(f"Sharpe ratio: {metrics['sharpe']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.1%}")
    print()
    print(f"Avg win: {metrics['avg_win']:.2%}")
    print(f"Avg loss: {metrics['avg_loss']:.2%}")
    print(f"Profit factor: {metrics['profit_factor']:.2f}")
    print()
    
    # Verdict
    if metrics['win_rate'] >= 0.55 and metrics['sharpe'] >= 0.8:
        print("VERDICT: ✅ PASS — Edge exists")
    elif metrics['win_rate'] >= 0.50:
        print("VERDICT: ⚠️ MARGINAL — Needs improvement")
    else:
        print("VERDICT: ❌ FAIL — No edge, kill it")
    
    # Show recent signals
    print("\n" + "-"*60)
    print("Recent BUY signals:")
    for s in buy_signals[-5:]:
        print(f"  {s.date.strftime('%Y-%m-%d')}: VIX={s.vix:.1f}, conf={s.confidence:.0%} — {s.reason}")
    
    print("\nRecent SELL signals:")
    for s in sell_signals[-5:]:
        print(f"  {s.date.strftime('%Y-%m-%d')}: VIX={s.vix:.1f}, conf={s.confidence:.0%} — {s.reason}")
    
    # Show trades
    if trades:
        print("\n" + "-"*60)
        print("Trades:")
        for t in trades:
            print(f"  {t.entry_date.strftime('%Y-%m-%d')} -> {t.exit_date.strftime('%Y-%m-%d')}: "
                  f"{t.return_pct:+.1%} ({t.reason})")
    
    print("="*60)


if __name__ == "__main__":
    main()
