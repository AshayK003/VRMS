"""Run walk-forward backtest."""
from __future__ import annotations

import logging
import sys

from src.backtest.walk_forward import run_walk_forward_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the backtest and print results."""
    symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'SBIN', 'ITC', 'BHARTIARTL', 'HCLTECH', 'ASIANPAINT'
    ]
    
    result = run_walk_forward_backtest(
        symbols=symbols,
        start_date='2023-01-01',
        end_date='2024-12-31',
        train_window=252,
        test_window=21,
        transaction_cost=0.003,
        target_pct=0.05,
        stop_loss_pct=0.05,
        threshold=0.3,
    )
    
    print("\n" + "="*60)
    print("VRMS — Walk-Forward Backtest Results")
    print("="*60)
    print(f"Period: 2023-01-01 to 2024-12-31")
    print(f"Stocks: {len(symbols)}")
    print(f"Train window: 252 days")
    print(f"Test window: 21 days")
    print(f"Transaction cost: 0.3%")
    print(f"Target: 5% | Stop loss: 5%")
    print()
    print(f"Total trades: {result.n_trades}")
    print(f"Wins: {result.n_wins}")
    print(f"Losses: {result.n_losses}")
    print(f"Win rate: {result.win_rate:.1%}")
    print()
    print(f"Total return: {result.total_return:.1%}")
    print(f"Sharpe ratio: {result.sharpe:.2f}")
    print(f"Max drawdown: {result.max_drawdown:.1%}")
    print()
    print(f"Avg win: {result.avg_win:.2%}")
    print(f"Avg loss: {result.avg_loss:.2%}")
    print(f"Profit factor: {result.profit_factor:.2f}")
    print()
    
    # Verdict
    if result.win_rate >= 0.55 and result.sharpe >= 0.8:
        print("VERDICT: ✅ PASS — Edge exists")
    elif result.win_rate >= 0.50:
        print("VERDICT: ⚠️ MARGINAL — Needs improvement")
    else:
        print("VERDICT: ❌ FAIL — No edge, kill it")
    print("="*60)


if __name__ == "__main__":
    main()
