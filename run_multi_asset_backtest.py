"""Backtest multi-asset VIX screener.

Tests whether screening Nifty 50 stocks improves returns
vs just trading Nifty index.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from src.screener.multi_asset import (
    fetch_vix, fetch_stock_data, NIFTY_50,
    get_vix_regime, compute_momentum, compute_conviction,
    StockPick,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def backtest_multi_asset(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    target_pct: float = 0.05,
    stop_loss_pct: float = 0.03,
    transaction_cost: float = 0.003,
) -> dict:
    """Backtest multi-asset screener.
    
    For each fear day (VIX > 18):
    1. Screen Nifty 50 stocks
    2. Buy top 5 picks
    3. Hold until target/stop/reverse signal
    4. Track returns
    
    Returns:
        Dict with backtest results
    """
    params = {
        'vix_high': 18,
        'vix_low': 14,
        'adx_threshold': 25,
        'target_pct': target_pct,
        'stop_loss_pct': stop_loss_pct,
    }
    
    # Fetch VIX
    vix_df = fetch_vix(period="3y")
    vix_df = vix_df[(vix_df.index >= start_date) & (vix_df.index <= end_date)]
    
    if vix_df.empty:
        return {"error": "No VIX data"}
    
    # Find fear days
    fear_days = vix_df[vix_df['VIX'] > 18].index
    
    if len(fear_days) == 0:
        return {"error": "No fear days found"}
    
    # Backtest each fear day
    all_trades = []
    
    for fear_day in fear_days:
        # Get VIX data up to this day
        vix_until = vix_df[vix_df.index <= fear_day]
        
        # Screen stocks
        picks = []
        for symbol in NIFTY_50:
            df = fetch_stock_data(symbol, period="6mo")
            if df.empty or len(df) < 50:
                continue
            
            # Filter to data up to fear day
            df = df[df.index <= fear_day]
            if len(df) < 50:
                continue
            
            mom = compute_momentum(df)
            if not mom:
                continue
            
            conviction, reason = compute_conviction(mom, "fear", params)
            
            if conviction < 30:
                continue
            
            price = mom['price']
            picks.append({
                'symbol': symbol,
                'conviction': conviction,
                'price': price,
                'target': price * (1 + target_pct),
                'stop': price * (1 - stop_loss_pct),
                'reason': reason,
            })
        
        if not picks:
            continue
        
        # Sort by conviction and take top 5
        picks.sort(key=lambda x: x['conviction'], reverse=True)
        top_picks = picks[:5]
        
        # Simulate trades for each pick
        for pick in top_picks:
            # Get future prices
            future_data = fetch_stock_data(pick['symbol'], period="1mo")
            future_data = future_data[future_data.index > fear_day]
            
            if future_data.empty:
                continue
            
            # Find exit
            exit_price = None
            exit_date = None
            exit_reason = ""
            
            for date, row in future_data.iterrows():
                price = row['Close']
                
                if price >= pick['target']:
                    exit_price = pick['target']
                    exit_date = date
                    exit_reason = "Target hit"
                    break
                
                if price <= pick['stop']:
                    exit_price = pick['stop']
                    exit_date = date
                    exit_reason = "Stop loss hit"
                    break
                
                # Exit after 20 days
                if (date - fear_day).days >= 20:
                    exit_price = price
                    exit_date = date
                    exit_reason = "Holding limit"
                    break
            
            if exit_price is None:
                # Use last available price
                exit_price = future_data.iloc[-1]['Close']
                exit_date = future_data.index[-1]
                exit_reason = "End of data"
            
            return_pct = (exit_price - pick['price']) / pick['price'] - transaction_cost
            
            all_trades.append({
                'entry_date': fear_day,
                'exit_date': exit_date,
                'symbol': pick['symbol'],
                'entry_price': pick['price'],
                'exit_price': exit_price,
                'return_pct': return_pct,
                'reason': exit_reason,
                'conviction': pick['conviction'],
            })
    
    # Calculate metrics
    if not all_trades:
        return {"error": "No trades executed"}
    
    trades_df = pd.DataFrame(all_trades)
    
    wins = trades_df[trades_df['return_pct'] > 0]
    losses = trades_df[trades_df['return_pct'] <= 0]
    
    win_rate = len(wins) / len(trades_df)
    
    total_return = 1.0
    for _, trade in trades_df.iterrows():
        total_return *= (1 + trade['return_pct'])
    total_return -= 1
    
    returns = trades_df['return_pct'].values
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / len(returns)) if np.std(returns) > 0 else 0
    
    return {
        'n_trades': len(trades_df),
        'win_rate': win_rate,
        'total_return': total_return,
        'sharpe': sharpe,
        'avg_win': wins['return_pct'].mean() if len(wins) > 0 else 0,
        'avg_loss': losses['return_pct'].mean() if len(losses) > 0 else 0,
        'trades': trades_df,
    }


def main():
    """Run the backtest."""
    print("Running multi-asset backtest...")
    print("This may take a few minutes (fetching data for 50 stocks)...")
    
    result = backtest_multi_asset()
    
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    
    print(f"\n{'='*60}")
    print("MULTI-ASSET VIX SCREENER — BACKTEST RESULTS")
    print("="*60)
    print(f"Period: 2023-01-01 to 2024-12-31")
    print(f"Target: 5% | Stop loss: 3% | Transaction cost: 0.3%")
    print()
    print(f"Total trades: {result['n_trades']}")
    print(f"Win rate: {result['win_rate']:.1%}")
    print(f"Total return: {result['total_return']:.1%}")
    print(f"Sharpe ratio: {result['sharpe']:.2f}")
    print(f"Avg win: {result['avg_win']:.2%}")
    print(f"Avg loss: {result['avg_loss']:.2%}")
    print()
    
    # Show trades
    print("Trades:")
    for _, trade in result['trades'].iterrows():
        print(f"  {trade['entry_date'].strftime('%Y-%m-%d')} -> {trade['exit_date'].strftime('%Y-%m-%d')}: "
              f"{trade['symbol']:<12} {trade['return_pct']:+.1%} ({trade['reason']})")
    
    print("="*60)


if __name__ == "__main__":
    main()
