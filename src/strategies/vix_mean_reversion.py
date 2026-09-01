"""VIX Mean-Reversion Strategy — Backtestable with available data.

Since FII/DII historical data is limited, this strategy uses VIX only.
VIX mean-reverts around 13-15 in India. This is a documented anomaly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A trading signal."""
    date: datetime
    direction: str  # "BUY", "SELL", "HOLD"
    vix: float
    confidence: float  # 0-1
    reason: str
    position_size: float  # 0-1 (fraction of capital)


@dataclass
class Trade:
    """A completed trade."""
    entry_date: datetime
    exit_date: datetime
    direction: str
    entry_price: float
    exit_price: float
    return_pct: float
    reason: str


def fetch_vix_history(period: str = "5y") -> pd.DataFrame:
    """Fetch VIX history from Yahoo Finance."""
    try:
        ticker = yf.Ticker("^INDIAVIX")
        df = ticker.history(period=period)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df = df.rename(columns={'Date': 'Date', 'Close': 'VIX'})
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.set_index('Date').sort_index()
        
        return df[['VIX']]
        
    except Exception as e:
        logger.error(f"Failed to fetch VIX: {e}")
        return pd.DataFrame()


def fetch_nifty_history(period: str = "5y") -> pd.DataFrame:
    """Fetch Nifty 50 history from Yahoo Finance."""
    try:
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(period=period)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'Date', 'Open': 'Open', 'High': 'High',
            'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.set_index('Date').sort_index()
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        logger.error(f"Failed to fetch Nifty: {e}")
        return pd.DataFrame()


def generate_signals(
    vix_df: pd.DataFrame,
    vix_high: float = 18.0,
    vix_low: float = 12.0,
    vix_mean: float = 14.0,
) -> list[Signal]:
    """Generate trading signals based on VIX mean-reversion.
    
    Args:
        vix_df: VIX history
        vix_high: VIX threshold for BUY (fear)
        vix_low: VIX threshold for SELL (complacency)
        vix_mean: Long-term VIX mean
        
    Returns:
        List of Signal objects
    """
    signals = []
    
    for date, row in vix_df.iterrows():
        vix = row['VIX']
        
        direction = "HOLD"
        confidence = 0.0
        reason = ""
        position_size = 0.0
        
        # VIX > high threshold → BUY (mean-reversion: VIX will fall)
        if vix > vix_high:
            direction = "BUY"
            # Confidence scales with distance from mean
            confidence = min((vix - vix_mean) / 10, 1.0)
            reason = f"VIX at {vix:.1f} (>{vix_high}) — market fearful, mean-reversion likely"
            position_size = confidence
        
        # VIX < low threshold → SELL (mean-reversion: VIX will rise)
        elif vix < vix_low:
            direction = "SELL"
            confidence = min((vix_mean - vix) / 5, 1.0)
            reason = f"VIX at {vix:.1f} (<{vix_low}) — market complacent, mean-reversion likely"
            position_size = confidence
        
        # VIX in neutral zone → HOLD
        else:
            direction = "HOLD"
            confidence = 0.0
            reason = f"VIX at {vix:.1f} (neutral zone)"
            position_size = 0.0
        
        signals.append(Signal(
            date=date,
            direction=direction,
            vix=vix,
            confidence=confidence,
            reason=reason,
            position_size=position_size,
        ))
    
    return signals


def run_backtest(
    signals: list[Signal],
    nifty_df: pd.DataFrame,
    target_pct: float = 0.05,
    stop_loss_pct: float = 0.05,
    transaction_cost: float = 0.003,
    holding_limit: int = 20,  # Max days to hold
) -> tuple[list[Trade], pd.DataFrame]:
    """Run backtest on signals.
    
    Args:
        signals: List of signals
        nifty_df: Nifty price history
        target_pct: Target gain
        stop_loss_pct: Stop loss
        transaction_cost: Round-trip cost
        holding_limit: Max holding period
        
    Returns:
        Tuple of (trades, equity_curve)
    """
    trades = []
    equity_curve = []
    current_position = None
    
    for signal in signals:
        if signal.direction == "HOLD":
            continue
        
        # Get Nifty price on signal date
        if signal.date not in nifty_df.index:
            continue
        
        price = nifty_df.loc[signal.date, 'Close']
        
        # Open position
        if current_position is None and signal.direction == "BUY":
            current_position = {
                'entry_date': signal.date,
                'entry_price': price,
                'direction': signal.direction,
                'target': price * (1 + target_pct),
                'stop': price * (1 - stop_loss_pct),
                'days_held': 0,
            }
        
        # Close position
        elif current_position is not None:
            current_position['days_held'] += 1
            exit_price = None
            exit_reason = ""
            
            # Check target
            if price >= current_position['target']:
                exit_price = current_position['target']
                exit_reason = "Target hit"
            
            # Check stop loss
            elif price <= current_position['stop']:
                exit_price = current_position['stop']
                exit_reason = "Stop loss hit"
            
            # Check holding limit
            elif current_position['days_held'] >= holding_limit:
                exit_price = price
                exit_reason = "Holding limit reached"
            
            # Check reverse signal
            elif signal.direction == "SELL":
                exit_price = price
                exit_reason = "Reverse signal"
            
            if exit_price is not None:
                return_pct = (exit_price - current_position['entry_price']) / current_position['entry_price']
                return_pct -= transaction_cost
                
                trades.append(Trade(
                    entry_date=current_position['entry_date'],
                    exit_date=signal.date,
                    direction=current_position['direction'],
                    entry_price=current_position['entry_price'],
                    exit_price=exit_price,
                    return_pct=return_pct,
                    reason=exit_reason,
                ))
                
                current_position = None
    
    # Calculate equity curve
    equity = 1.0
    for trade in trades:
        equity *= (1 + trade.return_pct)
        equity_curve.append({
            'date': trade.exit_date,
            'equity': equity,
            'return': trade.return_pct,
        })
    
    return trades, pd.DataFrame(equity_curve)


def calculate_metrics(trades: list[Trade]) -> dict:
    """Calculate performance metrics."""
    if not trades:
        return {
            'n_trades': 0, 'win_rate': 0.0, 'total_return': 0.0,
            'sharpe': 0.0, 'max_drawdown': 0.0, 'avg_win': 0.0,
            'avg_loss': 0.0, 'profit_factor': 0.0,
        }
    
    wins = [t for t in trades if t.return_pct > 0]
    losses = [t for t in trades if t.return_pct <= 0]
    
    win_rate = len(wins) / len(trades)
    
    total_return = 1.0
    for t in trades:
        total_return *= (1 + t.return_pct)
    total_return -= 1
    
    returns = [t.return_pct for t in trades]
    if np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 / len(trades))
    else:
        sharpe = 0.0
    
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for t in trades:
        equity *= (1 + t.return_pct)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd
    
    avg_win = np.mean([t.return_pct for t in wins]) if wins else 0.0
    avg_loss = np.mean([t.return_pct for t in losses]) if losses else 0.0
    
    gross_profits = sum(t.return_pct for t in wins)
    gross_losses = abs(sum(t.return_pct for t in losses))
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    return {
        'n_trades': len(trades), 'win_rate': win_rate, 'total_return': total_return,
        'sharpe': sharpe, 'max_drawdown': max_dd, 'avg_win': avg_win,
        'avg_loss': avg_loss, 'profit_factor': profit_factor,
    }
