"""Directional VIX Strategy for Indian Markets.

This strategy is DIRECTIONAL, not mean-reversion:
- VIX > 18 (fear) → BUY Nifty (fear is temporary, buy the dip)
- VIX < 12 (complacency) → SELL/exit (complacency is dangerous)
- VIX 12-18 → HOLD (no edge, stay invested)

Key insight: Indian markets have a bullish bias. Nifty goes up over time.
Buying when VIX is high (fear) captures the recovery.
Selling when VIX is low (complacency) avoids corrections.
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
    confidence: float
    reason: str
    position_size: float


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


def fetch_data(symbol: str, period: str = "5y") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        date_col = 'Date' if 'Date' in df.columns else 'Datetime'
        df = df.rename(columns={
            date_col: 'Date',
            'Open': 'Open', 'High': 'High', 'Low': 'Low',
            'Close': 'Close', 'Volume': 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.set_index('Date').sort_index()
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'INDIAVIX' in symbol or 'VIX' in symbol:
            df = df.rename(columns={'Close': 'VIX'})
            return df[['VIX']]
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def generate_directional_signals(
    vix_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    vix_high: float = 18.0,
    vix_low: float = 12.0,
) -> list[Signal]:
    """Generate DIRECTIONAL signals based on VIX.
    
    Logic:
    - VIX > 18 → Fear → BUY (buy the dip)
    - VIX < 12 → Complacency → SELL (take profits, avoid correction)
    - VIX 12-18 → Neutral → HOLD (stay invested)
    """
    # Merge VIX and Nifty
    merged = pd.concat([nifty_df, vix_df], axis=1, join='inner')
    merged = merged.dropna(subset=['Close', 'VIX'])
    
    if merged.empty:
        return []
    
    signals = []
    
    for date, row in merged.iterrows():
        vix = row['VIX']
        
        direction = "HOLD"
        confidence = 0.0
        reason = ""
        position_size = 0.0
        
        # VIX > high → Fear → BUY (buy the dip)
        if vix > vix_high:
            direction = "BUY"
            # Higher VIX = more fear = more confidence to buy
            confidence = min((vix - vix_high) / 10 + 0.5, 1.0)
            reason = f"VIX={vix:.1f} (fear zone) → BUY the dip"
            position_size = confidence
        
        # VIX < low → Complacency → SELL
        elif vix < vix_low:
            direction = "SELL"
            # Lower VIX = more complacency = more confidence to sell
            confidence = min((vix_low - vix) / 5 + 0.5, 1.0)
            reason = f"VIX={vix:.1f} (complacency zone) → SELL/exit"
            position_size = confidence
        
        # VIX neutral → HOLD
        else:
            direction = "HOLD"
            confidence = 0.0
            reason = f"VIX={vix:.1f} (neutral) → HOLD"
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


def run_directional_backtest(
    signals: list[Signal],
    nifty_df: pd.DataFrame,
    transaction_cost: float = 0.003,
) -> tuple[list[Trade], pd.DataFrame]:
    """Run directional backtest.
    
    Logic:
    - On BUY signal: Enter long position
    - On SELL signal: Exit long position
    - On HOLD signal: Do nothing
    """
    trades = []
    equity_curve = []
    current_position = None
    
    for signal in signals:
        if signal.direction == "HOLD":
            continue
        
        if signal.date not in nifty_df.index:
            continue
        
        price = nifty_df.loc[signal.date, 'Close']
        
        # BUY signal → Enter long
        if signal.direction == "BUY" and current_position is None:
            current_position = {
                'entry_date': signal.date,
                'entry_price': price,
            }
        
        # SELL signal → Exit long
        elif signal.direction == "SELL" and current_position is not None:
            return_pct = (price - current_position['entry_price']) / current_position['entry_price']
            return_pct -= transaction_cost
            
            trades.append(Trade(
                entry_date=current_position['entry_date'],
                exit_date=signal.date,
                direction="LONG",
                entry_price=current_position['entry_price'],
                exit_price=price,
                return_pct=return_pct,
                reason=signal.reason,
            ))
            
            current_position = None
    
    # Close any open position at the end
    if current_position is not None:
        last_date = nifty_df.index[-1]
        last_price = nifty_df.iloc[-1]['Close']
        return_pct = (last_price - current_position['entry_price']) / current_position['entry_price']
        return_pct -= transaction_cost
        
        trades.append(Trade(
            entry_date=current_position['entry_date'],
            exit_date=last_date,
            direction="LONG",
            entry_price=current_position['entry_price'],
            exit_price=last_price,
            return_pct=return_pct,
            reason="End of period",
        ))
    
    # Calculate equity curve
    equity = 1.0
    equity_curve_data = []
    for trade in trades:
        equity *= (1 + trade.return_pct)
        equity_curve_data.append({
            'date': trade.exit_date,
            'equity': equity,
            'return': trade.return_pct,
        })
    
    return trades, pd.DataFrame(equity_curve_data)


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
