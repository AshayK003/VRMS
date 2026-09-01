"""VIX + FII/DII Flow Strategy for Indian Markets.

Strategy logic:
1. VIX > 18 → Market fearful → BUY signal
2. VIX < 12 → Market complacent → SELL signal
3. FII/DII 5-day net flow confirms direction
4. Position size based on VIX distance from mean

This is a mean-reversion strategy that exploits the documented anomaly
that India VIX mean-reverts around 13-15.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

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
    fii_dii_net: float
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


def fetch_vix_history(period: str = "2y") -> pd.DataFrame:
    """Fetch VIX history from Yahoo Finance.
    
    Args:
        period: Data period
        
    Returns:
        DataFrame with Date, VIX columns
    """
    try:
        ticker = yf.Ticker("^INDIAVIX")
        df = ticker.history(period=period)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'Date',
            'Close': 'VIX'
        })
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.set_index('Date').sort_index()
        
        return df[['VIX']]
        
    except Exception as e:
        logger.error(f"Failed to fetch VIX: {e}")
        return pd.DataFrame()


def fetch_fii_dii_history(period: str = "2y") -> pd.DataFrame:
    """Fetch FII/DII flow history from NSE.
    
    Args:
        period: Data period
        
    Returns:
        DataFrame with Date, FII, DII, Net columns
    """
    try:
        from nsepython import nse_fiidii
        
        df = nse_fiidii()
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
        
        fii_rows = df[df['category'] == 'FII/FPI'].copy()
        dii_rows = df[df['category'] == 'DII'].copy()
        
        result = pd.DataFrame(index=df['date'].unique())
        result.index.name = 'Date'
        
        if not fii_rows.empty:
            fii_rows = fii_rows.set_index('date')
            result['FII'] = fii_rows['netValue'].astype(float)
        
        if not dii_rows.empty:
            dii_rows = dii_rows.set_index('date')
            result['DII'] = dii_rows['netValue'].astype(float)
        
        result['Net'] = result['FII'].fillna(0) + result['DII'].fillna(0)
        result = result.sort_index()
        
        return result[['FII', 'DII', 'Net']]
        
    except Exception as e:
        logger.error(f"Failed to fetch FII/DII: {e}")
        return pd.DataFrame()


def fetch_nifty_history(period: str = "2y") -> pd.DataFrame:
    """Fetch Nifty 50 history from Yahoo Finance.
    
    Args:
        period: Data period
        
    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume
    """
    try:
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(period=period)
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.set_index('Date').sort_index()
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        logger.error(f"Failed to fetch Nifty: {e}")
        return pd.DataFrame()


def generate_signals(
    vix_df: pd.DataFrame,
    fii_dii_df: pd.DataFrame,
    vix_high: float = 18.0,
    vix_low: float = 12.0,
    flow_threshold: float = 1000.0,
) -> list[Signal]:
    """Generate trading signals based on VIX + FII/DII.
    
    Args:
        vix_df: VIX history
        fii_dii_df: FII/DII flow history
        vix_high: VIX threshold for BUY
        vix_low: VIX threshold for SELL
        flow_threshold: FII/DII net flow threshold (Cr)
        
    Returns:
        List of Signal objects
    """
    signals = []
    
    # Merge VIX and FII/DII on date
    merged = pd.concat([vix_df, fii_dii_df], axis=1, join='inner')
    merged = merged.dropna()
    
    if merged.empty:
        return signals
    
    # Calculate rolling FII/DII flow (5-day)
    merged['FII_DII_5d'] = merged['Net'].rolling(5).sum()
    
    for date, row in merged.iterrows():
        vix = row['VIX']
        flow_5d = row.get('FII_DII_5d', 0)
        
        direction = "HOLD"
        confidence = 0.0
        reason = ""
        position_size = 0.0
        
        # VIX > high threshold → BUY
        if vix > vix_high:
            direction = "BUY"
            confidence = min((vix - vix_high) / 10, 1.0)  # Scale by VIX distance
            reason = f"VIX at {vix:.1f} (>{vix_high}) — market fearful"
            position_size = confidence
            
            # Confirm with FII/DII flow
            if flow_5d > flow_threshold:
                confidence = min(confidence + 0.2, 1.0)
                reason += f", FII/DII buying ₹{flow_5d:.0f}Cr"
            elif flow_5d < -flow_threshold:
                confidence = max(confidence - 0.2, 0.1)
                reason += f", FII/DII selling ₹{flow_5d:.0f}Cr (weaker)"
        
        # VIX < low threshold → SELL
        elif vix < vix_low:
            direction = "SELL"
            confidence = min((vix_low - vix) / 5, 1.0)
            reason = f"VIX at {vix:.1f} (<{vix_low}) — market complacent"
            position_size = confidence
            
            # Confirm with FII/DII flow
            if flow_5d < -flow_threshold:
                confidence = min(confidence + 0.2, 1.0)
                reason += f", FII/DII selling ₹{flow_5d:.0f}Cr"
            elif flow_5d > flow_threshold:
                confidence = max(confidence - 0.2, 0.1)
                reason += f", FII/DII buying ₹{flow_5d:.0f}Cr (weaker)"
        
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
            fii_dii_net=flow_5d,
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
) -> tuple[list[Trade], pd.DataFrame]:
    """Run backtest on signals.
    
    Args:
        signals: List of signals
        nifty_df: Nifty price history
        target_pct: Target gain
        stop_loss_pct: Stop loss
        transaction_cost: Round-trip cost
        
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
            }
        
        # Close position
        elif current_position is not None:
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
    """Calculate performance metrics.
    
    Args:
        trades: List of completed trades
        
    Returns:
        Dict with metrics
    """
    if not trades:
        return {
            'n_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
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
    
    # Max drawdown
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
        'n_trades': len(trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
    }
