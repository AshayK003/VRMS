"""Combined VIX + Momentum Strategy with Optimal Parameters.

Strategy logic:
1. VIX > vix_high (fear) → BUY (high conviction, mean-reversion)
2. VIX < vix_low (complacency) → SELL (take profits)
3. VIX neutral → Trade momentum (MA crossover + ADX filter)

Parameter optimization:
- vix_high: [16, 17, 18]
- vix_low: [12, 13, 14]
- ma_fast: [10, 20]
- ma_slow: [40, 50]
- adx_threshold: [20, 25]
- target_pct: [0.03, 0.05, 0.07]
- stop_loss_pct: [0.03, 0.05, 0.07]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from typing import Callable

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


@dataclass
class BacktestResult:
    """Backtest results."""
    trades: list[Trade]
    equity_curve: pd.DataFrame
    metrics: dict


def fetch_data(symbol: str, period: str = "3y") -> pd.DataFrame:
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


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute momentum features."""
    result = pd.DataFrame(index=df.index)
    
    # Moving averages
    result['MA_fast'] = df['Close'].rolling(20).mean()
    result['MA_slow'] = df['Close'].rolling(50).mean()
    
    # Price vs MAs
    result['above_MA_fast'] = (df['Close'] > result['MA_fast']).astype(int)
    result['above_MA_slow'] = (df['Close'] > result['MA_slow']).astype(int)
    
    # MA crossover
    result['MA_cross'] = (result['MA_fast'] > result['MA_slow']).astype(int)
    
    # ADX
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    
    result['ADX'] = np.nan
    result.iloc[1:, result.columns.get_loc('ADX')] = adx.values
    
    return result


def generate_signals(
    vix_df: pd.DataFrame,
    nifty_df: pd.DataFrame,
    params: dict,
) -> list[Signal]:
    """Generate combined VIX + momentum signals."""
    vix_high = params['vix_high']
    vix_low = params['vix_low']
    adx_threshold = params['adx_threshold']
    
    # Compute momentum features
    mom_df = compute_momentum_features(nifty_df)
    
    # Merge VIX + Nifty + Momentum
    merged = pd.concat([nifty_df, vix_df, mom_df], axis=1, join='inner')
    merged = merged.dropna(subset=['Close', 'VIX', 'MA_fast', 'MA_slow', 'ADX'])
    
    if merged.empty:
        return []
    
    signals = []
    
    for date, row in merged.iterrows():
        vix = row['VIX']
        adx = row.get('ADX', 0)
        above_ma_fast = row.get('above_MA_fast', 0)
        above_ma_slow = row.get('above_MA_slow', 0)
        ma_cross = row.get('MA_cross', 0)
        
        direction = "HOLD"
        confidence = 0.0
        reason = ""
        position_size = 0.0
        
        # VIX > high → Fear → BUY (mean-reversion)
        if vix > vix_high:
            direction = "BUY"
            confidence = min((vix - vix_high) / 10 + 0.5, 1.0)
            reason = f"VIX={vix:.1f} (fear) → BUY"
            position_size = confidence
        
        # VIX < low → Complacency → SELL
        elif vix < vix_low:
            direction = "SELL"
            confidence = min((vix_low - vix) / 5 + 0.5, 1.0)
            reason = f"VIX={vix:.1f} (complacent) → SELL"
            position_size = confidence
        
        # VIX neutral → Trade momentum
        else:
            if above_ma_fast and above_ma_slow and ma_cross and adx > adx_threshold:
                direction = "BUY"
                confidence = 0.4
                reason = f"VIX neutral + Momentum (ADX={adx:.0f})"
                position_size = 0.4
            elif not above_ma_fast and not above_ma_slow and not ma_cross:
                direction = "SELL"
                confidence = 0.4
                reason = f"VIX neutral + Momentum bearish"
                position_size = 0.4
            else:
                direction = "HOLD"
                reason = f"VIX neutral, no momentum"
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
    params: dict,
) -> BacktestResult:
    """Run backtest."""
    target_pct = params['target_pct']
    stop_loss_pct = params['stop_loss_pct']
    transaction_cost = params.get('transaction_cost', 0.003)
    
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
                'target': price * (1 + target_pct),
                'stop': price * (1 - stop_loss_pct),
                'days_held': 0,
            }
        
        # SELL signal → Exit long
        elif signal.direction == "SELL" and current_position is not None:
            current_position['days_held'] += 1
            exit_price = None
            exit_reason = ""
            
            if price >= current_position['target']:
                exit_price = current_position['target']
                exit_reason = "Target hit"
            elif price <= current_position['stop']:
                exit_price = current_position['stop']
                exit_reason = "Stop loss hit"
            elif signal.direction == "SELL":
                exit_price = price
                exit_reason = signal.reason
            
            if exit_price is not None:
                return_pct = (exit_price - current_position['entry_price']) / current_position['entry_price']
                return_pct -= transaction_cost
                
                trades.append(Trade(
                    entry_date=current_position['entry_date'],
                    exit_date=signal.date,
                    direction="LONG",
                    entry_price=current_position['entry_price'],
                    exit_price=exit_price,
                    return_pct=return_pct,
                    reason=exit_reason,
                ))
                
                current_position = None
    
    # Close any open position
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
    
    equity_curve = pd.DataFrame(equity_curve_data)
    metrics = calculate_metrics(trades)
    
    return BacktestResult(trades=trades, equity_curve=equity_curve, metrics=metrics)


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


def optimize_parameters(
    nifty_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    param_grid: dict | None = None,
) -> list[dict]:
    """Find optimal parameters using grid search.
    
    Args:
        nifty_df: Nifty OHLCV
        vix_df: VIX history
        param_grid: Parameter grid to search
        
    Returns:
        List of results sorted by Sharpe ratio
    """
    if param_grid is None:
        param_grid = {
            'vix_high': [16, 17, 18],
            'vix_low': [12, 13, 14],
            'adx_threshold': [20, 25],
            'target_pct': [0.03, 0.05, 0.07],
            'stop_loss_pct': [0.03, 0.05, 0.07],
        }
    
    results = []
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(product(*values))
    
    logger.info(f"Testing {len(combinations)} parameter combinations...")
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        params['transaction_cost'] = 0.003
        
        try:
            signals = generate_signals(vix_df, nifty_df, params)
            result = run_backtest(signals, nifty_df, params)
            
            results.append({
                'params': params,
                'metrics': result.metrics,
                'trades': len(result.trades),
            })
            
        except Exception as e:
            logger.warning(f"Failed {params}: {e}")
            continue
    
    # Sort by Sharpe ratio (descending)
    results.sort(key=lambda x: x['metrics']['sharpe'], reverse=True)
    
    return results
