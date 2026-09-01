"""VIX-Filtered Momentary Strategy for Indian Markets.

Combines VIX regime awareness with momentum edge:
- VIX > 18 (fear) + Momentum positive = BUY (high conviction)
- VIX < 12 (complacency) = SELL/avoid
- VIX 12-18 + Momentum positive = BUY (lower conviction)

Momentum signals:
- Price > 20-day MA (short-term trend)
- Price > 50-day MA (medium-term trend)
- ADX > 25 (trending market)
- Relative strength vs Nifty > 1.0 (outperforming)
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


def fetch_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance.
    
    Args:
        symbol: Yahoo ticker (e.g., '^NSEI', 'RELIANCE.NS')
        period: Data period
        
    Returns:
        DataFrame with Date, Open, High, Low, Close, Volume
    """
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
        
        # Ensure numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename Close to VIX for VIX data
        if 'INDIAVIX' in symbol or 'VIX' in symbol:
            df = df.rename(columns={'Close': 'VIX'})
            return df[['VIX']]
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        logger.error(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def compute_momentum_features(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute momentum features.
    
    Args:
        df: Stock OHLCV DataFrame
        benchmark: Benchmark OHLCV DataFrame (e.g., Nifty)
        
    Returns:
        DataFrame with momentum features
    """
    result = pd.DataFrame(index=df.index)
    
    # Moving averages
    result['MA20'] = df['Close'].rolling(20).mean()
    result['MA50'] = df['Close'].rolling(50).mean()
    
    # Price vs MAs
    result['above_MA20'] = (df['Close'] > result['MA20']).astype(int)
    result['above_MA50'] = (df['Close'] > result['MA50']).astype(int)
    
    # MA crossover (golden cross / death cross)
    result['MA_cross'] = (result['MA20'] > result['MA50']).astype(int)
    
    # Rate of change
    result['ROC_10'] = df['Close'].pct_change(10)
    result['ROC_20'] = df['Close'].pct_change(20)
    
    # ADX (trend strength)
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
    
    # Relative strength vs benchmark
    if benchmark is not None:
        common_idx = df.index.intersection(benchmark.index)
        if len(common_idx) > 20:
            stock_returns = df.loc[common_idx, 'Close'].pct_change(20)
            bench_returns = benchmark.loc[common_idx, 'Close'].pct_change(20)
            rs = stock_returns / (bench_returns + 1e-10)
            result['RS'] = rs.reindex(df.index)
        else:
            result['RS'] = 1.0
    else:
        result['RS'] = 1.0
    
    return result


def generate_signals(
    nifty_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    vix_high: float = 18.0,
    vix_low: float = 12.0,
    adx_threshold: float = 25.0,
) -> list[Signal]:
    """Generate trading signals based on VIX + momentum.
    
    Args:
        nifty_df: Nifty OHLCV DataFrame
        vix_df: VIX history
        vix_high: VIX threshold for fear (BUY zone)
        vix_low: VIX threshold for complacency (SELL zone)
        adx_threshold: ADX threshold for trending market
        
    Returns:
        List of Signal objects
    """
    # Compute momentum features
    mom_df = compute_momentum_features(nifty_df, nifty_df)
    
    # Merge with VIX
    merged = pd.concat([mom_df, vix_df], axis=1, join='inner')
    merged = merged.dropna()
    
    if merged.empty:
        return []
    
    signals = []
    
    for date, row in merged.iterrows():
        vix = row['VIX']
        adx = row.get('ADX', 0)
        above_ma20 = row.get('above_MA20', 0)
        above_ma50 = row.get('above_MA50', 0)
        ma_cross = row.get('MA_cross', 0)
        roc_20 = row.get('ROC_20', 0)
        
        direction = "HOLD"
        confidence = 0.0
        reason = ""
        position_size = 0.0
        
        # VIX > high threshold → Fear zone → Look for BUY
        if vix > vix_high:
            # Check momentum confirmation
            if above_ma20 and above_ma50 and ma_cross:
                direction = "BUY"
                confidence = 0.8
                reason = f"VIX={vix:.1f} (fear) + Momentum bullish (MA20>MA50, ADX={adx:.0f})"
                position_size = 0.8
            elif above_ma20 and above_ma50:
                direction = "BUY"
                confidence = 0.6
                reason = f"VIX={vix:.1f} (fear) + Price above MA20/MA50"
                position_size = 0.6
            elif adx > adx_threshold and roc_20 > 0:
                direction = "BUY"
                confidence = 0.5
                reason = f"VIX={vix:.1f} (fear) + Strong trend (ADX={adx:.0f})"
                position_size = 0.5
            else:
                direction = "HOLD"
                reason = f"VIX={vix:.1f} (fear) but no momentum confirmation"
        
        # VIX < low threshold → Complacency → SELL
        elif vix < vix_low:
            if not above_ma20 or not above_ma50:
                direction = "SELL"
                confidence = 0.7
                reason = f"VIX={vix:.1f} (complacent) + Momentum bearish"
                position_size = 0.7
            else:
                direction = "SELL"
                confidence = 0.5
                reason = f"VIX={vix:.1f} (complacent) — take profits"
                position_size = 0.5
        
        # VIX neutral → Trade momentum only
        else:
            if above_ma20 and above_ma50 and ma_cross and adx > adx_threshold:
                direction = "BUY"
                confidence = 0.4
                reason = f"VIX neutral + Strong momentum (ADX={adx:.0f})"
                position_size = 0.4
            elif not above_ma20 and not above_ma50:
                direction = "SELL"
                confidence = 0.4
                reason = f"VIX neutral + Momentum bearish"
                position_size = 0.4
            else:
                direction = "HOLD"
                reason = f"VIX neutral, no clear momentum"
        
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
    holding_limit: int = 20,
) -> tuple[list[Trade], pd.DataFrame]:
    """Run backtest on signals."""
    trades = []
    equity_curve = []
    current_position = None
    
    for signal in signals:
        if signal.direction == "HOLD":
            continue
        
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
            
            if price >= current_position['target']:
                exit_price = current_position['target']
                exit_reason = "Target hit"
            elif price <= current_position['stop']:
                exit_price = current_position['stop']
                exit_reason = "Stop loss hit"
            elif current_position['days_held'] >= holding_limit:
                exit_price = price
                exit_reason = "Holding limit reached"
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
