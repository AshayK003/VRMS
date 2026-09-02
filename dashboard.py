"""Streamlit dashboard for VRMS — wired to live pipeline.

Free, institutional-grade swing trading signals for Indian equity markets.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.screener.multi_asset import (
    fetch_vix, NIFTY_50,
    get_vix_regime, compute_momentum, compute_conviction,
)
from src.data.ohlcv import fetch_ohlcv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="VRMS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Live Data ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_live_signals(params: dict) -> tuple[list[dict], str, float]:
    """Get live signals from screener.
    
    Returns:
        Tuple of (signals_list, vix_regime, current_vix)
    """
    vix_df = fetch_vix(period="6mo")
    if vix_df.empty:
        return [], "unknown", 0.0

    vix_regime, current_vix = get_vix_regime(vix_df, params)

    if vix_regime == "complacency":
        return [], vix_regime, current_vix

    # Scan stocks
    picks = []
    for symbol in NIFTY_50:
        df = fetch_stock_data(symbol, period="6mo")
        if df.empty or len(df) < 50:
            continue

        mom = compute_momentum(df)
        if not mom:
            continue

        conviction, reason = compute_conviction(mom, vix_regime, params)

        if conviction < 30:
            continue

        price = mom['price']
        picks.append({
            'symbol': symbol.replace('.NS', ''),
            'conviction': conviction,
            'price': price,
            'target': price * (1 + params['target_pct']),
            'stop': price * (1 - params['stop_loss_pct']),
            'reason': reason,
            'mom_20': mom['roc_20'],
            'adx': mom['adx'],
            'ma20': mom['ma20'],
            'ma50': mom['ma50'],
        })

    picks.sort(key=lambda x: x['conviction'], reverse=True)
    return picks[:5], vix_regime, current_vix


@st.cache_data(ttl=300)
def fetch_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch stock data with fallback suffixes."""
    import yfinance as yf
    candidates = [symbol, f"{symbol}.NS", f"{symbol}.BO"]
    for sym in candidates:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period)
            if df is not None and not df.empty and len(df) >= 50:
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
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=300)
def get_backtest_metrics() -> dict:
    """Get backtest metrics from latest run."""
    # Load from saved results if available
    results_path = Path("data/backtest_results.csv")
    if results_path.exists():
        try:
            df = pd.read_csv(results_path)
            if not df.empty:
                wins = df[df['return_pct'] > 0]
                losses = df[df['return_pct'] <= 0]
                return {
                    'n_trades': len(df),
                    'win_rate': len(wins) / len(df) if len(df) > 0 else 0,
                    'total_return': (1 + df['return_pct']).prod() - 1,
                    'sharpe': df['return_pct'].mean() / df['return_pct'].std() * np.sqrt(252) if df['return_pct'].std() > 0 else 0,
                    'max_drawdown': calculate_max_drawdown(df['return_pct'].values),
                    'avg_win': wins['return_pct'].mean() if len(wins) > 0 else 0,
                    'avg_loss': losses['return_pct'].mean() if len(losses) > 0 else 0,
                    'n_wins': len(wins),
                    'n_losses': len(losses),
                }
        except Exception:
            pass

    # Default: run quick backtest
    return run_quick_backtest()


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns."""
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(np.min(drawdown))


@st.cache_data(ttl=300)
def run_quick_backtest() -> dict:
    """Run a quick backtest for dashboard display."""
    from run_multi_asset_backtest import backtest_multi_asset
    result = backtest_multi_asset()
    
    if "error" in result:
        return {
            'n_trades': 0, 'win_rate': 0, 'total_return': 0,
            'sharpe': 0, 'max_drawdown': 0, 'avg_win': 0, 'avg_loss': 0,
            'n_wins': 0, 'n_losses': 0,
        }
    
    return {
        'n_trades': result['n_trades'],
        'win_rate': result['win_rate'],
        'total_return': result['total_return'],
        'sharpe': result['sharpe'],
        'max_drawdown': 0.0,  # Not computed in backtest
        'avg_win': result['avg_win'],
        'avg_loss': result['avg_loss'],
        'n_wins': result['n_trades'] - len(result['trades'][result['trades']['return_pct'] <= 0]),
        'n_losses': len(result['trades'][result['trades']['return_pct'] <= 0]),
    }


# ─── Components ──────────────────────────────────────────────────────────────

def render_header():
    """Render the header section."""
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="margin: 0;">VRMS</h1>
            <p style="color: #64748b; margin: 0;">
                Volatility Regime Momentum Scanner
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_regime_gauge(vix: float, regime: str):
    """Render regime indicator."""
    regime_colors = {
        "complacency": "#22c55e",
        "neutral": "#3b82f6",
        "fear": "#f59e0b",
        "unknown": "#64748b",
    }
    color = regime_colors.get(regime, "#64748b")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Regime", regime.upper())
    with col2:
        st.metric("VIX", f"{vix:.1f}")
    with col3:
        st.metric("Signal Date", datetime.now().strftime("%Y-%m-%d"))
    with col4:
        st.metric("Stocks Scanned", "50")
    
    # Regime indicator bar
    st.markdown(
        f"""
        <div style="
            height: 4px;
            background: linear-gradient(90deg, #22c55e 0%, #3b82f6 33%, #f59e0b 66%, #ef4444 100%);
            border-radius: 2px;
            margin: 0.5rem 0;
            position: relative;
        ">
            <div style="
                position: absolute;
                top: -4px;
                left: {min(max((vix - 10) / 30 * 100, 0), 100)}%;
                width: 12px;
                height: 12px;
                background: white;
                border: 2px solid {color};
                border-radius: 50%;
                transform: translateX(-50%);
            "></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signals_table(signals: list[dict]):
    """Render the daily signals table."""
    st.subheader("Today's Top 5 Signals")
    
    if not signals:
        st.info("No signals today. Market conditions are unfavorable. Wait for VIX > 18 (fear) or strong momentum in neutral zone.")
        return
    
    df = pd.DataFrame(signals)
    
    if not df.empty:
        df['price'] = df['price'].apply(lambda x: f"₹{x:.2f}")
        df['target'] = df['target'].apply(lambda x: f"₹{x:.2f}")
        df['stop'] = df['stop'].apply(lambda x: f"₹{x:.2f}")
        df['mom_20'] = df['mom_20'].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
        df['adx'] = df['adx'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
        df['conviction'] = df['conviction'].apply(lambda x: f"{x:.0f}/100")
    
    df = df.rename(columns={
        'symbol': 'Symbol',
        'conviction': 'Conviction',
        'price': 'Entry',
        'target': 'Target',
        'stop': 'Stop Loss',
        'mom_20': 'Momentum',
        'adx': 'ADX',
        'reason': 'Reason',
    })
    
    display_cols = ['Symbol', 'Conviction', 'Entry', 'Target', 'Stop Loss', 'Momentum', 'ADX', 'Reason']
    df = df[[c for c in display_cols if c in df.columns]]
    
    st.dataframe(
        df,
        hide_index=True,
        width='stretch',
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", width="medium"),
            "Conviction": st.column_config.TextColumn("Conviction", width="small"),
            "Entry": st.column_config.TextColumn("Entry", width="small"),
            "Target": st.column_config.TextColumn("Target", width="small"),
            "Stop Loss": st.column_config.TextColumn("SL", width="small"),
            "Momentum": st.column_config.TextColumn("Momentum", width="small"),
            "ADX": st.column_config.TextColumn("ADX", width="small"),
            "Reason": st.column_config.TextColumn("Reason", width="large"),
        },
    )


def render_metrics(metrics: dict):
    """Render key performance metrics."""
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Win Rate", f"{metrics['win_rate']:.0%}")
    with col2:
        st.metric("Sharpe", f"{metrics['sharpe']:.2f}")
    with col3:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.0%}")
    with col4:
        st.metric("Total Return", f"{metrics['total_return']:.0%}")
    with col5:
        st.metric("Trades", f"{metrics['n_trades']}")
    
    if metrics['n_trades'] > 0:
        st.markdown(
            f"""
            <div style="padding: 1rem; background: #1e293b; border-radius: 8px; margin-top: 0.5rem;">
                <div style="display: flex; justify-content: space-between; font-size: 0.875rem;">
                    <span>Wins: <b>{metrics['n_wins']}</b> | Losses: <b>{metrics['n_losses']}</b></span>
                    <span>Avg Win: <b>{metrics['avg_win']:+.2%}</b> | Avg Loss: <b>{metrics['avg_loss']:+.2%}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_equity_curve():
    """Render equity curve from backtest results."""
    st.subheader("Equity Curve (Backtest)")
    
    results_path = Path("data/backtest_results.csv")
    if results_path.exists():
        try:
            df = pd.read_csv(results_path)
            if not df.empty:
                # Build equity curve from trades
                equity = [1.0]
                for _, trade in df.iterrows():
                    equity.append(equity[-1] * (1 + trade['return_pct']))
                
                df_eq = pd.DataFrame({
                    "Trade": range(len(equity)),
                    "Equity": equity,
                })
                
                # Calculate drawdown
                peak = df_eq["Equity"].cummax()
                df_eq["Drawdown"] = (df_eq["Equity"] - peak) / peak
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.line_chart(df_eq.set_index("Trade")["Equity"], width='stretch')
                with col2:
                    st.area_chart(df_eq.set_index("Trade")["Drawdown"], width='stretch', color="#ef4444")
                return
        except Exception:
            pass
    
    st.info("Run backtest to see equity curve.")


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    """Main app entry point."""
    # Sidebar
    with st.sidebar:
        st.markdown("### VRMS")
        st.markdown("---")
        
        st.markdown("**Settings**")
        risk_budget = st.slider(
            "Risk Budget (₹)",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
        )
        
        st.markdown("**Screener Parameters**")
        vix_high = st.slider("VIX High", 15.0, 25.0, 18.0, 0.5)
        vix_low = st.slider("VIX Low", 10.0, 18.0, 14.0, 0.5)
        adx_threshold = st.slider("ADX Threshold", 15, 40, 25)
        target_pct = st.slider("Target %", 0.02, 0.10, 0.05, 0.01)
        stop_loss_pct = st.slider("Stop Loss %", 0.02, 0.08, 0.03, 0.01)
        
        st.markdown("---")
        st.markdown("**Last Update**")
        st.markdown(f"`{datetime.now().strftime('%Y-%m-%d %H:%M')}`")
        
        if st.button("Refresh Signals", width='stretch'):
            st.cache_data.clear()
            st.rerun()
    
    # Get parameters
    params = {
        'vix_high': vix_high,
        'vix_low': vix_low,
        'adx_threshold': adx_threshold,
        'target_pct': target_pct,
        'stop_loss_pct': stop_loss_pct,
    }
    
    # Load live data
    signals, vix_regime, current_vix = get_live_signals(params)
    metrics = get_backtest_metrics()
    
    # Header
    render_header()
    
    # Row 1: Regime gauge
    render_regime_gauge(vix=current_vix, regime=vix_regime)
    
    st.markdown("---")
    
    # Row 2: Signals table
    render_signals_table(signals)
    
    st.markdown("---")
    
    # Row 3: Equity curve + metrics
    col1, col2 = st.columns([2, 1])
    with col1:
        render_equity_curve()
    with col2:
        render_metrics(metrics)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #64748b; font-size: 0.75rem;">
            VRMS — Volatility Regime Momentum Scanner • AGPL v3 • For educational purposes only
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
