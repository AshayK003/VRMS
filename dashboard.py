"""Streamlit dashboard for VRMS.

Free, institutional-grade swing trading signals for Indian equity markets.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Page config
st.set_page_config(
    page_title="VRMS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)


# ─── Session State ───────────────────────────────────────────────────────────

def init_session_state():
    """Initialize session state variables."""
    if "signals" not in st.session_state:
        st.session_state.signals = []
    if "regime" not in st.session_state:
        st.session_state.regime = "NORMAL"
    if "vix" not in st.session_state:
        st.session_state.vix = 14.2
    if "adx" not in st.session_state:
        st.session_state.adx = 28.5
    if "equity_curve" not in st.session_state:
        st.session_state.equity_curve = [1.0]
    if "win_rate" not in st.session_state:
        st.session_state.win_rate = 0.0
    if "sharpe" not in st.session_state:
        st.session_state.sharpe = 0.0
    if "max_drawdown" not in st.session_state:
        st.session_state.max_drawdown = 0.0
    if "paper_trades" not in st.session_state:
        st.session_state.paper_trades = []
    if "last_update" not in st.session_state:
        st.session_state.last_update = None


# ─── Data Loading ────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_sample_data() -> dict:
    """Load sample data for demonstration.
    
    In production, this will pull from the live pipeline.
    """
    # Sample signals
    signals = [
        {
            "symbol": "TATAMOTORS",
            "direction": "LONG",
            "probability": 0.72,
            "score": 0.68,
            "momentum": 0.08,
            "rs": 1.15,
            "stop_loss_pct": 0.05,
            "target_pct": 0.05,
            "risk_multiplier": 1.0,
        },
        {
            "symbol": "INFY",
            "direction": "LONG",
            "probability": 0.68,
            "score": 0.64,
            "momentum": 0.06,
            "rs": 1.12,
            "stop_loss_pct": 0.05,
            "target_pct": 0.05,
            "risk_multiplier": 1.0,
        },
        {
            "symbol": "RELIANCE",
            "direction": "LONG",
            "probability": 0.65,
            "score": 0.61,
            "momentum": 0.05,
            "rs": 1.08,
            "stop_loss_pct": 0.05,
            "target_pct": 0.05,
            "risk_multiplier": 1.0,
        },
        {
            "symbol": "HDFCBANK",
            "direction": "LONG",
            "probability": 0.63,
            "score": 0.59,
            "momentum": 0.04,
            "rs": 1.05,
            "stop_loss_pct": 0.05,
            "target_pct": 0.05,
            "risk_multiplier": 1.0,
        },
        {
            "symbol": "ICICIBANK",
            "direction": "LONG",
            "probability": 0.61,
            "score": 0.57,
            "momentum": 0.03,
            "rs": 1.02,
            "stop_loss_pct": 0.05,
            "target_pct": 0.05,
            "risk_multiplier": 1.0,
        },
    ]
    
    # Sample equity curve
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    
    # Sample metrics
    metrics = {
        "win_rate": 0.58,
        "sharpe": 1.23,
        "max_drawdown": 0.12,
        "total_return": 0.18,
        "n_trades": 45,
        "n_wins": 26,
        "n_losses": 19,
        "deflated_sharpe": 1.15,
        "win_rate_lower": 0.52,
        "win_rate_upper": 0.64,
    }
    
    return {
        "signals": signals,
        "equity_curve": equity,
        "metrics": metrics,
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


def render_regime_gauge(vix: float, adx: float, regime: str):
    """Render regime indicator."""
    # Regime color
    regime_colors = {
        "LOW-VOL": "#22c55e",
        "NORMAL": "#3b82f6",
        "HIGH-VOL": "#f59e0b",
        "SPIKE": "#ef4444",
    }
    color = regime_colors.get(regime, "#64748b")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Regime", regime)
    with col2:
        st.metric("VIX", f"{vix:.1f}")
    with col3:
        st.metric("ADX", f"{adx:.1f}")
    with col4:
        st.metric("Signal Date", datetime.now().strftime("%Y-%m-%d"))
    
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
        st.info("No signals today. Market conditions are unfavorable.")
        return
    
    # Convert to DataFrame for display
    df = pd.DataFrame(signals)
    
    # Format columns
    if not df.empty:
        df['probability'] = df['probability'].apply(lambda x: f"{x:.0%}")
        df['momentum'] = df['momentum'].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "—")
        df['rs'] = df['rs'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        df['stop_loss_pct'] = df['stop_loss_pct'].apply(lambda x: f"-{x:.0%}")
        df['target_pct'] = df['target_pct'].apply(lambda x: f"+{x:.0%}")
    
    # Rename columns for display
    df = df.rename(columns={
        'symbol': 'Symbol',
        'probability': 'Probability',
        'momentum': 'Momentum',
        'rs': 'Rel. Strength',
        'stop_loss_pct': 'Stop Loss',
        'target_pct': 'Target',
    })
    
    # Select display columns
    display_cols = ['Symbol', 'Probability', 'Momentum', 'Rel. Strength', 'Stop Loss', 'Target']
    df = df[[c for c in display_cols if c in df.columns]]
    
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", width="medium"),
            "Probability": st.column_config.TextColumn("Prob", width="small"),
            "Momentum": st.column_config.TextColumn("Momentum", width="small"),
            "Rel. Strength": st.column_config.TextColumn("RS", width="small"),
            "Stop Loss": st.column_config.TextColumn("SL", width="small"),
            "Target": st.column_config.TextColumn("Target", width="small"),
        },
    )


def render_equity_curve(equity_curve: list[float]):
    """Render equity curve with drawdown."""
    st.subheader("Equity Curve (Walk-Forward)")
    
    if len(equity_curve) < 2:
        st.info("Not enough data to display equity curve.")
        return
    
    df = pd.DataFrame({
        "Date": pd.date_range(start="2025-01-01", periods=len(equity_curve), freq="B"),
        "Equity": equity_curve,
    })
    
    # Calculate drawdown
    peak = df["Equity"].cummax()
    df["Drawdown"] = (df["Equity"] - peak) / peak
    
    # Two charts: equity + drawdown
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.line_chart(df.set_index("Date")["Equity"], use_container_width=True)
    
    with col2:
        # Drawdown area
        st.area_chart(df.set_index("Date")["Drawdown"], use_container_width=True, color="#ef4444")


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
    
    # Confidence intervals
    st.markdown(
        f"""
        <div style="padding: 1rem; background: #1e293b; border-radius: 8px; margin-top: 0.5rem;">
            <div style="display: flex; justify-content: space-between; font-size: 0.875rem;">
                <span>Deflated Sharpe: <b>{metrics['deflated_sharpe']:.2f}</b></span>
                <span>Win Rate CI: <b>{metrics['win_rate_lower']:.0%} – {metrics['win_rate_upper']:.0%}</b></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_win_rate_by_regime():
    """Render win rate breakdown by regime."""
    st.subheader("Win Rate by Regime")
    
    regime_data = {
        "Low-Vol": 0.62,
        "Normal": 0.58,
        "High-Vol": 0.45,
        "Spike": 0.35,
    }
    
    df = pd.DataFrame(
        list(regime_data.items()),
        columns=["Regime", "Win Rate"],
    )
    
    # Color coding
    colors = []
    for val in df["Win Rate"]:
        if val >= 0.6:
            colors.append("#22c55e")
        elif val >= 0.5:
            colors.append("#3b82f6")
        elif val >= 0.4:
            colors.append("#f59e0b")
        else:
            colors.append("#ef4444")
    
    st.bar_chart(
        df.set_index("Regime")["Win Rate"],
        use_container_width=True,
        color=colors,
    )


def render_paper_trading_tracker():
    """Render paper trading tracker."""
    st.subheader("Paper Trading")
    
    trades = st.session_state.paper_trades
    
    if not trades:
        st.info("No paper trades yet. Signals will be tracked automatically.")
        return
    
    df = pd.DataFrame(trades)
    st.dataframe(df, hide_index=True, use_container_width=True)


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    """Main app entry point."""
    init_session_state()
    
    # Load data
    data = load_sample_data()
    signals = data["signals"]
    equity_curve = data["equity_curve"]
    metrics = data["metrics"]
    
    # Header
    render_header()
    
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
        
        st.markdown("**Filters**")
        filter_vix = st.checkbox("Filter VIX > 22", value=True)
        filter_adx = st.checkbox("Filter ADX < 15", value=True)
        filter_expiry = st.checkbox("Block Expiry Day", value=True)
        
        st.markdown("---")
        st.markdown("**Last Update**")
        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.markdown(f"`{st.session_state.last_update}`")
        
        if st.button("Refresh Signals", use_container_width=True):
            st.rerun()
    
    # Main content
    # Row 1: Regime gauge
    render_regime_gauge(
        vix=st.session_state.vix,
        adx=st.session_state.adx,
        regime=st.session_state.regime,
    )
    
    st.markdown("---")
    
    # Row 2: Signals table
    render_signals_table(signals)
    
    st.markdown("---")
    
    # Row 3: Equity curve + metrics
    col1, col2 = st.columns([2, 1])
    with col1:
        render_equity_curve(equity_curve)
    with col2:
        render_metrics(metrics)
    
    st.markdown("---")
    
    # Row 4: Win rate by regime + paper trading
    col1, col2 = st.columns(2)
    with col1:
        render_win_rate_by_regime()
    with col2:
        render_paper_trading_tracker()
    
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
