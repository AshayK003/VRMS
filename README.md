# VRMS — Volatility Regime Momentum Scanner

**Free, open-source swing trading signal generator for Indian equity markets.**

VRMS uses VIX regime detection + momentum screening to generate institutional-grade swing trading signals for Nifty 50 stocks. No paid APIs, no subscriptions — just free data and transparent logic.

![CI](https://github.com/AshayK003/VRMS/workflows/VRMS%20CI/badge.svg)
![License](https://img.shields.io/badge/license-AGPL%20v3-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-green)

## Quick Start

```bash
# Clone
git clone https://github.com/AshayK003/VRMS.git
cd VRMS

# Install
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Run screener
python run_screener.py

# Run dashboard
streamlit run dashboard.py
```

## What It Does

1. **VIX Regime Detection** — Classifies market as fear (>18), neutral (14-18), or complacency (<14)
2. **Momentum Screening** — Ranks Nifty 50 stocks by conviction score (MA alignment + ADX + momentum)
3. **Signal Generation** — Top 5 picks with entry/target/stop-loss levels
4. **Paper Trading** — Track hypothetical P&L without real money
5. **Walk-Forward Backtest** — Expanding-window validation with no look-ahead bias

## Results

| Metric | Value |
|--------|-------|
| Backtest trades | 105 |
| Win rate | 65.7% |
| Total return | 505% |
| Sharpe ratio | 0.78 |

*Period: 2023-01-01 to 2024-12-31. Past performance ≠ future results.*

## Architecture

```
Data (yfinance) → Corporate Action Filter → Feature Engineering → XGBoost → Signals
                     ↓
              VIX Regime Detection → Momentum Screening → Paper Trading
```

## Project Structure

```
VRMS/
├── src/
│   ├── data/           # OHLCV fetching, corporate actions, constituents
│   ├── features/       # Feature engineering (8 features + PCA)
│   ├── models/         # XGBoost classifier
│   ├── screener/       # Multi-asset VIX + momentum screener
│   ├── strategies/     # 4 trading strategies
│   └── backtest/       # Walk-forward backtest engine
├── tests/              # 53 pytest tests
├── dashboard.py        # Streamlit dashboard
├── run_screener.py     # CLI screener
├── run_paper_trader.py # Paper trading tracker
├── daily_cron.py       # Daily update job
└── requirements.txt
```

## Configuration

Edit `src/screener/multi_asset.py`:

```python
DEFAULT_PARAMS = {
    'vix_high': 18,       # VIX fear threshold
    'vix_low': 14,        # VIX complacency threshold
    'adx_threshold': 25,  # Minimum ADX for trending markets
    'target_pct': 0.05,   # 5% target
    'stop_loss_pct': 0.03, # 3% stop loss
}
```

## Daily Automation

```bash
# Run daily (after market close)
python daily_cron.py

# View paper trading status
python run_paper_trader.py --status

# View trade history
python run_paper_trader.py --history
```

## Testing

```bash
python -m pytest tests/ -v
```

## License

AGPL v3 — Free and open source. Use at your own risk. For educational purposes only.

## Disclaimer

This is a research tool, not financial advice. Past performance does not guarantee future results. SEBI data shows 70% of intraday traders lose money. Always do your own research.
