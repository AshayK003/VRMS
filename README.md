# VRMS — Volatility Regime Momentum Scanner

**Free, open-source swing trading signal generator for Indian equity markets.**

VRMS uses VIX regime detection + momentum screening to generate institutional-grade swing trading signals for Nifty 50 stocks. No paid APIs, no subscriptions — just free data and transparent logic.

![License](https://img.shields.io/badge/license-AGPL%20v3-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-green)
![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen)

## Quick Start

```bash
git clone https://github.com/AshayK003/VRMS.git
cd VRMS
pip install -r requirements.txt
python run_screener.py
```

## Example Output

```
Fetching VIX data...
Current VIX: 11.7
VIX Regime: complacency (SELL zone)

Scanning Nifty 50 stocks...

No high-conviction picks found. Market conditions are unfavorable.
Wait for VIX > 18 (fear) or strong momentum in neutral zone.
```

When VIX > 18 (fear zone):

```
============================================================
MULTI-ASSET VIX SCREENER — BACKTEST RESULTS
============================================================
Period: 2023-01-01 to 2024-12-31
Target: 5% | Stop loss: 3% | Transaction cost: 0.3%

Total trades: 105
Win rate: 65.7%
Total return: 505%
Sharpe ratio: 0.78
============================================================
```

## Features

- **VIX Regime Detection** — Fear (>18), neutral (14-18), complacency (<14)
- **Momentum Screening** — MA alignment + ADX + rate of change
- **Corporate Action Filtering** — Demergers, splits, bonuses excluded from features
- **Paper Trading** — Track hypothetical P&L without real money
- **Walk-Forward Backtest** — Expanding-window validation, no look-ahead bias
- **Live Dashboard** — Streamlit UI with real-time signals
- **Daily Automation** — Cron job for daily signal generation

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        VRMS Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                  │
│  ├── yfinance (OHLCV) → Corporate Action Filter             │
│  ├── VIX (^INDIAVIX) → Regime Detection                     │
│  └── Nifty 50 Constituents (point-in-time)                  │
├─────────────────────────────────────────────────────────────┤
│  Feature Engineering                                         │
│  ├── Realized Volatility (5/10/20d)                         │
│  ├── Momentum (21/63d)                                      │
│  ├── Relative Strength (vs Nifty)                           │
│  ├── ADX, RSI, ATR                                          │
│  ├── Volume Ratio, Circuit Flag                             │
│  └── GARCH(1,1) Volatility                                  │
├─────────────────────────────────────────────────────────────┤
│  Model Layer                                                 │
│  ├── PCA (10 components)                                    │
│  └── XGBoost Classifier                                     │
├─────────────────────────────────────────────────────────────┤
│  Output Layer                                                │
│  ├── Top 5 Signals (entry/target/stop)                      │
│  ├── Paper Trading Tracker                                  │
│  └── Streamlit Dashboard                                    │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
VRMS/
├── src/
│   ├── data/           # OHLCV, corporate actions, constituents
│   ├── features/       # 8 features + PCA
│   ├── models/         # XGBoost classifier
│   ├── screener/       # Multi-asset VIX + momentum screener
│   ├── strategies/     # 4 trading strategies
│   └── backtest/       # Walk-forward engine
├── tests/              # 53 pytest tests
├── dashboard.py        # Streamlit UI
├── run_screener.py     # CLI screener
├── run_paper_trader.py # Paper trading
├── daily_cron.py       # Daily automation
└── requirements.txt
```

## Configuration

```python
# src/screener/multi_asset.py
DEFAULT_PARAMS = {
    'vix_high': 18,        # Fear threshold
    'vix_low': 14,         # Complacency threshold
    'adx_threshold': 25,   # Min ADX for trending
    'target_pct': 0.05,    # 5% target
    'stop_loss_pct': 0.03, # 3% stop loss
}
```

## Daily Automation

```bash
python daily_cron.py              # Run daily update
python run_paper_trader.py --status   # View positions
python run_paper_trader.py --history  # View closed trades
```

## Testing

```bash
python -m pytest tests/ -v
```

## Results

| Metric | Value |
|--------|-------|
| Trades | 105 |
| Win Rate | 65.7% |
| Total Return | 505% |
| Sharpe | 0.78 |

*Period: 2023-01-01 to 2024-12-31. Past performance ≠ future results.*

## License

AGPL v3 — Free and open source.

## Disclaimer

Research tool, not financial advice. SEBI data shows 70% of intraday traders lose money. Always do your own research.
