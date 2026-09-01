# Volatility Regime Momentum Scanner (VRMS)

> Free, institutional-grade swing trading signals for Indian equity markets.

## Overview

VRMS is a daily signal generator that produces Top 5 BUY signals for Nifty 50 stocks, filtered by volatility regime and confirmed by institutional flow.

**Core principles:**
- No look-ahead bias (expanding window validation)
- No survivorship bias (point-in-time constituents)
- Transaction-cost-aware backtesting
- Walk-forward validated performance

## Status

🚧 **Phase 1 Complete** — Foundation, Architecture, Plan
📅 **Next:** Phase 5 — Implementation (Days 2-8)

## Quick Start

```bash
# Clone
git clone https://github.com/AshayK003/VRMS.git
cd VRMS

# Install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py
```

## Project Structure

```
VRMS/
├── internals/          # Internal workflow (gitignored)
├── docs/               # Public documentation
├── src/                # Source code
├── data/               # Data storage
├── models/             # Trained models
├── features/           # Feature store
├── signals/            # Signal generation
├── backtest/           # Backtesting
├── tracker/            # Paper trading
└── dashboard.py        # Streamlit dashboard
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for full system design.

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Foundation | ✅ Complete |
| 2-3 | Strategy + Architecture | ✅ Complete |
| 4-5 | UI/UX + Implementation | 🔜 Pending |
| 6-7 | Verification + Pre-release | ⏳ Pending |
| 8 | Release | ⏳ Pending |

## License

AGPL v3 — Free to use, modify, and share. Cannot be closed-sourced.

## Disclaimer

For educational and research purposes only. Not financial advice. Past performance does not guarantee future results.
