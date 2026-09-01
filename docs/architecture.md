# VRMS Architecture

> System design for the Volatility Regime Momentum Scanner.

---

## System Overview

VRMS is a daily swing trading signal generator for Indian equity markets. It produces Top 5 BUY signals every morning at 8:30 AM for Nifty 50 stocks, filtered by volatility regime, confirmed by institutional flow.

**Core principle:** No look-ahead bias. No survivorship bias. Transaction-cost-aware. Walk-forward validated.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            DATA LAYER                                    │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │   nsepython  │  │   yfinance   │  │  Screener    │  │  News RSS   │ │
│  │   OHLCV      │  │   VIX, USD   │  │  Fundamentals│  │  Sentiment  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬──────┘ │
│         │                 │                 │                 │         │
│         └────────────────┴────────────────┴─────────────────┘         │
│                                      │                                   │
│                              ┌───────▼────────┐                         │
│                              │  Data Validator │                         │
│                              └───────┬────────┘                         │
└──────────────────────────────────────┼───────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────┐
│                        FEATURE ENGINEERING                               │
│                              (Expanding Window)                          │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │   Volatility    │  │    Momentum     │  │   Relative Strength     │  │
│  │   GARCH(1,1)    │  │   1M, 3M, 6M    │  │   vs Nifty, vs Sector   │  │
│  │   Realized Vol  │  │   (unadjusted)  │  │                         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │   Volume        │  │   FII/DII       │  │   Sentiment             │  │
│  │   20d ratio     │  │   5d/20d flow   │  │   SmartScore, events    │  │
│  │   Circuit flag  │  │   T-2 final     │  │   8:00 AM cutoff        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │   Regime        │  │   Fundamentals  │  │   Factors               │  │
│  │   ADX + VIX     │  │   P/E, ROE,     │  │   Market, size, mom,   │  │
│  │   HMM bull/bear │  │   Debt/Equity   │  │   vol, liquidity        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
│                                                                          │
│                                      │                                   │
│                              ┌───────▼────────┐                         │
│                              │      PCA       │                         │
│                              │   30 → 12-15   │                         │
│                              └───────┬────────┘                         │
└──────────────────────────────────────┼───────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────┐
│                         XGBOOST CLASSIFIER                               │
│                                                                          │
│  Input: 12-15 orthogonal features per stock                              │
│  Output: Probability of 5% upside in 5 days                             │
│  Training: Expanding window (2019-2024)                                  │
│  Validation: Walk-forward                                                │
│  Test: Hold-out (last 6 months, never tuned on)                          │
│  Retrain: Weekly (5 min on CPU)                                          │
│                                                                          │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────┐
│                        GOVERNANCE FILTERS                                │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │   VIX > 22      │  │   ADX < 15      │  │   A/D ratio < 0.7       │  │
│  │   Reduce 50%    │  │   Reduce 50%    │  │   Reduce 50%            │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │   FII/DII       │  │   Expiry day    │  │   Consecutive 3 losses   │  │
│  │   Contradicts   │  │   BLOCK         │  │   Kill switch            │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐                                │
│  │   Corporate     │  │   Circuit       │                                │
│  │   action day    │  │   filter day    │                                │
│  │   SKIP          │  │   SKIP          │                                │
│  └─────────────────┘  └─────────────────┘                                │
│                                                                          │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────┐
│                         SIGNAL GENERATOR                                 │
│                                                                          │
│  Filter: Low-vol or Normal-vol regime                                    │
│  Filter: FII/DII flow positive (5d)                                      │
│  Filter: Volume ratio > 1.2                                              │
│  Filter: Not corporate action day                                        │
│  Rank: Probability × momentum × relative strength                        │
│  Output: Top 5 BUY signals + stop loss + target                         │
│  Position size: risk_budget / (ATR × point_value)                        │
│                                                                          │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │
┌──────────────────────────────────────▼───────────────────────────────────┐
│                        STREAMLIT DASHBOARD                               │
│                                                                          │
│  Daily signal table (top 5)                                              │
│  Regime indicator (gauge chart)                                          │
│  Equity curve + drawdown (walk-forward)                                  │
│  Win rate by regime                                                      │
│  Deflated Sharpe + bootstrap confidence intervals                        │
│  Paper trading tracker                                                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
VRMS/
├── internals/                  # Gitignored — internal workflow
│   ├── constitution.md
│   ├── orchestrator.md
│   ├── approval-gates.md
│   ├── engineering_memory.md
│   └── session_context.json
│
├── docs/                       # Public-facing documentation
│   ├── plan.md
│   ├── architecture.md
│   └── README.md
│
├── src/                        # Source code
│   ├── data/                   # Data fetching and validation
│   │   ├── __init__.py
│   │   ├── ohlcv.py           # OHLCV data from nsepython
│   │   ├── fii_dii.py         # FII/DII data
│   │   ├── sentiment.py       # News sentiment
│   │   ├── fundamentals.py    # Screener fundamentals
│   │   └── validator.py       # Data validation layer
│   │
│   ├── features/              # Feature engineering
│   │   ├── __init__.py
│   │   ├── volatility.py      # GARCH, realized vol
│   │   ├── momentum.py        # 1M/3M/6M momentum
│   │   ├── relative_strength.py # vs Nifty, vs Sector
│   │   ├── volume.py          # Volume ratio, circuit flags
│   │   ├── regime.py          # ADX + VIX + HMM
│   │   ├── fii_dii.py         # FII/DII flow features
│   │   ├── sentiment.py       # SmartScore + event flags
│   │   ├── fundamentals.py    # P/E, ROE, Debt/Equity
│   │   ├── factors.py         # Factor decomposition
│   │   ├── pca.py             # PCA dimensionality reduction
│   │   └── engineering.py     # Main feature pipeline
│   │
│   ├── models/                # ML models
│   │   ├── __init__.py
│   │   └── xgboost.py         # XGBoost classifier
│   │
│   ├── signals/               # Signal generation
│   │   ├── __init__.py
│   │   ├── generator.py       # Signal generation logic
│   │   ├── governance.py      # Governance filters
│   │   └── sizing.py          # Position sizing
│   │
│   ├── backtest/              # Backtesting
│   │   ├── __init__.py
│   │   ├── engine.py          # Walk-forward backtest engine
│   │   ├── metrics.py         # Deflated Sharpe, bootstrap CI
│   │   └── costs.py           # Transaction cost model
│   │
│   └── tracker/               # Paper trading
│       ├── __init__.py
│       └── paper.py           # Paper trading tracker
│
├── features/                   # Feature store (generated)
│
├── models/                    # Trained models (generated)
│
├── data/                      # Data storage
│   ├── raw/                   # Raw fetched data
│   ├── processed/             # Processed data
│   ├── constituents.csv       # Point-in-time Nifty 50
│   └── corporate_actions.csv  # Corporate action calendar
│
├── dashboard.py               # Streamlit dashboard
├── requirements.txt           # Dependencies
└── .gitignore                 # Git ignore rules
```

---

## Data Flow

### Daily Signal Generation (8:30 AM)

```
1. Fetch OHLCV (previous day close) → validate
2. Fetch FII/DII (T-2 final) → validate
3. Fetch VIX, USD/INR → validate
4. Fetch sentiment (cutoff 8:00 AM) → validate
5. Compute features (expanding window) → PCA
6. XGBoost predict → probabilities
7. Governance filters → filtered signals
8. Rank by probability × momentum × RS
9. Output top 5 + SL + Target
10. Display on dashboard
```

### Weekly Retrain (Sunday 10:00 AM)

```
1. Fetch latest OHLCV for all Nifty 50
2. Update features (expanding window)
3. Retrain XGBoost (5 min on CPU)
4. Validate on hold-out set
5. Update model artifact
6. Log to engineering memory
```

---

## Feature List (30 → 12-15)

| # | Feature | Type | Source |
|---|---|---|---|
| 1 | GARCH(1,1) vol | Volatility | Computed |
| 2 | Realized vol 5d | Volatility | Computed |
| 3 | Realized vol 10d | Volatility | Computed |
| 4 | Realized vol 20d | Volatility | Computed |
| 5 | ADX | Regime | Computed |
| 6 | VIX level | Regime | yfinance |
| 7 | HMM regime | Regime | Computed |
| 8 | Momentum 1M | Momentum | Computed |
| 9 | Momentum 3M | Momentum | Computed |
| 10 | Momentum 6M | Momentum | Computed |
| 11 | RS vs Nifty | Relative Strength | Computed |
| 12 | RS vs Sector | Relative Strength | Computed |
| 13 | Volume ratio 20d | Volume | Computed |
| 14 | Circuit flag | Volume | Computed |
| 15 | FII flow 5d | FII/DII | nsepython |
| 16 | FII flow 20d | FII/DII | nsepython |
| 17 | DII flow 5d | FII/DII | nsepython |
| 18 | DII flow 20d | FII/DII | nsepython |
| 19 | FII/DII correlation | FII/DII | Computed |
| 20 | SmartScore | Sentiment | Sentiment Analyzer |
| 21-30 | 19 event flags | Sentiment | Sentiment Analyzer |
| 31 | P/E | Fundamental | Screener |
| 32 | ROE | Fundamental | Screener |
| 33 | Debt/Equity | Fundamental | Screener |
| 34 | Market factor | Factor | Computed |
| 35 | Size factor | Factor | Computed |
| 36 | Momentum factor | Factor | Computed |
| 37 | Volatility factor | Factor | Computed |
| 38 | Liquidity factor | Factor | Computed |

After PCA: 12-15 orthogonal features.

---

## Transaction Cost Model

| Cost | Rate | Applied |
|---|---|---|
| STT (equity delivery) | 0.1% | Both sides |
| Stamp duty | 0.015% | Both sides |
| Brokerage (Zerodha) | ₹20/order or 0.03% | Per order |
| Slippage (Nifty 50) | 0.05% | Entry + Exit |
| **Total round trip** | **0.3%** | |

---

## Walk-Forward Validation Protocol

```
For each date T in walk_forward_period:
  1. train = data[0:T-1]  (expanding window)
  2. test = data[T]
  3. Fit scaler on train only
  4. Fit PCA on train only
  5. Train XGBoost on train only
  6. Predict on test
  7. Record prediction vs actual
  8. Roll forward

Final metrics computed on ALL predictions (out-of-sample only).
```

---

## Label Definition

| Label | Definition |
|---|---|
| **1 (WIN)** | Close at T+5 ≥ 1.05 × Close at T AND no close ≤ 0.95 × Close at T in between |
| **0 (LOSS)** | Close at T+5 < 1.05 × Close at T OR any close ≤ 0.95 × Close at T |
| **Excluded** | Corporate action day, circuit filter day, missing data |

---

## Governance Filter Rules

| Rule | Condition | Action |
|---|---|---|
| VIX spike | VIX > 22 | Reduce 50% |
| ADX doldrums | ADX < 15 | Reduce 50% |
| Breadth weak | A/D ratio < 0.7 | Reduce 50% |
| FII/DII conflict | Bias contradicts signal | Reduce 50% |
| Expiry day | Is expiry day | BLOCK |
| Consecutive losses | 3 consecutive losses | Kill switch |
| Corporate action | Corp action day | SKIP |
| Circuit filter | Stock in circuit | SKIP |

---

## Position Sizing

```
position_size = risk_budget / (ATR × point_value)

Where:
  risk_budget = ₹10,000 (1-2% of ₹5L-1L portfolio)
  ATR = 14-day Average True Range
  point_value = 1 (equity cash)
```

---

## Dashboard Design

```
┌──────────────────────────────────────────────────────────────┐
│  VRMS — Volatility Regime Momentum Scanner                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  REGIME: [LOW-VOL] [NORMAL] [HIGH-VOL] [SPIKE]      │    │
│  │  VIX: 14.2 | ADX: 28.5 | HMM: BULL                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  TODAY'S TOP 5 SIGNALS                              │    │
│  │  ┌─────┬──────┬────┬─────┬──────┬──────┬──────┐    │    │
│  │  │Rank │Stock │Prob│Momen│RS   │SL    │Target│    │    │
│  │  ├─────┼──────┼────┼─────┼──────┼──────┼──────┤    │    │
│  │  │ 1   │ TATA │72% │ +8% │ 1.15 │ -3%  │ +6%  │    │    │
│  │  │ 2   │ INFY │68% │ +6% │ 1.12 │ -2%  │ +5%  │    │    │
│  │  │ ... │ ...  │... │ ... │ ...  │ ...  │ ...  │    │    │
│  │  └─────┴──────┴────┴─────┴──────┴──────┴──────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │  Equity Curve        │  │  Win Rate by Regime          │ │
│  │  [Line chart]        │  │  [Bar chart]                 │ │
│  │                      │  │  Low-vol: 62%                │ │
│  │  +25% YTD            │  │  Normal: 58%                 │ │
│  │                      │  │  High-vol: 45%               │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │  Deflated Sharpe     │  │  Bootstrap CI                │ │
│  │  1.23                │  │  Win rate: 58% ± 4%          │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | Stack compatibility |
| ML | XGBoost 2.0 | Fast training, interpretable |
| Data | nsepython, yfinance | Free, reliable |
| Dashboard | Streamlit 1.30 | Free hosting, Python-native |
| Persistence | SQLite | Zero infra, stdlib |
| Backtest | Custom | Walk-forward, cost-aware |
| Hosting | Streamlit Cloud | Free tier |
| Version control | Git + GitHub | Private repo |

---

## Security Considerations

| Concern | Mitigation |
|---|---|
| API keys | No API keys required |
| Data validation | Validate all inputs before use |
| No external data sharing | All data stays local |
| No user authentication | Single-user tool |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Dependency conflicts | Medium | High | Extract logic, unify deps |
| Look-ahead bias | High | Critical | Expanding window everywhere |
| Overfitting | High | High | PCA, walk-forward, hold-out |
| Data quality | Medium | High | Validation layer |
| Hardware limits | Low | Medium | Nifty 50 scope |
| Regime change | Medium | High | Weekly retrain |

---

## Success Criteria

| Metric | Threshold |
|---|---|
| Walk-forward win rate | >53% |
| Sharpe ratio (net of costs) | >0.8 |
| Max drawdown | <20% |
| Net expectancy per trade | >0.1% |
| Live paper trading win rate | >50% after 30 trades |

---

## Kill Criteria

| Metric | Threshold |
|---|---|
| Walk-forward win rate | <53% |
| Live paper win rate | <50% after 30 trades |
| Sharpe ratio | <0.8 |
| Max drawdown | >20% |
| Net expectancy | <0.1% per trade |
