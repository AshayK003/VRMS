# VRMS Engineering Plan

> Build plan for the Volatility Regime Momentum Scanner.

---

## Project Overview

**Name:** Volatility Regime Momentum Scanner (VRMS)
**Goal:** Generate daily swing trading signals for Nifty 50 stocks with >53% walk-forward win rate.
**Timeline:** 8 days
**Cost:** ₹0

---

## Phase Summary

| Phase | Focus | Duration | Status |
|---|---|---|---|
| 1 | Foundation | Day 1 | COMPLETE |
| 2 | Strategy & Research | — | SKIPPED (user domain knowledge) |
| 3 | Architecture & Planning | Day 1 | ACTIVE |
| 4 | UI/UX | — | PENDING |
| 5 | Implementation | Days 2-6 | PENDING |
| 6 | Verification | Day 7 | PENDING |
| 7 | Pre-release | Day 7 | PENDING |
| 8 | Release | Day 8 | PENDING |
| 9 | Operations | Ongoing | PENDING |
| 10 | Learning | Ongoing | PENDING |

---

## Day-by-Day Plan

### Day 1 (COMPLETE)

| Task | Deliverable | Status |
|---|---|---|
| Create project folder | `D:/Personal/projects/VRMS/` | ✅ |
| Create private GitHub repo | `github.com/AshayK003/VRMS` | ✅ |
| Write `.gitignore` | `.gitignore` | ✅ |
| Write Constitution | `internals/constitution.md` | ✅ |
| Write Orchestrator | `internals/orchestrator.md` | ✅ |
| Write Governance | `internals/approval-gates.md` | ✅ |
| Write Engineering Memory | `internals/engineering_memory.md` | ✅ |
| Write Engineering Plan | `docs/plan.md` | ✅ |
| Write Architecture | `docs/architecture.md` | PENDING |

### Day 2

| Task | Deliverable |
|---|---|
| Extract logic from existing repos | `src/` skeleton |
| Unify dependencies | `requirements.txt` |
| Build point-in-time constituent mapper | `data/constituents.csv` |
| Build corporate action flagger | `data/corporate_actions.csv` |
| Pull historical OHLCV data | `data/raw/ohlcv/` |

### Day 3

| Task | Deliverable |
|---|---|
| Build expanding-window feature engineering | `features/engineering.py` |
| Add volatility features (GARCH, realized) | Feature DataFrame |
| Add momentum features (1M/3M/6M) | Feature DataFrame |
| Add relative strength features | Feature DataFrame |
| Add volume features | Feature DataFrame |
| Add FII/DII features | Feature DataFrame |
| PCA to reduce to 12-15 features | Feature DataFrame |

### Day 4

| Task | Deliverable |
|---|---|
| Add sentiment features (from Sentiment Analyzer) | Feature DataFrame |
| Add fundamental features (P/E, ROE, Debt) | Feature DataFrame |
| Add FII/DII lag (T-2 final data) | Feature DataFrame |
| Add temporal alignment | Feature DataFrame |
| Add data validation layer | `features/validation.py` |
| Define labels precisely | Label DataFrame |

### Day 5

| Task | Deliverable |
|---|---|
| XGBoost training with expanding window | `models/xgb_v1.pkl` |
| Hyperparameter tuning (validation set only) | Best params |
| Feature importance analysis | Feature ranking |
| Walk-forward backtest | Backtest results |
| Transaction cost modeling | Cost-adjusted returns |

### Day 6

| Task | Deliverable |
|---|---|
| Governance filters (VIX, ADX, A/D, FII/DII, expiry) | `signals/governance.py` |
| Signal generator | `signals/generator.py` |
| Position sizing (volatility-adjusted) | `signals/sizing.py` |
| Deflated Sharpe calculation | `backtest/metrics.py` |
| Bootstrap confidence intervals | `backtest/metrics.py` |

### Day 7

| Task | Deliverable |
|---|---|
| Streamlit dashboard | `dashboard.py` |
| Equity curve visualization | Dashboard |
| Win rate by regime | Dashboard |
| Paper trading tracker | `tracker/paper.py` |
| Code review (self) | `reviews/CODE_REVIEW.md` |
| Logic review (self) | `reviews/LOGIC_REVIEW.md` |
| QA report | `reviews/QA_REPORT.md` |

### Day 8

| Task | Deliverable |
|---|---|
| Final cleanup | Clean code |
| Deploy to Streamlit Cloud | Live URL |
| GitHub push | Remote repo |
| Paper trading mode activation | Live tracking |

---

## Architecture (High-Level)

```
Data Layer → Feature Engineering → XGBoost → Governance → Signals → Dashboard
```

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Dependency conflicts | Medium | High | Extract logic, don't import |
| Look-ahead bias | High | Critical | Expanding window everywhere |
| Overfitting | High | High | PCA, walk-forward, hold-out set |
| Data quality | Medium | High | Data validation layer |
| Hardware limits | Low | Medium | Nifty 50 scope |

---

## Success Criteria

| Metric | Threshold |
|---|---|
| Walk-forward win rate | >53% |
| Sharpe ratio (net) | >0.8 |
| Max drawdown | <20% |
| Net expectancy | >0.1% per trade |
| Live paper win rate | >50% after 30 trades |

---

## Kill Criteria

| Metric | Threshold |
|---|---|
| Walk-forward win rate | <53% |
| Live paper win rate | <50% after 30 trades |
| Sharpe ratio | <0.8 |
| Max drawdown | >20% |
| Net expectancy | <0.1% per trade |
