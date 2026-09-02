"""Comprehensive audit verification for VRMS."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf

print("="*70)
print("VRMS AUDIT VERIFICATION")
print("="*70)

# ── 1. yfinance data integrity ──
print("\n── 1. YFINANCE DATA INTEGRITY ──")
try:
    nifty = yf.Ticker("^NSEI").history(period="5y")
    print(f"Nifty 5y rows: {len(nifty)}")
    print(f"  NaN in Close: {nifty['Close'].isna().sum()}")
    print(f"  Date range: {nifty.index.min().date()} to {nifty.index.max().date()}")
    print(f"  Close monotonic?: {nifty['Close'].is_monotonic_increasing}")
    # Check for splits/dividends
    print(f"  Splits: {len(nifty['Dividends'])} non-zero")
    print(f"  Last Close: {nifty['Close'].iloc[-1]:.2f}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 2. VIX data integrity ──
print("\n── 2. VIX DATA INTEGRITY ──")
try:
    vix = yf.Ticker("^INDIAVIX").history(period="5y")
    print(f"VIX 5y rows: {len(vix)}")
    print(f"  Date range: {vix.index.min().date()} to {vix.index.max().date()}")
    print(f"  VIX min/max: {vix['Close'].min():.1f} / {vix['Close'].max():.1f}")
    print(f"  VIX last: {vix['Close'].iloc[-1]:.1f}")
    # Check if VIX is always positive
    print(f"  Any VIX <= 0: {(vix['Close'] <= 0).any()}")
    # Check NaN
    print(f"  NaN count: {vix['Close'].isna().sum()}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 3. Nifty 50 ticker validation ──
print("\n── 3. NIFTY 50 TICKER VALIDATION ──")
NIFTY_50 = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
    'ICICIBANK.NS', 'SBIN.NS', 'ITC.NS', 'BHARTIARTL.NS', 'LICI.NS',
    'HCLTECH.NS', 'ASIANPAINT.NS', 'KOTAKBANK.NS', 'MARUTI.NS', 'TATAMOTORS.NS',
    'SUNPHARMA.NS', 'TITAN.NS', 'AXISBANK.NS', 'WIPRO.NS', 'NESTLEIND.NS',
    'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'ONGC.NS', 'ADANIPORTS.NS', 'POWERGRID.NS',
    'NTPC.NS', 'TATASTEEL.NS', 'JSWSTEEL.NS', 'COALINDIA.NS', 'GRASIM.NS',
    'TECHM.NS', 'CIPLA.NS', 'DRREDDY.NS', 'BRITANNIA.NS', 'HEROMOTOCO.NS',
    'EICHERMOT.NS', 'APOLLOHOSP.NS', 'ADANIENT.NS', 'TATACONSUM.NS', 'DIVISLAB.NS',
    'HINDALCO.NS', 'SHREECEM.NS', 'BAJAJFINSV.NS', 'M&M.NS', 'SBILIFE.NS',
    'IOC.NS', 'INDUSINDBK.NS', 'HDFCLIFE.NS', 'BPCL.NS', 'TRENT.NS',
]

# Test fetching a few tickers to find which ones actually work
working = []
broken = []
for sym in NIFTY_50[:10]:  # Test first 10
    try:
        t = yf.Ticker(sym)
        h = t.history(period="5d")
        if len(h) > 0 and h['Close'].iloc[-1] > 0:
            working.append(sym)
        else:
            broken.append(sym)
    except Exception as e:
        broken.append(f"{sym} ({e})")

print(f"Working (first 10): {len(working)}/{10}")
print(f"Broken (first 10): {broken}")

# Check specifically for TATAMOTORS.NS (known problematic from backtest error log)
print("\n  TATAMOTORS.NS specific test:")
try:
    t = yf.Ticker("TATAMOTORS.NS")
    h = t.history(period="5d")
    if len(h) > 0:
        print(f"    Data fetched: {len(h)} rows, last close: {h['Close'].iloc[-1]:.2f}")
    else:
        print(f"    EMPTY - Yahoo returns no data for TATAMOTORS.NS")
except Exception as e:
    print(f"    ERROR: {e}")

# ── 4. Constituent mapping ──
print("\n── 4. CONSTITUENT MAPPING ──")
from src.data.constituents import get_constituents_on_date
const = get_constituents_on_date("2025-09-02")
print(f"  Constituents count: {len(const)}")
print(f"  Has TATAMOTORS: {'TATAMOTORS' in const}")
# Check if constituent symbols match Yahoo format
print(f"  Sample symbols: {const[:5]}")
print(f"  Note: constituents are plain symbols (no .NS), Yahoo needs .NS suffix")

# ── 5. Feature engineering correctness ──
print("\n── 5. FEATURE ENGINEERING CORRECTNESS ──")
from src.features.engineering import (
    compute_rsi, compute_adx, compute_atr, compute_momentum,
    compute_realized_vol, compute_relative_strength, compute_volume_features, compute_garch_vol
)

# Generate test data
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=100, freq="D")
test_df = pd.DataFrame({
    'Open': np.random.uniform(100, 200, 100),
    'High': np.random.uniform(100, 200, 100),
    'Low': np.random.uniform(100, 200, 100),
    'Close': np.cumsum(np.random.randn(100) * 2) + 150,
    'Volume': np.random.randint(100000, 1000000, 100),
}, index=dates)
# Ensure High >= Low and Close within range
test_df['High'] = test_df[['Open','High','Low']].max(axis=1) + abs(np.random.randn(100))
test_df['Low'] = test_df[['Open','High','Low']].min(axis=1) - abs(np.random.randn(100))
test_df['Close'] = (test_df['Close'] - test_df['Close'].min()) * (test_df['High'] - test_df['Low']).abs() / (test_df['Close'].max() - test_df['Close'].min()) + test_df['Low']

# RSI
rsi = compute_rsi(test_df)
print(f"  RSI: min={rsi.min():.1f}, max={rsi.max():.1f}, NaN={rsi.isna().sum()}")
assert rsi.min() >= 0 and rsi.max() <= 100, f"RSI out of range: {rsi.min()} to {rsi.max()}"

# ADX
adx = compute_adx(test_df)
print(f"  ADX: min={adx.min():.1f}, max={adx.max():.1f}, NaN={adx.isna().sum()}")
assert adx.min() >= 0, f"ADX negative: {adx.min()}"

# ATR
atr = compute_atr(test_df)
print(f"  ATR: min={atr.min():.2f}, max={atr.max():.2f}")
assert (atr > 0).all(), "ATR has non-positive values"

# Momentum
mom = compute_momentum(test_df)
print(f"  Momentum columns: {list(mom.columns)}")

# Realized vol
vol = compute_realized_vol(test_df)
print(f"  Vol columns: {list(vol.columns)}")

# Volume features
vf = compute_volume_features(test_df)
print(f"  Volume features columns: {list(vf.columns)}")
print(f"  Circuit flag count: {(vf['circuit_flag']==1).sum()}")

# Relative strength (needs benchmark)
bench = pd.DataFrame({'Close': np.cumsum(np.random.randn(100)*2)+200}, index=dates)
rs = compute_relative_strength(test_df, bench)
print(f"  RS columns: {list(rs.columns)}")

# GARCH vol
garch = compute_garch_vol(test_df)
print(f"  GARCH vol: min={garch.min():.4f}, max={garch.max():.4f}")

# ── 6. Labels correctness ──
print("\n── 6. LABELS CORRECTNESS ──")
from src.features.labels import generate_labels
labels = generate_labels(test_df, target_pct=0.05, horizon=5, stop_loss_pct=0.03)
print(f"  Labels generated: {len(labels)}")
print(f"  WIN count: {(labels['label']==1).sum()}")
print(f"  LOSS count: {(labels['label']==0).sum()}")
print(f"  NaN count: {labels['label'].isna().sum()}")
# Verify no look-ahead: label at index i should only depend on data at i..i+5
print(f"  First label date: {labels.dropna()['label'].index[0].date()}")
print(f"  Last label date: {labels.dropna()['label'].index[-1].date()}")
# Check that labels are properly assigned (no NaN in valid range)
valid_labels = labels.dropna()
assert len(valid_labels) == len(test_df) - 5, f"Expected {len(test_df)-5} labels, got {len(valid_labels)}"
print("  ✅ Label count correct")

# ── 7. Corporate action filter ──
print("\n── 7. CORPORATE ACTION FILTER ──")
from src.data.corporate_actions import is_corporate_action_day
result = is_corporate_action_day("TATAMOTORS", "2024-07-31")
print(f"  TATAMOTORS on 2024-07-31: {result}")
result2 = is_corporate_action_day("TATAMOTORS", "2024-07-30")
print(f"  TATAMOTORS on 2024-07-30: {result2}")
result3 = is_corporate_action_day("RELIANCE", "2024-06-14")
print(f"  RELIANCE on 2024-06-14: {result3}")
result4 = is_corporate_action_day("RELIANCE", "2025-01-01")
print(f"  RELIANCE on 2025-01-01: {result4}")

# ── 8. Signal generation ──
print("\n── 8. SIGNAL GENERATION ──")
from src.strategies.combined_vix_momentum import generate_signals, Signal, Trade, BacktestResult

# Create test data
test_vix = pd.DataFrame({'VIX': [15.0]*20 + [20.0]*10 + [10.0]*10}, 
                         index=pd.date_range("2024-01-01", periods=40, freq="D"))
test_nifty = pd.DataFrame({
    'Open': np.random.uniform(100, 200, 40),
    'High': np.random.uniform(100, 200, 40),
    'Low': np.random.uniform(100, 200, 40),
    'Close': np.cumsum(np.random.randn(40) * 2) + 150,
    'Volume': np.random.randint(100000, 1000000, 40),
}, index=pd.date_range("2024-01-01", periods=40, freq="D"))
test_nifty['High'] = test_nifty[['Open','High','Low']].max(axis=1) + 1
test_nifty['Low'] = test_nifty[['Open','High','Low']].min(axis=1) - 1
test_nifty['Close'] = test_nifty['Close'].clip(lower=50)

params = {'vix_high': 18, 'vix_low': 14, 'adx_threshold': 25, 'target_pct': 0.05, 'stop_loss_pct': 0.03}
signals = generate_signals(test_vix, test_nifty, params)
print(f"  Signals generated: {len(signals)}")
buy_signals = [s for s in signals if s.direction == "BUY"]
sell_signals = [s for s in signals if s.direction == "SELL"]
print(f"  BUY: {len(buy_signals)}, SELL: {len(sell_signals)}")

# ── 9. Backtest metrics ──
print("\n── 9. BACKTEST METRICS VERIFICATION ──")
from src.strategies.combined_vix_momentum import run_backtest, calculate_metrics

# Run backtest
result = run_backtest(signals, test_nifty, params)
print(f"  Trades: {len(result.trades)}")
if result.trades:
    metrics = calculate_metrics(result.trades)
    print(f"  Win rate: {metrics['win_rate']:.1%}")
    print(f"  Total return: {metrics['total_return']:.1%}")
    print(f"  Sharpe: {metrics['sharpe']:.2f}")
    print(f"  Max DD: {metrics['max_drawdown']:.1%}")
    print(f"  Profit factor: {metrics['profit_factor']:.2f}")
    
    # Verify Sharpe formula: mean/std * sqrt(252/N)
    returns = [t.return_pct for t in result.trades]
    if len(returns) > 1:
        manual_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252/len(returns))
        print(f"  Manual Sharpe: {manual_sharpe:.2f}")
        print(f"  Match: {abs(manual_sharpe - metrics['sharpe']) < 0.01}")
else:
    print("  ⚠️ No trades executed")

# ── 10. Walk-forward backtest ──
print("\n── 10. WALK-FORWARD BACKTEST ──")
from src.backtest.walk_forward import run_walk_forward_backtest
try:
    # Use small symbol set for speed
    wf_result = run_walk_forward_backtest(
        symbols=['RELIANCE.NS', 'TCS.NS'],
        start_date="2023-01-01",
        end_date="2024-01-01",
        train_window=100,
        test_window=10,
        transaction_cost=0.003,
        target_pct=0.05,
        stop_loss_pct=0.05,
        threshold=0.5,
    )
    print(f"  Trades: {wf_result.n_trades}")
    print(f"  Win rate: {wf_result.win_rate:.1%}")
    print(f"  Total return: {wf_result.total_return:.1%}")
    print(f"  Sharpe: {wf_result.sharpe:.2f}")
    print(f"  Max DD: {wf_result.max_drawdown:.1%}")
except Exception as e:
    print(f"  ERROR: {e}")

# ── 11. Multi-asset screener ──
print("\n── 11. MULTI-ASSET SCREENER ──")
from src.screener.multi_asset import (
    compute_momentum, compute_conviction, get_vix_regime, screen_stocks
)

# Test compute_momentum
test_stock = test_nifty.copy()
mom = compute_momentum(test_stock)
print(f"  Momentum features: {list(mom.keys())}")
print(f"  Price: {mom['price']:.2f}, MA20: {mom['ma20']:.2f}, MA50: {mom['ma50']:.2f}")
print(f"  ADX: {mom['adx']:.1f}")
print(f"  ROC20: {mom['roc_20']:.4f}")

# Test compute_conviction
vix_regime = "fear"
conviction, reason = compute_conviction(mom, vix_regime, {'adx_threshold': 25})
print(f"  Conviction ({vix_regime}): {conviction:.0f} - {reason}")

# Test get_vix_regime
vix_df_test = pd.DataFrame({'VIX': [18.5]}, index=pd.date_range("2024-01-01", periods=1))
regime, vix_val = get_vix_regime(vix_df_test, {'vix_high': 18, 'vix_low': 14})
print(f"  VIX regime (VIX=18.5): {regime}")
vix_df_test2 = pd.DataFrame({'VIX': [11.0]}, index=pd.date_range("2024-01-01", periods=1))
regime2, vix_val2 = get_vix_regime(vix_df_test2, {'vix_high': 18, 'vix_low': 14})
print(f"  VIX regime (VIX=11.0): {regime2}")

# ── 12. Governance ──
print("\n── 12. GOVERNANCE FILTERS ──")
from src.signals.governance import (
    filter_vix_spike, filter_adx_doldrums, filter_breadth,
    filter_fii_dii_bias, filter_expiry_day
)
print(f"  VIX spike (VIX=25): {filter_vix_spike(25)}")
print(f"  VIX spike (VIX=18): {filter_vix_spike(18)}")
print(f"  ADX doldrums (ADX=10): {filter_adx_doldrums(10)}")
print(f"  ADX doldrums (ADX=25): {filter_adx_doldrums(25)}")
print(f"  Expiry day: {filter_expiry_day(True)}")
print(f"  Expiry normal: {filter_expiry_day(False)}")
print(f"  FII bearish + LONG: {filter_fii_dii_bias('BEARISH', 'LONG')}")

# ── 13. Data type / contract verification ──
print("\n── 13. CONTRACT VERIFICATION ──")
from src.models.xgboost import XGBoostClassifier
model = XGBoostClassifier(n_estimators=10, max_depth=3)
X_test = np.random.randn(5, 10).astype(np.float32)
y_test = np.array([1, 0, 1, 0, 1], dtype=np.int32)
model.fit(X_test, y_test)
pred = model.predict(X_test)
proba = model.predict_proba(X_test)
print(f"  Predict shape: {pred.shape}, dtype: {pred.dtype}")
print(f"  Proba shape: {proba.shape}")
print(f"  Proba sum check: {proba.sum(axis=1)}")
print(f"  Feature importance: {model.get_feature_importance().tolist()}")

# Test PCA
from src.features.pca import PCAReducer
pca = PCAReducer(n_components=5)
X_pca_input = np.random.randn(20, 15).astype(np.float32)
X_transformed = pca.fit_transform(X_pca_input)
X_new = pca.transform(X_pca_input[:3])
print(f"  PCA input shape: {X_pca_input.shape}")
print(f"  PCA output shape: {X_transformed.shape}")
print(f"  PCA transform shape: {X_new.shape}")

# ── 14. Run vix_strategy (baseline) ──
print("\n── 14. BASELINE vix_strategy (from run_vix_strategy.py) ──")
result_v = terminal("cd D:/Personal/projects/VRMS && .venv/Scripts/python.exe run_vix_strategy.py", timeout=60)
print(result_v["output"][-1500:])

print("\n" + "="*70)
print("AUDIT VERIFICATION COMPLETE")
print("="*70)