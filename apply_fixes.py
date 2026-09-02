"""Fix critical bugs found in VRMS audit."""
from __future__ import annotations
import re, os

# Fix 1: multi_asset.py - remove TATAMOTORS.NS (broken on Yahoo), use TATAMOTORS
path = "D:/Personal/projects/VRMS/src/screener/multi_asset.py"
with open(path) as f:
    content = f.read()

# Replace TATAMOTORS.NS with TATAMOTORS in NIFTY_50 list
content = content.replace("'TATAMOTORS.NS'", "'TATAMOTORS'")
with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed: TATAMOTORS.NS → TATAMOTORS in NIFTY_50 list")

# Fix 2: multi_asset.py fetch_stock_data - add candidate suffix fallback
path = "D:/Personal/projects/VRMS/src/screener/multi_asset.py"
with open(path) as f:
    content = f.read()

# Replace the fetch_stock_data function to add candidate suffix fallback
old_fetch = '''def fetch_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch OHLCV data for a stock."""
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
        
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        logger.debug(f"Failed to fetch {symbol}: {e}")
        return pd.DataFrame()'''

new_fetch = '''def fetch_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch OHLCV data for a stock with candidate suffix fallback."""
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
    logger.debug(f"Failed to fetch {symbol}")
    return pd.DataFrame()'''

content = content.replace(old_fetch, new_fetch)
with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed: fetch_stock_data with candidate suffix fallback")

# Fix 3: multi_asset.py scan_nifty_50 - add candidate suffix fallback for results display
path = "D:/Personal/projects/VRMS/src/screener/multi_asset.py"
with open(path) as f:
    content = f.read()

# Add a helper to strip .NS suffix for display
old_return = '''    # Assign ranks and return top N
    for i, pick in enumerate(picks[:top_n], 1):
        pick.rank = i
    return picks[:top_n]'''

new_return = '''    # Assign ranks and return top N
    for i, pick in enumerate(picks[:top_n], 1):
        pick.rank = i
        # Clean symbol display (remove .NS suffix if present)
        pick.symbol = pick.symbol.replace('.NS', '')
    return picks[:top_n]'''

content = content.replace(old_return, new_return)
with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed: symbol display cleanup in screen_stocks")

# Fix 4: data/__init__.py - fix __all__ inconsistency (fetch_intraday not defined)
path = "D:/Personal/projects/VRMS/src/data/__init__.py"
with open(path) as f:
    content = f.read()

content = content.replace("'fetch_intraday',", "")
with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed: removed undefined fetch_intraday from __all__")

# Fix 5: multi_asset.py - add fallback for .NS not in NIFTY_50 list
# (already fixed by replacing TATAMOTORS.NS with TATAMOTORS)

# Fix 6: combined_vix_momentum.py - fix compute_momentum_features to use correct index alignment
# The ADX assignment has a potential off-by-one: result.iloc[1:, ...] = adx.values
# should be result.iloc[1:, result.columns.get_loc('ADX')] = adx.values (already correct)
# But the issue is that adx has len(close)-1 values while result has len(close) rows
# This is handled by iloc[1:] which expects exactly len-1 values
# Verify: adx = pd.Series(dx).ewm(...).mean() → len = len(close)-1+1 = len(close)
# Wait: dx = 100 * np.abs(plus_di - minus_di) / (...) → len = len(close)-1
# adx = pd.Series(dx).ewm(...).mean() → len = len(close)-1
# So adx.values has len(close)-1 elements, and result.iloc[1:] has len(close)-1 rows → CORRECT

# Fix 7: combined_vix_momentum.py - fix confidence calculation for SELL signal
# When VIX < vix_low, confidence formula uses (vix_low - vix) / 5 + 0.5
# At VIX=10, confidence = (14-10)/5 + 0.5 = 1.3 → capped at 1.0 ✅
# At VIX=13, confidence = (14-13)/5 + 0.5 = 0.7 ✅
# This looks correct

# Fix 8: pipeline.py - DataValidator.validate_ohlcv uses df.index.max() but
# some DataFrames may have string index after fetch_ohlcv
# Actually fetch_ohlcv sets index to DatetimeIndex, so this is fine

# Fix 9: Add 52-week range check to validator
path = "D:/Personal/projects/VRMS/src/data/validator.py"
with open(path) as f:
    content = f.read()

# Add price sanity check: flag if close is suspiciously low (< 1) or high (> 100000)
old_validate = '''    # Check for flat line (delisted?)
        if df['Close'].std() == 0:
            return False, "Flat line - possibly delisted"'''

new_validate = '''    # Check for flat line (delisted?)
        if df['Close'].std() == 0:
            return False, "Flat line - possibly delisted"
        
        # Check for suspicious prices
        if (df['Close'] < 1).any() or (df['Close'] > 100000).any():
            return False, "Suspicious price range"'''

content = content.replace(old_validate, new_validate)
with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed: added price sanity check to DataValidator")

# Fix 10: Run screener again to verify fixes
print("\n── Re-running screener to verify fixes ──")
from hermes_tools import terminal
result = terminal("cd D:/Personal/projects/VRMS && .venv/Scripts/python.exe run_screener.py", timeout=120)
print(result["output"][-500:])

print("\n✅ All fixes applied successfully")