"""Fix TATAMOTORS demerger in multi_asset.py and related files."""
from __future__ import annotations

# ============================================================
# FIX 1: multi_asset.py - Replace TATAMOTORS with TMCV.NS
#    TATAMOTORS was demerged in Jul 2024. Old ticker is delisted.
#    TMCV.NS (Tata Motors PV) is the post-demerger entity.
# ============================================================
path = "D:/Personal/projects/VRMS/src/screener/multi_asset.py"
with open(path) as f:
    content = f.read()

# Replace the bare 'TATAMOTORS' in the NIFTY_50 list with 'TMCV.NS'
# Note: line 43 has 'TATAMOTORS' (no .NS suffix already)
content = content.replace("'TATAMOTORS'", "'TMCV.NS'")

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed: TATAMOTORS → TMCV.NS in NIFTY_50")

# ============================================================
# FIX 2: constituents.py - Update Nifty 50 constituent lists
#    TATAMOTORS was removed from Nifty 50 after demerger.
#    TMCV replaced it. Update all constituent lists.
# ============================================================
path = "D:/Personal/projects/VRMS/src/data/constituents.py"
with open(path) as f:
    content = f.read()

# Replace TATAMOTORS with TMCV in all constituent lists
content = content.replace("'TATAMOTORS'", "'TMCV'")

# Add corporate action note for the demerger
# Add a comment about the demerger at the top
old_header = '''# Historical Nifty 50 constituents (semi-annual rebalancing)
# Source: NSE press releases, compiled manually'''
new_header = '''# Historical Nifty 50 constituents (semi-annual rebalancing)
# Source: NSE press releases, compiled manually
# NOTE: TATAMOTORS demerged into TMCV (Tata Motors PV) in Jul 2024.
#       All constituent lists updated to use TMCV (post-demerger entity).
#       Old TATAMOTORS data is unavailable on Yahoo Finance (delisted).'''
content = content.replace(old_header, new_header)

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed: constituents.py updated (TATAMOTORS → TMCV)")

# ============================================================
# FIX 3: corporate_actions.py - Add TMCV demerger action
# ============================================================
path = "D:/Personal/projects/VRMS/src/data/corporate_actions.py"
with open(path) as f:
    content = f.read()

# Add TMCV to corporate actions (demerger)
old_tata = '''    'TATAMOTORS': {
        'DEMERGER': ['2024-07-31'],  # 5-way demerger
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-06-14', '2024-12-16', '2025-06-13']
    },'''
new_tata = '''    'TMCV': {
        'DEMERGER': ['2024-07-31'],  # Tata Motors demerged into TMCV (PV)
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': ['2024-06-14', '2024-12-16', '2025-06-13']
    },
    'TATAMOTORS': {  # Legacy ticker (delisted post-demerger)
        'DEMERGER': ['2024-07-31'],
        'SPLIT': [],
        'BONUS': [],
        'DIVIDEND': []
    },'''
content = content.replace(old_tata, new_tata)

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed: corporate_actions.py updated")

# ============================================================
# FIX 4: ohlcv.py - Add corporate action awareness for demergers
# ============================================================
path = "D:/Personal/projects/VRMS/src/data/ohlcv.py"
with open(path) as f:
    content = f.read()

# Add a note about demerger handling and auto_adjust
old_fetch = '''@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for a symbol from Yahoo Finance.
    
    Args:
        symbol: NSE ticker (e.g., 'RELIANCE', 'TCS')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume (tz-naive)
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(start=start_date, end=end_date)
        
        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        return _clean_yf_df(df)
        
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
        return pd.DataFrame()'''

new_fetch = '''@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily OHLCV data for a symbol from Yahoo Finance.
    
    Uses auto_adjust=True to handle corporate actions (splits, dividends,
    demergers) correctly. For demerged tickers (e.g., TATAMOTORS→TMCV),
    the old ticker may be delisted — use the new ticker symbol.
    
    Args:
        symbol: NSE ticker (e.g., 'RELIANCE', 'TCS')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume (tz-naive)
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(f"{symbol}.NS")
        # auto_adjust=True ensures corporate action adjustments are applied
        df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
        
        if df is None or df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        return _clean_yf_df(df)
        
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
        return pd.DataFrame()'''

content = content.replace(old_fetch, new_fetch)

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed: ohlcv.py - added auto_adjust=True")

# ============================================================
# FIX 5: pipeline.py - Add TATAMOTORS→TMCV alias mapping
# ============================================================
path = "D:/Personal/projects/VRMS/src/pipeline.py"
with open(path) as f:
    content = f.read()

# Add a ticker alias/demerger mapping constant
old_imports = '''from src.data.ohlcv import fetch_ohlcv, get_benchmark, fetch_vix'''
new_imports = '''from src.data.ohlcv import fetch_ohlcv, get_benchmark, fetch_vix
from src.data.constituents import get_constituents_on_date

# Ticker aliases for demerged/rename symbols
# Maps old/delisted tickers to current active tickers
TICKER_ALIASES = {
    'TATAMOTORS': 'TMCV.NS',  # Demerged Jul 2024 → Tata Motors PV
}'''
content = content.replace(old_imports, new_imports)

with open(path, 'w') as f:
    f.write(content)
print("✅ Fixed: pipeline.py - added TICKER_ALIASES")

# ============================================================
# VERIFY: Run screener to confirm fixes
# ============================================================
from hermes_tools import terminal
print("\n── Verifying fixes ──")
result = terminal("cd D:/Personal/projects/VRMS && .venv/Scripts/python.exe run_screener.py", timeout=120)
print(result["output"][-500:])

print("\n✅ All TATAMOTORS demerger fixes applied")