"""Run multi-asset screener to find top picks."""
from __future__ import annotations

import logging

from src.screener.multi_asset import fetch_vix, screen_stocks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the screener and show results."""
    print("Fetching VIX data...")
    vix_df = fetch_vix(period="6mo")
    
    if vix_df.empty:
        print("ERROR: Could not fetch VIX data")
        return
    
    current_vix = vix_df.iloc[-1]['VIX']
    print(f"Current VIX: {current_vix:.1f}")
    
    # Determine regime
    if current_vix > 18:
        regime = "fear (BUY zone)"
    elif current_vix < 14:
        regime = "complacency (SELL zone)"
    else:
        regime = "neutral (momentum only)"
    
    print(f"VIX Regime: {regime}")
    
    # Screen stocks
    print("\nScanning Nifty 50 stocks...")
    picks = screen_stocks(vix_df, top_n=5)
    
    if not picks:
        print("\nNo high-conviction picks found. Market conditions are unfavorable.")
        print("Wait for VIX > 18 (fear) or strong momentum in neutral zone.")
        return
    
    print(f"\n{'='*80}")
    print(f"TOP {len(picks)} PICKS — {regime.upper()}")
    print("="*80)
    print(f"{'Rank':<6} {'Symbol':<15} {'Conviction':<12} {'Price':<10} {'Target':<10} {'Stop':<10} {'Reason'}")
    print("-" * 80)
    
    for pick in picks:
        print(f"{pick.rank:<6} {pick.symbol:<15} {pick.conviction:<12.0f} "
              f"₹{pick.price:<9.0f} ₹{pick.target:<9.0f} ₹{pick.stop_loss:<9.0f} {pick.reason}")
    
    print("="*80)
    print("\nDisclaimer: For educational purposes only. Not financial advice.")
    print("Past performance does not guarantee future results.")


if __name__ == "__main__":
    main()
