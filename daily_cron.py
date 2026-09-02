"""Daily cron job for VRMS — runs screener + paper trading update.

Schedule: Daily at 6 PM IST (after market close).
On Windows: use Task Scheduler to run this script daily.
On Linux/Mac: add to crontab.

Usage:
    python daily_cron.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/daily_cron.log", mode="a"),
    ],
)
logger = logging.getLogger("daily_cron")


def is_market_open() -> bool:
    """Check if today is a market day (Mon-Fri, not a holiday)."""
    today = datetime.now()
    # Monday=0, Friday=4
    if today.weekday() >= 5:
        return False
    # TODO: Check NSE holiday calendar if needed
    return True


def run_daily_update():
    """Run daily screener + paper trading update."""
    logger.info("=" * 50)
    logger.info("Daily VRMS update started")
    logger.info("=" * 50)

    if not is_market_open():
        logger.info("Today is not a market day. Skipping.")
        return

    # Initialize variables
    vix = 0.0
    regime = "unknown"
    top5 = []

    # Step 1: Update paper trading positions
    logger.info("Updating paper trading positions...")
    try:
        from run_paper_trader import PaperTrader

        trader = PaperTrader()
        trader.update_positions()
        logger.info("Paper trading positions updated")
    except Exception as e:
        logger.error(f"Paper trading update failed: {e}")

    # Step 2: Run screener for today's signals
    logger.info("Running screener...")
    try:
        from run_paper_trader import fetch_vix, get_vix_regime, compute_conviction, compute_momentum, NIFTY_50
        import pandas as pd
        import numpy as np

        vix_df = fetch_vix(period="6mo")
        if vix_df.empty:
            logger.warning("No VIX data available")
            return

        regime, vix = get_vix_regime(vix_df, {"vix_high": 18, "vix_low": 14})
        logger.info(f"VIX: {vix:.1f}, Regime: {regime}")

        if regime == "complacency":
            logger.info("Complacency zone — no buys")
        else:
            # Scan for signals
            picks = []
            for symbol in NIFTY_50:
                from run_paper_trader import fetch_stock_data
                df = fetch_stock_data(symbol, period="6mo")
                if df.empty or len(df) < 50:
                    continue
                mom = compute_momentum(df)
                if not mom:
                    continue
                conviction, reason = compute_conviction(mom, regime, {"adx_threshold": 25})
                if conviction >= 30:
                    picks.append({
                        "symbol": symbol.replace(".NS", ""),
                        "conviction": conviction,
                        "price": mom["price"],
                        "reason": reason,
                    })

            picks.sort(key=lambda x: x["conviction"], reverse=True)
            top5 = picks[:5]

            if top5:
                logger.info(f"Top {len(top5)} signals:")
                for p in top5:
                    logger.info(f"  {p['symbol']:<12} Conviction: {p['conviction']:.0f} @ ₹{p['price']:.2f}")

                # Execute paper trades
                trader = PaperTrader()
                from run_paper_trader import PaperTrade
                signals = [
                    PaperTrade(
                        entry_date=datetime.now().strftime("%Y-%m-%d"),
                        symbol=p["symbol"],
                        entry_price=p["price"],
                        target_price=p["price"] * 1.05,
                        stop_price=p["price"] * 0.97,
                        conviction=p["conviction"],
                        vix_regime=regime,
                    )
                    for p in top5
                ]
                trader.execute_signals(signals)
            else:
                logger.info("No high-conviction signals today")

    except Exception as e:
        logger.error(f"Screener failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Step 3: Save daily snapshot
    logger.info("Saving daily snapshot...")
    try:
        snapshot = {
            "date": datetime.now().isoformat(),
            "vix": vix,
            "regime": regime,
            "signals_today": len(top5) if top5 else 0,
        }
        snapshot_path = Path("data/daily_snapshots.jsonl")
        with open(snapshot_path, "a") as f:
            f.write(json.dumps(snapshot) + "\n")
    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")

    logger.info("Daily VRMS update complete")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_daily_update()
