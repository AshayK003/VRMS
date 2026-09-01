"""Run VRMS pipeline standalone."""
from __future__ import annotations

import logging
import sys
from datetime import datetime

from src.pipeline import VRMSPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run the pipeline and print results."""
    pipeline = VRMSPipeline()
    
    logger.info("Starting VRMS pipeline...")
    result = pipeline.run_pipeline()
    
    print("\n" + "="*60)
    print("VRMS — Volatility Regime Momentum Scanner")
    print("="*60)
    print(f"Date: {result['date']}")
    print(f"VIX: {result['vix']}")
    print(f"ADX: {result['adx']}")
    print(f"Stocks analyzed: {result['n_stocks']}")
    print()
    
    print("Top 5 Signals:")
    print("-"*60)
    for i, signal in enumerate(result['signals'], 1):
        print(f"{i}. {signal['symbol']}")
        print(f"   Probability: {signal['probability']:.0%}")
        print(f"   Score: {signal['score']:.2f}")
        print(f"   Direction: {signal['direction']}")
        print(f"   Stop Loss: -{signal['stop_loss_pct']:.0%}")
        print(f"   Target: +{signal['target_pct']:.0%}")
        print()
    
    print("-"*60)
    print(f"Model metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
