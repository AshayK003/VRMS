"""Paper trading tracker for VRMS.

Tracks live signals vs actual outcomes without real money.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PaperTrade:
    """A single paper trade."""
    symbol: str
    entry_date: datetime
    entry_price: float
    direction: str = "LONG"
    stop_loss: float = 0.0
    target: float = 0.0
    exit_date: datetime | None = None
    exit_price: float | None = None
    result: str = "OPEN"  # OPEN, WIN, LOSS
    return_pct: float = 0.0


class PaperTradingTracker:
    """Track paper trades and compute performance."""
    
    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.trades: list[PaperTrade] = []
        self._load_trades()
    
    def add_trade(self, trade: PaperTrade) -> None:
        """Add a new paper trade.
        
        Args:
            trade: PaperTrade object
        """
        self.trades.append(trade)
        self._save_trades()
    
    def update_trade(self, symbol: str, exit_price: float, exit_date: datetime | None = None) -> None:
        """Update a trade with exit price.
        
        Args:
            symbol: Trade symbol
            exit_price: Exit price
            exit_date: Exit date (defaults to now)
        """
        for trade in self.trades:
            if trade.symbol == symbol and trade.result == "OPEN":
                trade.exit_price = exit_price
                trade.exit_date = exit_date or datetime.now()
                
                # Calculate return
                if trade.direction == "LONG":
                    trade.return_pct = (exit_price - trade.entry_price) / trade.entry_price
                
                # Determine result
                if trade.return_pct >= trade.target:
                    trade.result = "WIN"
                elif trade.return_pct <= -trade.stop_loss:
                    trade.result = "LOSS"
                else:
                    trade.result = "CLOSED"
                
                self._save_trades()
                break
    
    def get_open_trades(self) -> list[PaperTrade]:
        """Get all open trades."""
        return [t for t in self.trades if t.result == "OPEN"]
    
    def get_closed_trades(self) -> list[PaperTrade]:
        """Get all closed trades."""
        return [t for t in self.trades if t.result != "OPEN"]
    
    def get_win_rate(self) -> float:
        """Get win rate for closed trades."""
        closed = self.get_closed_trades()
        if not closed:
            return 0.0
        
        wins = sum(1 for t in closed if t.result == "WIN")
        return wins / len(closed)
    
    def get_total_return(self) -> float:
        """Get total return across all closed trades."""
        closed = self.get_closed_trades()
        if not closed:
            return 0.0
        
        total = sum(t.return_pct for t in closed)
        return total
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
        closed = self.get_closed_trades()
        open_trades = self.get_open_trades()
        
        if not closed:
            return {
                "win_rate": 0.0,
                "total_return": 0.0,
                "n_trades": 0,
                "n_wins": 0,
                "n_losses": 0,
                "open_trades": len(open_trades),
            }
        
        wins = sum(1 for t in closed if t.result == "WIN")
        losses = sum(1 for t in closed if t.result == "LOSS")
        
        return {
            "win_rate": wins / len(closed),
            "total_return": sum(t.return_pct for t in closed),
            "n_trades": len(closed),
            "n_wins": wins,
            "n_losses": losses,
            "open_trades": len(open_trades),
        }
    
    def _save_trades(self) -> None:
        """Save trades to CSV."""
        if not self.trades:
            return
        
        df = pd.DataFrame([
            {
                "symbol": t.symbol,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "direction": t.direction,
                "stop_loss": t.stop_loss,
                "target": t.target,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "result": t.result,
                "return_pct": t.return_pct,
            }
            for t in self.trades
        ])
        
        path = self.data_dir / "paper_trades.csv"
        df.to_csv(path, index=False)
    
    def _load_trades(self) -> None:
        """Load trades from CSV."""
        path = self.data_dir / "paper_trades.csv"
        if not path.exists():
            return
        
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                trade = PaperTrade(
                    symbol=row['symbol'],
                    entry_date=pd.Timestamp(row['entry_date']),
                    entry_price=row['entry_price'],
                    direction=row['direction'],
                    stop_loss=row['stop_loss'],
                    target=row['target'],
                    exit_date=pd.Timestamp(row['exit_date']) if pd.notna(row['exit_date']) else None,
                    exit_price=row['exit_price'] if pd.notna(row['exit_price']) else None,
                    result=row['result'],
                    return_pct=row['return_pct'],
                )
                self.trades.append(trade)
        except Exception as e:
            logger.error(f"Failed to load paper trades: {e}")
