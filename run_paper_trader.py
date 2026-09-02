"""Paper trading tracker for VRMS signals.

Tracks hypothetical trades based on screener signals without real money.
Stores trade log as JSON. Computes P&L, win rate, equity curve.

Usage:
    python run_paper_trader.py          # Run today's screener & log signals
    python run_paper_trader.py --status  # Show current positions & P&L
    python run_paper_trader.py --history # Show all closed trades
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.screener.multi_asset import (
    fetch_vix, NIFTY_50,
    get_vix_regime, compute_momentum, compute_conviction,
    StockPick,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default paper trading parameters
DEFAULT_PARAMS = {
    'vix_high': 18,
    'vix_low': 14,
    'adx_threshold': 25,
    'target_pct': 0.05,
    'stop_loss_pct': 0.03,
    'holding_limit': 20,
    'transaction_cost': 0.003,
    'initial_capital': 100000,  # ₹1 lakh
}

TRADE_LOG_PATH = Path("data/paper_trades.json")


@dataclass
class PaperTrade:
    """A paper trade (hypothetical position)."""
    entry_date: str
    symbol: str
    entry_price: float
    target_price: float
    stop_price: float
    conviction: float
    vix_regime: str
    status: str = "OPEN"  # OPEN, CLOSED
    exit_date: str = ""
    exit_price: float = 0.0
    return_pct: float = 0.0
    reason: str = ""


@dataclass
class PaperPortfolio:
    """Paper trading portfolio state."""
    capital: float
    cash: float
    positions: list[dict] = field(default_factory=list)
    closed_trades: list[dict] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)

    @property
    def total_value(self) -> float:
        position_value = sum(p.get('current_value', 0) for p in self.positions)
        return self.cash + position_value


class PaperTrader:
    """Paper trading tracker."""

    def __init__(
        self,
        trade_log_path: Path = TRADE_LOG_PATH,
        params: dict = None,
    ):
        self.trade_log_path = trade_log_path
        self.params = params or DEFAULT_PARAMS.copy()
        self.portfolio = self._load_or_create_portfolio()

    def _load_or_create_portfolio(self) -> PaperPortfolio:
        """Load portfolio from disk or create new one."""
        if self.trade_log_path.exists():
            try:
                with open(self.trade_log_path) as f:
                    data = json.load(f)
                return PaperPortfolio(
                    capital=data.get('capital', self.params['initial_capital']),
                    cash=data.get('cash', self.params['initial_capital']),
                    positions=data.get('positions', []),
                    closed_trades=data.get('closed_trades', []),
                    equity_curve=data.get('equity_curve', []),
                )
            except Exception as e:
                logger.warning(f"Failed to load portfolio: {e}. Creating new.")
        
        capital = self.params['initial_capital']
        return PaperPortfolio(
            capital=capital,
            cash=capital,
            positions=[],
            closed_trades=[],
            equity_curve=[{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'equity': capital,
            }],
        )

    def _save_portfolio(self):
        """Save portfolio to disk."""
        self.trade_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trade_log_path, 'w') as f:
            json.dump(asdict(self.portfolio), f, indent=2)

    def get_signals(self, date: datetime | None = None) -> list[PaperTrade]:
        """Generate trading signals for a given date.
        
        Args:
            date: Date to generate signals for (default: today)
            
        Returns:
            List of PaperTrade signals
        """
        if date is None:
            date = datetime.now()

        # Fetch VIX
        vix_df = fetch_vix(period="6mo")
        if vix_df.empty:
            logger.warning("No VIX data available")
            return []

        vix_regime, current_vix = get_vix_regime(vix_df, self.params)

        if vix_regime == "complacency":
            logger.info("VIX complacency zone — no buys")
            return []

        # Scan stocks
        trades = []
        for symbol in NIFTY_50:
            df = fetch_stock_data(symbol, period="6mo")
            if df.empty or len(df) < 50:
                continue

            mom = compute_momentum(df)
            if not mom:
                continue

            conviction, reason = compute_conviction(mom, vix_regime, self.params)

            if conviction < 30:
                continue

            price = mom['price']
            trades.append(PaperTrade(
                entry_date=date.strftime('%Y-%m-%d'),
                symbol=symbol.replace('.NS', ''),
                entry_price=price,
                target_price=price * (1 + self.params['target_pct']),
                stop_price=price * (1 - self.params['stop_loss_pct']),
                conviction=conviction,
                vix_regime=vix_regime,
            ))

        # Sort by conviction, take top 5
        trades.sort(key=lambda t: t.conviction, reverse=True)
        return trades[:5]

    def execute_signals(self, signals: list[PaperTrade]):
        """Execute paper trades for given signals."""
        if not signals:
            logger.info("No signals to execute")
            return

        for signal in signals:
            # Check if already in position
            if any(p['symbol'] == signal.symbol for p in self.portfolio.positions):
                logger.debug(f"Already in position: {signal.symbol}")
                continue

            # Check if we have enough cash (allocate ~20% per position)
            allocation = self.portfolio.cash * 0.2
            if allocation < signal.entry_price:
                logger.debug(f"Insufficient cash for {signal.symbol}")
                continue

            # Calculate shares (round down to nearest integer)
            shares = int(allocation / signal.entry_price)
            if shares == 0:
                continue

            cost = shares * signal.entry_price
            cost += cost * self.params['transaction_cost']

            position = {
                'symbol': signal.symbol,
                'entry_date': signal.entry_date,
                'entry_price': signal.entry_price,
                'shares': shares,
                'target_price': signal.target_price,
                'stop_price': signal.stop_price,
                'conviction': signal.conviction,
                'vix_regime': signal.vix_regime,
                'cost': cost,
                'current_value': cost,
            }

            self.portfolio.positions.append(position)
            self.portfolio.cash -= cost
            logger.info(
                f"BUY {shares} {signal.symbol} @ ₹{signal.entry_price:.2f} "
                f"(conviction: {signal.conviction:.0f})"
            )

        self._save_portfolio()

    def update_positions(self, date: datetime | None = None):
        """Update open positions (check exits)."""
        if date is None:
            date = datetime.now()

        closed = []
        for position in self.portfolio.positions:
            symbol = position['symbol']
            df = fetch_stock_data(symbol, period="1mo")

            if df.empty:
                continue

            # Get prices since entry
            df = df[df.index > position['entry_date']]
            if df.empty:
                continue

            current_price = df.iloc[-1]['Close']
            position['current_value'] = position['shares'] * current_price

            # Check exit conditions
            exit_price = None
            exit_reason = ""

            for _, row in df.iterrows():
                price = row['Close']

                if price >= position['target_price']:
                    exit_price = position['target_price']
                    exit_reason = "Target hit"
                    break

                if price <= position['stop_price']:
                    exit_price = position['stop_price']
                    exit_reason = "Stop loss hit"
                    break

            # Check holding limit
            entry_date = pd.Timestamp(position['entry_date'])
            days_held = (date - entry_date).days
            if days_held >= self.params['holding_limit'] and exit_price is None:
                exit_price = current_price
                exit_reason = "Holding limit"

            if exit_price is not None:
                gross = position['shares'] * exit_price
                cost = position['shares'] * position['entry_price']
                net_return = (gross - cost) / cost - self.params['transaction_cost']

                trade = PaperTrade(
                    entry_date=position['entry_date'],
                    symbol=position['symbol'],
                    entry_price=position['entry_price'],
                    target_price=position['target_price'],
                    stop_price=position['stop_price'],
                    conviction=position['conviction'],
                    vix_regime=position['vix_regime'],
                    status="CLOSED",
                    exit_date=date.strftime('%Y-%m-%d'),
                    exit_price=exit_price,
                    return_pct=net_return,
                    reason=exit_reason,
                )

                self.portfolio.closed_trades.append(asdict(trade))
                self.portfolio.cash += gross
                closed.append(position)

                logger.info(
                    f"SELL {position['symbol']} @ ₹{exit_price:.2f} "
                    f"({exit_reason}, P&L: {net_return:+.1%})"
                )

        # Remove closed positions
        for pos in closed:
            self.portfolio.positions.remove(pos)

        # Update equity curve
        self.portfolio.equity_curve.append({
            'date': date.strftime('%Y-%m-%d'),
            'equity': self.portfolio.total_value,
        })

        self._save_portfolio()

    def get_status(self) -> dict:
        """Get current portfolio status."""
        closed = self.portfolio.closed_trades
        open_positions = self.portfolio.positions

        # Compute metrics from closed trades
        if closed:
            returns = [t['return_pct'] for t in closed]
            wins = [r for r in returns if r > 0]
            losses = [r for r in returns if r <= 0]
            win_rate = len(wins) / len(returns) if returns else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            total_return = 1.0
            for r in returns:
                total_return *= (1 + r)
            total_return -= 1
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            total_return = 0

        return {
            'capital': self.portfolio.capital,
            'cash': self.portfolio.cash,
            'total_value': self.portfolio.total_value,
            'total_return': (self.portfolio.total_value - self.portfolio.capital) / self.portfolio.capital,
            'open_positions': len(open_positions),
            'closed_trades': len(closed),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'closed_return': total_return,
        }

    def print_status(self):
        """Print portfolio status to console."""
        status = self.get_status()

        print(f"\n{'='*60}")
        print("PAPER TRADING PORTFOLIO — STATUS")
        print("="*60)
        print(f"Capital: ₹{status['capital']:,.0f}")
        print(f"Cash: ₹{status['cash']:,.0f}")
        print(f"Total Value: ₹{status['total_value']:,.0f}")
        print(f"Total Return: {status['total_return']:+.1%}")
        print(f"Open Positions: {status['open_positions']}")
        print(f"Closed Trades: {status['closed_trades']}")
        if status['closed_trades'] > 0:
            print(f"Win Rate: {status['win_rate']:.1%}")
            print(f"Avg Win: {status['avg_win']:+.2%}")
            print(f"Avg Loss: {status['avg_loss']:+.2%}")
        print()

        if self.portfolio.positions:
            print("Open Positions:")
            for pos in self.portfolio.positions:
                current = pos['current_value']
                cost = pos['cost']
                pnl = (current - cost) / cost
                print(
                    f"  {pos['symbol']:<12} {pos['shares']} shares | "
                    f"Entry: ₹{pos['entry_price']:.2f} | "
                    f"Current: ₹{current:,.0f} | "
                    f"P&L: {pnl:+.1%}"
                )

        print("="*60)

    def print_history(self):
        """Print closed trade history."""
        if not self.portfolio.closed_trades:
            print("No closed trades yet.")
            return

        print(f"\n{'='*60}")
        print("CLOSED TRADE HISTORY")
        print("="*60)
        for trade in self.portfolio.closed_trades:
            print(
                f"  {trade['entry_date']} -> {trade['exit_date']}: "
                f"{trade['symbol']:<12} {trade['return_pct']:+.1%} "
                f"({trade['reason']})"
            )
        print("="*60)


def fetch_stock_data(symbol: str, period: str = "6mo") -> pd.DataFrame:
    """Fetch stock data with fallback suffixes."""
    import yfinance as yf
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
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Tracker")
    parser.add_argument('--status', action='store_true', help='Show portfolio status')
    parser.add_argument('--history', action='store_true', help='Show closed trade history')
    parser.add_argument('--signals', action='store_true', help='Show today\'s signals without executing')
    args = parser.parse_args()

    trader = PaperTrader()

    if args.status:
        trader.print_status()
        return

    if args.history:
        trader.print_history()
        return

    if args.signals:
        signals = trader.get_signals()
        if signals:
            print(f"\nToday's signals ({len(signals)}):")
            for s in signals:
                print(f"  {s.symbol:<12} Conviction: {s.conviction:.0f} | Entry: ₹{s.entry_price:.2f}")
        else:
            print("No signals today.")
        return

    # Daily run: update positions, then generate & execute new signals
    print("Running daily paper trading update...")
    trader.update_positions()
    signals = trader.get_signals()
    trader.execute_signals(signals)
    trader.print_status()


if __name__ == "__main__":
    main()
