"""Test paper trading module."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from run_paper_trader import PaperTrader, PaperTrade


class TestPaperTrade:
    def test_creation(self):
        trade = PaperTrade(
            entry_date="2024-01-01",
            symbol="TMCV",
            entry_price=100.0,
            target_price=105.0,
            stop_price=97.0,
            conviction=50,
            vix_regime="fear",
        )
        assert trade.status == "OPEN"
        assert trade.symbol == "TMCV"


class TestPaperTrader:
    def test_new_portfolio_creation(self):
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trades.json"
            trader = PaperTrader(trade_log_path=log_path)
            assert trader.portfolio.capital == 100000
            assert trader.portfolio.cash == 100000
            assert len(trader.portfolio.positions) == 0
    
    def test_save_and_load_portfolio(self):
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trades.json"
            trader = PaperTrader(trade_log_path=log_path)
            trader._save_portfolio()
            
            # Load back
            trader2 = PaperTrader(trade_log_path=log_path)
            assert trader2.portfolio.capital == trader.portfolio.capital
    
    def test_get_status_empty(self):
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trades.json"
            trader = PaperTrader(trade_log_path=log_path)
            status = trader.get_status()
            assert status['closed_trades'] == 0
            assert status['capital'] == 100000


class TestPaperTradeExecution:
    def test_execute_signals_empty(self):
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "trades.json"
            trader = PaperTrader(trade_log_path=log_path)
            trader.execute_signals([])
            assert len(trader.portfolio.positions) == 0
