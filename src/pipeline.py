"""End-to-end pipeline for VRMS.

Fetches data, computes features, trains model, generates signals.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.data.ohlcv import fetch_ohlcv, get_benchmark, fetch_vix
from src.data.constituents import get_constituents_on_date

# Ticker aliases for demerged/rename symbols
# Maps old/delisted tickers to current active tickers
TICKER_ALIASES = {
    'TATAMOTORS': 'TMCV.NS',  # Demerged Nov 2025 → Tata Motors PV
}
from src.data.fii_dii import fetch_fii_dii
from src.data.constituents import get_constituents_on_date
from src.data.validator import DataValidator
from src.features.engineering import (
    compute_realized_vol,
    compute_momentum,
    compute_relative_strength,
    compute_volume_features,
    compute_adx,
    compute_rsi,
    compute_atr,
    compute_garch_vol,
)
from src.features.labels import generate_labels
from src.features.pca import PCAReducer
from src.models.xgboost import XGBoostClassifier
from src.signals.generator import generate_signals
from src.signals.sizing import calc_position_size_with_governance

logger = logging.getLogger(__name__)


class VRMSPipeline:
    """End-to-end pipeline for signal generation."""
    
    def __init__(
        self,
        data_dir: str | Path = "data",
        model_dir: str | Path = "models",
        train_window: int = 252,
        n_features_pca: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_window = train_window
        self.n_features_pca = n_features_pca
        self.validator = DataValidator()
        
        self.model = XGBoostClassifier()
        self.pca = None  # Will be set during training
        
        self._is_trained = False
    
    def fetch_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols.
        
        Args:
            symbols: List of NSE tickers
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            
        Returns:
            Dict mapping symbol to OHLCV DataFrame
        """
        data = {}
        
        for symbol in symbols:
            logger.info(f"Fetching {symbol}...")
            df = fetch_ohlcv(symbol, start_date, end_date)
            
            is_valid, reason = self.validator.validate_ohlcv(df)
            if not is_valid:
                logger.warning(f"Invalid data for {symbol}: {reason}")
                continue
            
            data[symbol] = df
            
            # Save to disk
            path = self.data_dir / "raw" / f"{symbol}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path)
        
        logger.info(f"Fetched data for {len(data)} symbols")
        return data
    
    def compute_features(
        self,
        stock_data: dict[str, pd.DataFrame],
        benchmark: pd.DataFrame,
        fii_dii_df: pd.DataFrame | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Compute features for all stocks."""
        features = {}
        
        for symbol, df in stock_data.items():
            try:
                all_features = []
                
                # Volatility (short windows only to reduce NaN)
                vol = compute_realized_vol(df, windows=[5, 10, 20])
                all_features.append(vol)
                
                # GARCH vol
                garch = compute_garch_vol(df)
                all_features.append(pd.DataFrame({'garch_vol': garch}, index=df.index))
                
                # Momentum (short windows only)
                mom = compute_momentum(df, windows=[21, 63])
                all_features.append(mom)
                
                # Relative strength (short windows only)
                rs = compute_relative_strength(df, benchmark, windows=[21])
                all_features.append(rs)
                
                # Volume
                vol_feat = compute_volume_features(df)
                all_features.append(vol_feat)
                
                # ADX
                adx = compute_adx(df)
                all_features.append(pd.DataFrame({'adx': adx}, index=df.index))
                
                # RSI
                rsi = compute_rsi(df)
                all_features.append(pd.DataFrame({'rsi': rsi}, index=df.index))
                
                # ATR
                atr = compute_atr(df)
                all_features.append(pd.DataFrame({'atr': atr}, index=df.index))
                
                # Combine
                combined = pd.concat(all_features, axis=1)
                combined = combined.dropna(how='all')
                
                features[symbol] = combined
                
            except Exception as e:
                logger.error(f"Feature computation failed for {symbol}: {e}")
                continue
        
        logger.info(f"Computed features for {len(features)} symbols")
        return features
    
    def train(
        self,
        features: dict[str, pd.DataFrame],
        labels: dict[str, pd.DataFrame],
    ) -> dict:
        """Train XGBoost model.
        
        Args:
            features: Dict mapping symbol to feature DataFrame
            labels: Dict mapping symbol to label DataFrame
            
        Returns:
            Training metrics
        """
        # Combine all stocks into single training set
        all_X = []
        all_y = []
        
        for symbol in features:
            if symbol not in labels:
                continue
            
            X = features[symbol]
            y = labels[symbol]['label']
            
            # Align
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            # Drop NaN
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            all_X.append(X.values.astype(np.float32))
            all_y.append(y.values.astype(np.int32))
        
        if not all_X:
            raise ValueError("No training data available")
        
        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)
        
        logger.info(f"Training on {len(X_train)} samples")
        
        # PCA reduction using numpy
        # Drop NaN rows
        mask = ~np.isnan(X_train).any(axis=1)
        X_clean = X_train[mask]
        y_clean = y_train[mask]
        
        # Adjust n_components
        max_components = min(X_clean.shape[0], X_clean.shape[1])
        if self.n_features_pca > max_components:
            self.n_features_pca = max_components
        
        self.pca = PCA(n_components=self.n_features_pca)
        X_pca = self.pca.fit_transform(X_clean)
        
        # Train model
        self.model.fit(X_pca, y_clean)
        self._is_trained = True
        
        # Save model
        model_path = self.model_dir / "xgb_v1.json"
        self.model.save(model_path)
        
        # Feature importance
        importance = self.model.get_feature_importance()
        
        metrics = {
            "n_samples": len(X_pca),
            "n_features": X_pca.shape[1],
            "explained_variance": self.pca.explained_variance_ratio_.tolist(),
        }
        
        logger.info(f"Model trained: {X_pca.shape[1]} features, {len(X_pca)} samples")
        return metrics
    
    def predict(
        self,
        features: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate predictions for all stocks.
        
        Args:
            features: Dict mapping symbol to feature DataFrame
            
        Returns:
            DataFrame with predictions for each symbol
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train first.")
        
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Train the model first.")
        
        predictions = []
        
        for symbol, X in features.items():
            try:
                # Convert to numpy and drop NaN rows
                X_arr = X.values.astype(np.float32)
                mask = ~np.isnan(X_arr).any(axis=1)
                X_clean = X_arr[mask]
                
                if len(X_clean) == 0:
                    continue
                
                # Apply PCA transformation
                X_pca = self.pca.transform(X_clean)
                
                # Get latest row
                latest_X = X_pca[-1:]
                
                # Predict
                prob = self.model.predict_proba(latest_X)[0]
                
                predictions.append({
                    'symbol': symbol,
                    'probability': prob,
                    'momentum': X['mom_21d'].iloc[-1] if 'mom_21d' in X.columns and len(X) > 0 else 0,
                    'rs': X['rs_21d'].iloc[-1] if 'rs_21d' in X.columns and len(X) > 0 else 0,
                })
                
            except Exception as e:
                logger.error(f"Prediction failed for {symbol}: {e}")
                continue
        
        return pd.DataFrame(predictions)
    
    def generate_daily_signals(
        self,
        predictions: pd.DataFrame,
        vix: float | None = None,
        adx: float | None = None,
        top_n: int = 5,
    ) -> list[dict]:
        """Generate daily signals from predictions.
        
        Args:
            predictions: Prediction DataFrame
            vix: Current VIX
            adx: Current ADX
            top_n: Number of signals
            
        Returns:
            List of signal dicts
        """
        signals = generate_signals(
            predictions,
            top_n=top_n,
            vix=vix,
            adx=adx,
        )
        
        return signals
    
    def run_pipeline(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict:
        """Run the full pipeline.
        
        Args:
            symbols: List of symbols (default: Nifty 50)
            start_date: Start date (default: 5 years ago)
            end_date: End date (default: today)
            
        Returns:
            Dict with signals and metrics
        """
        if symbols is None:
            symbols = get_constituents_on_date(datetime.now())
        
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
        
        # Fetch data
        stock_data = self.fetch_data(symbols, start_date, end_date)
        benchmark = get_benchmark(start_date, end_date)
        fii_dii_df = fetch_fii_dii()
        
        # Compute features
        features = self.compute_features(stock_data, benchmark, fii_dii_df)
        
        # Generate labels
        labels = {}
        for symbol, df in stock_data.items():
            labels[symbol] = generate_labels(df)
        
        # Train model
        metrics = self.train(features, labels)
        
        # Generate predictions
        predictions = self.predict(features)
        
        # Get current market context
        vix = fetch_vix()
        adx = None
        if not benchmark.empty:
            adx_series = compute_adx(benchmark)
            adx = adx_series.iloc[-1] if len(adx_series) > 0 else None
        
        # Generate signals
        signals = self.generate_daily_signals(predictions, vix=vix, adx=adx)
        
        return {
            "signals": signals,
            "metrics": metrics,
            "vix": vix,
            "adx": adx,
            "n_stocks": len(stock_data),
            "date": end_date,
        }
