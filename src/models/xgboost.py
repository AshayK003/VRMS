"""XGBoost classifier for VRMS."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


class XGBoostClassifier:
    """XGBoost classifier with walk-forward training."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1,
        )
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Label array (n_samples,)
        """
        # Drop NaN rows
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask].astype(np.int32)
        
        self.model.fit(X_clean, y_clean)
        self._is_fitted = True
        
        logger.info(f"Model fitted on {len(X_clean)} samples")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels.
        
        Args:
            X: Feature array
            
        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit first.")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.
        
        Args:
            X: Feature array
            
        Returns:
            Probabilities of class 1 (WIN)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit first.")
        
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance.
        
        Returns:
            Series with feature importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        
        importance = self.model.feature_importances_
        return pd.Series(importance, index=[f'PC{i+1}' for i in range(len(importance))])
    
    def save(self, path: str | Path) -> None:
        """Save model to disk.
        
        Args:
            path: File path
        """
        self.model.save_model(str(path))
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str | Path) -> None:
        """Load model from disk.
        
        Args:
            path: File path
        """
        self.model.load_model(str(path))
        self._is_fitted = True
        logger.info(f"Model loaded from {path}")
