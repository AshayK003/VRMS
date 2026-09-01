"""XGBoost classifier for VRMS."""
from __future__ import annotations

import logging
from pathlib import Path

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
        )
        self._is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model.
        
        Args:
            X: Feature DataFrame
            y: Label Series
        """
        # Drop NaN rows
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        self.model.fit(X_clean, y_clean)
        self._is_fitted = True
        
        logger.info(f"Model fitted on {len(X_clean)} samples")
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict labels.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predicted labels
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit first.")
        
        predictions = self.model.predict(X)
        return pd.Series(predictions, index=X.index)
    
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Predict probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Probabilities of class 1 (WIN)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit first.")
        
        probas = self.model.predict_proba(X)[:, 1]
        return pd.Series(probas, index=X.index)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance.
        
        Returns:
            Series with feature importance scores
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.model.get_booster().feature_names)
    
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
