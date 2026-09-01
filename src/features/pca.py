"""PCA dimensionality reduction for VRMS.

Reduces features to orthogonal components.
Fit on training data only, transform test data.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class PCAReducer:
    """PCA reducer with fit/transform separation.
    
    Fit on training data only, transform both train and test.
    """
    
    def __init__(self, n_components: int = 10, variance_threshold: float = 0.95):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = PCA(n_components=n_components)
        self._is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit PCA on training data and transform."""
        clean_df = df.dropna()
        
        if len(clean_df) < 2:
            return df
        
        # Adjust n_components to not exceed min(n_samples, n_features)
        max_components = min(len(clean_df), len(clean_df.columns))
        if self.n_components > max_components:
            self.n_components = max_components
            self.pca = PCA(n_components=self.n_components)
        
        if len(clean_df) < self.n_components:
            return df
        
        # Use numpy to avoid memory issues
        X = clean_df.values.astype(np.float32)
        self.pca.fit(X)
        self._is_fitted = True
        
        # Transform
        transformed = self.pca.transform(X)
        
        # Create column names
        columns = [f'PC{i+1}' for i in range(self.n_components)]
        
        return pd.DataFrame(
            transformed,
            index=clean_df.index,
            columns=columns
        )
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted PCA.
        
        Args:
            df: Test feature DataFrame
            
        Returns:
            Transformed DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted. Call fit_transform first.")
        
        clean_df = df.dropna()
        X = clean_df.values.astype(np.float32)
        transformed = self.pca.transform(X)
        
        columns = [f'PC{i+1}' for i in range(self.n_components)]
        
        return pd.DataFrame(
            transformed,
            index=clean_df.index,
            columns=columns
        )
    
    def get_explained_variance(self) -> pd.Series:
        """Get explained variance ratio for each component.
        
        Returns:
            Series with variance ratios
        """
        if not self._is_fitted:
            raise RuntimeError("PCA not fitted.")
        
        return pd.Series(
            self.pca.explained_variance_ratio_,
            index=[f'PC{i+1}' for i in range(self.n_components)]
        )
