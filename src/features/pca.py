"""PCA dimensionality reduction for VRMS.

Reduces 30 features to 12-15 orthogonal components.
Fit on training data only, transform test data.
"""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class PCAReducer:
    """PCA reducer with fit/transform separation.
    
    Fit on training data only, transform both train and test.
    """
    
    def __init__(self, n_components: int = 15, variance_threshold: float = 0.95):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.pca = PCA(n_components=n_components)
        self._is_fitted = False
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit PCA on training data and transform.
        
        Args:
            df: Training feature DataFrame
            
        Returns:
            Transformed DataFrame with PC columns
        """
        # Drop NaN rows for fitting
        clean_df = df.dropna()
        
        if len(clean_df) < self.n_components:
            logger.warning(f"Not enough data for PCA: {len(clean_df)} < {self.n_components}")
            return df
        
        # Fit PCA
        self.pca.fit(clean_df)
        self._is_fitted = True
        
        # Transform
        transformed = self.pca.transform(clean_df)
        
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
        transformed = self.pca.transform(clean_df)
        
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
