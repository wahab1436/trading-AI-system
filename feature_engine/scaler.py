"""Feature scaling utilities for normalization and standardization"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureScalerConfig:
    """Configuration for feature scaling"""
    scaler_type: str = "standard"  # standard, minmax, robust
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    per_feature: bool = True


class FeatureScaler:
    """
    Handles feature scaling for SMC features.
    Supports multiple scaling strategies and maintains fit for production.
    """
    
    def __init__(
        self,
        config: Optional[FeatureScalerConfig] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.config = config or FeatureScalerConfig()
        self.feature_names = feature_names
        self.scaler = self._create_scaler()
        self.is_fitted = False
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        
    def _create_scaler(self):
        """Create the appropriate scaler based on config"""
        if self.config.scaler_type == "standard":
            return StandardScaler()
        elif self.config.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.config.scaler_type == "robust":
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")
            
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> 'FeatureScaler':
        """
        Fit scaler on training data
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            self for method chaining
        """
        # Convert to numpy if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = X
            
        # Fit scaler
        self.scaler.fit(X_array)
        self.is_fitted = True
        
        # Store feature ranges
        for i in range(X_array.shape[1]):
            feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            self.feature_ranges[feature_name] = (
                float(X_array[:, i].min()),
                float(X_array[:, i].max())
            )
            
        logger.info(f"Scaler fitted on {X_array.shape[0]} samples, {X_array.shape[1]} features")
        return self
        
    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform features using fitted scaler
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
            
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Apply scaling
        X_scaled = self.scaler.transform(X_array)
        
        # Apply clipping if configured
        if self.config.clip_min is not None:
            X_scaled = np.clip(X_scaled, self.config.clip_min, self.config.clip_max)
        elif self.config.clip_max is not None:
            X_scaled = np.clip(X_scaled, None, self.config.clip_max)
            
        return X_scaled
        
    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Fit scaler and transform in one step
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed numpy array
        """
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled features back to original space
        
        Args:
            X_scaled: Scaled feature matrix
            
        Returns:
            Original scale features
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
            
        return self.scaler.inverse_transform(X_scaled)
        
    def get_feature_statistics(self) -> Dict:
        """
        Get statistics about fitted features
        
        Returns:
            Dictionary with mean, std, min, max for each feature
        """
        if not self.is_fitted:
            return {}
            
        stats = {}
        
        if self.config.scaler_type == "standard":
            means = self.scaler.mean_
            stds = self.scaler.scale_
            
            for i, name in enumerate(self.feature_names or range(len(means))):
                stats[str(name)] = {
                    'mean': float(means[i]),
                    'std': float(stds[i]),
                    'min': self.feature_ranges.get(str(name), (None, None))[0],
                    'max': self.feature_ranges.get(str(name), (None, None))[1]
                }
        elif self.config.scaler_type == "minmax":
            mins = self.scaler.data_min_
            maxs = self.scaler.data_max_
            
            for i, name in enumerate(self.feature_names or range(len(mins))):
                stats[str(name)] = {
                    'min_raw': float(mins[i]),
                    'max_raw': float(maxs[i]),
                    'min_scaled': 0.0,
                    'max_scaled': 1.0
                }
                
        return stats
        
    def save(self, path: Path):
        """Save scaler to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'scaler': self.scaler,
            'config': self.config,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'feature_ranges': self.feature_ranges
        }
        
        with open(path / 'feature_scaler.pkl', 'wb') as f:
            pickle.dump(save_data, f)
            
        logger.info(f"Feature scaler saved to {path}")
        
    def load(self, path: Path):
        """Load scaler from disk"""
        path = Path(path)
        
        with open(path / 'feature_scaler.pkl', 'rb') as f:
            save_data = pickle.load(f)
            
        self.scaler = save_data['scaler']
        self.config = save_data['config']
        self.feature_names = save_data['feature_names']
        self.is_fitted = save_data['is_fitted']
        self.feature_ranges = save_data['feature_ranges']
        
        logger.info(f"Feature scaler loaded from {path}")


class OnlineFeatureScaler:
    """
    Online feature scaler that can be updated incrementally.
    Useful for production systems where new data arrives continuously.
    """
    
    def __init__(self, n_features: int, alpha: float = 0.01):
        """
        Args:
            n_features: Number of features
            alpha: Running average update rate (0-1)
        """
        self.n_features = n_features
        self.alpha = alpha
        self.running_mean = np.zeros(n_features)
        self.running_std = np.ones(n_features)
        self.n_samples = 0
        
    def update(self, X: np.ndarray):
        """
        Update running statistics with new samples
        
        Args:
            X: New feature samples (n_samples, n_features)
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        batch_mean = np.mean(X, axis=0)
        batch_std = np.std(X, axis=0)
        batch_n = X.shape[0]
        
        # Update running mean
        if self.n_samples == 0:
            self.running_mean = batch_mean
        else:
            self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * batch_mean
            
        # Update running std
        self.running_std = (1 - self.alpha) * self.running_std + self.alpha * batch_std
        self.running_std = np.maximum(self.running_std, 1e-8)  # Prevent division by zero
        
        self.n_samples += batch_n
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize features using running statistics
        
        Args:
            X: Features to standardize
            
        Returns:
            Standardized features
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        return (X - self.running_mean) / self.running_std
        
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform standardized features
        
        Args:
            X_scaled: Standardized features
            
        Returns:
            Original scale features
        """
        if len(X_scaled.shape) == 1:
            X_scaled = X_scaled.reshape(1, -1)
            
        return X_scaled * self.running_std + self.running_mean
        
    def get_state(self) -> Dict:
        """Get current state for serialization"""
        return {
            'running_mean': self.running_mean.tolist(),
            'running_std': self.running_std.tolist(),
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'alpha': self.alpha
        }
        
    def set_state(self, state: Dict):
        """Restore state from serialization"""
        self.running_mean = np.array(state['running_mean'])
        self.running_std = np.array(state['running_std'])
        self.n_samples = state['n_samples']
        self.n_features = state['n_features']
        self.alpha = state['alpha']


class FeatureNormalizer:
    """
    Handles per-feature normalization with outlier clipping.
    Useful for features with known ranges (e.g., distances in ATR units).
    """
    
    def __init__(self, feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Args:
            feature_bounds: Dictionary mapping feature names to (min, max) bounds
        """
        self.feature_bounds = feature_bounds or {}
        self._default_bounds = {
            'hh_hl_ratio': (0, 1),
            'lh_ll_ratio': (0, 1),
            'dist_nearest_bull_ob': (0, 20),
            'dist_nearest_bear_ob': (0, 20),
            'bull_ob_strength': (0, 5),
            'bear_ob_strength': (0, 5),
            'liq_high_distance': (0, 20),
            'liq_low_distance': (0, 20),
            'impulse_strength': (0, 3),
            'volatility_regime': (0, 1),
            'spread_pips': (0, 10),
            'time_of_day': (0, 1)
        }
        
    def normalize(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize features to [0, 1] range
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Normalized features
        """
        normalized = {}
        
        for name, value in features.items():
            # Get bounds
            if name in self.feature_bounds:
                min_val, max_val = self.feature_bounds[name]
            elif name in self._default_bounds:
                min_val, max_val = self._default_bounds[name]
            else:
                # No bounds defined, assume already normalized
                normalized[name] = value
                continue
                
            # Clip to bounds
            clipped = max(min_val, min(value, max_val))
            
            # Normalize
            if max_val > min_val:
                normalized[name] = (clipped - min_val) / (max_val - min_val)
            else:
                normalized[name] = 0.5
                
        return normalized
        
    def denormalize(self, normalized_features: Dict[str, float]) -> Dict[str, float]:
        """
        Convert normalized features back to original scale
        
        Args:
            normalized_features: Normalized feature values (0-1)
            
        Returns:
            Original scale features
        """
        original = {}
        
        for name, norm_value in normalized_features.items():
            if name in self.feature_bounds:
                min_val, max_val = self.feature_bounds[name]
            elif name in self._default_bounds:
                min_val, max_val = self._default_bounds[name]
            else:
                original[name] = norm_value
                continue
                
            original[name] = min_val + norm_value * (max_val - min_val)
            
        return original
        
    def clip_outliers(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Clip outlier values to reasonable bounds
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Clipped features
        """
        clipped = {}
        
        for name, value in features.items():
            if name in self.feature_bounds:
                min_val, max_val = self.feature_bounds[name]
            elif name in self._default_bounds:
                min_val, max_val = self._default_bounds[name]
            else:
                clipped[name] = value
                continue
                
            clipped[name] = max(min_val, min(value, max_val))
            
        return clipped


def save_scaler(scaler: FeatureScaler, path: Path):
    """Convenience function to save scaler"""
    scaler.save(path)
    
    
def load_scaler(path: Path) -> FeatureScaler:
    """Convenience function to load scaler"""
    scaler = FeatureScaler()
    scaler.load(path)
    return scaler
