"""Feature Engine Module - Builds and scales feature vectors for the fusion model"""

from .builder import FeatureBuilder, build_smc_feature_vector
from .scaler import FeatureScaler, load_scaler, save_scaler

__all__ = [
    'FeatureBuilder',
    'build_smc_feature_vector',
    'FeatureScaler',
    'load_scaler',
    'save_scaler'
]
