"""Fusion Model Module - XGBoost model combining CNN embeddings with SMC features"""

from .train_xgb import FusionModel, train_fusion_pipeline
from .shap_explainer import SHAPExplainer
from .infer import FusionInference, ModelEnsemble

__all__ = [
    'FusionModel',
    'train_fusion_pipeline',
    'SHAPExplainer',
    'FusionInference',
    'ModelEnsemble'
]
