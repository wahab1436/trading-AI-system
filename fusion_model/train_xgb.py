"""XGBoost fusion model training"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple, Dict, Optional
import pickle
import logging
from pathlib import Path
import mlflow

logger = logging.getLogger(__name__)


class FusionModel:
    """XGBoost model that fuses CNN embeddings with SMC features"""
    
    def __init__(
        self,
        cnn_embedding_dim: int = 1536,
        pca_components: int = 128,
        xgb_params: Optional[Dict] = None
    ):
        self.cnn_embedding_dim = cnn_embedding_dim
        self.pca_components = pca_components
        
        # PCA for dimensionality reduction
        self.pca = PCA(n_components=pca_components, whiten=False)
        self.scaler = StandardScaler()
        
        # XGBoost parameters
        self.xgb_params = xgb_params or {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'early_stopping_rounds': 30
        }
        
        self.model = None
        
    def prepare_features(
        self,
        cnn_embeddings: np.ndarray,
        smc_features: np.ndarray
    ) -> np.ndarray:
        """Prepare features by applying PCA to CNN embeddings"""
        
        # Apply PCA to reduce dimensionality
        if self.pca.components_.shape[0] == 0:
            # Fit PCA if not already fitted
            cnn_pca = self.pca.fit_transform(cnn_embeddings)
        else:
            cnn_pca = self.pca.transform(cnn_embeddings)
            
        # Combine with SMC features
        combined = np.hstack([cnn_pca, smc_features])
        
        # Scale features
        if self.scaler.mean_ is None:
            combined = self.scaler.fit_transform(combined)
        else:
            combined = self.scaler.transform(combined)
            
        return combined
        
    def train(
        self,
        cnn_embeddings: np.ndarray,
        smc_features: np.ndarray,
        labels: np.ndarray,
        validation_split: float = 0.15,
        time_series_cv: bool = True
    ) -> Dict:
        """Train the fusion model"""
        
        # Prepare features
        X = self.prepare_features(cnn_embeddings, smc_features)
        y = labels
        
        # Convert labels to 0-2 (XGBoost requires 0-indexed)
        label_map = {1: 0, -1: 1, 0: 2}
        y = np.array([label_map[l] for l in y])
        
        # Time series split for validation
        if time_series_cv:
            tscv = TimeSeriesSplit(n_splits=5, gap=50)
            
            cv_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBClassifier(**self.xgb_params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                score = model.score(X_val, y_val)
                cv_scores.append(score)
                
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV score: {np.mean(cv_scores):.4f}")
            
        # Train final model on all data
        self.model = xgb.XGBClassifier(**self.xgb_params)
        self.model.fit(X, y)
        
        # Feature importance
        importance = dict(zip(
            [f'pca_{i}' for i in range(self.pca_components)] + 
            [f'smc_{i}' for i in range(smc_features.shape[1])],
            self.model.feature_importances_
        ))
        
        return {
            'cv_scores': cv_scores if time_series_cv else None,
            'mean_cv_score': np.mean(cv_scores) if time_series_cv else None,
            'feature_importance': importance
        }
        
    def predict(
        self,
        cnn_embedding: np.ndarray,
        smc_features: np.ndarray
    ) -> Dict[str, float]:
        """Predict probabilities for a single sample"""
        
        X = self.prepare_features(
            cnn_embedding.reshape(1, -1),
            smc_features.reshape(1, -1)
        )
        
        probs = self.model.predict_proba(X)[0]
        
        # Map back to original labels: 0=BUY, 1=SELL, 2=NO_TRADE
        return {
            'buy_prob': probs[0],
            'sell_prob': probs[1],
            'notrade_prob': probs[2]
        }
        
    def predict_batch(
        self,
        cnn_embeddings: np.ndarray,
        smc_features: np.ndarray
    ) -> np.ndarray:
        """Predict probabilities for batch"""
        
        X = self.prepare_features(cnn_embeddings, smc_features)
        return self.model.predict_proba(X)
        
    def save(self, path: Path):
        """Save model to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        with open(path / 'fusion_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.model,
                'pca': self.pca,
                'scaler': self.scaler,
                'xgb_params': self.xgb_params,
                'pca_components': self.pca_components
            }, f)
            
        logger.info(f"Model saved to {path}")
        
    def load(self, path: Path):
        """Load model from disk"""
        path = Path(path)
        
        with open(path / 'fusion_model.pkl', 'rb') as f:
            data = pickle.load(f)
            
        self.model = data['model']
        self.pca = data['pca']
        self.scaler = data['scaler']
        self.xgb_params = data['xgb_params']
        self.pca_components = data['pca_components']
        
        logger.info(f"Model loaded from {path}")


def train_fusion_pipeline(
    cnn_embeddings_path: Path,
    smc_features_path: Path,
    labels_path: Path,
    output_path: Path
):
    """Complete training pipeline for fusion model"""
    
    # Load data
    cnn_embeddings = np.load(cnn_embedd
