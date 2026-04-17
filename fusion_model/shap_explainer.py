"""SHAP explainability for fusion model predictions"""

import shap
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """SHAP-based explainer for fusion model predictions"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
        # Create TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(model)
        
    def explain_prediction(
        self,
        features: np.ndarray,
        top_k: int = 10
    ) -> Dict:
        """Explain a single prediction"""
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(features.reshape(1, -1))
        
        # For multi-class, take the class with highest probability
        pred_probs = self.model.predict_proba(features.reshape(1, -1))[0]
        predicted_class = np.argmax(pred_probs)
        
        # Get SHAP values for predicted class
        shap_class = shap_values[predicted_class][0]
        
        # Create feature importance list
        importance = []
        for i, (name, value) in enumerate(zip(self.feature_names, shap_class)):
            importance.append({
                'feature': name,
                'shap_value': float(value),
                'abs_shap': abs(float(value))
            })
            
        # Sort by absolute SHAP value
        importance.sort(key=lambda x: x['abs_shap'], reverse=True)
        
        return {
            'predicted_class': ['BUY', 'SELL', 'NO_TRADE'][predicted_class],
            'probabilities': {
                'BUY': float(pred_probs[0]),
                'SELL': float(pred_probs[1]),
                'NO_TRADE': float(pred_probs[2])
            },
            'top_features': importance[:top_k]
        }
        
    def explain_batch(
        self,
        features: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:
        """Explain multiple predictions"""
        
        explanations = []
        for i in range(len(features)):
            explanations.append(self.explain_prediction(features[i], top_k))
        return explanations
        
    def get_global_importance(self) -> pd.DataFrame:
        """Calculate global feature importance"""
        
        # Use a small sample for global SHAP
        sample_size = min(1000, len(self.model._Booster.get_score()))
        
        # This would need a background dataset
        # Simplified version using tree feature importance
        importance = self.model.get_booster().get_score(importance_type='gain')
        
        df_importance = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)
        
        return df_importance
        
    def detect_drift(
        self,
        current_features: np.ndarray,
        reference_features: np.ndarray,
        threshold: float = 0.2
    ) -> Dict:
        """Detect feature drift using SHAP distributions"""
        
        # Calculate SHAP for current and reference
        current_shap = self.explainer.shap_values(current_features)
        reference_shap = self.explainer.shap_values(reference_features)
        
        # For multi-class, aggregate across classes
        current_abs = np.abs(np.array(current_shap)).mean(axis=0).mean(axis=0)
        reference_abs = np.abs(np.array(reference_shap)).mean(axis=0).mean(axis=0)
        
        # Calculate PSI-like drift
        drift_scores = {}
        drifted_features = []
        
        for i, name in enumerate(self.feature_names):
            if i < len(current_abs) and i < len(reference_abs):
                # Simple drift detection using distribution shift
                if reference_abs[i] > 0:
                    drift = abs(current_abs[i] - reference_abs[i]) / reference_abs[i]
                else:
                    drift = 0 if current_abs[i] == 0 else 1
                    
                drift_scores[name] = drift
                
                if drift > threshold:
                    drifted_features.append(name)
                    
        return {
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features,
            'drift_scores': drift_scores
        }
