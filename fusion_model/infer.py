"""Inference wrapper for fusion model with caching and batching support"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
import hashlib
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from fusion model inference"""
    signal: str  # BUY, SELL, NO_TRADE
    direction: int  # 1, -1, 0
    confidence: float
    buy_prob: float
    sell_prob: float
    notrade_prob: float
    timestamp: datetime
    model_version: str
    inference_time_ms: float
    feature_hash: str
    top_features: Optional[List[Dict]] = None
    
    def to_dict(self) -> dict:
        return {
            'signal': self.signal,
            'direction': self.direction,
            'confidence': self.confidence,
            'buy_prob': self.buy_prob,
            'sell_prob': self.sell_prob,
            'notrade_prob': self.notrade_prob,
            'timestamp': self.timestamp.isoformat(),
            'model_version': self.model_version,
            'inference_time_ms': self.inference_time_ms,
            'feature_hash': self.feature_hash,
            'top_features': self.top_features
        }


class PredictionCache:
    """LRU cache for predictions to avoid redundant computation"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 60):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.Lock()
        
    def _make_key(self, cnn_embedding_hash: str, smc_features_hash: str) -> str:
        """Create cache key from feature hashes"""
        return f"{cnn_embedding_hash}:{smc_features_hash}"
        
    def get(self, cnn_embedding_hash: str, smc_features_hash: str) -> Optional[InferenceResult]:
        """Get cached prediction if valid"""
        key = self._make_key(cnn_embedding_hash, smc_features_hash)
        
        with self.lock:
            if key in self.cache:
                result, timestamp = self.cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    # Move to end (most recent)
                    self.cache.move_to_end(key)
                    return result
                else:
                    # Expired
                    del self.cache[key]
        return None
        
    def put(self, cnn_embedding_hash: str, smc_features_hash: str, result: InferenceResult):
        """Cache prediction result"""
        key = self._make_key(cnn_embedding_hash, smc_features_hash)
        
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest
                self.cache.popitem(last=False)
            self.cache[key] = (result, datetime.now())
            
    def clear(self):
        """Clear all cached predictions"""
        with self.lock:
            self.cache.clear()
            
    def size(self) -> int:
        return len(self.cache)


class FeatureHasher:
    """Generate deterministic hashes from feature vectors for caching"""
    
    @staticmethod
    def hash_cnn_embedding(embedding: np.ndarray) -> str:
        """Create hash from CNN embedding vector"""
        # Round to 6 decimal places for consistency
        rounded = np.round(embedding, decimals=6)
        return hashlib.md5(rounded.tobytes()).hexdigest()
        
    @staticmethod
    def hash_smc_features(features: Dict) -> str:
        """Create hash from SMC feature dictionary"""
        # Sort keys for consistency
        feature_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(feature_str.encode()).hexdigest()
        
    @staticmethod
    def hash_combined(cnn_hash: str, smc_hash: str) -> str:
        """Create combined hash"""
        return hashlib.md5(f"{cnn_hash}:{smc_hash}".encode()).hexdigest()


class FusionInference:
    """Main inference wrapper for fusion model with caching and batching"""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        cache_size: int = 1000,
        cache_ttl_seconds: int = 60,
        confidence_threshold: float = 0.65,
        batch_size: int = 32
    ):
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        
        # Initialize components
        self.cache = PredictionCache(max_size=cache_size, ttl_seconds=cache_ttl_seconds)
        self.hasher = FeatureHasher()
        
        # Model will be loaded lazily
        self._model = None
        self._model_version = None
        self._model_loaded_time = None
        
        # Thread pool for batch processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_inference_time_ms': 0.0,
            'signal_counts': {'BUY': 0, 'SELL': 0, 'NO_TRADE': 0}
        }
        
    @property
    def model(self):
        """Lazy load model"""
        if self._model is None and self.model_path:
            self._load_model()
        return self._model
        
    def _load_model(self):
        """Load fusion model from disk"""
        try:
            from .train_xgb import FusionModel
            
            self._model = FusionModel()
            
            if self.model_path and self.model_path.exists():
                self._model.load(self.model_path)
                self._model_version = self.model_path.name
                logger.info(f"Loaded fusion model from {self.model_path}")
            else:
                # Try default path
                default_path = Path("models/fusion_model")
                if default_path.exists():
                    self._model.load(default_path)
                    self._model_version = default_path.name
                    logger.info(f"Loaded fusion model from {default_path}")
                else:
                    logger.warning("No fusion model found, using default predictions")
                    
            self._model_loaded_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None
            
    def predict_single(
        self,
        cnn_embedding: np.ndarray,
        smc_features: Dict[str, float],
        return_explanation: bool = False
    ) -> InferenceResult:
        """Get prediction for a single sample"""
        
        start_time = datetime.now()
        
        # Generate hashes for caching
        cnn_hash = self.hasher.hash_cnn_embedding(cnn_embedding)
        smc_hash = self.hasher.hash_smc_features(smc_features)
        
        # Check cache
        cached = self.cache.get(cnn_hash, smc_hash)
        if cached:
            self.stats['cache_hits'] += 1
            self.stats['total_predictions'] += 1
            return cached
            
        self.stats['cache_misses'] += 1
        
        # Convert SMC features to array
        smc_array = self._dict_to_array(smc_features)
        
        # Get prediction from model
        if self.model is not None:
            probs = self.model.predict(cnn_embedding, smc_array)
        else:
            # Fallback: simple heuristic
            probs = self._fallback_prediction(smc_features)
            
        # Determine signal
        signal, direction, confidence = self._determine_signal(probs)
        
        # Get explanation if requested
        top_features = None
        if return_explanation and self.model is not None:
            top_features = self._get_explanation(cnn_embedding, smc_array)
            
        inference_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        result = InferenceResult(
            signal=signal,
            direction=direction,
            confidence=confidence,
            buy_prob=probs['buy_prob'],
            sell_prob=probs['sell_prob'],
            notrade_prob=probs['notrade_prob'],
            timestamp=datetime.now(),
            model_version=self._model_version or "default",
            inference_time_ms=inference_time_ms,
            feature_hash=FeatureHasher.hash_combined(cnn_hash, smc_hash),
            top_features=top_features
        )
        
        # Cache result
        self.cache.put(cnn_hash, smc_hash, result)
        
        # Update statistics
        self.stats['total_predictions'] += 1
        self.stats['signal_counts'][signal] += 1
        self.stats['avg_inference_time_ms'] = (
            (self.stats['avg_inference_time_ms'] * (self.stats['total_predictions'] - 1) +
             inference_time_ms) / self.stats['total_predictions']
        )
        
        return result
        
    def predict_batch(
        self,
        cnn_embeddings: List[np.ndarray],
        smc_features_list: List[Dict[str, float]],
        return_explanations: bool = False
    ) -> List[InferenceResult]:
        """Get predictions for multiple samples with batching"""
        
        results = []
        
        # Process in batches
        for i in range(0, len(cnn_embeddings), self.batch_size):
            batch_end = min(i + self.batch_size, len(cnn_embeddings))
            batch_cnn = cnn_embeddings[i:batch_end]
            batch_smc = smc_features_list[i:batch_end]
            
            # Process batch
            batch_results = self._predict_batch_internal(
                batch_cnn, batch_smc, return_explanations
            )
            results.extend(batch_results)
            
        return results
        
    def _predict_batch_internal(
        self,
        cnn_embeddings: List[np.ndarray],
        smc_features_list: List[Dict[str, float]],
        return_explanations: bool
    ) -> List[InferenceResult]:
        """Internal batch prediction"""
        
        results = []
        
        if self.model is not None and len(cnn_embeddings) > 0:
            # Stack embeddings
            cnn_stack = np.vstack(cnn_embeddings)
            
            # Stack SMC features
            smc_stack = np.vstack([self._dict_to_array(smc) for smc in smc_features_list])
            
            # Batch prediction
            probs_batch = self.model.predict_batch(cnn_stack, smc_stack)
            
            for i, probs in enumerate(probs_batch):
                probs_dict = {
                    'buy_prob': probs[0],
                    'sell_prob': probs[1],
                    'notrade_prob': probs[2]
                }
                signal, direction, confidence = self._determine_signal(probs_dict)
                
                result = InferenceResult(
                    signal=signal,
                    direction=direction,
                    confidence=confidence,
                    buy_prob=probs[0],
                    sell_prob=probs[1],
                    notrade_prob=probs[2],
                    timestamp=datetime.now(),
                    model_version=self._model_version or "default",
                    inference_time_ms=0,
                    feature_hash=""
                )
                results.append(result)
                
        return results
        
    def _dict_to_array(self, smc_features: Dict[str, float]) -> np.ndarray:
        """Convert SMC feature dict to numpy array in consistent order"""
        
        # Define consistent feature order
        feature_order = [
            'hh_hl_ratio', 'lh_ll_ratio', 'bos_count_bull', 'bos_count_bear',
            'choch_detected', 'dist_nearest_bull_ob', 'dist_nearest_bear_ob',
            'fvg_bull_open', 'fvg_bear_open', 'liq_high_distance', 'liq_low_distance',
            'impulse_strength', 'market_state', 'session_code', 'htf_bias',
            'volatility_regime'
        ]
        
        arr = []
        for feature in feature_order:
            arr.append(smc_features.get(feature, 0.0))
            
        return np.array(arr)
        
    def _determine_signal(self, probs: Dict[str, float]) -> Tuple[str, int, float]:
        """Determine trade signal from probabilities"""
        
        buy_prob = probs['buy_prob']
        sell_prob = probs['sell_prob']
        notrade_prob = probs['notrade_prob']
        
        # Check if confidence is above threshold
        if buy_prob > self.confidence_threshold and buy_prob > sell_prob:
            return "BUY", 1, buy_prob
        elif sell_prob > self.confidence_threshold and sell_prob > buy_prob:
            return "SELL", -1, sell_prob
        else:
            # No trade - return highest probability but below threshold
            max_prob = max(buy_prob, sell_prob, notrade_prob)
            if max_prob == buy_prob:
                return "NO_TRADE", 0, buy_prob
            elif max_prob == sell_prob:
                return "NO_TRADE", 0, sell_prob
            else:
                return "NO_TRADE", 0, notrade_prob
                
    def _fallback_prediction(self, smc_features: Dict[str, float]) -> Dict[str, float]:
        """Fallback heuristic when model is not available"""
        
        # Simple heuristic based on SMC features
        buy_score = 0.0
        sell_score = 0.0
        
        # Structure bias
        if smc_features.get('hh_hl_ratio', 0.5) > 0.6:
            buy_score += 0.3
        if smc_features.get('lh_ll_ratio', 0.5) > 0.6:
            sell_score += 0.3
            
        # Order blocks
        if smc_features.get('dist_nearest_bull_ob', 10) < 2:
            buy_score += 0.2
        if smc_features.get('dist_nearest_bear_ob', 10) < 2:
            sell_score += 0.2
            
        # FVG
        if smc_features.get('fvg_bull_open', 0):
            buy_score += 0.15
        if smc_features.get('fvg_bear_open', 0):
            sell_score += 0.15
            
        # HTF bias
        htf_bias = smc_features.get('htf_bias', 0)
        if htf_bias == 1:
            buy_score += 0.2
        elif htf_bias == -1:
            sell_score += 0.2
            
        # Normalize
        total = buy_score + sell_score
        if total > 0:
            buy_prob = buy_score / total * 0.8  # Max 80% confidence
            sell_prob = sell_score / total * 0.8
        else:
            buy_prob = 0.33
            sell_prob = 0.33
            
        return {
            'buy_prob': buy_prob,
            'sell_prob': sell_prob,
            'notrade_prob': 1 - buy_prob - sell_prob
        }
        
    def _get_explanation(self, cnn_embedding: np.ndarray, smc_array: np.ndarray) -> List[Dict]:
        """Get SHAP explanation for prediction"""
        
        try:
            from .shap_explainer import SHAPExplainer
            
            if hasattr(self.model, 'model') and self.model.model is not None:
                # Create feature names
                feature_names = [f'pca_{i}' for i in range(128)] + \
                               [f'smc_{i}' for i in range(len(smc_array))]
                
                explainer = SHAPExplainer(self.model.model, feature_names)
                
                # Combine features
                combined = np.hstack([cnn_embedding[:128], smc_array])
                
                explanation = explainer.explain_prediction(combined, top_k=5)
                return explanation.get('top_features', [])
                
        except Exception as e:
            logger.warning(f"Failed to get SHAP explanation: {e}")
            
        return []
        
    def get_stats(self) -> Dict:
        """Get inference statistics"""
        return {
            **self.stats,
            'cache_size': self.cache.size(),
            'model_loaded': self._model is not None,
            'model_version': self._model_version,
            'confidence_threshold': self.confidence_threshold
        }
        
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_inference_time_ms': 0.0,
            'signal_counts': {'BUY': 0, 'SELL': 0, 'NO_TRADE': 0}
        }
        self.cache.clear()
        
    def reload_model(self, model_path: Optional[Path] = None):
        """Reload model from disk"""
        if model_path:
            self.model_path = Path(model_path)
        self._model = None
        self._load_model()
        self.cache.clear()
        logger.info("Model reloaded")


class ModelEnsemble:
    """Ensemble of multiple fusion models for robust predictions"""
    
    def __init__(self, model_paths: List[Path], weights: Optional[List[float]] = None):
        """
        Initialize ensemble with multiple models
        
        Args:
            model_paths: List of paths to fusion models
            weights: Optional weights for each model (default: equal)
        """
        self.models = []
        self.model_versions = []
        
        for path in model_paths:
            try:
                from .train_xgb import FusionModel
                model = FusionModel()
                model.load(Path(path))
                self.models.append(model)
                self.model_versions.append(path.name)
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {e}")
                
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        self.weights = weights
        
        logger.info(f"Ensemble initialized with {len(self.models)} models")
        
    def predict(
        self,
        cnn_embedding: np.ndarray,
        smc_features: Dict[str, float]
    ) -> Dict[str, float]:
        """Weighted average prediction from all models"""
        
        if not self.models:
            return {'buy_prob': 0.33, 'sell_prob': 0.33, 'notrade_prob': 0.34}
            
        # Convert SMC features to array (use same order as FusionInference)
        inference = FusionInference()
        smc_array = inference._dict_to_array(smc_features)
        
        total_probs = np.zeros(3)
        total_weight = sum(self.weights)
        
        for model, weight in zip(self.models, self.weights):
            probs = model.predict(cnn_embedding, smc_array)
            total_probs[0] += probs['buy_prob'] * weight
            total_probs[1] += probs['sell_prob'] * weight
            total_probs[2] += probs['notrade_prob'] * weight
            
        total_probs /= total_weight
        
        return {
            'buy_prob': total_probs[0],
            'sell_prob': total_probs[1],
            'notrade_prob': total_probs[2]
        }
        
    def predict_with_uncertainty(
        self,
        cnn_embedding: np.ndarray,
        smc_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Get prediction with uncertainty (std dev across models)"""
        
        all_buy = []
        all_sell = []
        all_notrade = []
        
        inference = FusionInference()
        smc_array = inference._dict_to_array(smc_features)
        
        for model in self.models:
            probs = model.predict(cnn_embedding, smc_array)
            all_buy.append(probs['buy_prob'])
            all_sell.append(probs['sell_prob'])
            all_notrade.append(probs['notrade_prob'])
            
        return {
            'buy_prob': np.mean(all_buy),
            'sell_prob': np.mean(all_sell),
            'notrade_prob': np.mean(all_notrade),
            'buy_std': np.std(all_buy),
            'sell_std': np.std(all_sell),
            'notrade_std': np.std(all_notrade),
            'uncertainty': np.mean([np.std(all_buy), np.std(all_sell), np.std(all_notrade)])
        }


# Convenience function for quick inference
def quick_predict(
    cnn_embedding: np.ndarray,
    smc_features: Dict[str, float],
    model_path: Optional[Path] = None
) -> InferenceResult:
    """
    Quick single prediction using default or specified model
    
    Args:
        cnn_embedding: CNN embedding vector
        smc_features: Dictionary of SMC features
        model_path: Optional custom model path
        
    Returns:
        InferenceResult object
    """
    inference = FusionInference(model_path=model_path)
    return inference.predict_single(cnn_embedding, smc_features)


# Async inference support
async def async_predict(
    cnn_embedding: np.ndarray,
    smc_features: Dict[str, float],
    model_path: Optional[Path] = None
) -> InferenceResult:
    """
    Async version of quick_predict for non-blocking inference
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            quick_predict,
            cnn_embedding,
            smc_features,
            model_path
        )
    return result
