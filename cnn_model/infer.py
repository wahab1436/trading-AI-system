"""Inference utilities for CNN model"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import time
from dataclasses import dataclass

from .model import ChartCNN

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from CNN inference"""
    embedding: np.ndarray  # (1536,) feature vector
    probabilities: Dict[str, float]  # BUY, SELL, NO_TRADE probs
    predicted_class: str
    confidence: float
    inference_time_ms: float
    success: bool
    error: Optional[str] = None


class CNNInference:
    """Optimized inference wrapper for CNN model"""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: str = "cuda",
        input_size: int = 380,
        batch_size: int = 32
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.input_size = input_size
        self.batch_size = batch_size
        
        # Initialize model
        self.model = ChartCNN(
            architecture="efficientnet-b3",
            pretrained=False,
            embedding_dim=1536,
            dropout=0.3
        )
        
        # Load weights if provided
        if model_path and model_path.exists():
            self.load_model(model_path)
        else:
            logger.warning("No model weights loaded - using untrained model")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Image transforms (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Warm up CUDA if available
        if self.device == "cuda":
            self._warmup()
            
    def _warmup(self):
        """Warm up CUDA with dummy inference"""
        dummy = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        logger.info("CUDA warmup complete")
        
    def load_model(self, model_path: Path):
        """Load model weights from file"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded model from {model_path} (epoch {checkpoint.get('epoch', 'unknown')})")
            elif isinstance(checkpoint, dict) and any(k.startswith('backbone') for k in checkpoint.keys()):
                self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded model weights from {model_path}")
            else:
                # Try loading as full state dict
                self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded model from {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
            
    def preprocess_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess single image for inference"""
        
        # Load image from various input types
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            img = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        # Apply transforms
        tensor = self.transform(img)
        return tensor.unsqueeze(0)  # Add batch dimension
        
    def preprocess_batch(self, images: List[Union[str, Path, Image.Image]]) -> torch.Tensor:
        """Preprocess batch of images"""
        
        batch_tensors = []
        for img in images:
            tensor = self.preprocess_image(img)
            batch_tensors.append(tensor)
            
        return torch.cat(batch_tensors, dim=0)
        
    @torch.no_grad()
    def predict_single(self, image: Union[str, Path, Image.Image, np.ndarray]) -> InferenceResult:
        """Run inference on a single image"""
        
        start_time = time.time()
        
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image).to(self.device)
            
            # Inference
            embedding, logits = self.model(input_tensor)
            
            # Calculate probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Map to labels: 0=BUY, 1=SELL, 2=NO_TRADE
            label_map = {0: 'BUY', 1: 'SELL', 2: 'NO_TRADE'}
            predicted_idx = np.argmax(probs)
            
            inference_time = (time.time() - start_time) * 1000
            
            return InferenceResult(
                embedding=embedding.cpu().numpy()[0],
                probabilities={
                    'BUY': float(probs[0]),
                    'SELL': float(probs[1]),
                    'NO_TRADE': float(probs[2])
                },
                predicted_class=label_map[predicted_idx],
                confidence=float(probs[predicted_idx]),
                inference_time_ms=inference_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return InferenceResult(
                embedding=np.zeros(1536),
                probabilities={'BUY': 0.0, 'SELL': 0.0, 'NO_TRADE': 1.0},
                predicted_class='NO_TRADE',
                confidence=0.0,
                inference_time_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )
            
    @torch.no_grad()
    def predict_batch(self, images: List[Union[str, Path, Image.Image]]) -> List[InferenceResult]:
        """Run inference on multiple images (batched for efficiency)"""
        
        start_time = time.time()
        results = []
        
        try:
            # Process in batches
            for i in range(0, len(images), self.batch_size):
                batch = images[i:i + self.batch_size]
                batch_tensors = self.preprocess_batch(batch).to(self.device)
                
                embeddings, logits = self.model(batch_tensors)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                label_map = {0: 'BUY', 1: 'SELL', 2: 'NO_TRADE'}
                
                for j, (emb, prob) in enumerate(zip(embeddings.cpu().numpy(), probs)):
                    predicted_idx = np.argmax(prob)
                    results.append(InferenceResult(
                        embedding=emb,
                        probabilities={
                            'BUY': float(prob[0]),
                            'SELL': float(prob[1]),
                            'NO_TRADE': float(prob[2])
                        },
                        predicted_class=label_map[predicted_idx],
                        confidence=float(prob[predicted_idx]),
                        inference_time_ms=(time.time() - start_time) * 1000 / len(images),
                        success=True
                    ))
                    
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            # Return failed results for all
            for _ in images:
                results.append(InferenceResult(
                    embedding=np.zeros(1536),
                    probabilities={'BUY': 0.0, 'SELL': 0.0, 'NO_TRADE': 1.0},
                    predicted_class='NO_TRADE',
                    confidence=0.0,
                    inference_time_ms=0,
                    success=False,
                    error=str(e)
                ))
                
        return results
        
    def extract_embedding(self, image: Union[str, Path, Image.Image]) -> Optional[np.ndarray]:
        """Extract only the embedding vector (faster)"""
        
        try:
            input_tensor = self.preprocess_image(image).to(self.device)
            embedding, _ = self.model(input_tensor)
            return embedding.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return None
            
    def extract_embeddings_batch(self, images: List[Union[str, Path, Image.Image]]) -> List[np.ndarray]:
        """Extract embeddings for multiple images"""
        
        embeddings = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            batch_tensors = self.preprocess_batch(batch).to(self.device)
            
            batch_embeddings, _ = self.model(batch_tensors)
            embeddings.extend(batch_embeddings.cpu().numpy())
            
        return embeddings
        
    def get_model_info(self) -> Dict:
        """Get model information"""
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'architecture': 'efficientnet-b3',
            'embedding_dim': 1536,
            'device': self.device,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_size': self.input_size,
            'batch_size': self.batch_size
        }
        
    def benchmark(self, num_runs: int = 100) -> Dict:
        """Benchmark inference speed"""
        
        # Create dummy image
        dummy_img = Image.new('RGB', (self.input_size, self.input_size), color='black')
        
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            _ = self.predict_single(dummy_img)
            times.append((time.time() - start) * 1000)
            
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'num_runs': num_runs
        }
