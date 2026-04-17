"""TorchServe handler for serving CNN model in production"""

import torch
import torch.nn.functional as F
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import numpy as np
import io
import base64
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class CNNHandler(BaseHandler):
    """
    TorchServe handler for CNN chart pattern recognition model.
    
    Handles:
    - Image preprocessing (base64 or raw bytes)
    - Model inference
    - Embedding extraction
    - Prediction formatting
    """
    
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.input_size = 380
        self.embedding_dim = 1536
        
    def initialize(self, context):
        """Initialize model and preprocessing"""
        
        self.manifest = context.manifest
        properties = context.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id")
            else "cpu"
        )
        
        # Load model
        model_dir = properties.get("model_dir")
        self.model = self._load_model(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.initialized = True
        logger.info(f"CNNHandler initialized on {self.device}")
        
    def _load_model(self, model_dir: str):
        """Load PyTorch model from model directory"""
        
        import sys
        from pathlib import Path
        
        # Add model directory to path
        sys.path.append(model_dir)
        
        # Import model class
        from cnn_model.model import ChartCNN
        
        model = ChartCNN(
            architecture="efficientnet-b3",
            pretrained=False,
            embedding_dim=self.embedding_dim,
            dropout=0.3
        )
        
        # Load state dict
        model_pt_path = Path(model_dir) / "model.pt"
        if model_pt_path.exists():
            state_dict = torch.load(model_pt_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            elif isinstance(state_dict, dict) and 'backbone' in state_dict:
                model.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
                
            logger.info(f"Loaded model from {model_pt_path}")
        else:
            logger.warning(f"No model found at {model_pt_path}, using untrained model")
            
        return model
        
    def preprocess(self, data: List[Dict]) -> torch.Tensor:
        """
        Preprocess incoming request data.
        
        Expected input formats:
        1. Base64 encoded image: {"body": "base64_string"}
        2. Image bytes: {"body": bytes}
        3. Image URL: {"url": "http://..."}
        """
        
        images = []
        
        for row in data:
            # Handle different input formats
            if "body" in row:
                body = row["body"]
                
                # Check if base64 encoded
                if isinstance(body, str):
                    # Decode base64
                    if body.startswith('data:image'):
                        # Remove data URL prefix
                        body = body.split(',')[1]
                    image_bytes = base64.b64decode(body)
                elif isinstance(body, bytes):
                    image_bytes = body
                else:
                    raise ValueError(f"Unsupported body type: {type(body)}")
                    
                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                
            elif "url" in row:
                # Download image from URL
                import requests
                response = requests.get(row["url"], timeout=5)
                image = Image.open(io.BytesIO(response.content)).convert('RGB')
                
            else:
                raise ValueError("Missing 'body' or 'url' in request")
                
            # Apply transforms
            tensor = self.transform(image)
            images.append(tensor)
            
        # Stack into batch
        batch = torch.stack(images)
        return batch.to(self.device)
        
    def inference(self, data: torch.Tensor) -> Dict[str, Any]:
        """
        Run model inference.
        
        Returns:
            Dictionary with embeddings and logits
        """
        
        with torch.no_grad():
            embeddings, logits = self.model(data)
            
        return {
            'embeddings': embeddings.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }
        
    def postprocess(self, inference_output: Dict[str, Any]) -> List[Dict]:
        """
        Format inference output for API response.
        """
        
        embeddings = inference_output['embeddings']
        logits = inference_output['logits']
        
        # Calculate probabilities
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        
        # Label mapping
        label_map = {0: 'BUY', 1: 'SELL', 2: 'NO_TRADE'}
        
        results = []
        for i in range(len(embeddings)):
            pred_idx = np.argmax(probs[i])
            
            results.append({
                'embedding': embeddings[i].tolist(),
                'embedding_shape': list(embeddings[i].shape),
                'probabilities': {
                    'BUY': float(probs[i][0]),
                    'SELL': float(probs[i][1]),
                    'NO_TRADE': float(probs[i][2])
                },
                'prediction': label_map[pred_idx],
                'confidence': float(probs[i][pred_idx])
            })
            
        return results


class EmbeddingOnlyHandler(CNNHandler):
    """
    Handler that only returns embeddings (faster, for fusion model).
    """
    
    def postprocess(self, inference_output: Dict[str, Any]) -> List[Dict]:
        """Return only embeddings"""
        
        embeddings = inference_output['embeddings']
        
        return [
            {
                'embedding': emb.tolist(),
                'embedding_shape': list(emb.shape)
            }
            for emb in embeddings
        ]


class BatchHandler(CNNHandler):
    """
    Handler optimized for batch processing.
    """
    
    def __init__(self):
        super().__init__()
        self.max_batch_size = 64
        self.max_batch_delay = 100  # milliseconds
        
    def preprocess(self, data: List[Dict]) -> torch.Tensor:
        """Batch preprocessing with size validation"""
        
        if len(data) > self.max_batch_size:
            logger.warning(f"Batch size {len(data)} exceeds max {self.max_batch_size}, truncating")
            data = data[:self.max_batch_size]
            
        return super().preprocess(data)
        
    def inference(self, data: torch.Tensor) -> Dict[str, Any]:
        """Optimized batch inference"""
        
        # Use mixed precision for batch inference
        with torch.no_grad(), torch.cuda.amp.autocast():
            embeddings, logits = self.model(data)
            
        return {
            'embeddings': embeddings.cpu().numpy(),
            'logits': logits.cpu().numpy()
        }


# Configuration for model archive (MAR file)
# Save this as model-config.yaml
"""
{
    "model_name": "cnn_chart_model",
    "model_version": "1.0",
    "handler": "cnn_model.serve.handler:CNNHandler",
    "requirements_file": "requirements.txt",
    "runtime": "python3",
    "min_workers": 1,
    "max_workers": 4,
    "batch_size": 32,
    "max_batch_delay": 100,
    "response_timeout": 60
}
"""

# To create model archive:
# torch-model-archiver \
#     --model-name cnn_chart_model \
#     --version 1.0 \
#     --model-file cnn_model/model.py \
#     --serialized-file models/cnn_best.pt \
#     --handler cnn_model/serve/handler.py:CNNHandler \
#     --extra-files config/model_config.yaml \
#     --requirements-file requirements.txt \
#     --export-path model_store/
#
# To serve:
# torchserve --start --model-store model_store/ --models cnn_chart_model.mar
