"""EfficientNet-B3 CNN model for chart pattern recognition"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ChartCNN(nn.Module):
    """CNN model for extracting embeddings from chart images"""
    
    def __init__(
        self,
        architecture: str = "efficientnet-b3",
        pretrained: bool = True,
        embedding_dim: int = 1536,
        dropout: float = 0.3,
        num_classes: int = 3  # BUY, SELL, NO_TRADE
    ):
        super().__init__()
        
        self.architecture = architecture
        self.embedding_dim = embedding_dim
        
        # Load base model
        if architecture == "efficientnet-b3":
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
            # Remove classifier
            self.backbone.classifier = nn.Identity()
        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
            
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Auxiliary classification head (for supervised training)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for custom layers"""
        for m in self.embedding_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            embeddings: (B, embedding_dim)
            logits: (B, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Generate embeddings
        embeddings = self.embedding_head(features)
        
        # Classification logits
        logits = self.classifier(embeddings)
        
        return embeddings, logits
        
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract only embeddings (for inference)"""
        features = self.backbone(x)
        return self.embedding_head(features)


class MultiTimeframeCNN(nn.Module):
    """CNN that processes multiple timeframes simultaneously"""
    
    def __init__(
        self,
        timeframes: list = ["15m", "1h", "4h"],
        embedding_dim: int = 1536,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.timeframes = timeframes
        self.cnn_encoders = nn.ModuleDict()
        
        for tf in timeframes:
            self.cnn_encoders[tf] = ChartCNN(
                architecture="efficientnet-b3",
                pretrained=True,
                embedding_dim=embedding_dim // len(timeframes),
                dropout=dropout
            )
            
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
    def forward(self, images: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multiple timeframe images
        
        Args:
            images: Dict of {timeframe: image_tensor}
            
        Returns:
            fused_embeddings: Combined embeddings
            logits: Classification logits
        """
        all_embeddings = []
        
        for tf in self.timeframes:
            if tf in images:
                embeddings, logits = self.cnn_encoders[tf](images[tf])
                all_embeddings.append(embeddings)
                
        # Concatenate embeddings
        fused = torch.cat(all_embeddings, dim=1)
        
        # Final fusion
        fused_embeddings = self.fusion_layer(fused)
        
        return fused_embeddings
