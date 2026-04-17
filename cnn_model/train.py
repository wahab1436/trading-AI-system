"""Training script for CNN model"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import logging
from typing import Tuple, Dict, Optional
import mlflow

from .model import ChartCNN

logger = logging.getLogger(__name__)


class ChartDataset(Dataset):
    """Dataset for chart images"""
    
    def __init__(
        self,
        image_dir: Path,
        transform=None,
        label_map: Dict = {1: 0, -1: 1, 0: 2}  # BUY=0, SELL=1, NO_TRADE=2
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_map = label_map
        
        # Load image paths and labels
        self.images = []
        self.labels = []
        
        for img_path in self.image_dir.glob("*.png"):
            # Extract label from filename
            # Format: {symbol}_{timestamp}_label_{label}.png
            label_str = img_path.stem.split('_label_')[-1]
            label = int(label_str)
            
            self.images.append(img_path)
            self.labels.append(self.label_map[label])
            
        logger.info(f"Loaded {len(self.images)} images from {image_dir}")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class CNNTrainer:
    """Trainer for CNN model"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 0.0001,
        weight_decay: float = 1e-4,
        mixed_precision: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.mixed_precision = mixed_precision
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler() if mixed_precision else None
        
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with autocast():
                    embeddings, logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                embeddings, logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
            total_loss += loss.item()
            
            # Accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100 * correct / total
            })
            
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100 * correct / total
        }
        
    def validate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """Validate model"""
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                embeddings, logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro')
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100 * correct / total,
            'f1_score': f1
        }
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 7
    ) -> Dict:
        """Full training loop"""
        
        best_val_f1 = 0
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)
            
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1_score'])
            
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val F1: {val_metrics['f1_score']:.4f}"
            )
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_f1': val_metrics['f1_score']
            }, step=epoch)
            
            # Early stopping
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                patience_counter = 0
                self.save_checkpoint('best_model.pt', val_metrics)
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
        return history
