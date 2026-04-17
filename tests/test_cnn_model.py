"""Unit tests for CNN model module"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile

from cnn_model.model import ChartCNN, MultiTimeframeCNN
from cnn_model.infer import CNNInference, InferenceResult
from cnn_model.train import ChartDataset


class TestChartCNN:
    """Test ChartCNN model"""
    
    def test_model_initialization(self):
        """Test model creates correctly"""
        model = ChartCNN(
            architecture="efficientnet-b3",
            pretrained=False,
            embedding_dim=1536,
            dropout=0.3
        )
        
        assert model.embedding_dim == 1536
        assert model.architecture == "efficientnet-b3"
        
    def test_forward_pass(self):
        """Test forward pass works"""
        model = ChartCNN()
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 380, 380)
        
        embeddings, logits = model(dummy_input)
        
        assert embeddings.shape == (batch_size, 1536)
        assert logits.shape == (batch_size, 3)
        
    def test_extract_embeddings(self):
        """Test embedding extraction"""
        model = ChartCNN()
        dummy_input = torch.randn(1, 3, 380, 380)
        
        embeddings = model.extract_embeddings(dummy_input)
        
        assert embeddings.shape == (1, 1536)
        
    def test_multi_timeframe_model(self):
        """Test multi-timeframe model"""
        model = MultiTimeframeCNN(
            timeframes=["15m", "1h", "4h"],
            embedding_dim=1536
        )
        
        images = {
            "15m": torch.randn(2, 3, 380, 380),
            "1h": torch.randn(2, 3, 380, 380),
            "4h": torch.randn(2, 3, 380, 380)
        }
        
        embeddings = model(images)
        
        assert embeddings.shape == (2, 1536)


class TestCNNInference:
    """Test CNN inference wrapper"""
    
    def test_initialization(self):
        """Test inference initialization"""
        inference = CNNInference()
        
        assert inference.device in ["cuda", "cpu"]
        assert inference.input_size == 380
        assert inference.batch_size == 32
        
    def test_preprocess_image(self):
        """Test image preprocessing"""
        inference = CNNInference()
        
        # Create dummy image
        img = Image.new('RGB', (500, 500), color='black')
        
        tensor = inference.preprocess_image(img)
        
        assert tensor.shape == (1, 3, 380, 380)
        assert tensor.dtype == torch.float32
        
    def test_preprocess_image_from_path(self):
        """Test preprocessing from file path"""
        
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            img = Image.new('RGB', (500, 500), color='black')
            img.save(tmp.name)
            
            inference = CNNInference()
            tensor = inference.preprocess_image(tmp.name)
            
            assert tensor.shape == (1, 3, 380, 380)
            
    def test_predict_single(self):
        """Test single image prediction"""
        inference = CNNInference()
        img = Image.new('RGB', (500, 500), color='black')
        
        result = inference.predict_single(img)
        
        assert isinstance(result, InferenceResult)
        assert result.success
        assert result.embedding.shape == (1536,)
        assert set(result.probabilities.keys()) == {'BUY', 'SELL', 'NO_TRADE'}
        assert sum(result.probabilities.values()) == pytest.approx(1.0)
        
    def test_extract_embedding(self):
        """Test embedding extraction"""
        inference = CNNInference()
        img = Image.new('RGB', (500, 500), color='black')
        
        embedding = inference.extract_embedding(img)
        
        assert embedding is not None
        assert embedding.shape == (1536,)
        
    def test_get_model_info(self):
        """Test model info retrieval"""
        inference = CNNInference()
        info = inference.get_model_info()
        
        assert info['architecture'] == 'efficientnet-b3'
        assert info['embedding_dim'] == 1536
        assert 'total_params' in info
        assert 'trainable_params' in info
        
    def test_benchmark(self):
        """Test benchmark runs without error"""
        inference = CNNInference()
        results = inference.benchmark(num_runs=10)
        
        assert 'mean_ms' in results
        assert 'std_ms' in results
        assert results['num_runs'] == 10


class TestChartDataset:
    """Test dataset loading"""
    
    def test_dataset_initialization(self, tmp_path):
        """Test dataset creation"""
        
        # Create dummy images
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        
        for i, label in enumerate([1, -1, 0]):
            img_path = images_dir / f"test_label_{label}.png"
            img = Image.new('RGB', (380, 380), color='black')
            img.save(img_path)
            
        dataset = ChartDataset(images_dir)
        
        assert len(dataset) == 3
        
    def test_dataset_getitem(self, tmp_path):
        """Test dataset item retrieval"""
        
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        
        img_path = images_dir / "test_label_1.png"
        img = Image.new('RGB', (380, 380), color='black')
        img.save(img_path)
        
        dataset = ChartDataset(images_dir)
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 380, 380)
        assert label in [0, 1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
