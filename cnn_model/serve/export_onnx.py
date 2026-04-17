"""Export PyTorch model to ONNX format for optimized inference"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Tuple
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from ..model import ChartCNN

logger = logging.getLogger(__name__)


class ONNXExporter:
    """Export CNN model to ONNX format"""
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        input_size: int = 380,
        embedding_dim: int = 1536,
        device: str = "cuda"
    ):
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Initialize model
        self.model = ChartCNN(
            architecture="efficientnet-b3",
            pretrained=False,
            embedding_dim=embedding_dim,
            dropout=0.3
        )
        
        if model_path and model_path.exists():
            self.load_model(model_path)
            
        self.model.eval()
        
    def load_model(self, model_path: Path):
        """Load PyTorch model weights"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        logger.info(f"Loaded model from {model_path}")
        
    def export(
        self,
        output_path: Path,
        opset_version: int = 14,
        dynamic_batch: bool = True,
        quantize: bool = False
    ) -> Path:
        """Export model to ONNX format"""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, self.input_size, self.input_size)
        
        # Dynamic axes for variable batch size
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        } if dynamic_batch else None
        
        # Export to ONNX
        onnx_path = output_path.with_suffix('.onnx')
        
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['embedding', 'logits'],
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        logger.info(f"Exported ONNX model to {onnx_path}")
        
        # Validate ONNX model
        self._validate_onnx(onnx_path)
        
        # Quantize if requested (smaller size, faster inference)
        if quantize:
            quantized_path = output_path.parent / f"{output_path.stem}_quantized.onnx"
            self._quantize_model(onnx_path, quantized_path)
            logger.info(f"Quantized model saved to {quantized_path}")
            return quantized_path
            
        return onnx_path
        
    def _validate_onnx(self, onnx_path: Path):
        """Validate exported ONNX model"""
        
        # Load and check model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Test inference with ONNX Runtime
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, self.input_size, self.input_size).astype(np.float32)
        
        # Run inference
        outputs = ort_session.run(
            ['embedding', 'logits'],
            {'input': dummy_input}
        )
        
        embedding, logits = outputs
        
        logger.info(f"ONNX validation passed - embedding shape: {embedding.shape}, logits shape: {logits.shape}")
        
    def _quantize_model(self, input_path: Path, output_path: Path):
        """Quantize ONNX model for smaller size and faster CPU inference"""
        
        quantize_dynamic(
            model_input=str(input_path),
            model_output=str(output_path),
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8
        )
        
    def export_with_pca(
        self,
        model_path: Path,
        pca_model_path: Path,
        output_path: Path
    ):
        """Export combined model with PCA for embedding reduction"""
        
        # This creates a more complex export with PCA integration
        # For now, export separately
        
        onnx_path = self.export(model_path, output_path)
        
        # Save PCA parameters separately
        import pickle
        with open(pca_model_path, 'rb') as f:
            pca = pickle.load(f)
            
        pca_params_path = output_path.parent / f"{output_path.stem}_pca_params.pkl"
        with open(pca_params_path, 'wb') as f:
            pickle.dump(pca, f)
            
        logger.info(f"Saved PCA parameters to {pca_params_path}")
        
        return onnx_path, pca_params_path


class ONNXInference:
    """Optimized inference using ONNX Runtime"""
    
    def __init__(
        self,
        onnx_path: Path,
        device: str = "cpu",
        use_cuda: bool = False
    ):
        self.onnx_path = Path(onnx_path)
        
        # Configure ONNX Runtime session
        providers = []
        
        if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.embedding_name = self.session.get_outputs()[0].name
        self.logits_name = self.session.get_outputs()[1].name
        
        logger.info(f"ONNX Runtime session created on {providers[0] if providers else 'CPU'}")
        
    def predict(self, image_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference with ONNX Runtime"""
        
        # Ensure correct shape and type
        if image_tensor.ndim == 3:
            image_tensor = image_tensor[np.newaxis, ...]
            
        image_tensor = image_tensor.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(
            [self.embedding_name, self.logits_name],
            {self.input_name: image_tensor}
        )
        
        return outputs[0], outputs[1]  # embedding, logits
        
    def predict_batch(self, image_tensors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch inference with ONNX Runtime"""
        
        image_tensors = image_tensors.astype(np.float32)
        
        outputs = self.session.run(
            [self.embedding_name, self.logits_name],
            {self.input_name: image_tensors}
        )
        
        return outputs[0], outputs[1]
        
    def benchmark(self, num_runs: int = 100, batch_size: int = 1) -> Dict:
        """Benchmark ONNX inference speed"""
        
        import time
        
        # Create dummy input
        dummy_input = np.random.randn(batch_size, 3, 380, 380).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = self.predict(dummy_input)
            
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.predict(dummy_input)
            times.append((time.time() - start) * 1000)
            
        import numpy as np
        times = np.array(times)
        
        return {
            'mean_ms': float(np.mean(times)),
            'std_ms': float(np.std(times)),
            'min_ms': float(np.min(times)),
            'max_ms': float(np.max(times)),
            'p95_ms': float(np.percentile(times, 95)),
            'batch_size': batch_size,
            'num_runs': num_runs
        }


def export_model_cli():
    """Command-line interface for ONNX export"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Export CNN model to ONNX")
    parser.add_argument("--input", type=str, required=True, help="Input PyTorch model path")
    parser.add_argument("--output", type=str, required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--quantize", action="store_true", help="Quantize model")
    parser.add_argument("--input-size", type=int, default=380, help="Input image size")
    
    args = parser.parse_args()
    
    exporter = ONNXExporter(
        model_path=Path(args.input),
        input_size=args.input_size
    )
    
    output_path = exporter.export(
        output_path=Path(args.output),
        opset_version=args.opset,
        quantize=args.quantize
    )
    
    print(f"Model exported to {output_path}")
    
    # Test inference
    onnx_inference = ONNXInference(output_path)
    benchmark = onnx_inference.benchmark()
    print(f"Inference benchmark: {benchmark['mean_ms']:.2f}ms mean")


if __name__ == "__main__":
    export_model_cli()
