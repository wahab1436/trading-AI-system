"""Image validator - ensures all rendered images meet specifications"""

import hashlib
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImageValidationResult:
    """Result of image validation"""
    is_valid: bool
    file_path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    hash: str = ""
    dimensions: Tuple[int, int] = (0, 0)
    mode: str = ""
    mean_color: Tuple[float, float, float] = (0, 0, 0)


class ImageValidator:
    """Validates chart images against strict specifications"""
    
    # Expected specifications
    EXPECTED_WIDTH = 380
    EXPECTED_HEIGHT = 380
    EXPECTED_MODE = "RGB"
    EXPECTED_BACKGROUND_COLOR = (0, 0, 0)  # Pure black
    ALLOWED_COLOR_VARIANCE = 10  # Allow small RGB differences
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize validator
        
        Args:
            strict_mode: If True, any warning becomes an error
        """
        self.strict_mode = strict_mode
        self.manifest_path = None
        
    def validate_image(self, image_path: Path) -> ImageValidationResult:
        """Validate a single image file"""
        
        result = ImageValidationResult(
            is_valid=True,
            file_path=str(image_path)
        )
        
        try:
            # Open image
            img = Image.open(image_path)
            
            # Check dimensions
            if img.size != (self.EXPECTED_WIDTH, self.EXPECTED_HEIGHT):
                error = f"Wrong dimensions: {img.size} != ({self.EXPECTED_WIDTH}, {self.EXPECTED_HEIGHT})"
                if self.strict_mode:
                    result.errors.append(error)
                else:
                    result.warnings.append(error)
                result.is_valid = False
            result.dimensions = img.size
            
            # Check mode
            if img.mode != self.EXPECTED_MODE:
                error = f"Wrong mode: {img.mode} != {self.EXPECTED_MODE}"
                if self.strict_mode:
                    result.errors.append(error)
                else:
                    result.warnings.append(error)
                result.is_valid = False
            result.mode = img.mode
            
            # Convert to RGB if needed for color checks
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Check background color (corners should be black)
            corners = [
                img.getpixel((0, 0)),           # Top-left
                (img.getpixel((self.EXPECTED_WIDTH - 1, 0))),  # Top-right
                (img.getpixel((0, self.EXPECTED_HEIGHT - 1))),  # Bottom-left
                (img.getpixel((self.EXPECTED_WIDTH - 1, self.EXPECTED_HEIGHT - 1)))  # Bottom-right
            ]
            
            for corner in corners:
                if not self._is_close_to_black(corner):
                    warning = f"Corner not black: {corner}"
                    if self.strict_mode:
                        result.errors.append(warning)
                    else:
                        result.warnings.append(warning)
                    result.is_valid = False
                    break
                    
            # Calculate mean color
            img_array = np.array(img)
            result.mean_color = tuple(img_array.mean(axis=(0, 1)) / 255)
            
            # Compute hash
            result.hash = self.compute_hash(img)
            
            # Check for corrupted/blank images
            if img_array.std() < 5:
                error = "Image has near-zero variance (may be blank or corrupted)"
                result.errors.append(error)
                result.is_valid = False
                
            # Check for excessive brightness (indicates rendering issue)
            if img_array.mean() > 240:
                warning = "Image unusually bright"
                if self.strict_mode:
                    result.errors.append(warning)
                else:
                    result.warnings.append(warning)
                result.is_valid = False
                
        except Exception as e:
            result.errors.append(f"Failed to open/validate: {str(e)}")
            result.is_valid = False
            
        return result
        
    def validate_directory(
        self,
        directory: Path,
        max_workers: int = 8,
        sample_size: Optional[int] = None
    ) -> Dict:
        """Validate all images in a directory"""
        
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
            
        # Get all PNG images
        images = list(directory.glob("*.png")) + list(directory.glob("*.jpg"))
        
        if sample_size and sample_size < len(images):
            import random
            images = random.sample(images, sample_size)
            
        logger.info(f"Validating {len(images)} images in {directory}")
        
        results = {
            'total': len(images),
            'valid': 0,
            'invalid': 0,
            'errors': [],
            'warnings': [],
            'hashes': {},
            'duplicates': [],
            'dimension_issues': [],
            'color_issues': []
        }
        
        # Validate in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.validate_image, img): img for img in images}
            
            for future in as_completed(futures):
                result = future.result()
                
                if result.is_valid:
                    results['valid'] += 1
                else:
                    results['invalid'] += 1
                    
                # Track hash for duplicate detection
                if result.hash:
                    if result.hash in results['hashes']:
                        results['duplicates'].append({
                            'original': results['hashes'][result.hash],
                            'duplicate': result.file_path
                        })
                    else:
                        results['hashes'][result.hash] = result.file_path
                        
                # Collect issues
                for error in result.errors:
                    results['errors'].append({
                        'file': result.file_path,
                        'error': error
                    })
                    
                for warning in result.warnings:
                    results['warnings'].append({
                        'file': result.file_path,
                        'warning': warning
                    })
                    
                # Track dimension issues
                if result.dimensions != (self.EXPECTED_WIDTH, self.EXPECTED_HEIGHT):
                    results['dimension_issues'].append({
                        'file': result.file_path,
                        'dimensions': result.dimensions
                    })
                    
        # Summary
        logger.info(f"Validation complete: {results['valid']}/{results['total']} valid")
        logger.info(f"Found {len(results['duplicates'])} duplicate images")
        logger.info(f"Found {len(results['errors'])} errors, {len(results['warnings'])} warnings")
        
        return results
        
    def compute_hash(self, image: Image.Image) -> str:
        """Compute perceptual hash of image for duplicate detection"""
        
        # Resize to 16x16 for consistent hashing
        img_small = image.resize((16, 16), Image.Resampling.LANCZOS)
        
        # Convert to grayscale
        if img_small.mode != 'L':
            img_small = img_small.convert('L')
            
        # Get pixel values
        pixels = np.array(img_small).flatten()
        
        # Compute median
        median = np.median(pixels)
        
        # Create hash: 1 if pixel > median, else 0
        hash_bits = (pixels > median).astype(int)
        
        # Convert to hex
        hash_hex = ''.join(str(bit) for bit in hash_bits)
        hash_hex = hex(int(hash_hex, 2))[2:]
        
        return hash_hex
        
    def _is_close_to_black(self, color: Tuple[int, int, int]) -> bool:
        """Check if color is close to black"""
        return all(c < self.ALLOWED_COLOR_VARIANCE for c in color)
        
    def generate_manifest(
        self,
        directory: Path,
        output_path: Optional[Path] = None
    ) -> Dict:
        """Generate a manifest file for the image dataset"""
        
        directory = Path(directory)
        images = list(directory.glob("*.png"))
        
        manifest = {
            'dataset_path': str(directory.absolute()),
            'created_at': str(import_datetime()),
            'total_images': len(images),
            'specifications': {
                'width': self.EXPECTED_WIDTH,
                'height': self.EXPECTED_HEIGHT,
                'mode': self.EXPECTED_MODE,
                'background_color': self.EXPECTED_BACKGROUND_COLOR
            },
            'images': []
        }
        
        for img_path in images:
            # Extract label from filename
            label = self._extract_label_from_filename(img_path)
            
            # Compute hash
            img = Image.open(img_path)
            img_hash = self.compute_hash(img)
            
            manifest['images'].append({
                'filename': img_path.name,
                'path': str(img_path.relative_to(directory.parent) if directory.parent != directory else img_path),
                'label': label,
                'hash': img_hash,
                'size': img_path.stat().st_size
            })
            
        # Save manifest if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
                
            logger.info(f"Manifest saved to {output_path}")
            
        return manifest
        
    def _extract_label_from_filename(self, filepath: Path) -> int:
        """Extract label from filename (format: *_label_{label}.png)"""
        try:
            # Pattern: ..._label_1.png
            label_str = filepath.stem.split('_label_')[-1]
            return int(label_str)
        except (ValueError, IndexError):
            return 0  # Default to NO_TRADE
            
    def verify_dataset_split(
        self,
        train_dir: Path,
        val_dir: Path,
        test_dir: Path
    ) -> Dict:
        """Verify that dataset splits have no overlapping images"""
        
        def get_image_hashes(directory: Path) -> Dict[str, str]:
            hashes = {}
            for img_path in directory.glob("*.png"):
                img = Image.open(img_path)
                hash_val = self.compute_hash(img)
                hashes[hash_val] = str(img_path)
            return hashes
            
        train_hashes = get_image_hashes(train_dir)
        val_hashes = get_image_hashes(val_dir)
        test_hashes = get_image_hashes(test_dir)
        
        # Find overlaps
        train_val_overlap = set(train_hashes.keys()) & set(val_hashes.keys())
        train_test_overlap = set(train_hashes.keys()) & set(test_hashes.keys())
        val_test_overlap = set(val_hashes.keys()) & set(test_hashes.keys())
        
        result = {
            'train_count': len(train_hashes),
            'val_count': len(val_hashes),
            'test_count': len(test_hashes),
            'train_val_overlap': len(train_val_overlap),
            'train_test_overlap': len(train_test_overlap),
            'val_test_overlap': len(val_test_overlap),
            'is_clean': len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0,
            'overlap_details': {
                'train_val': [train_hashes[h] for h in train_val_overlap],
                'train_test': [train_hashes[h] for h in train_test_overlap],
                'val_test': [val_hashes[h] for h in val_test_overlap]
            }
        }
        
        if not result['is_clean']:
            logger.warning(f"Dataset splits have overlaps! {result}")
            
        return result
        
    def check_consistency(
        self,
        image1: Path,
        image2: Path,
        tolerance: int = 5
    ) -> Dict:
        """Check consistency between two images (should be identical if same candle window)"""
        
        img1 = Image.open(image1)
        img2 = Image.open(image2)
        
        # Convert to arrays
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        
        # Calculate difference
        diff = np.abs(arr1.astype(int) - arr2.astype(int))
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        # Check if within tolerance
        is_consistent = max_diff <= tolerance
        
        return {
            'is_consistent': is_consistent,
            'max_difference': int(max_diff),
            'mean_difference': float(mean_diff),
            'image1': str(image1),
            'image2': str(image2)
        }


class HashValidator:
    """Validates image integrity using hash matching"""
    
    def __init__(self, manifest_path: Path):
        """
        Initialize with a manifest file
        
        Args:
            manifest_path: Path to manifest JSON file
        """
        self.manifest_path = Path(manifest_path)
        self.manifest = self._load_manifest()
        self.image_validator = ImageValidator()
        
    def _load_manifest(self) -> Dict:
        """Load manifest from file"""
        with open(self.manifest_path, 'r') as f:
            return json.load(f)
            
    def verify_integrity(self, image_dir: Optional[Path] = None) -> Dict:
        """Verify all images match their hashes in manifest"""
        
        results = {
            'total': len(self.manifest['images']),
            'matched': 0,
            'mismatched': 0,
            'missing': 0,
            'corrupted': 0,
            'details': []
        }
        
        base_dir = Path(self.manifest['dataset_path']) if not image_dir else Path(image_dir)
        
        for img_info in self.manifest['images']:
            img_path = base_dir / img_info['filename']
            
            if not img_path.exists():
                results['missing'] += 1
                results['details'].append({
                    'filename': img_info['filename'],
                    'status': 'missing',
                    'expected_hash': img_info['hash']
                })
                continue
                
            try:
                img = Image.open(img_path)
                current_hash = self.image_validator.compute_hash(img)
                
                if current_hash == img_info['hash']:
                    results['matched'] += 1
                else:
                    results['mismatched'] += 1
                    results['details'].append({
                        'filename': img_info['filename'],
                        'status': 'hash_mismatch',
                        'expected_hash': img_info['hash'],
                        'actual_hash': current_hash
                    })
                    
            except Exception as e:
                results['corrupted'] += 1
                results['details'].append({
                    'filename': img_info['filename'],
                    'status': 'corrupted',
                    'error': str(e)
                })
                
        # Calculate integrity score
        total_valid = results['matched']
        results['integrity_score'] = total_valid / results['total'] if results['total'] > 0 else 0
        
        logger.info(f"Hash validation: {results['matched']}/{results['total']} matched ({results['integrity_score']:.1%})")
        
        return results
        
    def find_duplicates(self) -> List[Dict]:
        """Find duplicate images using manifest hashes"""
        
        hash_map = {}
        duplicates = []
        
        for img_info in self.manifest['images']:
            img_hash = img_info['hash']
            
            if img_hash in hash_map:
                duplicates.append({
                    'hash': img_hash,
                    'images': [hash_map[img_hash], img_info['filename']]
                })
            else:
                hash_map[img_hash] = img_info['filename']
                
        return duplicates
        
    def get_dataset_stats(self) -> Dict:
        """Get statistics about the dataset from manifest"""
        
        labels = [img['label'] for img in self.manifest['images']]
        
        from collections import Counter
        label_counts = Counter(labels)
        
        return {
            'total_images': len(self.manifest['images']),
            'label_distribution': dict(label_counts),
            'label_percentages': {
                label: count / len(labels) * 100 
                for label, count in label_counts.items()
            },
            'created_at': self.manifest.get('created_at'),
            'dataset_path': self.manifest.get('dataset_path')
        }


def import_datetime():
    """Helper to import datetime only when needed"""
    from datetime import datetime
    return datetime.utcnow().isoformat()


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate chart images")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--strict", action="store_true", help="Strict mode (warnings become errors)")
    parser.add_argument("--generate-manifest", action="store_true", help="Generate manifest file")
    parser.add_argument("--output", type=str, default="image_manifest.json", help="Output manifest path")
    
    args = parser.parse_args()
    
    validator = ImageValidator(strict_mode=args.strict)
    
    if args.generate_manifest:
        manifest = validator.generate_manifest(Path(args.dir), Path(args.output))
        print(f"Manifest generated: {len(manifest['images'])} images")
    else:
        results = validator.validate_directory(Path(args.dir))
        print(f"\nValidation Results:")
        print(f"  Valid: {results['valid']}/{results['total']}")
        print(f"  Invalid: {results['invalid']}")
        print(f"  Duplicates: {len(results['duplicates'])}")
        print(f"  Errors: {len(results['errors'])}")
        print(f"  Warnings: {len(results['warnings'])}")
