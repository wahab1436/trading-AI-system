"""Image augmentation strategies for training data"""

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import random
from typing import Tuple, Optional
import albumentations as A


class ChartAugmenter:
    """Apply augmentations to chart images (training only)"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        
        # Define augmentation pipeline using albumentations
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=0, p=0.3),
            A.CoarseDropout(max_holes=1, max_height=20, max_width=50, p=0.2),
        ])
        
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply augmentations to image"""
        
        if random.random() > self.p:
            return image
            
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply transforms
        augmented = self.transform(image=img_array)
        
        # Convert back to PIL Image
        return Image.fromarray(augmented['image'])
        
    def augment_dataset(self, images: list, labels: list) -> Tuple[list, list]:
        """Augment entire dataset"""
        
        augmented_images = []
        augmented_labels = []
        
        for img, label in zip(images, labels):
            # Original
            augmented_images.append(img)
            augmented_labels.append(label)
            
            # Augmented version
            if random.random() < 0.7:  # 70% chance to add augmented version
                aug_img = self(img)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
                
        return augmented_images, augmented_labels


class TimeConsistencyValidator:
    """Ensures augmentations don't break temporal consistency"""
    
    def validate_augmentation(self, original: Image.Image, augmented: Image.Image) -> bool:
        """Validate that augmentation didn't corrupt price patterns"""
        
        # Check that key features remain visible
        # This is a simplified check
        
        orig_array = np.array(original)
        aug_array = np.array(augmented)
        
        # Check that brightness isn't extreme
        aug_brightness = aug_array.mean()
        if aug_brightness < 10 or aug_brightness > 245:
            return False
            
        # Check that contrast is reasonable
        aug_std = aug_array.std()
        if aug_std < 5:
            return False
            
        return True
