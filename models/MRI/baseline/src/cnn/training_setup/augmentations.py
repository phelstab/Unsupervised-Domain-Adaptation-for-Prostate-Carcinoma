"""
Data augmentation transforms for 3D MRI prostate cancer imaging.

Simple, standard augmentations as recommended for medical imaging UDA:
- Gaussian noise: Simulates acquisition noise variations
- Brightness adjustment: Simulates intensity variations across scanners
- Contrast adjustment: Simulates contrast differences between protocols
- Blur (Gaussian): Simulates resolution differences

Hyperparameters tuned for normalized MRI data (mean ~0, std ~1).

Usage:
    from training_setup.augmentations import MRIAugmentation, AugmentationConfig
    
    config = AugmentationConfig(
        gaussian_noise=True,
        brightness=True,
        contrast=True,
        blur=True
    )
    transform = MRIAugmentation(config)


SOURCE: https://github.com/google-research/uda/blob/960684e363251772a5938451d4d2bc0f1da9e24b/image/randaugment/augmentation_transforms.py#L4
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import scipy.ndimage


@dataclass
class AugmentationConfig:
    """Configuration for MRI augmentations.
    
    All augmentations are disabled by default for backward compatibility.
    Enable each individually via command line arguments.
    
    Hyperparameters are tuned for normalized MRI data (mean ~0, std ~1).
    Values are conservative to avoid unrealistic transformations.
    """
    # Gaussian noise: adds random noise to simulate acquisition variations
    gaussian_noise: bool = False
    gaussian_noise_std: float = 0.05  # 5% of normalized range - subtle noise
    gaussian_noise_prob: float = 0.5  # Apply with 50% probability
    
    # Brightness adjustment: shifts intensity values
    brightness: bool = False
    brightness_range: tuple = (-0.1, 0.1)  # ±10% shift for normalized data
    brightness_prob: float = 0.5
    
    # Contrast adjustment: scales intensity values around mean
    contrast: bool = False
    contrast_range: tuple = (0.9, 1.1)  # ±10% scaling
    contrast_prob: float = 0.5
    
    # Gaussian blur: simulates resolution variations
    blur: bool = False
    blur_sigma_range: tuple = (0.1, 0.5)  # Subtle blur, preserve lesion details
    blur_prob: float = 0.3  # Lower probability - blur can hurt small lesion detection
    
    # Per-channel augmentation: apply augmentations independently to each MRI sequence
    # If False, same augmentation is applied to all channels (T2W, ADC, HBV together)
    per_channel: bool = False
    
    def any_enabled(self) -> bool:
        """Check if any augmentation is enabled."""
        return any([
            self.gaussian_noise,
            self.brightness,
            self.contrast,
            self.blur
        ])
    
    def get_enabled_names(self) -> List[str]:
        """Get list of enabled augmentation names."""
        enabled = []
        if self.gaussian_noise:
            enabled.append(f"gaussian_noise(std={self.gaussian_noise_std})")
        if self.brightness:
            enabled.append(f"brightness(range={self.brightness_range})")
        if self.contrast:
            enabled.append(f"contrast(range={self.contrast_range})")
        if self.blur:
            enabled.append(f"blur(sigma={self.blur_sigma_range})")
        return enabled


class MRIAugmentation:
    """Augmentation pipeline for 3D MRI volumes.
    
    Designed for prostate cancer MRI with 3 sequences (T2W, ADC, HBV).
    Input shape: (C, D, H, W) where C=3 (sequences), D=20 (slices), H=W=256
    
    All augmentations preserve the diagnostic quality while adding
    realistic variations seen across different MRI scanners/protocols.
    """
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """Apply augmentation pipeline to MRI volume.
        
        Args:
            data: MRI volume of shape (C, D, H, W) - already normalized
            
        Returns:
            Augmented volume of same shape
        """
        if not self.config.any_enabled():
            return data
        
        # Work with a copy to avoid modifying original
        data = data.copy()
        
        # Apply augmentations
        if self.config.gaussian_noise and np.random.rand() < self.config.gaussian_noise_prob:
            data = self._add_gaussian_noise(data)
        
        if self.config.brightness and np.random.rand() < self.config.brightness_prob:
            data = self._adjust_brightness(data)
        
        if self.config.contrast and np.random.rand() < self.config.contrast_prob:
            data = self._adjust_contrast(data)
        
        if self.config.blur and np.random.rand() < self.config.blur_prob:
            data = self._apply_blur(data)
        
        return data
    
    def _add_gaussian_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to simulate acquisition variations.
        
        For normalized MRI data, std of 0.05 adds subtle noise
        that's realistic for scanner variations.
        """
        if self.config.per_channel:
            # Different noise for each sequence
            for c in range(data.shape[0]):
                noise = np.random.normal(0, self.config.gaussian_noise_std, data[c].shape)
                data[c] = data[c] + noise
        else:
            # Same noise pattern for all channels
            noise = np.random.normal(0, self.config.gaussian_noise_std, data.shape)
            data = data + noise
        
        return data
    
    def _adjust_brightness(self, data: np.ndarray) -> np.ndarray:
        """Adjust brightness by adding a constant offset.
        
        Simulates global intensity variations between scanners.
        For normalized data, range of ±0.1 is appropriate.
        """
        shift = np.random.uniform(*self.config.brightness_range)
        
        if self.config.per_channel:
            # Different shift for each sequence
            for c in range(data.shape[0]):
                channel_shift = np.random.uniform(*self.config.brightness_range)
                data[c] = data[c] + channel_shift
        else:
            data = data + shift
        
        return data
    
    def _adjust_contrast(self, data: np.ndarray) -> np.ndarray:
        """Adjust contrast by scaling intensity around mean.
        
        Simulates contrast variations between protocols.
        For normalized data, scaling by 0.9-1.1 is appropriate.
        
        Formula: output = mean + scale * (input - mean)
        """
        if self.config.per_channel:
            # Different scaling for each sequence
            for c in range(data.shape[0]):
                scale = np.random.uniform(*self.config.contrast_range)
                mean = data[c].mean()
                data[c] = mean + scale * (data[c] - mean)
        else:
            # Same scaling for all channels
            scale = np.random.uniform(*self.config.contrast_range)
            for c in range(data.shape[0]):
                mean = data[c].mean()
                data[c] = mean + scale * (data[c] - mean)
        
        return data
    
    def _apply_blur(self, data: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to simulate resolution variations.
        
        Uses conservative sigma values (0.1-0.5) to preserve
        diagnostic details while adding subtle smoothing.
        
        Blur is applied in 3D but with smaller sigma in depth
        direction due to larger slice spacing in MRI.
        """
        sigma = np.random.uniform(*self.config.blur_sigma_range)
        
        # Apply per channel (each sequence may have different characteristics)
        for c in range(data.shape[0]):
            # Sigma is (depth, height, width) - less blur in depth due to slice spacing
            sigma_3d = (sigma * 0.5, sigma, sigma)
            data[c] = scipy.ndimage.gaussian_filter(data[c], sigma=sigma_3d)
        
        return data
    
    def __repr__(self) -> str:
        enabled = self.config.get_enabled_names()
        if enabled:
            return f"MRIAugmentation({', '.join(enabled)})"
        return "MRIAugmentation(none)"


def create_augmentation_transform(
    gaussian_noise: bool = False,
    brightness: bool = False, 
    contrast: bool = False,
    blur: bool = False,
    per_channel: bool = False
) -> Optional[MRIAugmentation]:
    """Factory function to create augmentation transform.
    
    Returns None if no augmentations are enabled.
    
    Args:
        gaussian_noise: Enable Gaussian noise (std=0.05)
        brightness: Enable brightness adjustment (±10%)
        contrast: Enable contrast adjustment (±10%)
        blur: Enable Gaussian blur (sigma 0.1-0.5)
        per_channel: Apply augmentations independently per MRI sequence
        
    Returns:
        MRIAugmentation instance or None
    """
    config = AugmentationConfig(
        gaussian_noise=gaussian_noise,
        brightness=brightness,
        contrast=contrast,
        blur=blur,
        per_channel=per_channel
    )
    
    if not config.any_enabled():
        return None
    
    return MRIAugmentation(config)
