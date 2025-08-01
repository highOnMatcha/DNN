"""
Anti-blur improvements for Pokemon sprite generation.

This module provides specialized loss functions and training techniques
to combat blurriness in generated sprites and improve pixel-art quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional
import numpy as np


class AntiBlurLoss(nn.Module):
    """Specialized loss function to combat blurriness in generated images."""
    
    def __init__(self, alpha: float = 1.0, beta: float = 0.5):
        super().__init__()
        self.alpha = alpha  # Edge preservation weight
        self.beta = beta    # High frequency enhancement weight
        
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute anti-blur loss combining edge preservation and high-frequency enhancement.
        
        Args:
            generated: Generated images [B, C, H, W]
            target: Target images [B, C, H, W]
            
        Returns:
            Anti-blur loss value
        """
        # Edge preservation loss
        edge_loss = self._edge_loss(generated, target)
        
        # High frequency enhancement loss
        freq_loss = self._frequency_loss(generated, target)
        
        return self.alpha * edge_loss + self.beta * freq_loss
    
    def _edge_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute edge preservation loss using Sobel operators."""
        # Sobel X filter
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=generated.device).view(1, 1, 3, 3)
        # Sobel Y filter  
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=generated.device).view(1, 1, 3, 3)
        
        # Expand for all channels
        sobel_x = sobel_x.expand(3, 1, 3, 3)
        sobel_y = sobel_y.expand(3, 1, 3, 3)
        
        # Compute edges
        gen_edges_x = F.conv2d(generated, sobel_x, padding=1, groups=3)
        gen_edges_y = F.conv2d(generated, sobel_y, padding=1, groups=3)
        gen_edges = torch.sqrt(gen_edges_x**2 + gen_edges_y**2 + 1e-8)
        
        target_edges_x = F.conv2d(target, sobel_x, padding=1, groups=3)
        target_edges_y = F.conv2d(target, sobel_y, padding=1, groups=3)
        target_edges = torch.sqrt(target_edges_x**2 + target_edges_y**2 + 1e-8)
        
        return F.l1_loss(gen_edges, target_edges)
    
    def _frequency_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute high-frequency enhancement loss using FFT."""
        # Convert to grayscale for frequency analysis
        gen_gray = 0.299 * generated[:, 0] + 0.587 * generated[:, 1] + 0.114 * generated[:, 2]
        target_gray = 0.299 * target[:, 0] + 0.587 * target[:, 1] + 0.114 * target[:, 2]
        
        # Compute FFT
        gen_fft = torch.fft.fft2(gen_gray)
        target_fft = torch.fft.fft2(target_gray)
        
        # Focus on high frequencies (center of spectrum)
        h, w = gen_fft.shape[-2:]
        center_h, center_w = h // 2, w // 2
        
        # Create high-pass filter
        y, x = torch.meshgrid(torch.arange(h, device=generated.device), 
                             torch.arange(w, device=generated.device), indexing='ij')
        distance = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
        high_pass = (distance > min(h, w) * 0.1).float()  # Keep frequencies > 10% of max
        
        # Apply high-pass filter
        gen_high_freq = gen_fft * high_pass
        target_high_freq = target_fft * high_pass
        
        # Compute loss on high-frequency components
        return F.mse_loss(torch.abs(gen_high_freq), torch.abs(target_high_freq))


class PixelArtLoss(nn.Module):
    """Specialized loss for pixel-art style preservation."""
    
    def __init__(self, sharpness_weight: float = 2.0, color_weight: float = 1.0):
        super().__init__()
        self.sharpness_weight = sharpness_weight
        self.color_weight = color_weight
        
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute pixel-art preservation loss.
        
        Args:
            generated: Generated images [B, C, H, W]
            target: Target images [B, C, H, W]
            
        Returns:
            Pixel-art loss value
        """
        # Sharpness loss (penalize smooth gradients)
        sharpness_loss = self._sharpness_loss(generated, target)
        
        # Color quantization loss (encourage discrete colors)
        color_loss = self._color_quantization_loss(generated, target)
        
        return self.sharpness_weight * sharpness_loss + self.color_weight * color_loss
    
    def _sharpness_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Penalize smooth gradients to encourage sharp edges."""
        # Compute local gradients
        gen_grad_x = torch.abs(generated[:, :, :, 1:] - generated[:, :, :, :-1])
        gen_grad_y = torch.abs(generated[:, :, 1:, :] - generated[:, :, :-1, :])
        
        target_grad_x = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_grad_y = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        
        # Encourage generated gradients to match target gradients
        loss_x = F.mse_loss(gen_grad_x, target_grad_x)
        loss_y = F.mse_loss(gen_grad_y, target_grad_y)
        
        return loss_x + loss_y
    
    def _color_quantization_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Encourage discrete color values typical of pixel art."""
        # Quantize to limited color palette (simulate pixel art)
        levels = 16  # Reduce to 16 levels per channel
        
        # Quantize generated images
        gen_quantized = torch.round(generated * (levels - 1)) / (levels - 1)
        
        # Encourage generated images to be close to quantized version
        quantization_loss = F.mse_loss(generated, gen_quantized.detach())
        
        # Also compare quantized versions
        target_quantized = torch.round(target * (levels - 1)) / (levels - 1)
        palette_loss = F.mse_loss(gen_quantized, target_quantized)
        
        return quantization_loss + palette_loss


class ImprovedPerceptualLoss(nn.Module):
    """Improved perceptual loss with anti-blur enhancements."""
    
    def __init__(self, layers: list = [3, 8, 15, 22], weights: list = [1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        # Use new weights parameter instead of deprecated pretrained
        try:
            from torchvision.models import VGG16_Weights
            self.vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except ImportError:
            # Fallback for older torchvision versions
            self.vgg = models.vgg16(pretrained=True).features
            
        self.layers = layers
        self.weights = weights
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        # Use full precision for better quality
        self.vgg = self.vgg.float()
        
    def forward(self, input_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are in [0, 1] range
        if input_img.max() > 1.0 or input_img.min() < 0.0:
            input_img = (input_img + 1) / 2
        if target_img.max() > 1.0 or target_img.min() < 0.0:
            target_img = (target_img + 1) / 2
            
        # Resize to 224x224 for VGG
        input_img = F.interpolate(input_img, size=(224, 224), mode='bilinear', align_corners=False)
        target_img = F.interpolate(target_img, size=(224, 224), mode='bilinear', align_corners=False)
        
        loss = torch.tensor(0.0, device=input_img.device, dtype=torch.float32)
        x = input_img
        y = target_img
        
        for i, layer in enumerate(list(self.vgg.children())):
            x = layer(x)
            y = layer(y)
            
            if i in self.layers:
                weight = self.weights[self.layers.index(i)]
                # Use L1 loss instead of MSE for sharper features
                loss += weight * F.l1_loss(x, y)
                
        return loss


class CombinedAntiBlurLoss(nn.Module):
    """Combined loss function to combat blurriness comprehensively."""
    
    def __init__(self, 
                 l1_weight: float = 100.0,
                 anti_blur_weight: float = 10.0,
                 pixel_art_weight: float = 5.0,
                 perceptual_weight: float = 1.0):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.anti_blur_weight = anti_blur_weight
        self.pixel_art_weight = pixel_art_weight
        self.perceptual_weight = perceptual_weight
        
        # Initialize loss components
        self.l1_loss = nn.L1Loss()
        self.anti_blur_loss = AntiBlurLoss()
        self.pixel_art_loss = PixelArtLoss()
        self.perceptual_loss = ImprovedPerceptualLoss()
        
    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined anti-blur loss.
        
        Returns:
            Total loss and individual loss components for logging
        """
        # Basic L1 loss
        l1 = self.l1_loss(generated, target)
        
        # Anti-blur components
        anti_blur = self.anti_blur_loss(generated, target)
        pixel_art = self.pixel_art_loss(generated, target)
        perceptual = self.perceptual_loss(generated, target)
        
        # Combine losses
        total_loss = (self.l1_weight * l1 + 
                     self.anti_blur_weight * anti_blur +
                     self.pixel_art_weight * pixel_art +
                     self.perceptual_weight * perceptual)
        
        # Return loss components for logging
        loss_components = {
            'l1_loss': l1.item(),
            'anti_blur_loss': anti_blur.item(),
            'pixel_art_loss': pixel_art.item(),
            'perceptual_loss': perceptual.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_components
