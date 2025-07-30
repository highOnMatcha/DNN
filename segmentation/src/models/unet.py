"""
U-Net model implementation for semantic segmentation.

This module implements the U-Net architecture from scratch, which is a popular
encoder-decoder network with skip connections for semantic segmentation tasks.
The implementation includes both a custom U-Net and wrapper for pre-trained models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import segmentation_models_pytorch as smp


class DoubleConv(nn.Module):
    """
    Double convolution block used in U-Net.
    
    Consists of two consecutive convolutions with batch normalization and ReLU.
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        # If bilinear, use normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net implementation from scratch.
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        bilinear: Whether to use bilinear interpolation for upsampling
    """
    
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class SegmentationModel(nn.Module):
    """
    Wrapper class for segmentation models.
    
    Supports both custom U-Net and pre-trained models from segmentation_models_pytorch.
    """
    
    def __init__(
        self,
        architecture: str = "unet",
        encoder_name: str = "resnet50",
        encoder_weights: Optional[str] = "imagenet",
        in_channels: int = 3,
        classes: int = 21,
        **kwargs
    ):
        super().__init__()
        
        self.architecture = architecture
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.classes = classes
        
        if architecture == "unet_custom":
            # Use our custom U-Net implementation
            self.model = UNet(
                n_channels=in_channels,
                n_classes=classes,
                bilinear=kwargs.get('bilinear', False)
            )
        elif architecture == "unet":
            # Use segmentation_models_pytorch U-Net
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        elif architecture == "deeplabv3":
            # Use DeepLabV3
            self.model = smp.DeepLabV3(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        elif architecture == "deeplabv3plus":
            # Use DeepLabV3+
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        elif architecture == "fpn":
            # Use Feature Pyramid Network
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_encoder(self):
        """Get the encoder part of the model if available."""
        if hasattr(self.model, 'encoder'):
            return self.model.encoder
        return None
    
    def get_decoder(self):
        """Get the decoder part of the model if available."""
        if hasattr(self.model, 'decoder'):
            return self.model.decoder
        return None


def create_segmentation_model(config) -> SegmentationModel:
    """
    Factory function to create a segmentation model based on configuration.
    
    Args:
        config: Configuration object with model parameters
        
    Returns:
        SegmentationModel instance
    """
    model = SegmentationModel(
        architecture=config.architecture,
        encoder_name=config.encoder_name,
        encoder_weights=config.encoder_weights,
        in_channels=config.in_channels,
        classes=config.classes
    )
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count the total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> Dict:
    """
    Get a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        
    Returns:
        Dictionary with model summary information
    """
    total_params, trainable_params = count_parameters(model)
    
    # Calculate model size in MB
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": size_mb,
        "input_size": input_size
    }
