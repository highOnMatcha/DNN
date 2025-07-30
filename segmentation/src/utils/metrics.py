"""
Metrics and evaluation utilities for segmentation tasks.

This module provides comprehensive evaluation metrics for semantic segmentation,
including IoU, Dice coefficient, pixel accuracy, and class-wise metrics.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class SegmentationMetrics:
    """
    Comprehensive metrics calculator for semantic segmentation.
    
    Args:
        num_classes: Number of segmentation classes
        ignore_index: Class index to ignore in calculations (e.g., background)
        average: Averaging method ('macro', 'weighted', 'none')
    """
    
    def __init__(
        self,
        num_classes: int,
        ignore_index: Optional[int] = None,
        average: str = 'macro'
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.average = average
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.total_samples = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with a batch of predictions and targets.
        
        Args:
            predictions: Model predictions (B, C, H, W) or (B, H, W)
            targets: Ground truth masks (B, H, W)
        """
        # Convert predictions to class indices if needed
        if predictions.dim() == 4:  # (B, C, H, W)
            predictions = torch.argmax(predictions, dim=1)
        
        # Flatten tensors
        predictions = predictions.view(-1).cpu().numpy()
        targets = targets.view(-1).cpu().numpy()
        
        # Create mask for valid pixels
        if self.ignore_index is not None:
            valid_mask = targets != self.ignore_index
            predictions = predictions[valid_mask]
            targets = targets[valid_mask]
        
        # Update confusion matrix
        cm = confusion_matrix(
            targets,
            predictions,
            labels=list(range(self.num_classes))
        )
        self.confusion_matrix += cm
        self.total_samples += len(targets)
    
    def compute_iou(self) -> Dict[str, float]:
        """
        Compute Intersection over Union (IoU) metrics.
        
        Returns:
            Dictionary with IoU metrics
        """
        # Avoid division by zero
        eps = 1e-8
        
        # Calculate IoU for each class
        intersection = np.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0) -
            intersection
        )
        
        iou_per_class = intersection / (union + eps)
        
        # Calculate mean IoU based on averaging method
        if self.average == 'macro':
            mean_iou = np.mean(iou_per_class)
        elif self.average == 'weighted':
            weights = self.confusion_matrix.sum(axis=1)
            mean_iou = np.average(iou_per_class, weights=weights)
        else:
            mean_iou = iou_per_class
        
        return {
            'iou_per_class': iou_per_class,
            'mean_iou': mean_iou
        }
    
    def compute_dice(self) -> Dict[str, float]:
        """
        Compute Dice coefficient metrics.
        
        Returns:
            Dictionary with Dice metrics
        """
        eps = 1e-8
        
        # Calculate Dice for each class
        intersection = np.diag(self.confusion_matrix)
        dice_denominator = (
            self.confusion_matrix.sum(axis=1) +
            self.confusion_matrix.sum(axis=0)
        )
        
        dice_per_class = (2.0 * intersection) / (dice_denominator + eps)
        
        # Calculate mean Dice
        if self.average == 'macro':
            mean_dice = np.mean(dice_per_class)
        elif self.average == 'weighted':
            weights = self.confusion_matrix.sum(axis=1)
            mean_dice = np.average(dice_per_class, weights=weights)
        else:
            mean_dice = dice_per_class
        
        return {
            'dice_per_class': dice_per_class,
            'mean_dice': mean_dice
        }
    
    def compute_pixel_accuracy(self) -> float:
        """
        Compute overall pixel accuracy.
        
        Returns:
            Pixel accuracy as float
        """
        correct_pixels = np.diag(self.confusion_matrix).sum()
        total_pixels = self.confusion_matrix.sum()
        return correct_pixels / (total_pixels + 1e-8)
    
    def compute_class_accuracy(self) -> Dict[str, float]:
        """
        Compute per-class accuracy (recall).
        
        Returns:
            Dictionary with class accuracies
        """
        eps = 1e-8
        class_accuracy = np.diag(self.confusion_matrix) / (
            self.confusion_matrix.sum(axis=1) + eps
        )
        
        if self.average == 'macro':
            mean_accuracy = np.mean(class_accuracy)
        elif self.average == 'weighted':
            weights = self.confusion_matrix.sum(axis=1)
            mean_accuracy = np.average(class_accuracy, weights=weights)
        else:
            mean_accuracy = class_accuracy
        
        return {
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': mean_accuracy
        }
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics and return as a dictionary.
        
        Returns:
            Dictionary with all computed metrics
        """
        iou_metrics = self.compute_iou()
        dice_metrics = self.compute_dice()
        pixel_acc = self.compute_pixel_accuracy()
        class_acc_metrics = self.compute_class_accuracy()
        
        return {
            'pixel_accuracy': pixel_acc,
            'mean_iou': iou_metrics['mean_iou'],
            'mean_dice': dice_metrics['mean_dice'],
            'mean_class_accuracy': class_acc_metrics['mean_class_accuracy'],
            'iou_per_class': iou_metrics['iou_per_class'],
            'dice_per_class': dice_metrics['dice_per_class'],
            'class_accuracy': class_acc_metrics['class_accuracy']
        }


def dice_loss(predictions: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Compute Dice loss for segmentation.
    
    Args:
        predictions: Model predictions (B, C, H, W)
        targets: Ground truth masks (B, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice loss as tensor
    """
    # Apply softmax to predictions
    predictions = F.softmax(predictions, dim=1)
    
    # Convert targets to one-hot encoding
    targets_one_hot = F.one_hot(targets, num_classes=predictions.size(1))
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
    
    # Calculate intersection and union
    intersection = (predictions * targets_one_hot).sum(dim=(2, 3))
    union = predictions.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
    
    # Calculate Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    # Return Dice loss (1 - Dice coefficient)
    return 1 - dice.mean()


def focal_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.0,
    gamma: float = 2.0,
    ignore_index: Optional[int] = None
) -> torch.Tensor:
    """
    Compute Focal loss for addressing class imbalance.
    
    Args:
        predictions: Model predictions (B, C, H, W)
        targets: Ground truth masks (B, H, W)
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
        ignore_index: Class index to ignore
        
    Returns:
        Focal loss as tensor
    """
    ce_loss = F.cross_entropy(predictions, targets, ignore_index=ignore_index, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def combined_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
    ignore_index: Optional[int] = None
) -> torch.Tensor:
    """
    Combine Cross-Entropy and Dice loss.
    
    Args:
        predictions: Model predictions (B, C, H, W)
        targets: Ground truth masks (B, H, W)
        ce_weight: Weight for cross-entropy loss
        dice_weight: Weight for dice loss
        ignore_index: Class index to ignore
        
    Returns:
        Combined loss as tensor
    """
    ce_loss = F.cross_entropy(predictions, targets, ignore_index=ignore_index)
    dice_loss_val = dice_loss(predictions, targets)
    
    return ce_weight * ce_loss + dice_weight * dice_loss_val


def visualize_predictions(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    save_path: Optional[str] = None,
    num_samples: int = 4
) -> None:
    """
    Visualize model predictions alongside ground truth.
    
    Args:
        images: Input images (B, C, H, W)
        predictions: Model predictions (B, C, H, W)
        targets: Ground truth masks (B, H, W)
        class_names: List of class names
        save_path: Optional path to save the visualization
        num_samples: Number of samples to visualize
    """
    # Convert predictions to class indices
    pred_masks = torch.argmax(predictions, dim=1)
    
    # Select samples to visualize
    num_samples = min(num_samples, images.size(0))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]
    
    for i in range(num_samples):
        # Denormalize image
        img = images[i].cpu()
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std[:, None, None] + mean[:, None, None]
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        pred_mask = pred_masks[i].cpu().numpy()
        target_mask = targets[i].cpu().numpy()
        
        # Plot image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(target_mask, cmap='tab20', vmin=0, vmax=len(class_names)-1)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(pred_mask, cmap='tab20', vmin=0, vmax=len(class_names)-1)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Optional path to save the plot
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        cm = confusion_matrix
        title = 'Confusion Matrix'
        fmt = 'd'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def calculate_flops(model: torch.nn.Module, input_size: Tuple[int, ...]) -> int:
    """
    Calculate FLOPs (Floating Point Operations) for a model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
        
    Returns:
        Number of FLOPs
    """
    try:
        from thop import profile
        input_tensor = torch.randn(1, *input_size)
        flops, params = profile(model, inputs=(input_tensor,))
        return flops
    except ImportError:
        print("thop package not installed. Cannot calculate FLOPs.")
        return 0
