#!/usr/bin/env python3
"""
Inference script for semantic segmentation models.

This script provides functionality to load trained models and perform
inference on new images, with visualization and results saving capabilities.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Add the src directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from core.trainer import SegmentationTrainer
from config.settings import get_config, PASCAL_VOC_CLASSES, PASCAL_VOC_COLORS
from core.logging_config import get_logger

# Initialize logger
logger = get_logger(__name__)


class SegmentationPredictor:
    """
    Inference class for semantic segmentation models.
    
    Args:
        model_path: Path to trained model checkpoint
        config_name: Configuration name used for training
        device: Device to run inference on
    """
    
    def __init__(
        self, 
        model_path: str, 
        config_name: str = "default",
        device: Optional[str] = None
    ):
        self.config = get_config(config_name)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize trainer and load model
        self.trainer = SegmentationTrainer(self.config)
        self.trainer.load_checkpoint(model_path)
        self.trainer.model.eval()
        
        # Setup preprocessing transform
        self.transform = A.Compose([
            A.Resize(height=self.config.image_size[0], width=self.config.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess input image for inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (preprocessed_tensor, original_image)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_image
    
    def predict_single_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict segmentation mask for a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (predicted_mask, original_image)
        """
        image_tensor, original_image = self.preprocess_image(image_path)
        
        with torch.no_grad():
            # Run inference
            predictions = self.trainer.model(image_tensor.to(self.device))
            predicted_mask = torch.argmax(predictions, dim=1).squeeze().cpu().numpy()
        
        return predicted_mask, original_image
    
    def predict_batch(self, image_paths: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Predict segmentation masks for multiple images.
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            List of (predicted_mask, original_image) tuples
        """
        results = []
        
        for image_path in image_paths:
            try:
                mask, original = self.predict_single_image(image_path)
                results.append((mask, original))
                logger.info(f"Processed: {image_path}")
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                results.append((None, None))
        
        return results
    
    def create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Create colored visualization of segmentation mask.
        
        Args:
            mask: Predicted segmentation mask
            
        Returns:
            Colored mask as numpy array
        """
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id in range(len(PASCAL_VOC_COLORS)):
            colored_mask[mask == class_id] = PASCAL_VOC_COLORS[class_id]
        
        return colored_mask
    
    def visualize_prediction(
        self,
        original_image: np.ndarray,
        predicted_mask: np.ndarray,
        save_path: Optional[str] = None,
        overlay_alpha: float = 0.5
    ) -> None:
        """
        Visualize prediction results.
        
        Args:
            original_image: Original input image
            predicted_mask: Predicted segmentation mask
            save_path: Optional path to save visualization
            overlay_alpha: Alpha value for mask overlay
        """
        # Create colored mask
        colored_mask = self.create_colored_mask(predicted_mask)
        
        # Resize colored mask to match original image size
        if original_image.shape[:2] != predicted_mask.shape:
            colored_mask = cv2.resize(
                colored_mask, 
                (original_image.shape[1], original_image.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predicted mask
        axes[1].imshow(predicted_mask, cmap='tab20')
        axes[1].set_title('Predicted Segmentation')
        axes[1].axis('off')
        
        # Overlay
        overlay = original_image.astype(float)
        overlay = overlay * (1 - overlay_alpha) + colored_mask.astype(float) * overlay_alpha
        overlay = overlay.astype(np.uint8)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # Add colorbar legend
        # Create a small subplot for the legend
        unique_classes = np.unique(predicted_mask)
        legend_text = [f"{PASCAL_VOC_CLASSES[cls]}" for cls in unique_classes if cls < len(PASCAL_VOC_CLASSES)]
        legend_colors = [np.array(PASCAL_VOC_COLORS[cls])/255.0 for cls in unique_classes if cls < len(PASCAL_VOC_COLORS)]
        
        # Add legend as text
        legend_str = "Classes detected:\n" + "\n".join(legend_text)
        plt.figtext(0.02, 0.02, legend_str, fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def get_class_statistics(self, mask: np.ndarray) -> dict:
        """
        Get statistics about detected classes in the mask.
        
        Args:
            mask: Predicted segmentation mask
            
        Returns:
            Dictionary with class statistics
        """
        unique_classes, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        stats = {}
        for cls, count in zip(unique_classes, counts):
            if cls < len(PASCAL_VOC_CLASSES):
                stats[PASCAL_VOC_CLASSES[cls]] = {
                    'pixel_count': int(count),
                    'percentage': float(count / total_pixels * 100)
                }
        
        return stats


def main():
    """Main function for inference script."""
    parser = argparse.ArgumentParser(description="Run inference on segmentation models")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory containing images"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name used for training"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda/cpu)"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization plots"
    )
    
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save predicted masks as images"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize predictor
    predictor = SegmentationPredictor(
        model_path=args.model,
        config_name=args.config,
        device=args.device
    )
    
    # Get input images
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            str(p) for p in input_path.iterdir() 
            if p.suffix.lower() in image_extensions
        ]
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Process images
    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        
        try:
            # Run prediction
            predicted_mask, original_image = predictor.predict_single_image(image_path)
            
            # Get class statistics
            stats = predictor.get_class_statistics(predicted_mask)
            logger.info("Class statistics:")
            for class_name, class_stats in stats.items():
                logger.info(f"  {class_name}: {class_stats['percentage']:.2f}% ({class_stats['pixel_count']} pixels)")
            
            # Generate output filename
            image_name = Path(image_path).stem
            
            # Save mask if requested
            if args.save_masks:
                mask_path = output_dir / f"{image_name}_mask.png"
                colored_mask = predictor.create_colored_mask(predicted_mask)
                cv2.imwrite(str(mask_path), cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
                logger.info(f"Mask saved to {mask_path}")
            
            # Create visualization if requested
            if args.visualize:
                viz_path = output_dir / f"{image_name}_prediction.png"
                predictor.visualize_prediction(
                    original_image,
                    predicted_mask,
                    save_path=str(viz_path)
                )
        
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
    
    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
