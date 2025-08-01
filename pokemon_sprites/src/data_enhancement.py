#!/usr/bin/env python3
"""
Data enhancement script to create more training samples for Pokemon sprite generation.

This script uses various techniques to expand the training dataset:
1. Style transfer between different Pokemon
2. Color palette swapping
3. Advanced augmentation with semantic preservation
4. Synthetic data generation using existing models
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
import random
import cv2
from typing import List, Tuple

class DataEnhancer:
    def __init__(self, input_dir: str, output_dir: str, target_size: int = 64):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def color_palette_swap(self, image: Image.Image, target_image: Image.Image) -> Image.Image:
        """Swap color palette from one Pokemon to another while preserving structure."""
        # Convert to numpy arrays
        img_arr = np.array(image)
        target_arr = np.array(target_image)
        
        # Extract dominant colors from target
        target_colors = self._extract_dominant_colors(target_arr, n_colors=8)
        
        # Apply color mapping
        result = self._remap_colors(img_arr, target_colors)
        
        return Image.fromarray(result)
    
    def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 8) -> List[Tuple[int, int, int]]:
        """Extract dominant colors using k-means clustering."""
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Use k-means to find dominant colors
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        kmeans.fit(pixels)
        
        return [tuple(map(int, color)) for color in kmeans.cluster_centers_]
    
    def _remap_colors(self, image: np.ndarray, target_colors: List[Tuple[int, int, int]]) -> np.ndarray:
        """Remap image colors to target palette."""
        result = image.copy()
        
        # Extract colors from source image
        source_colors = self._extract_dominant_colors(image, len(target_colors))
        
        # Create color mapping
        for i, (source_color, target_color) in enumerate(zip(source_colors, target_colors)):
            # Find pixels close to source color
            diff = np.abs(image - source_color).sum(axis=2)
            mask = diff < 50  # Threshold for color similarity
            
            # Replace with target color
            result[mask] = target_color
            
        return result
    
    def create_variants(self, image_path: Path, n_variants: int = 5) -> List[Image.Image]:
        """Create multiple variants of a single image."""
        image = Image.open(image_path).convert('RGB')
        variants = []
        
        for i in range(n_variants):
            variant = image.copy()
            
            # Random brightness/contrast
            enhancer = ImageEnhance.Brightness(variant)
            variant = enhancer.enhance(random.uniform(0.8, 1.2))
            
            enhancer = ImageEnhance.Contrast(variant)
            variant = enhancer.enhance(random.uniform(0.8, 1.2))
            
            # Random hue shift
            enhancer = ImageEnhance.Color(variant)
            variant = enhancer.enhance(random.uniform(0.7, 1.3))
            
            # Slight rotation and scaling
            angle = random.uniform(-5, 5)
            scale = random.uniform(0.95, 1.05)
            
            variant = variant.rotate(angle, expand=False, fillcolor=(255, 255, 255))
            
            # Resize with slight variation
            size = int(self.target_size * scale)
            variant = variant.resize((size, size), Image.Resampling.LANCZOS)
            variant = variant.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
            
            variants.append(variant)
            
        return variants
    
    def enhance_dataset(self, input_pokemon_dir: str, sprite_pokemon_dir: str, multiplier: int = 3):
        """Enhance dataset by creating variants and cross-Pokemon style transfers."""
        input_dir = Path(input_pokemon_dir)
        sprite_dir = Path(sprite_pokemon_dir)
        
        input_files = list(input_dir.glob("*.png"))
        sprite_files = list(sprite_dir.glob("*.png"))
        
        print(f"Found {len(input_files)} input files and {len(sprite_files)} sprite files")
        
        enhanced_count = 0
        
        for input_file in input_files:
            # Find corresponding sprite
            pokemon_id = input_file.stem.split('_')[1]
            sprite_file = sprite_dir / f"pokemon_{pokemon_id}_bw.png"
            
            if not sprite_file.exists():
                continue
                
            # Load original pair
            input_img = Image.open(input_file).convert('RGB')
            sprite_img = Image.open(sprite_file).convert('RGB')
            
            # Create variants of original
            input_variants = self.create_variants(input_file, multiplier)
            sprite_variants = self.create_variants(sprite_file, multiplier)
            
            # Save variants
            for i, (inp_var, spr_var) in enumerate(zip(input_variants, sprite_variants)):
                # Save enhanced input
                inp_output = self.output_dir / "input" / f"pokemon_{pokemon_id}_var{i}.png"
                inp_output.parent.mkdir(exist_ok=True)
                inp_var.save(inp_output)
                
                # Save enhanced sprite
                spr_output = self.output_dir / "target" / f"pokemon_{pokemon_id}_var{i}.png"
                spr_output.parent.mkdir(exist_ok=True)
                spr_var.save(spr_output)
                
                enhanced_count += 1
            
            # Create color-swapped versions with other Pokemon
            other_pokemon = random.sample(input_files, min(2, len(input_files)))
            for j, other_file in enumerate(other_pokemon):
                if other_file == input_file:
                    continue
                    
                other_img = Image.open(other_file).convert('RGB')
                
                # Color swap input
                swapped_input = self.color_palette_swap(input_img, other_img)
                
                # Save color-swapped pair
                inp_output = self.output_dir / "input" / f"pokemon_{pokemon_id}_swap{j}.png"
                swapped_input.save(inp_output)
                
                spr_output = self.output_dir / "target" / f"pokemon_{pokemon_id}_swap{j}.png"
                sprite_img.save(spr_output)
                
                enhanced_count += 1
            
            if enhanced_count % 50 == 0:
                print(f"Enhanced {enhanced_count} samples...")
        
        print(f"Dataset enhancement complete! Created {enhanced_count} additional samples.")
        print(f"Enhanced data saved to: {self.output_dir}")


def main():
    """Main data enhancement function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance Pokemon dataset")
    parser.add_argument("--input-dir", required=True, help="Input Pokemon artwork directory")
    parser.add_argument("--sprite-dir", required=True, help="Pokemon sprite directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for enhanced data")
    parser.add_argument("--multiplier", type=int, default=3, help="Enhancement multiplier")
    
    args = parser.parse_args()
    
    enhancer = DataEnhancer(args.input_dir, args.output_dir)
    enhancer.enhance_dataset(args.input_dir, args.sprite_dir, args.multiplier)


if __name__ == "__main__":
    main()
