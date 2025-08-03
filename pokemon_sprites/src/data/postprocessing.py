"""
Post-processing utilities for Pokemon sprite generation.

This module contains classes and functions for converting ARGB model outputs
to optimized P format sprites suitable for game deployment.
"""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict, Optional, Union, Any


class ARGBToPaletteConverter:
    """
    Post-processing converter for ARGB outputs to optimal P format sprites.
    
    Converts 4-channel RGBA images to palette-based P format with optimized
    color palettes, transparency preservation, and compression benefits.
    """
    
    def __init__(self, max_colors: int = 256, preserve_transparency: bool = True, 
                 optimize_palette: bool = True):
        """
        Initialize the ARGB to palette converter.
        
        Args:
            max_colors: Maximum number of colors in palette (1-256)
            preserve_transparency: Whether to preserve transparency information
            optimize_palette: Whether to use K-means optimization for palette
        """
        self.max_colors = min(max(max_colors, 1), 256)
        self.preserve_transparency = preserve_transparency
        self.optimize_palette = optimize_palette
        self.global_palette = None
        
    def _extract_colors(self, rgba_image: Image.Image) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Extract unique colors from RGBA image, separating transparent pixels.
        
        Args:
            rgba_image: PIL Image in RGBA format
            
        Returns:
            Tuple of (opaque_pixels, transparent_pixels) as numpy arrays
        """
        img_array = np.array(rgba_image)
        h, w, c = img_array.shape
        pixels = img_array.reshape(-1, c)
        
        if self.preserve_transparency:
            alpha_mask = pixels[:, 3] == 0
            transparent_pixels = pixels[alpha_mask]
            opaque_pixels = pixels[~alpha_mask]
            return opaque_pixels, transparent_pixels if len(transparent_pixels) > 0 else None
        
        return pixels, None
    
    def _optimize_palette_kmeans(self, pixels: np.ndarray, n_colors: int) -> np.ndarray:
        """
        Use K-means clustering to find optimal color palette.
        
        Args:
            pixels: Array of pixel values (N, 4) for RGBA
            n_colors: Number of colors for palette
            
        Returns:
            Optimized palette as numpy array (n_colors, 4)
        """
        if len(pixels) == 0:
            return np.array([]).reshape(0, 4)
            
        if len(pixels) <= n_colors:
            return np.unique(pixels, axis=0)
        
        # Use only RGB for clustering (alpha handled separately)
        rgb_pixels = pixels[:, :3] if pixels.shape[1] == 4 else pixels
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(rgb_pixels)
        
        # Get cluster centers and add alpha channel
        centers = kmeans.cluster_centers_
        if pixels.shape[1] == 4:
            # Add most common alpha value for each cluster
            labels = kmeans.labels_
            alpha_values = []
            for i in range(n_colors):
                cluster_pixels = pixels[labels == i]
                if len(cluster_pixels) > 0:
                    alpha_vals, counts = np.unique(cluster_pixels[:, 3], return_counts=True)
                    most_common_alpha = alpha_vals[np.argmax(counts)]
                    alpha_values.append(most_common_alpha)
                else:
                    alpha_values.append(255)
            
            centers = np.column_stack([centers, alpha_values])
        
        return centers.astype(np.uint8)
    
    def _assign_to_palette(self, pixels: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """
        Assign pixels to closest palette colors using Euclidean distance.
        
        Args:
            pixels: Array of pixel values to assign
            palette: Palette colors to assign to
            
        Returns:
            Array of palette indices for each pixel
        """
        if len(palette) == 0:
            return np.zeros(len(pixels), dtype=np.uint8)
        
        distances = cdist(pixels, palette, metric='euclidean')
        indices = np.argmin(distances, axis=1)
        
        return indices.astype(np.uint8)
    
    def convert_single_image(self, rgba_image: Union[Image.Image, np.ndarray], 
                           custom_palette: Optional[np.ndarray] = None) -> Tuple[Image.Image, np.ndarray]:
        """
        Convert single RGBA image to P format with optimal palette.
        
        Args:
            rgba_image: Input RGBA image (PIL Image or numpy array)
            custom_palette: Optional custom palette to use
            
        Returns:
            Tuple of (p_format_image, palette_used)
        """
        if not isinstance(rgba_image, Image.Image):
            rgba_image = Image.fromarray(rgba_image, 'RGBA')
        
        # Extract colors
        opaque_pixels, transparent_pixels = self._extract_colors(rgba_image)
        
        if custom_palette is not None:
            palette = custom_palette
        else:
            # Reserve space for transparency if needed
            max_opaque_colors = self.max_colors - (1 if self.preserve_transparency and transparent_pixels is not None else 0)
            
            # Optimize palette for opaque colors
            if len(opaque_pixels) > 0 and self.optimize_palette:
                palette = self._optimize_palette_kmeans(opaque_pixels, max_opaque_colors)
            else:
                palette = np.unique(opaque_pixels, axis=0)[:max_opaque_colors]
            
            # Add transparency color if needed
            if self.preserve_transparency and transparent_pixels is not None:
                transparent_color = np.array([[0, 0, 0, 0]], dtype=np.uint8)
                palette = np.vstack([transparent_color, palette])
        
        # Convert image
        img_array = np.array(rgba_image)
        h, w, c = img_array.shape
        pixels = img_array.reshape(-1, c)
        
        # Assign pixels to palette
        indices = self._assign_to_palette(pixels, palette)
        
        # Create P format image
        indexed_array = indices.reshape(h, w)
        p_image = Image.fromarray(indexed_array, 'P')
        
        # Set palette (PIL expects RGB format for palette)
        palette_rgb = palette[:, :3].flatten()
        # Pad palette to 768 bytes (256 * 3) if needed
        if len(palette_rgb) < 768:
            palette_rgb = np.pad(palette_rgb, (0, 768 - len(palette_rgb)), 'constant')
        
        p_image.putpalette(palette_rgb.tolist())
        
        # Handle transparency - find transparent color index
        if self.preserve_transparency and len(palette) > 0:
            # Find the index of transparent pixels (alpha == 0)
            transparent_indices = np.where(palette[:, 3] == 0)[0]
            if len(transparent_indices) > 0:
                p_image.info['transparency'] = int(transparent_indices[0])
        
        return p_image, palette
    
    def convert_batch(self, rgba_images: List[Union[Image.Image, np.ndarray]], 
                     use_global_palette: bool = True) -> Tuple[List[Image.Image], Optional[np.ndarray]]:
        """
        Convert batch of RGBA images with consistent palette.
        
        Args:
            rgba_images: List of RGBA images to convert
            use_global_palette: Whether to use a global palette for all images
            
        Returns:
            Tuple of (converted_p_images, global_palette_used)
        """
        if use_global_palette and self.global_palette is None:
            # Build global palette from all images
            all_pixels = []
            for img in rgba_images:
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img, 'RGBA')
                opaque_pixels, _ = self._extract_colors(img)
                if len(opaque_pixels) > 0:
                    all_pixels.append(opaque_pixels)
            
            if all_pixels:
                combined_pixels = np.vstack(all_pixels)
                max_colors = self.max_colors - (1 if self.preserve_transparency else 0)
                palette = self._optimize_palette_kmeans(combined_pixels, max_colors)
                
                if self.preserve_transparency:
                    transparent_color = np.array([[0, 0, 0, 0]], dtype=np.uint8)
                    palette = np.vstack([transparent_color, palette])
                
                self.global_palette = palette
        
        # Convert all images with consistent palette
        results = []
        for img in rgba_images:
            p_img, _ = self.convert_single_image(img, self.global_palette if use_global_palette else None)
            results.append(p_img)
        
        return results, self.global_palette
    
    def p_to_rgba_for_display(self, p_image: Image.Image) -> Image.Image:
        """
        Convert P format image back to RGBA for proper display with transparency.
        
        Args:
            p_image: P format image with transparency info
            
        Returns:
            RGBA image suitable for display
        """
        # Convert to RGBA to preserve transparency
        if 'transparency' in p_image.info:
            # P format with transparency
            rgba_image = p_image.convert('RGBA')
        else:
            # P format without transparency - convert via RGB then add alpha
            rgb_image = p_image.convert('RGB')
            rgba_image = rgb_image.convert('RGBA')
        
        return rgba_image
    
    def analyze_compression(self, rgba_image: Union[Image.Image, np.ndarray], 
                          p_image: Image.Image) -> Dict[str, Union[int, float]]:
        """
        Analyze compression benefits of P format conversion.
        
        Args:
            rgba_image: Original RGBA image
            p_image: Converted P format image
            
        Returns:
            Dictionary with compression analysis metrics
        """
        if not isinstance(rgba_image, Image.Image):
            rgba_image = Image.fromarray(rgba_image, 'RGBA')
        
        # Calculate theoretical sizes
        rgba_size = rgba_image.size[0] * rgba_image.size[1] * 4  # 4 bytes per pixel
        p_size = p_image.size[0] * p_image.size[1] * 1 + 768  # 1 byte per pixel + palette
        
        compression_ratio = rgba_size / p_size if p_size > 0 else 0
        size_reduction = (1 - p_size / rgba_size) * 100 if rgba_size > 0 else 0
        
        return {
            'rgba_size_bytes': rgba_size,
            'p_size_bytes': p_size,
            'compression_ratio': compression_ratio,
            'size_reduction_percent': size_reduction,
            'palette_colors': len(set(p_image.getdata()))
        }


class SpritePostProcessor:
    """
    Complete post-processing pipeline for sprite generation outputs.
    
    Handles conversion from model ARGB outputs to game-ready sprite formats
    with optimization and quality analysis.
    """
    
    def __init__(self, converter_config: Optional[Dict] = None):
        """
        Initialize sprite post-processor.
        
        Args:
            converter_config: Configuration for ARGBToPaletteConverter
        """
        config = converter_config or {
            'max_colors': 64,
            'preserve_transparency': True,
            'optimize_palette': True
        }
        self.converter = ARGBToPaletteConverter(**config)
    
    def process_single_sprite(self, rgba_sprite: Union[Image.Image, np.ndarray], 
                            return_formats: List[str] = ['rgba', 'p']) -> Dict[str, Any]:
        """
        Process single sprite output from model.
        
        Args:
            rgba_sprite: ARGB output from model
            return_formats: List of formats to return ('rgba', 'p')
            
        Returns:
            Dictionary with processed sprites and analysis
        """
        results = {}
        
        if 'rgba' in return_formats:
            if isinstance(rgba_sprite, np.ndarray):
                rgba_sprite = Image.fromarray(rgba_sprite, 'RGBA')
            results['rgba'] = rgba_sprite
        
        if 'p' in return_formats:
            p_sprite, palette = self.converter.convert_single_image(rgba_sprite)
            results['p'] = p_sprite
            results['palette'] = palette
            results['compression_analysis'] = self.converter.analyze_compression(rgba_sprite, p_sprite)
        
        return results
    
    def process_sprite_batch(self, rgba_sprites: List[Union[Image.Image, np.ndarray]], 
                           return_formats: List[str] = ['rgba', 'p']) -> Dict[str, Any]:
        """
        Process batch of sprite outputs with consistent palette.
        
        Args:
            rgba_sprites: List of ARGB outputs from model
            return_formats: List of formats to return
            
        Returns:
            Dictionary with processed sprite batches and analysis
        """
        results: Dict[str, Any] = {'batch_size': len(rgba_sprites)}
        
        if 'rgba' in return_formats:
            results['rgba_sprites'] = rgba_sprites
        
        if 'p' in return_formats:
            p_sprites, global_palette = self.converter.convert_batch(rgba_sprites, use_global_palette=True)
            results['p_sprites'] = p_sprites
            results['global_palette'] = global_palette
            
            # Analyze compression for first sprite
            if len(rgba_sprites) > 0:
                analysis = self.converter.analyze_compression(rgba_sprites[0], p_sprites[0])
                results['sample_compression_analysis'] = analysis
        
        return results
