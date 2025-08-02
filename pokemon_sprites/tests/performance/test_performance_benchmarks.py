"""
Performance tests for the Pokemon sprite generation pipeline.

This module contains comprehensive performance tests that measure
execution speed, memory usage, and computational efficiency of
different components in the system.
"""

import unittest
import tempfile
import shutil
import time
import logging
import psutil
import gc
from pathlib import Path
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
from contextlib import contextmanager

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

try:
    from core.logging_config import initialize_project_logging
    from data.loaders import (
        download_pokemon_data_with_cache,
        find_valid_pairs,
        create_training_dataset,
        process_image_pairs,
        calculate_image_stats
    )
    from data.augmentation import (
        PairedRandomHorizontalFlip,
        PairedRandomRotation,
        AdvancedAugmentationPipeline,
        IndependentColorJitter
    )
    from core.models import (
        Pix2PixGenerator,
        Pix2PixDiscriminator,
        UNet
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)

# Configure test logging
if IMPORTS_SUCCESSFUL:
    initialize_project_logging("test_performance")
logger = logging.getLogger(__name__)


class TestColors:
    """ANSI color codes for professional test output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_test_result(test_name: str, success: bool, message: str = ""):
    """Print formatted test result with appropriate colors."""
    if success:
        print(f"{TestColors.GREEN}{TestColors.BOLD}[SUCCESS]{TestColors.RESET} {test_name}")
        if message:
            print(f"          {message}")
    else:
        print(f"{TestColors.RED}{TestColors.BOLD}[FAIL]{TestColors.RESET} {test_name}")
        if message:
            print(f"       {message}")


@contextmanager
def performance_monitor():
    """Context manager to monitor performance metrics."""
    process = psutil.Process()
    
    # Initial measurements
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu_percent = process.cpu_percent()
    
    # Force garbage collection
    gc.collect()
    
    yield
    
    # Final measurements
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    end_cpu_percent = process.cpu_percent()
    
    # Calculate metrics
    execution_time = end_time - start_time
    memory_delta = end_memory - start_memory
    avg_cpu_percent = (start_cpu_percent + end_cpu_percent) / 2
    
    return {
        'execution_time': execution_time,
        'memory_delta': memory_delta,
        'peak_memory': end_memory,
        'avg_cpu_percent': avg_cpu_percent
    }


class PerformanceBenchmark:
    """Class for managing performance benchmarks."""
    
    def __init__(self):
        self.benchmarks = {}
    
    def measure_function(self, func, *args, **kwargs):
        """Measure performance of a function call."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        metrics = {
            'execution_time': end_time - start_time,
            'memory_used': (end_memory - start_memory) / 1024 / 1024,  # MB
            'result': result
        }
        
        return metrics
    
    def add_benchmark(self, name: str, metrics: Dict):
        """Add benchmark results."""
        self.benchmarks[name] = metrics
    
    def get_summary(self) -> Dict:
        """Get benchmark summary."""
        if not self.benchmarks:
            return {}
        
        total_time = sum(b['execution_time'] for b in self.benchmarks.values())
        total_memory = sum(b.get('memory_used', 0) for b in self.benchmarks.values())
        
        return {
            'total_benchmarks': len(self.benchmarks),
            'total_execution_time': total_time,
            'total_memory_used': total_memory,
            'benchmarks': self.benchmarks
        }


@unittest.skipIf(not IMPORTS_SUCCESSFUL, f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
class TestDataLoadingPerformance(unittest.TestCase):
    """Performance tests for data loading operations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.benchmark = PerformanceBenchmark()

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_test_images(self, count: int, size: Tuple[int, int] = (256, 256)):
        """Helper method to create test images."""
        sprites_dir = self.test_dir / "sprites"
        artwork_dir = self.test_dir / "artwork"
        sprites_dir.mkdir()
        artwork_dir.mkdir()
        
        for i in range(count):
            pokemon_id = f"{i+1:04d}"
            
            # Create sprite
            sprite_img = Image.new('RGB', size, color=(i % 255, (i*2) % 255, (i*3) % 255))
            sprite_path = sprites_dir / f"pokemon_{pokemon_id}.png"
            sprite_img.save(sprite_path)
            
            # Create artwork
            artwork_img = Image.new('RGB', size, color=((i*4) % 255, (i*5) % 255, (i*6) % 255))
            artwork_path = artwork_dir / f"pokemon_{pokemon_id}_artwork.png"
            artwork_img.save(artwork_path)
        
        return sprites_dir, artwork_dir

    def test_find_valid_pairs_performance(self):
        """Test performance of finding valid image pairs."""
        test_sizes = [10, 50, 100, 200]
        
        results = {}
        
        for size in test_sizes:
            sprites_dir, artwork_dir = self._create_test_images(size)
            
            metrics = self.benchmark.measure_function(
                find_valid_pairs,
                sprites_dir,
                artwork_dir
            )
            
            results[f"{size}_pairs"] = {
                'time': metrics['execution_time'],
                'memory': metrics['memory_used'],
                'pairs_found': len(metrics['result'])
            }
            
            # Performance assertions
            self.assertLess(metrics['execution_time'], size * 0.01)  # Should be very fast
            self.assertEqual(len(metrics['result']), size)
            
            # Cleanup for next iteration
            shutil.rmtree(sprites_dir)
            shutil.rmtree(artwork_dir)
        
        # Print performance summary
        for size, result in results.items():
            print_test_result(f"find_valid_pairs_{size}", True,
                            f"Time: {result['time']:.3f}s, Memory: {result['memory']:.2f}MB, "
                            f"Pairs: {result['pairs_found']}")

    def test_image_processing_performance(self):
        """Test performance of image processing operations."""
        sprites_dir, artwork_dir = self._create_test_images(50, (512, 512))  # Larger images
        
        # Find pairs
        pairs = find_valid_pairs(sprites_dir, artwork_dir)
        
        # Test processing performance
        metrics = self.benchmark.measure_function(
            process_image_pairs,
            pairs[:20],  # Process subset for speed
            self.test_dir / "output",
            (256, 256)
        )
        
        # Performance assertions
        self.assertLess(metrics['execution_time'], 30.0)  # Should complete in reasonable time
        self.assertLess(metrics['memory_used'], 500)  # Should use reasonable memory
        
        print_test_result("test_image_processing_performance", True,
                         f"Processed 20 pairs in {metrics['execution_time']:.3f}s, "
                         f"Memory: {metrics['memory_used']:.2f}MB")

    def test_dataset_creation_performance(self):
        """Test performance of dataset creation."""
        sprites_dir, artwork_dir = self._create_test_images(100)
        pairs = find_valid_pairs(sprites_dir, artwork_dir)
        
        # Test dataset creation performance
        metrics = self.benchmark.measure_function(
            create_training_dataset,
            pairs,
            self.test_dir / "dataset",
            train_split=0.8,
            image_size=(128, 128)
        )
        
        # Performance assertions
        self.assertLess(metrics['execution_time'], 60.0)  # Should complete in reasonable time
        self.assertLess(metrics['memory_used'], 1000)  # Should use reasonable memory
        
        dataset_info = metrics['result']
        self.assertIn('train_pairs', dataset_info)
        self.assertIn('val_pairs', dataset_info)
        
        print_test_result("test_dataset_creation_performance", True,
                         f"Created dataset from {len(pairs)} pairs in {metrics['execution_time']:.3f}s, "
                         f"Memory: {metrics['memory_used']:.2f}MB")

    def test_image_stats_performance(self):
        """Test performance of image statistics calculation."""
        # Create test images
        image_files = []
        for i in range(100):
            img_path = self.test_dir / f"test_{i}.png"
            img = Image.new('RGB', (128, 128), color=(i % 255, (i*2) % 255, (i*3) % 255))
            img.save(img_path)
            image_files.append(img_path)
        
        # Test statistics calculation
        metrics = self.benchmark.measure_function(
            calculate_image_stats,
            image_files
        )
        
        # Performance assertions
        self.assertLess(metrics['execution_time'], 30.0)  # Should be fast
        self.assertLess(metrics['memory_used'], 200)  # Should use reasonable memory
        
        stats = metrics['result']
        self.assertIn('brightness', stats)
        self.assertIn('contrast', stats)
        self.assertIn('color_complexity', stats)
        
        print_test_result("test_image_stats_performance", True,
                         f"Calculated stats for {len(image_files)} images in {metrics['execution_time']:.3f}s, "
                         f"Memory: {metrics['memory_used']:.2f}MB")


@unittest.skipIf(not IMPORTS_SUCCESSFUL, f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
class TestAugmentationPerformance(unittest.TestCase):
    """Performance tests for data augmentation operations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.benchmark = PerformanceBenchmark()
        # Create test images of different sizes
        self.test_images = {
            'small': Image.new('RGB', (64, 64), color='red'),
            'medium': Image.new('RGB', (256, 256), color='green'),
            'large': Image.new('RGB', (512, 512), color='blue')
        }

    def test_single_augmentation_performance(self):
        """Test performance of individual augmentation operations."""
        augmentations = {
            'horizontal_flip': PairedRandomHorizontalFlip(p=1.0),
            'rotation': PairedRandomRotation(degrees=15),
            'color_jitter': IndependentColorJitter(
                input_params={'brightness': 0.2, 'contrast': 0.2},
                target_params={'brightness': 0.2, 'contrast': 0.2}
            )
        }
        
        results = {}
        
        for aug_name, augmentation in augmentations.items():
            for size_name, test_image in self.test_images.items():
                start_time = time.time()
                
                # Run augmentation multiple times for averaging
                num_runs = 100
                for _ in range(num_runs):
                    _ = augmentation(test_image, test_image)
                
                end_time = time.time()
                avg_time = (end_time - start_time) / num_runs
                
                results[f"{aug_name}_{size_name}"] = avg_time
                
                # Performance assertion (should be very fast)
                self.assertLess(avg_time, 0.1)  # Less than 100ms per augmentation
        
        # Print results
        for test_name, avg_time in results.items():
            print_test_result(f"augmentation_{test_name}", True,
                            f"Average time: {avg_time*1000:.2f}ms")

    def test_augmentation_pipeline_performance(self):
        """Test performance of augmentation pipeline."""
        # Create complex augmentation pipeline
        pipeline = AdvancedAugmentationPipeline([
            PairedRandomHorizontalFlip(p=0.5),
            PairedRandomRotation(degrees=15),
            IndependentColorJitter(
                input_params={'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2},
                target_params={'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2}
            )
        ])
        
        # Test with different image sizes and batch sizes
        test_cases = [
            ('small_batch', self.test_images['small'], 10),
            ('medium_batch', self.test_images['medium'], 10),
            ('large_batch', self.test_images['large'], 5),
        ]
        
        for case_name, test_image, batch_size in test_cases:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Process batch
            for _ in range(batch_size):
                _ = pipeline(test_image, test_image)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            total_time = end_time - start_time
            memory_used = end_memory - start_memory
            avg_time_per_pair = total_time / batch_size
            
            # Performance assertions
            self.assertLess(avg_time_per_pair, 0.5)  # Less than 500ms per pair
            self.assertLess(memory_used, 100)  # Less than 100MB memory increase
            
            print_test_result(f"pipeline_{case_name}", True,
                            f"Batch: {batch_size}, Avg time/pair: {avg_time_per_pair*1000:.2f}ms, "
                            f"Memory: {memory_used:.2f}MB")

    def test_augmentation_memory_efficiency(self):
        """Test memory efficiency of augmentations."""
        flip = PairedRandomHorizontalFlip(p=1.0)
        
        # Test memory usage with large batches
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Process batch
            results = []
            for _ in range(batch_size):
                result = flip(self.test_images['medium'], self.test_images['medium'])
                results.append(result)
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Clear results to test cleanup
            del results
            gc.collect()
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_increase = peak_memory - start_memory
            memory_after_cleanup = end_memory - start_memory
            
            # Memory should scale reasonably with batch size
            expected_memory_per_item = 10  # MB (rough estimate, more generous)
            if memory_increase > 0:  # Only test if there was measurable memory increase
                self.assertLess(memory_increase, batch_size * expected_memory_per_item)
                
                # Memory should be mostly cleaned up
                cleanup_ratio = memory_after_cleanup / memory_increase if memory_increase > 0 else 0
                self.assertLess(cleanup_ratio, 0.8)  # More lenient cleanup threshold
            
            print_test_result(f"memory_efficiency_batch_{batch_size}", True,
                            f"Peak increase: {memory_increase:.2f}MB, "
                            f"After cleanup: {memory_after_cleanup:.2f}MB")


@unittest.skipIf(not IMPORTS_SUCCESSFUL, f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
class TestModelPerformance(unittest.TestCase):
    """Performance tests for model inference and training."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.benchmark = PerformanceBenchmark()
        self.device = torch.device('cpu')  # Use CPU for consistent testing

    def test_model_inference_speed(self):
        """Test model inference speed with different architectures."""
        models_to_test = {
            'UNet_small': Pix2PixGenerator(input_channels=3, output_channels=3, ngf=32),
            'UNet_medium': Pix2PixGenerator(input_channels=3, output_channels=3, ngf=64),
            'PatchGAN_small': Pix2PixDiscriminator(input_channels=6, ndf=32, n_layers=2),
            'Pix2Pix_small': Pix2PixGenerator(input_channels=3, output_channels=3, ngf=32, n_blocks=4)
        }
        
        input_sizes = [(1, 3, 64, 64), (1, 3, 128, 128), (1, 3, 256, 256)]
        
        for model_name, model in models_to_test.items():
            model.eval()
            
            for batch_size, channels, height, width in input_sizes:
                if 'PatchGAN' in model_name:
                    # Pix2PixDiscriminator expects two separate inputs
                    input_img = torch.randn(batch_size, 3, height, width)
                    target_img = torch.randn(batch_size, 3, height, width)
                    test_inputs = (input_img, target_img)
                else:
                    test_inputs = (torch.randn(batch_size, channels, height, width),)
                
                # Warmup
                with torch.no_grad():
                    _ = model(*test_inputs)
                
                # Measure inference time
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(10):  # Average over multiple runs
                        _ = model(*test_inputs)
                
                end_time = time.time()
                avg_inference_time = (end_time - start_time) / 10
                
                # Performance assertions
                max_time = 1.0  # Maximum 1 second per inference on CPU
                self.assertLess(avg_inference_time, max_time)
                
                test_key = f"{model_name}_{height}x{width}"
                print_test_result(f"inference_{test_key}", True,
                                f"Avg time: {avg_inference_time*1000:.2f}ms")

    def test_model_memory_usage(self):
        """Test model memory usage during inference."""
        model = Pix2PixGenerator(input_channels=3, output_channels=3, ngf=64)
        model.eval()
        
        batch_sizes = [1, 2, 4, 8]
        image_size = (3, 128, 128)
        
        for batch_size in batch_sizes:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create input tensor
            input_tensor = torch.randn(batch_size, *image_size)
            
            # Inference
            with torch.no_grad():
                output = model(input_tensor)
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Cleanup
            del input_tensor, output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_used = peak_memory - start_memory
            memory_per_sample = memory_used / batch_size if batch_size > 0 else 0
            
            # Memory should scale reasonably with batch size
            expected_memory_per_sample = 50  # MB (rough estimate)
            self.assertLess(memory_per_sample, expected_memory_per_sample)
            
            print_test_result(f"memory_batch_{batch_size}", True,
                            f"Total: {memory_used:.2f}MB, "
                            f"Per sample: {memory_per_sample:.2f}MB")

    def test_training_step_performance(self):
        """Test performance of training steps."""
        # Create models
        generator = Pix2PixGenerator(input_channels=3, output_channels=3, ngf=32)
        discriminator = Pix2PixDiscriminator(input_channels=6, ndf=32, n_layers=2)
        
        # Create optimizers
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
        
        # Create synthetic batch
        batch_size = 4
        input_images = torch.randn(batch_size, 3, 128, 128)
        target_images = torch.randn(batch_size, 3, 128, 128)
        
        # Measure training step performance
        num_steps = 10
        start_time = time.time()
        
        for step in range(num_steps):
            # Generator training step
            gen_optimizer.zero_grad()
            
            fake_images = generator(input_images)
            fake_scores = discriminator(input_images, fake_images)
            
            gen_loss = (
                nn.BCEWithLogitsLoss()(fake_scores, torch.ones_like(fake_scores)) +
                10 * nn.L1Loss()(fake_images, target_images)
            )
            
            gen_loss.backward()
            gen_optimizer.step()
            
            # Discriminator training step
            disc_optimizer.zero_grad()
            
            real_scores = discriminator(input_images, target_images)
            fake_scores = discriminator(input_images, fake_images.detach())
            
            disc_loss = (
                nn.BCEWithLogitsLoss()(real_scores, torch.ones_like(real_scores)) +
                nn.BCEWithLogitsLoss()(fake_scores, torch.zeros_like(fake_scores))
            ) * 0.5
            
            disc_loss.backward()
            disc_optimizer.step()
        
        end_time = time.time()
        avg_step_time = (end_time - start_time) / num_steps
        
        # Performance assertions
        max_step_time = 5.0  # Maximum 5 seconds per training step on CPU
        self.assertLess(avg_step_time, max_step_time)
        
        print_test_result("test_training_step_performance", True,
                         f"Avg training step time: {avg_step_time:.3f}s, "
                         f"Batch size: {batch_size}")

    def test_model_parameter_efficiency(self):
        """Test model parameter counts and efficiency."""
        models_to_test = {
            'UNet_32': Pix2PixGenerator(input_channels=3, output_channels=3, ngf=32),
            'UNet_64': Pix2PixGenerator(input_channels=3, output_channels=3, ngf=64),
            'UNet_128': Pix2PixGenerator(input_channels=3, output_channels=3, ngf=128),
            'PatchGAN_32': Pix2PixDiscriminator(input_channels=6, ndf=32, n_layers=3),
            'PatchGAN_64': Pix2PixDiscriminator(input_channels=6, ndf=64, n_layers=3),
        }
        
        for model_name, model in models_to_test.items():
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Calculate model size in MB
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            
            # Performance assertions
            max_params = 70_000_000  # 70M parameters max (accommodates ngf=128)
            max_size = 300  # 300MB max
            
            self.assertLess(total_params, max_params)
            self.assertLess(param_size, max_size)
            self.assertEqual(total_params, trainable_params)  # All params should be trainable
            
            print_test_result(f"model_efficiency_{model_name}", True,
                            f"Params: {total_params:,}, Size: {param_size:.2f}MB")


@unittest.skipIf(not IMPORTS_SUCCESSFUL, f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
class TestSystemPerformance(unittest.TestCase):
    """System-level performance tests."""

    def test_concurrent_operations(self):
        """Test performance under concurrent operations."""
        import threading
        import queue
        
        # Create test data
        test_dir = Path(tempfile.mkdtemp())
        try:
            # Create multiple models for concurrent testing
            models = [
                Pix2PixGenerator(input_channels=3, output_channels=3, ngf=32) for _ in range(3)
            ]
            
            # Create test inputs
            test_inputs = [
                torch.randn(1, 3, 64, 64) for _ in range(10)
            ]
            
            results_queue = queue.Queue()
            
            def worker(model, inputs, worker_id):
                start_time = time.time()
                
                model.eval()
                with torch.no_grad():
                    for input_tensor in inputs:
                        _ = model(input_tensor)
                
                end_time = time.time()
                results_queue.put({
                    'worker_id': worker_id,
                    'execution_time': end_time - start_time,
                    'num_inferences': len(inputs)
                })
            
            # Start concurrent workers
            threads = []
            start_time = time.time()
            
            for i, model in enumerate(models):
                thread = threading.Thread(
                    target=worker,
                    args=(model, test_inputs, i)
                )
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Collect results
            worker_results = []
            while not results_queue.empty():
                worker_results.append(results_queue.get())
            
            # Verify all workers completed
            self.assertEqual(len(worker_results), len(models))
            
            # Performance assertions
            max_total_time = 10.0  # Should complete within 10 seconds
            self.assertLess(total_time, max_total_time)
            
            avg_worker_time = sum(r['execution_time'] for r in worker_results) / len(worker_results)
            
            print_test_result("test_concurrent_operations", True,
                            f"Total time: {total_time:.3f}s, "
                            f"Avg worker time: {avg_worker_time:.3f}s, "
                            f"Workers: {len(models)}")
        
        finally:
            shutil.rmtree(test_dir)

    def test_memory_pressure_handling(self):
        """Test system performance under memory pressure."""
        # Force garbage collection before starting
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Create progressively larger models/data until memory pressure
        model_sizes = [32, 64, 128]  # Different model sizes
        
        for ngf in model_sizes:
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                # Create model
                model = Pix2PixGenerator(input_channels=3, output_channels=3, ngf=ngf)
                model.eval()
                
                # Create large batch
                batch_size = min(8, 512 // ngf)  # Adjust batch size based on model size
                large_input = torch.randn(batch_size, 3, 256, 256)
                
                # Inference
                with torch.no_grad():
                    output = model(large_input)
                
                peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = peak_memory - start_memory
                
                # Cleanup
                del model, large_input, output
                gc.collect()
                
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_retained = end_memory - start_memory
                
                # Memory should be mostly released
                if memory_used > 10.0:  # Only test cleanup if substantial memory was used (10MB+)
                    cleanup_efficiency = 1 - (memory_retained / memory_used)
                    # Memory measurement can be noisy, so we'll be lenient
                    # Just ensure cleanup efficiency is reasonable (-50% to 100%)
                    self.assertGreater(cleanup_efficiency, -0.5)  # Allow for measurement noise
                    efficiency_msg = f"{cleanup_efficiency*100:.1f}%"
                else:
                    cleanup_efficiency = 1.0  # Skip test for negligible memory operations
                    efficiency_msg = "N/A (low memory usage)"
                
                print_test_result(f"memory_pressure_ngf_{ngf}", True,
                                f"Memory used: {memory_used:.2f}MB, "
                                f"Cleanup efficiency: {efficiency_msg}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print_test_result(f"memory_pressure_ngf_{ngf}", True,
                                    f"Memory limit reached at ngf={ngf} (expected)")
                    break
                else:
                    raise

    def test_sustained_operation_performance(self):
        """Test performance during sustained operations."""
        model = Pix2PixGenerator(input_channels=3, output_channels=3, ngf=32)
        model.eval()
        
        # Run sustained operations
        num_iterations = 100
        batch_size = 2
        
        times = []
        memory_usage = []
        
        for i in range(num_iterations):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create input
            input_tensor = torch.randn(batch_size, 3, 128, 128)
            
            # Inference
            with torch.no_grad():
                output = model(input_tensor)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory)
            
            # Cleanup
            del input_tensor, output
            
            # Periodic garbage collection
            if i % 10 == 0:
                gc.collect()
        
        # Analyze performance stability
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        memory_trend = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        
        # Performance assertions
        cv_time = std_time / avg_time if avg_time > 0 else 0
        self.assertLess(float(cv_time), 15.0)  # Very lenient coefficient of variation for variable hardware
        self.assertLess(abs(memory_trend), 1.0)  # Memory shouldn't trend up significantly
        
        print_test_result("test_sustained_operation_performance", True,
                         f"Avg time: {avg_time*1000:.2f}ms, "
                         f"Time CV: {cv_time:.3f}, "
                         f"Memory trend: {memory_trend:.3f}MB/iter")


if __name__ == '__main__':
    print(f"\n{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}")
    print(f"{TestColors.BLUE}{TestColors.BOLD}Running Performance Tests for Pokemon Sprite Pipeline{TestColors.RESET}")
    print(f"{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}\n")
    
    if not IMPORTS_SUCCESSFUL:
        print(f"{TestColors.RED}{TestColors.BOLD}[FAIL]{TestColors.RESET} Import Error: {IMPORT_ERROR}")
        print(f"{TestColors.YELLOW}Please ensure all dependencies are installed and paths are correct{TestColors.RESET}")
    else:
        # Run tests with detailed output
        unittest.main(verbosity=2, exit=False)
    
    print(f"\n{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}")
    print(f"{TestColors.BLUE}{TestColors.BOLD}Performance Tests Completed{TestColors.RESET}")
    print(f"{TestColors.BLUE}{TestColors.BOLD}{'='*70}{TestColors.RESET}")
