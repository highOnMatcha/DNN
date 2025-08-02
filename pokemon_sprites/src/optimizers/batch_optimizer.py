"""
Batch Size Optimizer - Memory and Performance Based

Finds optimal batch sizes by testing memory usage and training speed
with actual models, not just theoretical calculations.
"""

import torch
import torch.nn as nn
import time
import json
from pathlib import Path
import sys
import gc

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))
from core.models import Pix2PixGenerator, Pix2PixDiscriminator


class BatchSizeOptimizer:
    """Find optimal batch size through memory and speed testing"""
    
    def __init__(self, generator, discriminator, device='cuda'):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
    def find_optimal_batch_size(self, initial_max_batch_size=64, input_size=(3, 256, 256), output_size=(4, 256, 256)):
        """Find optimal batch size by testing memory usage and training speed with dynamic extension"""
        
        print("Testing memory usage and training speed...")
        
        # Start with initial batch sizes
        test_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        test_batch_sizes = [bs for bs in test_batch_sizes if bs <= initial_max_batch_size]
        
        results = {
            'tested_batch_sizes': [],
            'memory_usage_mb': [],
            'training_time_ms': [],
            'successful': [],
            'optimal_batch_size': 4,
            'max_stable_batch_size': 4,
            'recommended_batch_size': 4
        }
        
        print("Batch Size | Memory (MB) | Time (ms) | Status")
        print("-" * 50)
        
        last_successful_batch = 0
        
        for batch_size in test_batch_sizes:
            try:
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Record initial memory
                if torch.cuda.is_available():
                    initial_memory = torch.cuda.memory_allocated(self.device) / 1024**2
                else:
                    initial_memory = 0
                
                # Create test data
                artwork = torch.randn(batch_size, *input_size).to(self.device)
                sprite = torch.randn(batch_size, *output_size).to(self.device)
                
                # Setup for training test
                gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
                disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
                criterion_GAN = nn.MSELoss()
                criterion_L1 = nn.L1Loss()
                
                # Time the training step
                start_time = time.time()
                
                # Train discriminator
                disc_optimizer.zero_grad()
                disc_real = self.discriminator(artwork, sprite)
                real_label = torch.ones_like(disc_real, device=self.device)
                loss_disc_real = criterion_GAN(disc_real, real_label)
                
                fake_sprite = self.generator(artwork)
                disc_fake = self.discriminator(artwork, fake_sprite.detach())
                fake_label = torch.zeros_like(disc_fake, device=self.device)
                loss_disc_fake = criterion_GAN(disc_fake, fake_label)
                
                loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
                loss_disc.backward()
                disc_optimizer.step()
                
                # Train generator
                gen_optimizer.zero_grad()
                disc_fake = self.discriminator(artwork, fake_sprite)
                real_label_gen = torch.ones_like(disc_fake, device=self.device)
                loss_gen_GAN = criterion_GAN(disc_fake, real_label_gen)
                loss_gen_L1 = criterion_L1(fake_sprite, sprite) * 100
                loss_gen = loss_gen_GAN + loss_gen_L1
                loss_gen.backward()
                gen_optimizer.step()
                
                end_time = time.time()
                training_time = (end_time - start_time) * 1000  # ms
                
                # Record peak memory
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**2
                    memory_used = peak_memory - initial_memory
                    torch.cuda.reset_peak_memory_stats(self.device)
                else:
                    memory_used = 0
                
                # Record results
                results['tested_batch_sizes'].append(batch_size)
                results['memory_usage_mb'].append(memory_used)
                results['training_time_ms'].append(training_time)
                results['successful'].append(True)
                
                print(f"{batch_size:9d} | {memory_used:10.1f} | {training_time:8.1f} | ✓ Success")
                
                # Update max stable and track last successful
                results['max_stable_batch_size'] = batch_size
                last_successful_batch = batch_size
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{batch_size:9d} | {'N/A':>10} | {'N/A':>8} | ✗ OOM")
                    break
                else:
                    print(f"{batch_size:9d} | {'N/A':>10} | {'N/A':>8} | ✗ Error")
            except Exception as e:
                print(f"{batch_size:9d} | {'N/A':>10} | {'N/A':>8} | ✗ Error")
        
        # Dynamic extension: test larger batch sizes if max tested was successful
        print(f"\nDynamic extension check: last_successful={last_successful_batch}, max_tested={max(test_batch_sizes) if test_batch_sizes else 0}")
        if last_successful_batch > 0 and last_successful_batch == max(test_batch_sizes):
            print(f"\nExtending tests beyond {last_successful_batch} since max tested was successful...")
            
            # Test progressively larger batch sizes
            extended_sizes = []
            next_batch = last_successful_batch * 2
            
            while next_batch <= 256:  # Reduced cap to 256 to avoid excessive testing
                extended_sizes.append(next_batch)
                next_batch *= 2
            
            for batch_size in extended_sizes:
                try:
                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Record initial memory
                    if torch.cuda.is_available():
                        initial_memory = torch.cuda.memory_allocated(self.device) / 1024**2
                    else:
                        initial_memory = 0
                    
                    # Create test data
                    artwork = torch.randn(batch_size, *input_size).to(self.device)
                    sprite = torch.randn(batch_size, *output_size).to(self.device)
                    
                    # Setup for training test
                    gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
                    disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
                    criterion_GAN = nn.MSELoss()
                    criterion_L1 = nn.L1Loss()
                    
                    # Time the training step
                    start_time = time.time()
                    
                    # Quick training test (abbreviated version)
                    fake_sprite = self.generator(artwork)
                    disc_fake = self.discriminator(artwork, fake_sprite)
                    real_label = torch.ones_like(disc_fake, device=self.device)
                    loss = nn.MSELoss()(disc_fake, real_label)
                    loss.backward()
                    
                    end_time = time.time()
                    training_time = (end_time - start_time) * 1000  # ms
                    
                    # Record peak memory
                    if torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**2
                        memory_used = peak_memory - initial_memory
                        torch.cuda.reset_peak_memory_stats(self.device)
                    else:
                        memory_used = 0
                    
                    # Record results
                    results['tested_batch_sizes'].append(batch_size)
                    results['memory_usage_mb'].append(memory_used)
                    results['training_time_ms'].append(training_time)
                    results['successful'].append(True)
                    
                    print(f"{batch_size:9d} | {memory_used:10.1f} | {training_time:8.1f} | ✓ Success (extended)")
                    
                    # Update max stable
                    results['max_stable_batch_size'] = batch_size
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"{batch_size:9d} | {'N/A':>10} | {'N/A':>8} | ✗ OOM (limit found)")
                        break
                    else:
                        print(f"{batch_size:9d} | {'N/A':>10} | {'N/A':>8} | ✗ Error")
                        break
                except Exception as e:
                    print(f"{batch_size:9d} | {'N/A':>10} | {'N/A':>8} | ✗ Error")
                    break
        
        # Find optimal batch size
        successful_indices = [i for i, success in enumerate(results['successful']) if success]
        
        if successful_indices:
            successful_batch_sizes = [results['tested_batch_sizes'][i] for i in successful_indices]
            successful_times = [results['training_time_ms'][i] for i in successful_indices]
            
            # Most efficient (best time per sample)
            time_per_sample = [time/batch_size for time, batch_size in zip(successful_times, successful_batch_sizes)]
            optimal_idx = successful_indices[time_per_sample.index(min(time_per_sample))]
            results['optimal_batch_size'] = results['tested_batch_sizes'][optimal_idx]
            
            # Conservative recommendation
            max_stable = results['max_stable_batch_size']
            results['recommended_batch_size'] = max(4, int(max_stable * 0.75))
            
            print(f"\nResults:")
            print(f"Max stable: {results['max_stable_batch_size']}")
            print(f"Most efficient: {results['optimal_batch_size']}")
            print(f"Recommended: {results['recommended_batch_size']}")
        
        return results


def optimize_batch_sizes(config_path, test_input_size=(3, 256, 256), test_output_size=(4, 256, 256)):
    """Optimize batch sizes using memory testing"""
    
    print("BATCH SIZE OPTIMIZATION")
    print("Testing memory usage with models")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Config load error: {e}")
        return {}
    
    model_configs = config.get('pix2pix_models', {})
    if not model_configs:
        return {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    batch_recommendations = {}
    
    for model_name, model_config in model_configs.items():
        try:
            print(f"\n--- Testing batch sizes for: {model_name} ---")
            
            # Extract model parameters from config
            params = model_config.get('parameters', {})
            gen_params = params.get('generator', {})
            disc_params = params.get('discriminator', {})
            
            # Create models with config parameters
            generator = Pix2PixGenerator(
                input_channels=gen_params.get('input_channels', 3),
                output_channels=gen_params.get('output_channels', 4),
                ngf=gen_params.get('ngf', 64),
                n_blocks=gen_params.get('n_blocks', 9),
                norm_layer=gen_params.get('norm_layer', 'instance'),
                dropout=gen_params.get('dropout', 0.3)
            )
            discriminator = Pix2PixDiscriminator(
                input_channels=disc_params.get('input_channels', 7),
                ndf=disc_params.get('ndf', 64),
                n_layers=disc_params.get('n_layers', 3),
                norm_layer=disc_params.get('norm_layer', 'instance'),
                use_spectral_norm=disc_params.get('use_spectral_norm', False)
            )
            
            # Test batch sizes
            optimizer = BatchSizeOptimizer(generator, discriminator, str(device))
            results = optimizer.find_optimal_batch_size(
                initial_max_batch_size=64,
                input_size=test_input_size,
                output_size=test_output_size
            )
            
            batch_recommendations[model_name] = {
                'recommended_batch_size': results['recommended_batch_size'],
                'optimal_batch_size': results['optimal_batch_size'],
                'max_stable_batch_size': results['max_stable_batch_size'],
                'testing_method': 'memory_testing'
            }
            
            print(f"✓ Completed for {model_name}")
            
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
            batch_recommendations[model_name] = {
                'recommended_batch_size': 4,
                'optimal_batch_size': 4,
                'max_stable_batch_size': 4,
                'testing_method': 'fallback'
            }
    
    return batch_recommendations
