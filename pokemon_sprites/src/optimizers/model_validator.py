"""
Model Validator - Model Instantiation Testing

Validates model configurations by creating and testing the models,
not just checking configuration syntax.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
import sys
import traceback

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))
from core.models import Pix2PixGenerator, Pix2PixDiscriminator


def validate_all_configurations(config_path):
    """Validate all model configurations by actually creating and testing models"""
    
    print("MODEL VALIDATION")
    print("Creating and testing models")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Config load error: {e}")
        return {}
    
    model_configs = config.get('pix2pix_models', {})
    if not model_configs:
        print("No model configurations found")
        return {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Validating {len(model_configs)} model configurations...\n")
    
    validation_results = {}
    
    for model_name, model_config in model_configs.items():
        try:
            print(f"Validating {model_name}...")
            
            results = {
                'model_name': model_name,
                'config_valid': True,
                'generator_created': False,
                'discriminator_created': False,
                'forward_pass_works': False,
                'backward_pass_works': False,
                'errors': [],
                'parameter_count': {'generator': 0, 'discriminator': 0}
            }
            
            # Extract model parameters from config
            params = model_config.get('parameters', {})
            gen_params = params.get('generator', {})
            disc_params = params.get('discriminator', {})
            
            # Test generator creation
            try:
                generator = Pix2PixGenerator(
                    input_channels=gen_params.get('input_channels', 3),
                    output_channels=gen_params.get('output_channels', 4),
                    ngf=gen_params.get('ngf', 64),
                    n_blocks=gen_params.get('n_blocks', 9),
                    norm_layer=gen_params.get('norm_layer', 'instance'),
                    dropout=gen_params.get('dropout', 0.3)
                )
                results['generator_created'] = True
                gen_param_count = sum(p.numel() for p in generator.parameters())
                results['parameter_count']['generator'] = gen_param_count
                print(f"  ✓ Generator created: {gen_param_count:,} parameters")
            except Exception as e:
                results['errors'].append(f"Generator creation failed: {str(e)}")
                print(f"  ✗ Generator creation failed: {e}")
                validation_results[model_name] = results
                continue
            
            # Test discriminator creation
            try:
                discriminator = Pix2PixDiscriminator(
                    input_channels=disc_params.get('input_channels', 7),
                    ndf=disc_params.get('ndf', 64),
                    n_layers=disc_params.get('n_layers', 3),
                    norm_layer=disc_params.get('norm_layer', 'instance'),
                    use_spectral_norm=disc_params.get('use_spectral_norm', False)
                )
                results['discriminator_created'] = True
                disc_param_count = sum(p.numel() for p in discriminator.parameters())
                results['parameter_count']['discriminator'] = disc_param_count
                print(f"  ✓ Discriminator created: {disc_param_count:,} parameters")
            except Exception as e:
                results['errors'].append(f"Discriminator creation failed: {str(e)}")
                print(f"  ✗ Discriminator creation failed: {e}")
                validation_results[model_name] = results
                continue
            
            # Move to device
            try:
                generator = generator.to(device)
                discriminator = discriminator.to(device)
                print(f"  ✓ Models moved to {device}")
            except Exception as e:
                results['errors'].append(f"Device transfer failed: {str(e)}")
                print(f"  ✗ Device transfer failed: {e}")
                validation_results[model_name] = results
                continue
            
            # Test forward pass
            try:
                generator.eval()
                discriminator.eval()
                
                test_artwork = torch.randn(2, 3, 256, 256).to(device)
                test_sprite = torch.randn(2, 4, 256, 256).to(device)
                
                with torch.no_grad():
                    fake_sprite = generator(test_artwork)
                    disc_real = discriminator(test_artwork, test_sprite)
                    disc_fake = discriminator(test_artwork, fake_sprite)
                
                results['forward_pass_works'] = True
                print(f"  ✓ Forward pass successful")
                print(f"    Generator: {test_artwork.shape} → {fake_sprite.shape}")
                print(f"    Discriminator: {test_artwork.shape} + {test_sprite.shape} → {disc_real.shape}")
                
            except Exception as e:
                results['errors'].append(f"Forward pass failed: {str(e)}")
                print(f"  ✗ Forward pass failed: {e}")
                validation_results[model_name] = results
                continue
            
            # Test backward pass
            try:
                generator.train()
                discriminator.train()
                
                gen_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
                disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
                criterion_GAN = nn.MSELoss()
                criterion_L1 = nn.L1Loss()
                
                test_artwork = torch.randn(2, 3, 256, 256).to(device)
                test_sprite = torch.randn(2, 4, 256, 256).to(device)
                
                # Train discriminator
                disc_optimizer.zero_grad()
                disc_real = discriminator(test_artwork, test_sprite)
                real_label = torch.ones_like(disc_real).to(device)
                loss_disc_real = criterion_GAN(disc_real, real_label)
                
                fake_sprite = generator(test_artwork)
                disc_fake = discriminator(test_artwork, fake_sprite.detach())
                fake_label = torch.zeros_like(disc_fake).to(device)
                loss_disc_fake = criterion_GAN(disc_fake, fake_label)
                
                loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
                loss_disc.backward()
                disc_optimizer.step()
                
                # Train generator
                gen_optimizer.zero_grad()
                disc_fake = discriminator(test_artwork, fake_sprite)
                real_label_gen = torch.ones_like(disc_fake).to(device)
                loss_gen_GAN = criterion_GAN(disc_fake, real_label_gen)
                loss_gen_L1 = criterion_L1(fake_sprite, test_sprite) * 100
                loss_gen = loss_gen_GAN + loss_gen_L1
                loss_gen.backward()
                gen_optimizer.step()
                
                results['backward_pass_works'] = True
                print(f"  ✓ Backward pass successful")
                print(f"    Generator loss: {loss_gen.item():.4f}")
                print(f"    Discriminator loss: {loss_disc.item():.4f}")
                
            except Exception as e:
                results['errors'].append(f"Backward pass failed: {str(e)}")
                print(f"  ✗ Backward pass failed: {e}")
            
            validation_results[model_name] = results
            
            # Summary for this model
            if results['forward_pass_works'] and results['backward_pass_works']:
                print(f"  ✅ {model_name}: VALID")
            else:
                print(f"  ❌ {model_name}: INVALID")
            
            print()
            
        except Exception as e:
            print(f"Error validating {model_name}: {e}")
            validation_results[model_name] = {
                'model_name': model_name,
                'config_valid': False,
                'errors': [f"Validation crashed: {str(e)}"]
            }
    
    # Summary report
    print("VALIDATION SUMMARY")
    print("=" * 50)
    
    valid_models = [name for name, results in validation_results.items() 
                   if results.get('forward_pass_works', False) and results.get('backward_pass_works', False)]
    invalid_models = [name for name, results in validation_results.items() 
                     if not (results.get('forward_pass_works', False) and results.get('backward_pass_works', False))]
    
    print(f"Valid models: {len(valid_models)}")
    for model in valid_models:
        print(f"  ✓ {model}")
    
    if invalid_models:
        print(f"Invalid models: {len(invalid_models)}")
        for model in invalid_models:
            print(f"  ✗ {model}")
    
    total_params = sum(results.get('parameter_count', {}).get('generator', 0) + 
                      results.get('parameter_count', {}).get('discriminator', 0)
                      for results in validation_results.values() 
                      if results.get('forward_pass_works', False))
    
    print(f"\nTotal parameters across all valid models: {total_params:,}")
    
    return validation_results


def optimize_model_config(config_path):
    """Main function to validate all model configurations"""
    return validate_all_configurations(config_path)
