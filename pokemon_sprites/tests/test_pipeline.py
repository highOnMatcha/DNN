#!/usr/bin/env python3
"""
Test script to validate the refactored Pokemon sprite generation pipeline.

Performs basic functionality tests without full training to ensure the
refactored system works correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import torch
from PIL import Image
import tempfile
import shutil

def test_config_loading():
    """Test configuration loading and validation."""
    print("Testing configuration loading...")
    
    try:
        from config.settings import (
            get_model_config, 
            get_training_config, 
            list_available_models,
            get_available_training_configs
        )
        
        # Test available configs
        models = list_available_models()
        configs = get_available_training_configs()
        
        print(f"Available models: {models}")
        print(f"Available training configs: {configs}")
        
        # Test specific config loading
        model_config = get_model_config("pix2pix-small")
        training_config = get_training_config("test")
        
        assert model_config is not None, "Failed to load model config"
        assert training_config is not None, "Failed to load training config"
        assert training_config.epochs == 5, "Test config epochs should be 5"
        
        print("✓ Configuration loading works")
        return True
        
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        return False


def test_augmentation_pipeline():
    """Test data augmentation pipeline."""
    print("Testing augmentation pipeline...")
    
    try:
        from data.augmentation import get_augmentation_config, AUGMENTATION_PRESETS
        
        # Create dummy images
        dummy_input = Image.new('RGB', (64, 64), color='red')
        dummy_target = Image.new('RGB', (64, 64), color='blue')
        
        # Test different augmentation levels
        for level in ['light', 'standard', 'production', 'none']:
            aug_pipeline = get_augmentation_config(level, 64)
            
            # Apply augmentation
            aug_input, aug_target = aug_pipeline(dummy_input, dummy_target)
            
            assert isinstance(aug_input, Image.Image), f"Input should be PIL Image for level {level}"
            assert isinstance(aug_target, Image.Image), f"Target should be PIL Image for level {level}"
            assert aug_input.size == (64, 64), f"Input size should be (64, 64) for level {level}"
            assert aug_target.size == (64, 64), f"Target size should be (64, 64) for level {level}"
        
        # Test presets mapping
        assert "test" in AUGMENTATION_PRESETS, "test preset should exist"
        assert "development" in AUGMENTATION_PRESETS, "development preset should exist"
        assert "production" in AUGMENTATION_PRESETS, "production preset should exist"
        
        print("✓ Augmentation pipeline works")
        return True
        
    except Exception as e:
        print(f"✗ Augmentation pipeline failed: {e}")
        return False


def test_dataset_creation():
    """Test Pokemon dataset creation.""" 
    print("Testing dataset creation...")
    
    try:
        from train import PokemonDataset
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            train_input_dir = temp_path / "train" / "input"
            train_target_dir = temp_path / "train" / "target"
            val_input_dir = temp_path / "val" / "input"
            val_target_dir = temp_path / "val" / "target"
            
            for dir_path in [train_input_dir, train_target_dir, val_input_dir, val_target_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Create dummy images
            for i in range(3):
                dummy_img = Image.new('RGB', (64, 64), color='red')
                dummy_img.save(train_input_dir / f"pokemon_{i}.png")
                dummy_img.save(train_target_dir / f"pokemon_{i}.png")
                dummy_img.save(val_input_dir / f"pokemon_{i}.png")
                dummy_img.save(val_target_dir / f"pokemon_{i}.png")
            
            # Test dataset creation
            train_dataset = PokemonDataset(
                str(temp_path), 
                split="train", 
                image_size=64, 
                augmentation_level="light"
            )
            
            val_dataset = PokemonDataset(
                str(temp_path),
                split="val",
                image_size=64,
                augmentation_level="none"
            )
            
            assert len(train_dataset) == 3, "Train dataset should have 3 samples"
            assert len(val_dataset) == 3, "Val dataset should have 3 samples"
            
            # Test sample retrieval
            input_tensor, target_tensor = train_dataset[0]
            assert isinstance(input_tensor, torch.Tensor), "Input should be tensor"
            assert isinstance(target_tensor, torch.Tensor), "Target should be tensor"
            assert input_tensor.shape == (3, 64, 64), "Input tensor shape should be (3, 64, 64)"
            assert target_tensor.shape == (3, 64, 64), "Target tensor shape should be (3, 64, 64)"
            
            print("✓ Dataset creation works")
            return True
            
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("Testing model creation...")
    
    try:
        from core.models import create_model
        from config.settings import get_model_config
        
        model_config = get_model_config("pix2pix-small")
        models = create_model(model_config)
        
        assert isinstance(models, dict), "Models should be returned as dict"
        assert "generator" in models, "Generator should be in models dict"
        assert "discriminator" in models, "Discriminator should be in models dict"
        
        # Test generator forward pass
        generator = models["generator"]
        dummy_input = torch.randn(1, 3, 64, 64)
        output = generator(dummy_input)
        
        assert output.shape == (1, 3, 64, 64), "Generator output shape should match input"
        
        print("✓ Model creation works")
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_trainer_initialization():
    """Test trainer initialization."""
    print("Testing trainer initialization...")
    
    try:
        from core.trainer import PokemonSpriteTrainer
        from config.settings import get_model_config, get_training_config
        
        model_config = get_model_config("pix2pix-small")
        training_config = get_training_config("test")
        
        trainer = PokemonSpriteTrainer(model_config, training_config, wandb_run=None)
        
        assert trainer.models is not None, "Trainer should have models"
        assert trainer.optimizers is not None, "Trainer should have optimizers"
        assert trainer.device is not None, "Trainer should have device"
        
        print("✓ Trainer initialization works")
        return True
        
    except Exception as e:
        print(f"✗ Trainer initialization failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Pokemon Sprite Generation Pipeline Tests")
    print("=" * 50)
    
    tests = [
        test_config_loading,
        test_augmentation_pipeline,
        test_dataset_creation,
        test_model_creation, 
        test_trainer_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Pipeline is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
