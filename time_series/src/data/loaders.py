"""
Data loading and dataset management utilities.

This module provides utilities for loading stock data, creating datasets,
and managing data for training and evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from utils.logging import get_logger

logger = get_logger(__name__)


class StockDataset(Dataset):
    """PyTorch Dataset for stock price sequences."""
    
    def __init__(self, 
                 sequences: np.ndarray, 
                 targets: np.ndarray,
                 transform=None):
        """
        Initialize stock dataset.
        
        Args:
            sequences: Input sequences of shape (n_samples, seq_len, n_features)
            targets: Target values of shape (n_samples,)
            transform: Optional transform to apply to sequences
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.transform = transform
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
            
        return sequence, target


class StockDataLoader:
    """Data loader for stock price data."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing stock data files
        """
        self.data_dir = Path(data_dir)
        
    def load_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load stock data for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with stock data or None if not found
        """
        # Try different file formats
        for ext in ['csv', 'parquet', 'h5']:
            file_path = self.data_dir / f"{symbol}_latest.{ext}"
            
            if file_path.exists():
                try:
                    if ext == 'csv':
                        data = pd.read_csv(file_path)
                    elif ext == 'parquet':
                        data = pd.read_parquet(file_path)
                    elif ext == 'h5':
                        data = pd.read_hdf(file_path, key='data')
                    
                    # Ensure timestamp is datetime
                    if 'timestamp' in data.columns:
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                    
                    logger.info(f"Loaded {len(data)} records for {symbol} from {file_path}")
                    return data
                    
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {str(e)}")
                    continue
        
        logger.error(f"No data file found for symbol {symbol}")
        return None
    
    def load_multiple_symbols(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to their data
        """
        data_dict = {}
        
        for symbol in symbols:
            data = self.load_stock_data(symbol)
            if data is not None:
                data_dict[symbol] = data
        
        logger.info(f"Loaded data for {len(data_dict)} symbols")
        return data_dict
    
    def create_combined_dataset(self, 
                               symbols: List[str],
                               config) -> Dict[str, Any]:
        """
        Create a combined dataset from multiple symbols.
        
        Args:
            symbols: List of stock symbols
            config: Data configuration
            
        Returns:
            Combined dataset dictionary
        """
        from data.preprocessing import FeatureEngineer
        
        all_sequences = []
        all_targets = []
        all_symbols = []
        
        for symbol in symbols:
            try:
                data = self.load_stock_data(symbol)
                if data is None:
                    continue
                
                # Process data
                feature_engineer = FeatureEngineer(config)
                processed = feature_engineer.process_data(data)
                
                # Add to combined dataset
                X = processed['X_train']
                y = processed['y_train']
                
                all_sequences.append(X)
                all_targets.append(y)
                all_symbols.extend([symbol] * len(X))
                
                # Also add validation data
                X_val = processed['X_val']
                y_val = processed['y_val']
                
                all_sequences.append(X_val)
                all_targets.append(y_val)
                all_symbols.extend([symbol] * len(X_val))
                
                logger.info(f"Added {len(X) + len(X_val)} samples from {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                continue
        
        if not all_sequences:
            raise ValueError("No valid data found for any symbols")
        
        # Combine all data
        X_combined = np.vstack(all_sequences)
        y_combined = np.hstack(all_targets)
        
        # Shuffle the combined data
        indices = np.random.permutation(len(X_combined))
        X_combined = X_combined[indices]
        y_combined = y_combined[indices]
        symbols_combined = np.array(all_symbols)[indices]
        
        # Split into train/val
        train_size = int(len(X_combined) * 0.8)
        
        return {
            'X_train': X_combined[:train_size],
            'y_train': y_combined[:train_size],
            'X_val': X_combined[train_size:],
            'y_val': y_combined[train_size:],
            'symbols_train': symbols_combined[:train_size],
            'symbols_val': symbols_combined[train_size:],
            'feature_columns': processed['feature_columns'],
            'scalers': processed['scalers']
        }


def create_data_loaders(X_train: np.ndarray,
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       batch_size: int = 32,
                       num_workers: int = 2,
                       shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch data loaders from numpy arrays.
    
    Args:
        X_train: Training sequences
        y_train: Training targets
        X_val: Validation sequences
        y_val: Validation targets
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader


def save_processed_data(data: Dict[str, Any], 
                       save_path: str,
                       format: str = 'numpy') -> None:
    """
    Save processed data to disk.
    
    Args:
        data: Processed data dictionary
        save_path: Path to save data
        format: Save format ('numpy', 'pickle', 'hdf5')
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if format == 'numpy':
        # Save arrays as .npy files
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np.save(save_path / f"{key}.npy", value)
            elif key == 'scalers':
                import pickle
                with open(save_path / "scalers.pkl", 'wb') as f:
                    pickle.dump(value, f)
            elif key == 'feature_columns':
                import json
                with open(save_path / "feature_columns.json", 'w') as f:
                    json.dump(value, f)
                    
    elif format == 'pickle':
        import pickle
        with open(save_path / "processed_data.pkl", 'wb') as f:
            pickle.dump(data, f)
            
    elif format == 'hdf5':
        import h5py
        with h5py.File(save_path / "processed_data.h5", 'w') as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
    
    logger.info(f"Processed data saved to {save_path} in {format} format")


def load_processed_data(load_path: str, 
                       format: str = 'numpy') -> Dict[str, Any]:
    """
    Load processed data from disk.
    
    Args:
        load_path: Path to load data from
        format: Data format ('numpy', 'pickle', 'hdf5')
        
    Returns:
        Processed data dictionary
    """
    load_path = Path(load_path)
    
    if format == 'numpy':
        data = {}
        
        # Load arrays
        for npy_file in load_path.glob("*.npy"):
            key = npy_file.stem
            data[key] = np.load(npy_file)
        
        # Load scalers
        scalers_file = load_path / "scalers.pkl"
        if scalers_file.exists():
            import pickle
            with open(scalers_file, 'rb') as f:
                data['scalers'] = pickle.load(f)
        
        # Load feature columns
        features_file = load_path / "feature_columns.json"
        if features_file.exists():
            import json
            with open(features_file, 'r') as f:
                data['feature_columns'] = json.load(f)
                
    elif format == 'pickle':
        import pickle
        with open(load_path / "processed_data.pkl", 'rb') as f:
            data = pickle.load(f)
            
    elif format == 'hdf5':
        import h5py
        data = {}
        with h5py.File(load_path / "processed_data.h5", 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
    
    logger.info(f"Processed data loaded from {load_path}")
    return data


class DataValidator:
    """Validator for stock data quality and consistency."""
    
    @staticmethod
    def validate_ohlcv_data(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate OHLCV data for consistency.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Validation results dictionary
        """
        results = {
            'total_records': len(data),
            'issues': [],
            'warnings': []
        }
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            results['issues'].append(f"Missing columns: {missing_columns}")
            return results
        
        # Check for invalid OHLC relationships
        invalid_high = (data['high'] < data['low']) | (data['high'] < data['open']) | (data['high'] < data['close'])
        invalid_low = (data['low'] > data['open']) | (data['low'] > data['close'])
        
        if invalid_high.any():
            results['issues'].append(f"Invalid high prices: {invalid_high.sum()} records")
        
        if invalid_low.any():
            results['issues'].append(f"Invalid low prices: {invalid_low.sum()} records")
        
        # Check for missing values
        null_counts = data[required_columns].isnull().sum()
        if null_counts.any():
            results['warnings'].append(f"Null values found: {dict(null_counts[null_counts > 0])}")
        
        # Check for zero or negative values
        for col in ['open', 'high', 'low', 'close']:
            negative_values = (data[col] <= 0).sum()
            if negative_values > 0:
                results['issues'].append(f"Non-positive {col} values: {negative_values} records")
        
        zero_volume = (data['volume'] == 0).sum()
        if zero_volume > 0:
            results['warnings'].append(f"Zero volume records: {zero_volume}")
        
        # Check for extreme price movements (>50% in one day)
        if 'timestamp' in data.columns:
            price_changes = data['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()
            if extreme_changes > 0:
                results['warnings'].append(f"Extreme price changes (>50%): {extreme_changes} records")
        
        return results
    
    @staticmethod
    def validate_sequences(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Validate sequence data for LSTM training.
        
        Args:
            X: Input sequences
            y: Target values
            
        Returns:
            Validation results dictionary
        """
        results = {
            'sequence_shape': X.shape,
            'target_shape': y.shape,
            'issues': [],
            'warnings': []
        }
        
        # Check shapes
        if len(X) != len(y):
            results['issues'].append(f"Sequence and target lengths don't match: {len(X)} vs {len(y)}")
        
        # Check for NaN values
        if np.isnan(X).any():
            nan_count = np.isnan(X).sum()
            results['issues'].append(f"NaN values in sequences: {nan_count}")
        
        if np.isnan(y).any():
            nan_count = np.isnan(y).sum()
            results['issues'].append(f"NaN values in targets: {nan_count}")
        
        # Check for infinite values
        if np.isinf(X).any():
            inf_count = np.isinf(X).sum()
            results['issues'].append(f"Infinite values in sequences: {inf_count}")
        
        if np.isinf(y).any():
            inf_count = np.isinf(y).sum()
            results['issues'].append(f"Infinite values in targets: {inf_count}")
        
        # Check value ranges (assuming normalized data should be roughly in [0, 1])
        if X.min() < -10 or X.max() > 10:
            results['warnings'].append(f"Unusual sequence value range: [{X.min():.3f}, {X.max():.3f}]")
        
        if y.min() < -10 or y.max() > 10:
            results['warnings'].append(f"Unusual target value range: [{y.min():.3f}, {y.max():.3f}]")
        
        return results
