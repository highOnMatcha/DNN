"""
Streaming data utilities for handling large datasets that don't fit in memory.

This module provides streaming data capabilities for training on datasets
that are too large to load into memory at once. It supports streaming from
both PostgreSQL databases and remote sources, with efficient batching and
automatic dataset splitting.
"""

import os
import sys
import psutil
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import deque

import pandas as pd
import torch
from torch.utils.data import DataLoader, IterableDataset
from sqlalchemy import create_engine, text
from transformers import PreTrainedTokenizer

# Add project root to path for imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from data.loaders import DatabaseManager


@dataclass
class StreamingConfig:
    """Configuration for streaming dataset operations."""
    batch_size: int = 1000  # Size of chunks fetched from database
    max_length: int = 512
    buffer_size: int = 10000
    prefetch_factor: int = 2
    num_workers: int = 0
    train_split: float = 0.9
    shuffle_buffer: int = 1000
    # Memory allocation for optimal fetching
    cache_memory_percent: float = 0.1  # Use 10% of available RAM for data chunks
    # Remote dataset configuration
    remote_dataset_url: Optional[str] = None  # URL for remote datasets
    table_name: str = "alpaca_gpt4_dataset"  # Database table name
    # Monte Carlo sampling configuration
    monte_carlo_sample_size: int = 1000  # Number of rows to sample for memory estimation
    memory_safety_margin: float = 0.2  # 20% safety margin for memory fluctuations
    # Sample limiting for testing
    max_samples: Optional[int] = None  # Maximum number of samples to process


class StreamingDatasetIterator:
    """Iterator for streaming data from database or remote source with intelligent prefetching.
    
    This iterator implements a prefetching strategy to minimize database round trips by 
    fetching large chunks of data and serving them incrementally to the training process.
    The cache acts as a prefetch buffer to reduce database latency, not for data reuse.
    """
    
    def __init__(self, source: str, config: StreamingConfig, split: str = "train", 
                 cached_dataset_info: Optional[Dict[str, Any]] = None):
        """
        Initialize streaming iterator with caching.
        
        Args:
            source: Data source - 'database' or 'remote'
            config: Streaming configuration
            split: Dataset split - 'train' or 'eval'
            cached_dataset_info: Pre-computed dataset info to avoid redundant SQL queries
        """
        self.source = source
        self.config = config
        self.split = split
        self.current_offset = 0
        self.total_rows = None
        self.is_exhausted = False
        self.cached_dataset_info = cached_dataset_info
        
        # Simple buffer variables for single optimal chunk
        self.cache = deque()  # Single optimal chunk storage
        self.cache_memory_used = 0  # Track memory usage in bytes
        self.current_cache_offset = 0  # Track position within current chunk
        self.samples_yielded = 0
        
        # Split-specific configuration
        self.split_start_offset = 0
        self.split_total_rows = 0
        
        print(f"Initializing streaming with Monte Carlo optimal fetching for {split} split")
        
        if source == "database":
            self.db_manager = DatabaseManager()
            self._init_database_streaming()
        elif source == "remote":
            self._init_remote_streaming()
        else:
            raise ValueError(f"Unsupported source: {source}")
    
    def _calculate_optimal_fetch_size(self) -> int:
        """Calculate how many rows to fetch based on available RAM allocation using Monte Carlo sampling."""
        available_memory = psutil.virtual_memory().available
        allocated_memory = int(available_memory * self.config.cache_memory_percent)
        
        # Monte Carlo sampling to estimate actual row size
        sample_size = min(self.config.monte_carlo_sample_size, self.total_rows // 10) if self.total_rows else self.config.monte_carlo_sample_size
        
        try:
            engine = create_engine(self.db_manager.connection_string)
            
            # Sample random rows to estimate memory usage
            query = f"""
            SELECT * FROM {self.config.table_name} 
            ORDER BY RANDOM() 
            LIMIT {sample_size}
            """
            
            sample_df = pd.read_sql_query(query, engine)
            actual_memory_per_row = self._estimate_dataframe_memory(sample_df) / len(sample_df)
            
            # Add configurable safety margin for memory fluctuations
            estimated_bytes_per_row = int(actual_memory_per_row * (1 + self.config.memory_safety_margin))
            optimal_rows = allocated_memory // estimated_bytes_per_row
            
            print(f"Available RAM: {available_memory / (1024**3):.2f} GB")
            print(f"Allocated memory ({self.config.cache_memory_percent:.1%}): {allocated_memory / (1024**3):.2f} GB")
            print(f"Monte Carlo sample: {sample_size} rows, {actual_memory_per_row:.0f} bytes/row")
            print(f"With safety margin: {estimated_bytes_per_row:.0f} bytes/row")
            print(f"Optimal rows to fetch: {optimal_rows:,}")
            
            return optimal_rows
            
        except Exception as e:
            print(f"Monte Carlo sampling failed: {e}")
            # Fallback to conservative estimate
            fallback_bytes_per_row = 2048  
            optimal_rows = allocated_memory // fallback_bytes_per_row
            print(f"Using fallback estimate: {optimal_rows:,} rows")
            return optimal_rows
    
    def _estimate_dataframe_memory(self, df: pd.DataFrame) -> int:
        """Estimate memory usage of a DataFrame in bytes."""
        return df.memory_usage(deep=True).sum()
    
    def _fetch_optimal_database_chunk(self) -> Optional[pd.DataFrame]:
        """Fetch optimal amount of data that fits in allocated RAM using SQL limits."""
        # Calculate optimal fetch size using Monte Carlo sampling
        optimal_rows = self._calculate_optimal_fetch_size()
        
        # Check if we've reached the end of our split
        if self.current_offset >= self.split_total_rows:
            return None
        
        # Adjust optimal_rows to not exceed split boundaries
        remaining_in_split = self.split_total_rows - self.current_offset
        actual_rows_to_fetch = min(optimal_rows, remaining_in_split)
        
        try:
            engine = create_engine(self.db_manager.connection_string)
            
            # Use SQL LIMIT and OFFSET to handle boundaries automatically
            # Add split_start_offset to current_offset for proper positioning
            absolute_offset = self.split_start_offset + self.current_offset
            
            query = f"""
            SELECT * FROM {self.config.table_name} 
            LIMIT {actual_rows_to_fetch} OFFSET {absolute_offset}
            """
            
            df = pd.read_sql_query(query, engine)
            
            if df.empty:
                return None
                
            chunk_memory = self._estimate_dataframe_memory(df)
            print(f"Fetched {self.split} chunk: {len(df):,} rows (split offset {self.current_offset}, absolute offset {absolute_offset}) - {chunk_memory / (1024**3):.3f} GB")
            
            self.current_offset += len(df)
            return df
            
        except Exception as e:
            print(f"Error fetching optimal database chunk: {e}")
            return None
    
    def _init_database_streaming(self):
        """Initialize database streaming with SQL-based split handling."""
        try:
            # Use cached dataset info if available to avoid redundant SQL queries
            if self.cached_dataset_info and 'total_rows' in self.cached_dataset_info:
                self.total_rows = self.cached_dataset_info['total_rows']
                print(f"Using cached dataset info: {self.total_rows} total rows")
            else:
                # Only make SQL query if we don't have cached info
                engine = create_engine(self.db_manager.connection_string)
                with engine.connect() as connection:
                    result = connection.execute(text(f"SELECT COUNT(*) FROM {self.config.table_name}"))
                    row = result.fetchone()
                    if row is not None:
                        self.total_rows = row[0]
                        print(f"Database contains {self.total_rows} total rows")
                    else:
                        raise ValueError("Could not get row count from database")
            
            # Calculate split boundaries
            train_rows = int(self.total_rows * self.config.train_split)
            
            if self.split == "train":
                self.split_start_offset = 0
                self.split_total_rows = train_rows
            else:  # eval
                self.split_start_offset = train_rows
                self.split_total_rows = self.total_rows - train_rows
            
            # Apply max_samples limit if specified
            if self.config.max_samples is not None:
                self.split_total_rows = min(self.split_total_rows, self.config.max_samples)
            
            # Start from beginning of split
            self.current_offset = 0
            print(f"Streaming {self.split} split: {self.split_total_rows} rows (absolute offset {self.split_start_offset}-{self.split_start_offset + self.split_total_rows})")
            print(f"Ready for optimal fetching with {self.config.cache_memory_percent:.1%} RAM allocation")
            
        except Exception as e:
            print(f"Error initializing database streaming: {e}")
            raise
    
    def _init_remote_streaming(self):
        """Initialize remote streaming from configurable source."""
        if self.config.remote_dataset_url:
            self.dataset_url = self.config.remote_dataset_url
            print(f"Remote streaming initialized with URL: {self.dataset_url}")
        else:
            print("Remote streaming initialized but no URL configured")
            self.dataset_url = None
    
    def __iter__(self):
        """Return self as iterator."""
        return self
    
    def __next__(self) -> pd.DataFrame:
        """Get next batch of data from cache or fetch new chunk."""
        if self.is_exhausted:
            raise StopIteration
            
        if self.source == "database":
            return self._next_database_batch()
        elif self.source == "remote":
            return self._next_remote_batch()
    
    def _next_database_batch(self) -> pd.DataFrame:
        """Get next batch from database using optimal memory allocation."""
        # Check if we've reached the maximum samples limit
        if (self.config.max_samples is not None and 
            self.samples_yielded >= self.config.max_samples):
            self.is_exhausted = True
            raise StopIteration
        
        # Check if we need a new chunk (buffer is empty or exhausted)
        if not self.cache or self.current_cache_offset >= len(self.cache[0]):
            # Clear any existing data
            self.cache.clear()
            self.cache_memory_used = 0
            self.current_cache_offset = 0
            
            # Fetch optimal chunk that fits in allocated RAM
            chunk = self._fetch_optimal_database_chunk()
            
            if chunk is None or chunk.empty:
                self.is_exhausted = True
                raise StopIteration
                
            # Store the chunk
            self.cache.append(chunk)
            chunk_memory = self._estimate_dataframe_memory(chunk)
            self.cache_memory_used = chunk_memory
            
            print(f"Loaded new {self.split} chunk: {len(chunk):,} rows, {chunk_memory / (1024**3):.3f} GB")
        
        # Get current chunk from RAM
        current_chunk = self.cache[0]
        
        # Calculate batch size for this request
        remaining_in_chunk = len(current_chunk) - self.current_cache_offset
        batch_size = min(self.config.batch_size, remaining_in_chunk)
        
        # Also limit by max_samples if specified
        if self.config.max_samples is not None:
            remaining_samples = self.config.max_samples - self.samples_yielded
            batch_size = min(batch_size, remaining_samples)
        
        if batch_size <= 0:
            self.is_exhausted = True
            raise StopIteration
        
        # Extract the batch from RAM
        start_idx = self.current_cache_offset
        end_idx = start_idx + batch_size
        batch_df = current_chunk.iloc[start_idx:end_idx].copy()
        
        self.current_cache_offset += batch_size
        self.samples_yielded += len(batch_df)
        
        #remaining_samples = len(current_chunk) - self.current_cache_offset
        #total_limit = f"/{self.config.max_samples}" if self.config.max_samples else ""
        #print(f"Serving {self.split} batch: {len(batch_df)} rows (total yielded: {self.samples_yielded:,}{total_limit}, remaining in RAM: {remaining_samples:,})")
        
        return batch_df
    
    def _next_remote_batch(self) -> pd.DataFrame:
        """Get next batch from remote source."""
        # This is a placeholder implementation
        # In practice, you'd implement chunked reading from parquet files
        # For now, we'll raise StopIteration to indicate no remote streaming yet
        self.is_exhausted = True
        raise StopIteration


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for text data that doesn't fit in memory."""
    
    def __init__(self, source: str, tokenizer: PreTrainedTokenizer, 
                 config: StreamingConfig, split: str = "train", 
                 cached_dataset_info: Optional[Dict[str, Any]] = None):
        """
        Initialize streaming text dataset.
        
        Args:
            source: Data source - 'database' or 'remote'
            tokenizer: HuggingFace tokenizer for text processing
            config: Streaming configuration
            split: Dataset split - 'train' or 'eval'
            cached_dataset_info: Pre-computed dataset info to avoid redundant SQL queries
        """
        self.source = source
        self.tokenizer = tokenizer
        self.config = config
        self.split = split
        self.cached_dataset_info = cached_dataset_info
        
    def __iter__(self):
        """Iterate over tokenized samples."""
        data_iterator = StreamingDatasetIterator(self.source, self.config, self.split, self.cached_dataset_info)
        samples_yielded = 0
        
        for batch_df in data_iterator:
            # Process each text in the batch
            for text in batch_df['text']:
                # Check sample limit before processing
                if (self.config.max_samples is not None and 
                    samples_yielded >= self.config.max_samples):
                    return
                
                # Tokenize the text
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                # Yield tokenized sample
                yield {
                    'input_ids': tokenized['input_ids'].squeeze(),
                    'attention_mask': tokenized['attention_mask'].squeeze()
                }
                
                samples_yielded += 1


class StreamingDataManager:
    """Manager for streaming data operations."""
    
    def __init__(self, config: StreamingConfig):
        """
        Initialize streaming data manager.
        
        Args:
            config: Streaming configuration
        """
        self.config = config
        
    def create_streaming_datasets(self, source: str, tokenizer: PreTrainedTokenizer) -> Tuple[StreamingTextDataset, StreamingTextDataset]:
        """
        Create streaming train and evaluation datasets.
        
        Args:
            source: Data source - 'database' or 'remote'
            tokenizer: HuggingFace tokenizer
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        # Get dataset info once and cache it to avoid redundant SQL queries
        cached_info = self.get_dataset_info(source=source) if source == 'database' else None
        
        # Create separate configs for train and eval
        train_config = self.config
        
        # For evaluation, use much smaller sample limit to avoid infinite loops
        eval_config = StreamingConfig(
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            buffer_size=self.config.buffer_size,
            prefetch_factor=self.config.prefetch_factor,
            num_workers=self.config.num_workers,
            train_split=self.config.train_split,
            shuffle_buffer=self.config.shuffle_buffer,
            cache_memory_percent=self.config.cache_memory_percent,
            remote_dataset_url=self.config.remote_dataset_url,
            table_name=self.config.table_name,
            monte_carlo_sample_size=self.config.monte_carlo_sample_size,
            memory_safety_margin=self.config.memory_safety_margin,
            max_samples=min(50, self.config.max_samples // 10) if self.config.max_samples else 50  # Limit eval to 50 samples or 1/10 of max_samples
        )
        
        train_dataset = StreamingTextDataset(source, tokenizer, train_config, "train", cached_info)
        eval_dataset = StreamingTextDataset(source, tokenizer, eval_config, "eval", cached_info)
        
        return train_dataset, eval_dataset
    
    def create_streaming_dataloaders(self, source: str, tokenizer: PreTrainedTokenizer) -> Tuple[DataLoader, DataLoader]:
        """
        Create streaming data loaders for training.
        
        Args:
            source: Data source - 'database' or 'remote'
            tokenizer: HuggingFace tokenizer
            
        Returns:
            Tuple of (train_dataloader, eval_dataloader)
        """
        train_dataset, eval_dataset = self.create_streaming_datasets(source, tokenizer)
        
        def collate_fn(batch):
            """Collate function for batching tokenized samples with padding."""
            input_ids = [item['input_ids'] for item in batch]
            attention_mask = [item['attention_mask'] for item in batch]
            
            # Pad sequences to the same length
            max_length = max(len(seq) for seq in input_ids)
            
            padded_input_ids = []
            padded_attention_mask = []
            
            for i, seq in enumerate(input_ids):
                # Pad input_ids
                padding_length = max_length - len(seq)
                padded_seq = torch.cat([seq, torch.full((padding_length,), tokenizer.pad_token_id)])
                padded_input_ids.append(padded_seq)
                
                # Pad attention_mask
                padded_mask = torch.cat([attention_mask[i], torch.zeros(padding_length)])
                padded_attention_mask.append(padded_mask)
            
            # Stack the padded sequences
            input_ids_tensor = torch.stack(padded_input_ids)
            attention_mask_tensor = torch.stack(padded_attention_mask)
            
            # Create labels with proper padding token handling
            # Set padding tokens to -100 so they are ignored in loss calculation
            # This matches the behavior of DataCollatorForLanguageModeling
            labels_tensor = input_ids_tensor.clone()
            labels_tensor[labels_tensor == tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': input_ids_tensor,
                'attention_mask': attention_mask_tensor,
                'labels': labels_tensor  # For language modeling with proper padding handling
            }
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
            num_workers=self.config.num_workers
        )
        
        return train_dataloader, eval_dataloader
    
    def get_dataset_info(self, source: str) -> Dict[str, Any]:
        """
        Get information about the streaming dataset.
        
        Args:
            source: Data source - 'database' or 'remote'
            
        Returns:
            Dictionary with dataset information
        """
        if source == "database":
            db_manager = DatabaseManager()
            try:
                engine = create_engine(db_manager.connection_string)
                with engine.connect() as connection:
                    # Get total count
                    result = connection.execute(text(f"SELECT COUNT(*) FROM {self.config.table_name}"))
                    row = result.fetchone()
                    total_rows = row[0] if row else 0
                    
                    # Get sample for column info
                    sample_df = pd.read_sql_query(f"SELECT * FROM {self.config.table_name} LIMIT 5", engine)
                    
                    split_idx = int(total_rows * self.config.train_split)
                    
                    return {
                        'source': 'database',
                        'total_rows': total_rows,
                        'train_rows': split_idx,
                        'eval_rows': total_rows - split_idx,
                        'columns': sample_df.columns.tolist(),
                        'dtypes': sample_df.dtypes.to_dict(),
                        'streaming': True,
                        'batch_size': self.config.batch_size
                    }
                    
            except Exception as e:
                print(f"Error getting database info: {e}")
                return {'source': 'database', 'error': str(e)}
                
        elif source == "remote":
            return {
                'source': 'remote',
                'streaming': True,
                'note': 'Remote streaming not yet implemented',
                'batch_size': self.config.batch_size
            }
        else:
            return {'error': f'Unknown source: {source}'}


def get_streaming_manager(config: Optional[StreamingConfig] = None) -> StreamingDataManager:
    """
    Factory function to create a StreamingDataManager instance.
    
    Args:
        config: Optional streaming configuration. If None, uses default.
        
    Returns:
        Initialized StreamingDataManager instance
    """
    if config is None:
        config = StreamingConfig()
    
    return StreamingDataManager(config)