"""
Dataset utilities for handling dialog datasets.

This module provides comprehensive dataset management functionality including
downloading, loading, preprocessing, and database operations for machine learning
datasets. It supports both local file storage and PostgreSQL database integration
for scalable data management.
"""

import os
from typing import Optional, Dict, Any, Literal

import pandas as pd
from sqlalchemy import create_engine, text

# Import logging
import sys
from pathlib import Path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
from core.logging_config import get_logger

# Initialize module logger
logger = get_logger(__name__)


class DatasetManager:
    """
    Manages dataset operations including download, storage, and preprocessing.
    
    This class provides a comprehensive interface for handling machine learning
    datasets, including downloading from remote sources, local file management,
    and basic dataset analysis. It's designed to work with the Alpaca-GPT4
    dataset but can be extended for other datasets.
    
    Attributes:
        data_dir: Directory for storing dataset files.
        dataset_url: URL for downloading the dataset.
        dataset_file: Local path to the dataset file.
    """
    
    def __init__(self, data_dir: str = "data") -> None:
        """
        Initialize DatasetManager with data directory.
        
        Args:
            data_dir: Directory path for storing dataset files.
        """
        self.data_dir = data_dir
        self.dataset_url = "hf://datasets/vicgalle/alpaca-gpt4/data/train-00000-of-00001-6ef3991c06080e14.parquet"
        self.dataset_file = os.path.join(self.data_dir, "alpaca-gpt4.csv")
        
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_dataset(self, force_download: bool = False, save_to_disk: bool = False, 
                    prefer_database: bool = True, shuffle_database: bool = True) -> pd.DataFrame:
        """
        Load the Alpaca-GPT4 dataset prioritizing database and remote sources.
        
        This method now prioritizes loading from database first, then remote sources,
        and only falls back to local CSV files. This removes the dependency on
        downloading and storing CSV files locally.
        
        Args:
            force_download: Whether to force re-download even if local sources exist.
            save_to_disk: Whether to save the downloaded dataset to local storage (deprecated).
            prefer_database: Whether to prefer database over file sources.
            shuffle_database: Whether to randomly shuffle data when loading from database.
        
        Returns:
            DataFrame containing the loaded dataset.
        """
        # Priority 1: Try database if available and preferred
        if prefer_database and not force_download:
            try:
                db_manager = DatabaseManager()
                df = db_manager.load_from_database(shuffle=shuffle_database)
                if df is not None and len(df) > 0:
                    print(f"Loaded dataset from database: {len(df)} rows{'(shuffled)' if shuffle_database else ''}")
                    return df
                else:
                    print("Database contains no data, trying other sources...")
            except Exception as e:
                print(f"Database unavailable ({e}), trying other sources...")
        
        # Priority 2: Try local CSV file if it exists
        if os.path.exists(self.dataset_file) and not force_download:
            print(f"Loading dataset from local file: {self.dataset_file}")
            df = pd.read_csv(self.dataset_file)
            
            # Apply shuffling to CSV data if requested
            if shuffle_database:
                df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                print("Applied random shuffling to CSV data")
            
            # Optionally save to database if loaded from file
            if prefer_database:
                try:
                    db_manager = DatabaseManager()
                    if db_manager.save_to_database(df):
                        print("Dataset also saved to database for future streaming")
                except Exception as e:
                    print(f"Could not save to database: {e}")
            
            return df
        
        # Priority 3: Download from remote source
        print(f"Downloading dataset from remote source: {self.dataset_url}")
        df = pd.read_parquet(self.dataset_url)
        
        # Apply shuffling to downloaded data if requested
        if shuffle_database:
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            print("Applied random shuffling to downloaded data")
        
        # Save to database first (for streaming), then optionally to disk
        if prefer_database:
            try:
                db_manager = DatabaseManager()
                if db_manager.save_to_database(df):
                    print("Dataset saved to database for streaming support")
            except Exception as e:
                print(f"Could not save to database: {e}")
        
        # Save to disk only if explicitly requested (deprecated)
        if save_to_disk:
            df.to_csv(self.dataset_file, index=False)
            logger.info(f"Dataset also saved to disk: {self.dataset_file}")
            logger.warning("CSV storage is deprecated, use database for streaming")
        else:
            logger.info("Dataset loaded in memory only (recommended for streaming workflows)")
        
        return df
    
    def get_dataset_size_info(self) -> Dict[str, Any]:
        """
        Get information about dataset size and disk usage.
        
        Returns:
            Dictionary containing size information including local file size,
            estimated download size, and storage details.
        """
        info = {
            'local_file_exists': os.path.exists(self.dataset_file),
            'local_file_size_mb': 0,
            'data_directory': self.data_dir,
            'estimated_download_size_mb': 85  # Approximate size of the dataset
        }
        
        if info['local_file_exists']:
            info['local_file_size_mb'] = os.path.getsize(self.dataset_file) / 1024**2
        
        return info
    
    def print_size_info(self) -> None:
        """
        Print dataset size information in a formatted display.
        
        Displays information about local file existence, file sizes,
        and estimated download requirements.
        """
        info = self.get_dataset_size_info()
        
        print("Dataset Size Information:")
        print(f"Data directory: {info['data_directory']}")
        print(f"Local file exists: {info['local_file_exists']}")
        
        if info['local_file_exists']:
            print(f"Local file size: {info['local_file_size_mb']:.1f} MB")
        else:
            print(f"Estimated download size: {info['estimated_download_size_mb']} MB")
            print("File will be downloaded on first use")
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive dataset information and statistics.
        
        Args:
            df: DataFrame to analyze.
        
        Returns:
            Dictionary containing dataset shape, columns, data types,
            null counts, and memory usage information.
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        return info
    
    def print_dataset_summary(self, df: pd.DataFrame) -> None:
        """
        Print comprehensive dataset summary including statistics and sample data.
        
        Args:
            df: DataFrame to summarize.
        """
        info = self.get_dataset_info(df)
        
        print("Dataset Summary:")
        print(f"Shape: {info['shape']}")
        print(f"Columns: {info['columns']}")
        print(f"Memory usage: {info['memory_usage'] / 1024**2:.2f} MB")
        
        print("\nData types:")
        for col, dtype in info['dtypes'].items():
            print(f"  {col}: {dtype}")
        
        print("\nNull values:")
        for col, null_count in info['null_counts'].items():
            print(f"  {col}: {null_count}")
        
        print("\nFirst 3 rows:")
        print(df.head(3))


class DatabaseManager:
    """
    Manages PostgreSQL database operations for dataset storage and retrieval.
    
    This class provides a comprehensive interface for database operations
    including saving DataFrames to PostgreSQL tables, loading data from
    tables, and checking table existence. It handles connection management
    and error handling for robust database operations.
    
    Attributes:
        connection_string: PostgreSQL connection string built from environment variables.
    """
    
    def __init__(self) -> None:
        """
        Initialize DatabaseManager with environment-based connection configuration.
        
        Reads database connection parameters from environment variables
        and constructs the connection string.
        """
        self.connection_string = self._build_connection_string()
    
    def _build_connection_string(self) -> str:
        """
        Build PostgreSQL connection string from environment variables.
        
        Reads database configuration from environment variables including
        host, port, database name, username, and password.
        
        Returns:
            Complete PostgreSQL connection string.
        
        Raises:
            ValueError: If required environment variables are not set.
        """
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'llm')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        
        if not db_user or not db_password:
            raise ValueError("Please set DB_USER and DB_PASSWORD environment variables")
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def save_to_database(self, df: pd.DataFrame, table_name: str = "alpaca_gpt4_dataset", 
                        if_exists: Literal['replace', 'append', 'fail'] = 'replace') -> bool:
        """
        Save DataFrame to PostgreSQL database table.
        
        Saves the provided DataFrame to a PostgreSQL table with proper
        data cleaning and verification of the upload.
        
        Args:
            df: DataFrame to save to the database.
            table_name: Name of the database table to create/update.
            if_exists: How to handle existing table ('replace', 'append', 'fail').
        
        Returns:
            True if save operation succeeded, False otherwise.
        """
        try:
            engine = create_engine(self.connection_string)
            
            # Clean the dataframe
            df_clean = df.copy()
            df_clean['input'] = df_clean['input'].fillna('')
            
            df_clean.to_sql(
                table_name, 
                engine, 
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )
            
            print(f"Successfully saved {len(df_clean)} rows to table '{table_name}'")
            
            # Verify the upload
            with engine.connect() as connection:
                result = connection.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row = result.fetchone()
                if row is not None:
                    count = row[0]
                    print(f"Verified: {count} rows in database table")
            
            return True
            
        except Exception as e:
            print(f"Error saving to PostgreSQL: {e}")
            return False
    
    def load_from_database(self, table_name: str = "alpaca_gpt4_dataset", 
                          shuffle: bool = False) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from PostgreSQL database table.
        
        Retrieves data from the specified table and returns it as a DataFrame.
        
        Args:
            table_name: Name of the database table to load from.
            shuffle: Whether to randomly shuffle the data during loading.
        
        Returns:
            DataFrame containing the table data, or None if load failed.
        """
        try:
            engine = create_engine(self.connection_string)
            
            if shuffle:
                # Load with random ordering for better training diversity
                query = f"SELECT * FROM {table_name} ORDER BY RANDOM()"
                df = pd.read_sql_query(query, engine)
                print(f"Successfully loaded {len(df)} rows from table '{table_name}' (shuffled)")
            else:
                df = pd.read_sql_table(table_name, engine)
                print(f"Successfully loaded {len(df)} rows from table '{table_name}'")
            
            return df
            
        except Exception as e:
            print(f"Error loading from PostgreSQL: {e}")
            return None
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check.
        
        Returns:
            True if table exists, False otherwise.
        """
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as connection:
                result = connection.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :table_name)"
                ), {"table_name": table_name})
                row = result.fetchone()
                return row[0] if row is not None else False
        except Exception:
            return False


def get_dataset_manager(data_dir: str = "data") -> DatasetManager:
    """
    Factory function to create a DatasetManager instance.
    
    This function provides a convenient way to create DatasetManager instances
    with custom data directories while maintaining consistent initialization.
    
    Args:
        data_dir: Directory path for dataset storage.
    
    Returns:
        Initialized DatasetManager instance.
    """
    return DatasetManager(data_dir)


def get_database_manager() -> DatabaseManager:
    """
    Factory function to create a DatabaseManager instance.
    
    This function provides a convenient way to create DatabaseManager instances
    with environment-based configuration.
    
    Returns:
        Initialized DatabaseManager instance.
    """
    return DatabaseManager()
