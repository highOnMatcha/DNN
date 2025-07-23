"""
Dataset utilities for handling dialog datasets.
Functions for downloading, loading, and managing dataset storage.
"""

import pandas as pd
import os
from sqlalchemy import create_engine, text
from typing import Optional


class DatasetManager:
    """Manages dataset operations including download, storage, and database operations."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize DatasetManager with data directory."""
        self.data_dir = data_dir
        self.dataset_url = "hf://datasets/vicgalle/alpaca-gpt4/data/train-00000-of-00001-6ef3991c06080e14.parquet"
        self.dataset_file = os.path.join(self.data_dir, "alpaca-gpt4.csv")
        
        os.makedirs(self.data_dir, exist_ok=True)
    
    def load_dataset(self, force_download: bool = False, save_to_disk: bool = True) -> pd.DataFrame:
        """Load the Alpaca-GPT4 dataset from local file or download if not available."""
        if os.path.exists(self.dataset_file) and not force_download:
            print(f"Loading dataset from local file: {self.dataset_file}")
            df = pd.read_csv(self.dataset_file)
        else:
            print(f"Downloading dataset from {self.dataset_url}")
            df = pd.read_parquet(self.dataset_url)
            
            if save_to_disk:
                df.to_csv(self.dataset_file, index=False)
                print(f"Dataset saved to disk: {self.dataset_file}")
                print(f"Dataset size: {os.path.getsize(self.dataset_file) / 1024**2:.1f} MB")
            else:
                print("Dataset loaded in memory only (not saved to disk)")
        
        return df
    
    def get_dataset_size_info(self) -> dict:
        """
        Get information about dataset size and disk usage.
        
        Returns:
            dict: Size information including local file size, estimated download size, etc.
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
    
    def print_size_info(self):
        """Print dataset size information."""
        info = self.get_dataset_size_info()
        
        print("Dataset Size Information:")
        print(f"Data directory: {info['data_directory']}")
        print(f"Local file exists: {info['local_file_exists']}")
        
        if info['local_file_exists']:
            print(f"Local file size: {info['local_file_size_mb']:.1f} MB")
        else:
            print(f"Estimated download size: {info['estimated_download_size_mb']} MB")
            print("File will be downloaded on first use")
    
    def get_dataset_info(self, df: pd.DataFrame) -> dict:
        """Get dataset information."""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        return info
    
    def print_dataset_summary(self, df: pd.DataFrame):
        """Print dataset summary."""
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
    """Manages PostgreSQL database operations for the dataset."""
    
    def __init__(self):
        """Initialize DatabaseManager with environment variables."""
        self.connection_string = self._build_connection_string()
    
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables."""
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'llm')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        
        if not db_user or not db_password:
            raise ValueError("Please set DB_USER and DB_PASSWORD environment variables")
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def save_to_database(self, df: pd.DataFrame, table_name: str = "alpaca_gpt4_dataset", 
                        if_exists: str = 'replace') -> bool:
        """Save DataFrame to PostgreSQL database."""
        try:
            engine = create_engine(self.connection_string)
            
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
                count = result.fetchone()[0]
                print(f"Verified: {count} rows in database table")
            
            return True
            
        except Exception as e:
            print(f"Error saving to PostgreSQL: {e}")
            return False
    
    def load_from_database(self, table_name: str = "alpaca_gpt4_dataset") -> Optional[pd.DataFrame]:
        """Load DataFrame from PostgreSQL database."""
        try:
            engine = create_engine(self.connection_string)
            df = pd.read_sql_table(table_name, engine)
            print(f"Successfully loaded {len(df)} rows from table '{table_name}'")
            return df
            
        except Exception as e:
            print(f"Error loading from PostgreSQL: {e}")
            return None
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as connection:
                result = connection.execute(text(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)"
                ), (table_name,))
                return result.fetchone()[0]
        except Exception:
            return False


def get_dataset_manager(data_dir: str = "data") -> DatasetManager:
    """Factory function to create a DatasetManager instance."""
    return DatasetManager(data_dir)


def get_database_manager() -> DatabaseManager:
    """Factory function to create a DatabaseManager instance."""
    return DatabaseManager()
