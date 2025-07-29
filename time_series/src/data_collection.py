#!/usr/bin/env python3
"""Stock data collection and preprocessing pipeline."""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import pandas as pd
import yfinance as yf
import numpy as np
from tqdm import tqdm
import click

from config.settings import list_available_symbols, get_symbol_info
from utils.logging import get_logger

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


class StockDataCollector:
    """Collects and processes stock data from various sources."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_yahoo_finance(self, 
                             symbol: str, 
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             period: str = "2y") -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Period to download if dates not specified
            
        Returns:
            DataFrame with stock data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date)
            else:
                data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            # Reset index to make date a column
            data.reset_index(inplace=True)
            
            # Handle different possible date column names
            if 'Date' in data.columns:
                data.rename(columns={'Date': 'timestamp'}, inplace=True)
            elif 'date' in data.columns:
                data.rename(columns={'date': 'timestamp'}, inplace=True)
            else:
                # If no date column found, use the index
                data['timestamp'] = data.index
            
            logger.info(f"Downloaded {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean stock data.
        
        Args:
            data: Raw stock data
            
        Returns:
            Cleaned and validated data
        """
        if data.empty:
            return data
        
        # Check for required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Remove rows with NaN values in essential columns
        data = data.dropna(subset=['open', 'high', 'low', 'close'])
        
        # Check for data consistency
        invalid_rows = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_rows.any():
            logger.warning(f"Found {invalid_rows.sum()} rows with invalid OHLC data")
            data = data[~invalid_rows]
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Data validation complete. {len(data)} valid records remaining")
        return data
    
    def save_data(self, data: pd.DataFrame, symbol: str, file_format: str = "csv"):
        """
        Save stock data to file.
        
        Args:
            data: Stock data to save
            symbol: Stock symbol
            file_format: File format ('csv', 'parquet', 'hdf5')
        """
        if data.empty:
            logger.warning(f"No data to save for {symbol}")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{timestamp}"
        
        if file_format == "csv":
            filepath = self.data_dir / f"{filename}.csv"
            data.to_csv(filepath, index=False)
        elif file_format == "parquet":
            filepath = self.data_dir / f"{filename}.parquet"
            data.to_parquet(filepath, index=False)
        elif file_format == "hdf5":
            filepath = self.data_dir / f"{filename}.h5"
            data.to_hdf(filepath, key='data', mode='w')
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        logger.info(f"Saved {len(data)} records to {filepath}")
        
        # Also save latest version
        latest_filepath = self.data_dir / f"{symbol}_latest.{file_format}"
        if file_format == "csv":
            data.to_csv(latest_filepath, index=False)
        elif file_format == "parquet":
            data.to_parquet(latest_filepath, index=False)
        elif file_format == "hdf5":
            data.to_hdf(latest_filepath, key='data', mode='w')
    
    def load_existing_data(self, symbol: str, file_format: str = "csv") -> Optional[pd.DataFrame]:
        """
        Load existing data for a symbol.
        
        Args:
            symbol: Stock symbol
            file_format: File format to load
            
        Returns:
            Existing data or None if not found
        """
        filepath = self.data_dir / f"{symbol}_latest.{file_format}"
        
        if not filepath.exists():
            return None
        
        try:
            if file_format == "csv":
                data = pd.read_csv(filepath)
            elif file_format == "parquet":
                data = pd.read_parquet(filepath)
            elif file_format == "hdf5":
                data = pd.read_hdf(filepath, key='data')
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Convert timestamp to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            logger.info(f"Loaded {len(data)} existing records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading existing data for {symbol}: {str(e)}")
            return None
    
    def update_data(self, symbol: str, file_format: str = "csv") -> pd.DataFrame:
        """
        Update existing data with new records.
        
        Args:
            symbol: Stock symbol
            file_format: File format
            
        Returns:
            Updated data
        """
        existing_data = self.load_existing_data(symbol, file_format)
        
        if existing_data is None or existing_data.empty:
            logger.info(f"No existing data for {symbol}, downloading full history")
            return self.download_yahoo_finance(symbol)
        
        # Get the last date in existing data
        last_date = existing_data['timestamp'].max()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        if start_date >= end_date:
            logger.info(f"Data for {symbol} is already up to date")
            return existing_data
        
        logger.info(f"Updating {symbol} data from {start_date} to {end_date}")
        new_data = self.download_yahoo_finance(symbol, start_date, end_date)
        
        if new_data.empty:
            logger.info(f"No new data available for {symbol}")
            return existing_data
        
        # Combine existing and new data
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        combined_data = combined_data.drop_duplicates(subset=['timestamp', 'symbol'])
        combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Added {len(new_data)} new records for {symbol}")
        return combined_data
    
    def collect_multiple_symbols(self, 
                                symbols: List[str],
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                update: bool = False,
                                file_format: str = "csv") -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            update: Whether to update existing data
            file_format: File format to save
            
        Returns:
            Dictionary mapping symbols to their data
        """
        results = {}
        
        for symbol in tqdm(symbols, desc="Downloading stock data"):
            try:
                if update:
                    data = self.update_data(symbol, file_format)
                else:
                    data = self.download_yahoo_finance(symbol, start_date, end_date)
                
                if not data.empty:
                    validated_data = self.validate_data(data)
                    if not validated_data.empty:
                        self.save_data(validated_data, symbol, file_format)
                        results[symbol] = validated_data
                    else:
                        logger.warning(f"No valid data for {symbol} after validation")
                else:
                    logger.warning(f"No data downloaded for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
                
        logger.info(f"Successfully collected data for {len(results)} symbols")
        return results


def parse_date(date_str: str) -> str:
    """Parse and validate date string."""
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


@click.command()
@click.option('--symbols', '-s', multiple=True, help='Stock symbols to download (e.g., AAPL GOOGL)')
@click.option('--all', 'download_all', is_flag=True, help='Download all available symbols')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--days', type=int, help='Number of days to download from today (alternative to date range)')
@click.option('--update', is_flag=True, help='Update existing data with new records')
@click.option('--format', 'file_format', default='csv', type=click.Choice(['csv', 'parquet', 'hdf5']), 
              help='File format to save data')
@click.option('--data-dir', default='data/raw', help='Directory to save data')
@click.option('--list-symbols', is_flag=True, help='List all available symbols')
def main(symbols, download_all, start_date, end_date, days, update, file_format, data_dir, list_symbols):
    """Download and process stock data."""
    
    if list_symbols:
        print("Available stock symbols:")
        for symbol in list_available_symbols():
            print(f"  {symbol}: {get_symbol_info(symbol)}")
        return
    
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize collector
    collector = StockDataCollector(data_dir)
    
    # Determine symbols to download
    if download_all:
        symbols_to_download = list_available_symbols()
    elif symbols:
        symbols_to_download = list(symbols)
    else:
        symbols_to_download = ['AAPL', 'GOOGL', 'META', 'TSLA']  # Default symbols
        logger.info(f"No symbols specified, using default: {symbols_to_download}")
    
    available_symbols = list_available_symbols()
    invalid_symbols = [s for s in symbols_to_download if s not in available_symbols]
    if invalid_symbols:
        logger.warning(f"Invalid symbols (will be skipped): {invalid_symbols}")
        symbols_to_download = [s for s in symbols_to_download if s in available_symbols]
    
    if not symbols_to_download:
        logger.error("No valid symbols to download")
        return
    
    # Handle date parameters
    if days:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        logger.info(f"Downloading {days} days of data from {start_date} to {end_date}")
    
    # Download data
    logger.info(f"Starting data collection for symbols: {symbols_to_download}")
    results = collector.collect_multiple_symbols(
        symbols_to_download,
        start_date=start_date,
        end_date=end_date,
        update=update,
        file_format=file_format
    )
    
    # Summary
    if results:
        print(f"\nData collection complete!")
        print(f"Successfully downloaded data for {len(results)} symbols:")
        for symbol, data in results.items():
            date_range = f"{data['timestamp'].min().date()} to {data['timestamp'].max().date()}"
            print(f"  {symbol}: {len(data)} records ({date_range})")
    else:
        print("No data was successfully downloaded.")


if __name__ == "__main__":
    main()
