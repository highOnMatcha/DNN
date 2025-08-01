"""
Data preprocessing and feature engineering for stock price prediction.

This module provides comprehensive data preprocessing capabilities including
technical indicator calculation, feature engineering, data normalization,
and sequence generation for LSTM model training.

Author: Time Series Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from pathlib import Path
import logging
import warnings

from utils.logging import get_logger

# Suppress warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = get_logger(__name__)

# Constants for validation
MIN_SEQUENCE_LENGTH = 5
MAX_SEQUENCE_LENGTH = 1000
REQUIRED_OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']
SUPPORTED_SCALERS = {
    'minmax': MinMaxScaler,
    'standard': StandardScaler, 
    'robust': RobustScaler
}


class TechnicalIndicators:
    """
    Calculate technical indicators for stock data.
    
    This class provides static methods to compute various technical analysis
    indicators including moving averages, momentum indicators, volatility
    measures, volume indicators, and trend analysis.
    """
    
    @staticmethod
    def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add moving average indicators to stock data.
        
        Computes Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)
        for various time windows.
        
        Args:
            data (pd.DataFrame): Stock data with 'close' column
            
        Returns:
            pd.DataFrame: Data with added moving average columns
            
        Raises:
            KeyError: If required columns are missing
            ValueError: If data is insufficient for calculations
        """
        if 'close' not in data.columns:
            raise KeyError("Data must contain 'close' column")
        
        if len(data) < 200:
            logger.warning(f"Data length ({len(data)}) may be insufficient for 200-day moving averages")
        
        data = data.copy()
        
        try:
            # Simple Moving Averages
            sma_windows = [5, 10, 20, 50, 100, 200]
            for window in sma_windows:
                if len(data) >= window:
                    data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
                else:
                    logger.warning(f"Skipping SMA_{window}: insufficient data")
            
            # Exponential Moving Averages
            ema_spans = [12, 26, 50]
            for span in ema_spans:
                if len(data) >= span:
                    data[f'ema_{span}'] = data['close'].ewm(span=span).mean()
                else:
                    logger.warning(f"Skipping EMA_{span}: insufficient data")
                    
            logger.debug("Successfully computed moving averages")
            
        except Exception as e:
            logger.error(f"Error computing moving averages: {e}")
            raise
        
        return data
    
    @staticmethod
    def add_momentum_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based technical indicators.
        
        Computes RSI, MACD, Stochastic Oscillator, Williams %R, and CCI
        using robust implementations.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with added momentum indicators
            
        Raises:
            KeyError: If required columns are missing
        """
        required_cols = ['high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}")
        
        data = data.copy()
        
        try:
            # RSI calculation
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()  # type: ignore
            loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()  # type: ignore
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD calculation
            exp1 = data['close'].ewm(span=12).mean()
            exp2 = data['close'].ewm(span=26).mean()
            data['macd'] = exp1 - exp2
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            # Stochastic Oscillator
            low_min = data['low'].rolling(window=14).min()
            high_max = data['high'].rolling(window=14).max()
            data['stoch_k'] = 100 * (data['close'] - low_min) / (high_max - low_min)
            data['stoch_d'] = data['stoch_k'].rolling(window=3).mean()
            
            # Williams %R
            data['williams_r'] = -100 * (high_max - data['close']) / (high_max - low_min)
            
            # CCI (Commodity Channel Index)
            tp = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = tp.rolling(window=14).mean()
            mad = tp.rolling(window=14).apply(lambda x: np.mean(np.abs(x - x.mean())))
            data['cci'] = (tp - sma_tp) / (0.015 * mad)
            
            logger.debug("Successfully computed momentum indicators")
            
        except Exception as e:
            logger.error(f"Error computing momentum indicators: {e}")
            raise
        
        return data
    
    @staticmethod
    def add_volatility_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based indicators.
        
        Computes Bollinger Bands, Average True Range, and Keltner Channels
        using robust implementations.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with added volatility indicators
        """
        data = data.copy()
        
        try:
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            bb_middle = data['close'].rolling(window=bb_period).mean()
            bb_std_dev = data['close'].rolling(window=bb_period).std()
            data['bb_upper'] = bb_middle + (bb_std_dev * bb_std)
            data['bb_middle'] = bb_middle
            data['bb_lower'] = bb_middle - (bb_std_dev * bb_std)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_percent'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Average True Range (ATR)
            high_low = data['high'] - data['low']
            high_close_prev = np.abs(data['high'] - data['close'].shift(1))
            low_close_prev = np.abs(data['low'] - data['close'].shift(1))
            true_range_series = pd.Series(np.maximum(high_low, np.maximum(high_close_prev, low_close_prev)))
            data['atr'] = true_range_series.rolling(window=14).mean()
            
            # Keltner Channel
            kc_period = 20
            kc_multiplier = 2
            kc_middle = data['close'].ewm(span=kc_period).mean()
            data['kc_upper'] = kc_middle + (kc_multiplier * data['atr'])
            data['kc_middle'] = kc_middle
            data['kc_lower'] = kc_middle - (kc_multiplier * data['atr'])
            
            logger.debug("Successfully computed volatility indicators")
            
        except Exception as e:
            logger.error(f"Error computing volatility indicators: {e}")
            raise
        
        return data
    
    @staticmethod
    def add_volume_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.
        
        Computes On-Balance Volume, Volume Weighted Average Price,
        Money Flow Index, A/D Line, and Chaikin Money Flow.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with added volume indicators
        """
        if 'volume' not in data.columns:
            logger.warning("Volume column not found, skipping volume indicators")
            return data
        
        data = data.copy()
        
        try:
            # On-Balance Volume (OBV)
            obv = np.where(data['close'] > data['close'].shift(1), 
                          data['volume'], 
                          np.where(data['close'] < data['close'].shift(1), 
                                  -data['volume'], 0)).cumsum()
            data['obv'] = obv
            
            # Volume Weighted Average Price (VWAP)
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            cumulative_typical_price_volume = (typical_price * data['volume']).cumsum()
            cumulative_volume = data['volume'].cumsum()
            data['vwap'] = cumulative_typical_price_volume / cumulative_volume
            
            # Money Flow Index (MFI)
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            money_flow = typical_price * data['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfi_ratio = positive_flow / negative_flow
            data['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
            # Accumulation/Distribution Line
            clv = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
            clv = clv.fillna(0)
            data['ad_line'] = (clv * data['volume']).cumsum()
            
            # Chaikin Money Flow
            ad_volume = clv * data['volume']
            data['cmf'] = ad_volume.rolling(window=20).sum() / data['volume'].rolling(window=20).sum()
            
            logger.debug("Successfully computed volume indicators")
            
        except Exception as e:
            logger.error(f"Error computing volume indicators: {e}")
            raise
        
        return data
    
    @staticmethod
    def add_trend_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based indicators.
        
        Computes Parabolic SAR, ADX, Aroon, and basic Ichimoku components.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with added trend indicators
        """
        data = data.copy()
        
        try:
            # Simplified Parabolic SAR
            # This is a basic implementation - full PSAR is more complex
            data['psar'] = data['close'].rolling(window=14).mean()  # Simplified
            
            # Average Directional Index (ADX) - simplified
            # Calculate directional movement
            plus_dm = data['high'].diff()
            minus_dm = data['low'].diff()
            plus_dm = plus_dm.where(plus_dm > 0, 0.0)  # type: ignore
            minus_dm = minus_dm.where(minus_dm < 0, 0.0).abs()  # type: ignore
            
            # True Range for ADX
            high_low = data['high'] - data['low']
            high_close_prev = np.abs(data['high'] - data['close'].shift(1))
            low_close_prev = np.abs(data['low'] - data['close'].shift(1))
            true_range_series = pd.Series(np.maximum(high_low, np.maximum(high_close_prev, low_close_prev)))
            
            # Directional Indicators
            plus_di = 100 * (plus_dm.rolling(14).mean() / true_range_series.rolling(14).mean())
            minus_di = 100 * (minus_dm.rolling(14).mean() / true_range_series.rolling(14).mean())
            
            data['adx_pos'] = plus_di
            data['adx_neg'] = minus_di
            
            # ADX calculation
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            data['adx'] = pd.Series(dx).rolling(14).mean()
            
            # Aroon indicators
            aroon_period = 25
            aroon_up = data['high'].rolling(aroon_period).apply(
                lambda x: (aroon_period - x.argmax()) / aroon_period * 100
            )
            aroon_down = data['low'].rolling(aroon_period).apply(
                lambda x: (aroon_period - x.argmin()) / aroon_period * 100
            )
            data['aroon_up'] = aroon_up
            data['aroon_down'] = aroon_down
            data['aroon_indicator'] = aroon_up - aroon_down
            
            # Simplified Ichimoku components
            # Conversion Line (Tenkan-sen)
            period9_high = data['high'].rolling(9).max()
            period9_low = data['low'].rolling(9).min()
            data['ichimoku_conversion'] = (period9_high + period9_low) / 2
            
            # Base Line (Kijun-sen)
            period26_high = data['high'].rolling(26).max()
            period26_low = data['low'].rolling(26).min()
            data['ichimoku_base'] = (period26_high + period26_low) / 2
            
            # Leading Span A (Senkou Span A)
            data['ichimoku_a'] = ((data['ichimoku_conversion'] + data['ichimoku_base']) / 2).shift(26)
            
            # Leading Span B (Senkou Span B)
            period52_high = data['high'].rolling(52).max()
            period52_low = data['low'].rolling(52).min()
            data['ichimoku_b'] = ((period52_high + period52_low) / 2).shift(26)
            
            logger.debug("Successfully computed trend indicators")
            
        except Exception as e:
            logger.error(f"Error computing trend indicators: {e}")
            raise
        
        return data


class FeatureEngineer:
    """
    Feature engineering for stock price data.
    
    This class provides comprehensive feature engineering capabilities
    including price-based features, time features, lag features, and
    rolling statistics.
    """
    
    def __init__(self, config: Any) -> None:
        """
        Initialize feature engineer.
        
        Args:
            config: Data configuration object containing feature engineering parameters
        """
        self.config = config
        self.scalers: Dict[str, Any] = {}
        
    def add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features to the dataset.
        
        Computes various price transformations including returns, ratios,
        gaps, and price positions within daily ranges.
        
        Args:
            data (pd.DataFrame): Stock data with OHLCV columns
            
        Returns:
            pd.DataFrame: Data with added price features
            
        Raises:
            KeyError: If required columns are missing
        """
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise KeyError(f"Missing required columns for price features: {missing_cols}")
        
        data = data.copy()
        
        try:
            # Price changes and returns
            data['price_change'] = data['close'].diff()
            data['price_change_pct'] = data['close'].pct_change()
            data['high_low_pct'] = (data['high'] - data['low']) / data['close']
            data['open_close_pct'] = (data['close'] - data['open']) / data['open']
            
            # Log returns (handle zeros and negatives safely)
            close_shifted = data['close'].shift(1)
            valid_mask = (close_shifted > 0) & (data['close'] > 0)
            data['log_return'] = np.where(valid_mask, 
                                        np.log(data['close'] / close_shifted), 
                                        0)
            
            # Price position within the day's range
            range_size = data['high'] - data['low']
            data['price_position'] = np.where(range_size > 0,
                                            (data['close'] - data['low']) / range_size,
                                            0.5)  # Middle position if no range
            
            # Gap analysis
            data['gap'] = data['open'] - data['close'].shift(1)
            data['gap_pct'] = np.where(data['close'].shift(1) > 0,
                                     data['gap'] / data['close'].shift(1),
                                     0)
            
            # Price acceleration (second derivative)
            data['price_acceleration'] = data['price_change'].diff()
            
            logger.debug("Successfully computed price features")
            
        except Exception as e:
            logger.error(f"Error computing price features: {e}")
            raise
        
        return data
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features to the dataset.
        
        Computes cyclical time encodings and market session indicators.
        
        Args:
            data (pd.DataFrame): Stock data with timestamp column
            
        Returns:
            pd.DataFrame: Data with added time features
            
        Raises:
            KeyError: If timestamp column is missing
        """
        if 'timestamp' not in data.columns:
            # Try to use index if it's a datetime index
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.copy()
                data['timestamp'] = data.index
            else:
                raise KeyError("Data must contain 'timestamp' column or have DatetimeIndex")
        
        data = data.copy()
        
        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Basic time components
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['day_of_month'] = data['timestamp'].dt.day
            data['week_of_year'] = data['timestamp'].dt.isocalendar().week
            data['month'] = data['timestamp'].dt.month
            data['quarter'] = data['timestamp'].dt.quarter
            data['year'] = data['timestamp'].dt.year
            
            # Cyclical encoding for time features (handles periodicity properly)
            data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
            data['day_of_month_sin'] = np.sin(2 * np.pi * data['day_of_month'] / 31)
            data['day_of_month_cos'] = np.cos(2 * np.pi * data['day_of_month'] / 31)
            
            # Market session indicators (assuming US market hours)
            data['hour'] = data['timestamp'].dt.hour
            data['is_market_open'] = ((data['hour'] >= 9) & (data['hour'] <= 16) & 
                                     (data['day_of_week'] < 5)).astype(int)
            
            # Holiday and special date indicators (basic)
            data['is_month_end'] = (data['day_of_month'] >= 28).astype(int)
            data['is_quarter_end'] = (data['month'] % 3 == 0).astype(int)
            data['is_year_end'] = (data['month'] == 12).astype(int)
            
            logger.debug("Successfully computed time features")
            
        except Exception as e:
            logger.error(f"Error computing time features: {e}")
            raise
        
        return data
    
    def add_lag_features(self, data: pd.DataFrame, lags: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Add lagged features to the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            lags (Optional[List[int]]): List of lag periods, defaults to [1, 2, 3, 5, 10]
            
        Returns:
            pd.DataFrame: Data with added lag features
        """
        if lags is None:
            lags = [1, 2, 3, 5, 10]
        
        data = data.copy()
        
        try:
            # Add lagged values for key price columns
            price_cols = ['close', 'open', 'high', 'low']
            available_price_cols = [col for col in price_cols if col in data.columns]
            
            for col in available_price_cols:
                for lag in lags:
                    if len(data) > lag:  # Only add if we have enough data
                        data[f'{col}_lag_{lag}'] = data[col].shift(lag)
            
            # Add lagged returns
            if 'price_change_pct' in data.columns:
                for lag in lags:
                    if len(data) > lag:
                        data[f'return_lag_{lag}'] = data['price_change_pct'].shift(lag)
            
            logger.debug(f"Successfully computed lag features for lags: {lags}")
            
        except Exception as e:
            logger.error(f"Error computing lag features: {e}")
            raise
        
        return data
    
    def add_rolling_features(self, data: pd.DataFrame, windows: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Add rolling window statistics.
        
        Args:
            data (pd.DataFrame): Input data
            windows (Optional[List[int]]): List of window sizes, defaults to [5, 10, 20]
            
        Returns:
            pd.DataFrame: Data with added rolling features
        """
        if windows is None:
            windows = [5, 10, 20]
        
        data = data.copy()
        
        try:
            # Rolling statistics for returns
            if 'price_change_pct' in data.columns:
                for window in windows:
                    if len(data) >= window:
                        data[f'return_std_{window}'] = data['price_change_pct'].rolling(window).std()
                        data[f'return_skew_{window}'] = data['price_change_pct'].rolling(window).skew()
            
            # Rolling statistics for volume
            if 'volume' in data.columns:
                for window in windows:
                    if len(data) >= window:
                        data[f'volume_mean_{window}'] = data['volume'].rolling(window).mean()
                        data[f'volume_std_{window}'] = data['volume'].rolling(window).std()
            
            # Rolling high/low features
            if all(col in data.columns for col in ['high', 'low', 'close']):
                for window in windows:
                    if len(data) >= window:
                        rolling_high = data['high'].rolling(window).max()
                        rolling_low = data['low'].rolling(window).min()
                        data[f'high_low_ratio_{window}'] = rolling_high / rolling_low
                        data[f'close_to_high_{window}'] = data['close'] / rolling_high
                        data[f'close_to_low_{window}'] = data['close'] / rolling_low
            
            logger.debug(f"Successfully computed rolling features for windows: {windows}")
            
        except Exception as e:
            logger.error(f"Error computing rolling features: {e}")
            raise
        
        return data


class DataPreprocessor:
    """
    Comprehensive data preprocessing for stock price prediction.
    
    This class handles data normalization, scaling, validation, and
    preparation for machine learning models.
    """
    
    def __init__(self, config: Any) -> None:
        """
        Initialize data preprocessor.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self.scalers: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        
    def normalize_features(self, 
                          train_data: pd.DataFrame,
                          val_data: Optional[pd.DataFrame] = None,
                          test_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, ...]:
        """
        Normalize features using specified scaling method.
        
        Args:
            train_data (pd.DataFrame): Training data
            val_data (Optional[pd.DataFrame]): Validation data
            test_data (Optional[pd.DataFrame]): Test data
            
        Returns:
            Tuple[pd.DataFrame, ...]: Normalized datasets
            
        Raises:
            ValueError: If normalization method is not supported
        """
        if self.config.normalization_method not in SUPPORTED_SCALERS:
            raise ValueError(f"Unsupported normalization method: {self.config.normalization_method}")
        
        scaler_class = SUPPORTED_SCALERS[self.config.normalization_method]
        
        try:
            # Initialize scaler
            scaler = scaler_class()
            
            # Fit on training data
            train_scaled = train_data.copy()
            numeric_columns = train_data.select_dtypes(include=[np.number]).columns
            
            train_scaled[numeric_columns] = scaler.fit_transform(train_data[numeric_columns])
            self.scalers['feature_scaler'] = scaler
            
            results = [train_scaled]
            
            # Transform validation data if provided
            if val_data is not None:
                val_scaled = val_data.copy()
                val_scaled[numeric_columns] = scaler.transform(val_data[numeric_columns])
                results.append(val_scaled)
            
            # Transform test data if provided
            if test_data is not None:
                test_scaled = test_data.copy()
                test_scaled[numeric_columns] = scaler.transform(test_data[numeric_columns])
                results.append(test_scaled)
            
            logger.info(f"Successfully normalized features using {self.config.normalization_method}")
            
            return tuple(results)
            
        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate input data for completeness and quality.
        
        Args:
            data (pd.DataFrame): Data to validate
            
        Returns:
            Dict[str, Any]: Validation results and statistics
        """
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'statistics': {}
        }
        
        try:
            # Check for required columns
            missing_cols = [col for col in REQUIRED_OHLCV_COLUMNS if col not in data.columns]
            if missing_cols:
                validation_results['errors'].append(f"Missing required columns: {missing_cols}")
                validation_results['valid'] = False
            
            # Check data length
            if len(data) < MIN_SEQUENCE_LENGTH:
                validation_results['errors'].append(f"Insufficient data: {len(data)} rows (minimum: {MIN_SEQUENCE_LENGTH})")
                validation_results['valid'] = False
            
            # Check for missing values
            missing_counts = data.isnull().sum()
            if missing_counts.any():
                validation_results['warnings'].append(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
            
            # Check for duplicate timestamps
            if 'timestamp' in data.columns:
                duplicates = data['timestamp'].duplicated().sum()
                if duplicates > 0:
                    validation_results['warnings'].append(f"Found {duplicates} duplicate timestamps")
            
            # Check for data anomalies
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in data.columns:
                    # Check for infinite values
                    inf_count = np.isinf(data[col]).sum()
                    if inf_count > 0:
                        validation_results['warnings'].append(f"Found {inf_count} infinite values in {col}")
                    
                    # Check for zero/negative prices (problematic for log calculations)
                    if col in ['open', 'high', 'low', 'close']:
                        non_positive = (data[col] <= 0).sum()
                        if non_positive > 0:
                            validation_results['warnings'].append(f"Found {non_positive} non-positive values in {col}")
            
            # Compute basic statistics
            validation_results['statistics'] = {
                'row_count': len(data),
                'column_count': len(data.columns),
                'numeric_columns': len(numeric_cols),
                'missing_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
            }
            
            logger.info(f"Data validation completed. Valid: {validation_results['valid']}")
            
        except Exception as e:
            logger.error(f"Error during data validation: {e}")
            validation_results['valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
        
        return validation_results


class SequenceGenerator:
    """
    Generate sequences for LSTM training from time series data.
    
    This class handles the creation of input sequences and target values
    for training time series prediction models.
    """
    
    def __init__(self, sequence_length: int, prediction_horizon: int = 1) -> None:
        """
        Initialize sequence generator.
        
        Args:
            sequence_length (int): Length of input sequences
            prediction_horizon (int): Number of steps to predict ahead
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not MIN_SEQUENCE_LENGTH <= sequence_length <= MAX_SEQUENCE_LENGTH:
            raise ValueError(f"sequence_length must be between {MIN_SEQUENCE_LENGTH} and {MAX_SEQUENCE_LENGTH}")
        
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
    def create_sequences(self, 
                        data: pd.DataFrame, 
                        target_column: str = 'close',
                        feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data (pd.DataFrame): Time series data
            target_column (str): Column to predict
            feature_columns (Optional[List[str]]): Features to use, None for all numeric
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (sequences, targets)
            
        Raises:
            ValueError: If data is insufficient or columns are missing
        """
        if len(data) < self.sequence_length + self.prediction_horizon:
            raise ValueError(f"Insufficient data: need at least {self.sequence_length + self.prediction_horizon} rows")
        
        if target_column not in data.columns:
            raise KeyError(f"Target column '{target_column}' not found in data")
        
        try:
            # Select feature columns
            if feature_columns is None:
                feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Validate feature columns exist
            missing_features = [col for col in feature_columns if col not in data.columns]
            if missing_features:
                raise KeyError(f"Missing feature columns: {missing_features}")
            
            # Extract features and targets
            features = data[feature_columns].values
            targets = data[target_column].values
            
            # Create sequences
            sequences = []
            sequence_targets = []
            
            for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
                # Input sequence
                seq = features[i:i + self.sequence_length]
                sequences.append(seq)
                
                # Target value (prediction_horizon steps ahead)
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                sequence_targets.append(targets[target_idx])
            
            sequences = np.array(sequences)
            sequence_targets = np.array(sequence_targets)
            
            logger.info(f"Created {len(sequences)} sequences of shape {sequences.shape}")
            
            return sequences, sequence_targets
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            raise


def load_and_process_data(symbol: str, 
                         data_config: Any,
                         data_dir: str = "data/raw") -> Dict[str, Any]:
    """
    Load and process stock data for training.
    
    This is the main entry point for data preprocessing, combining
    all preprocessing steps into a single function.
    
    Args:
        symbol (str): Stock symbol to load
        data_config: Data configuration object
        data_dir (str): Directory containing raw data files
        
    Returns:
        Dict[str, Any]: Processed data and metadata
        
    Raises:
        FileNotFoundError: If data file is not found
        ValueError: If data processing fails
    """
    try:
        # Load raw data - try multiple filename patterns
        data_path = Path(data_dir) / f"{symbol}.csv"
        if not data_path.exists():
            # Try alternative naming patterns
            data_path = Path(data_dir) / f"{symbol}_latest.csv"
            if not data_path.exists():
                # List available files for debugging
                available_files = list(Path(data_dir).glob(f"{symbol}*.csv"))
                if available_files:
                    # Use the most recent file
                    data_path = sorted(available_files)[-1]
                    logger.info(f"Using data file: {data_path}")
                else:
                    raise FileNotFoundError(f"Data file not found: {data_path}. No files matching {symbol}*.csv in {data_dir}")
        
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} rows of data for {symbol} from {data_path}")
        
        # Initialize processors
        feature_engineer = FeatureEngineer(data_config)
        preprocessor = DataPreprocessor(data_config)
        
        # Validate initial data
        validation_results = preprocessor.validate_data(data)
        if not validation_results['valid']:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")
        
        # Handle timestamp column
        if 'timestamp' in data.columns:
            # Convert timestamp to datetime if not already
            # Use utc=True to handle mixed timezones
            data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
            # Convert to naive datetime (remove timezone info)
            data['timestamp'] = data['timestamp'].dt.tz_localize(None)
        elif 'Date' in data.columns:
            data['timestamp'] = pd.to_datetime(data['Date'])
        else:
            data['timestamp'] = pd.to_datetime(data.index)
        
        logger.info(f"Timestamp column processed: {data['timestamp'].dtype}")
        
        # Feature engineering
        if hasattr(data_config, 'include_technical_indicators') and data_config.include_technical_indicators:
            data = TechnicalIndicators.add_moving_averages(data)
            data = TechnicalIndicators.add_momentum_indicators(data)
            data = TechnicalIndicators.add_volatility_indicators(data)
            data = TechnicalIndicators.add_volume_indicators(data)
            data = TechnicalIndicators.add_trend_indicators(data)
        
        # Add engineered features
        data = feature_engineer.add_price_features(data)
        data = feature_engineer.add_time_features(data)
        
        # Remove rows with NaN values created by indicators
        initial_length = len(data)
        data = data.dropna()
        final_length = len(data)
        
        if final_length < initial_length:
            logger.info(f"Dropped {initial_length - final_length} rows with NaN values")
        
        # Split data
        train_size = int(len(data) * data_config.train_split)
        val_size = int(len(data) * data_config.val_split)
        
        train_data = data[:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # Normalize features
        train_normalized, val_normalized, test_normalized = preprocessor.normalize_features(
            train_data, val_data, test_data
        )
        
        # Create sequences
        sequence_generator = SequenceGenerator(
            data_config.sequence_length,
            data_config.prediction_horizon
        )
        
        feature_columns = data_config.features if hasattr(data_config, 'features') else None
        
        train_sequences, train_targets = sequence_generator.create_sequences(
            train_normalized, data_config.target_column, feature_columns
        )
        val_sequences, val_targets = sequence_generator.create_sequences(
            val_normalized, data_config.target_column, feature_columns
        )
        test_sequences, test_targets = sequence_generator.create_sequences(
            test_normalized, data_config.target_column, feature_columns
        )
        
        # Prepare result
        result = {
            'X_train': train_sequences,
            'y_train': train_targets,
            'X_val': val_sequences,
            'y_val': val_targets,
            'X_test': test_sequences,
            'y_test': test_targets,
            'train_sequences': train_sequences,
            'train_targets': train_targets,
            'val_sequences': val_sequences,
            'val_targets': val_targets,
            'test_sequences': test_sequences,
            'test_targets': test_targets,
            'scaler': preprocessor.scalers.get('feature_scaler'),
            'feature_columns': feature_columns or data.select_dtypes(include=[np.number]).columns.tolist(),
            'raw_data': data,
            'train_data': train_normalized,
            'val_data': val_normalized,
            'test_data': test_normalized,
            'validation_results': validation_results,
            'metadata': {
                'symbol': symbol,
                'total_samples': len(data),
                'train_samples': len(train_sequences),
                'val_samples': len(val_sequences),
                'test_samples': len(test_sequences),
                'sequence_length': data_config.sequence_length,
                'prediction_horizon': data_config.prediction_horizon,
                'feature_count': train_sequences.shape[2] if len(train_sequences.shape) > 2 else 0
            }
        }
        
        logger.info(f"Successfully processed data for {symbol}")
        logger.info(f"Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)} sequences")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing data for {symbol}: {e}")
        raise
