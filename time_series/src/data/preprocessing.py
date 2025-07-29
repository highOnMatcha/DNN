"""Data preprocessing and feature engineering for stock price prediction."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import ta
from pathlib import Path

from utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """Calculate technical indicators for stock data."""
    
    @staticmethod
    def add_moving_averages(data: pd.DataFrame) -> pd.DataFrame:
        """Add moving average indicators."""
        data = data.copy()
        
        for window in [5, 10, 20, 50, 100, 200]:
            data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
        
        for span in [12, 26, 50]:
            data[f'ema_{span}'] = data['close'].ewm(span=span).mean()
        
        return data
    
    @staticmethod
    def add_momentum_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators."""
        data = data.copy()
        
        data['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
        
        macd = ta.trend.MACD(data['close'])
        data['macd'] = macd.macd()
        data['macd_signal'] = macd.macd_signal()
        data['macd_hist'] = macd.macd_diff()
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        data['williams_r'] = ta.momentum.WilliamsRIndicator(data['high'], data['low'], data['close']).williams_r()
        
        # Commodity Channel Index
        data['cci'] = ta.trend.CCIIndicator(data['high'], data['low'], data['close']).cci()
        
        return data
    
    @staticmethod
    def add_volatility_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators."""
        data = data.copy()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(data['close'])
        data['bb_upper'] = bb.bollinger_hband()
        data['bb_middle'] = bb.bollinger_mavg()
        data['bb_lower'] = bb.bollinger_lband()
        data['bb_width'] = bb.bollinger_wband()
        data['bb_percent'] = bb.bollinger_pband()
        
        # Average True Range
        data['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
        
        # Keltner Channel
        kc = ta.volatility.KeltnerChannel(data['high'], data['low'], data['close'])
        data['kc_upper'] = kc.keltner_channel_hband()
        data['kc_middle'] = kc.keltner_channel_mband()
        data['kc_lower'] = kc.keltner_channel_lband()
        
        return data
    
    @staticmethod
    def add_volume_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        data = data.copy()
        
        # On-Balance Volume
        data['obv'] = ta.volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
        
        # Volume Weighted Average Price
        data['vwap'] = ta.volume.VolumeWeightedAveragePrice(
            high=data['high'], low=data['low'], close=data['close'], volume=data['volume']
        ).volume_weighted_average_price()
        
        # Money Flow Index
        data['mfi'] = ta.volume.MFIIndicator(data['high'], data['low'], data['close'], data['volume']).money_flow_index()
        
        # Accumulation/Distribution Line
        data['ad_line'] = ta.volume.AccDistIndexIndicator(data['high'], data['low'], data['close'], data['volume']).acc_dist_index()
        
        # Chaikin Money Flow
        data['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(data['high'], data['low'], data['close'], data['volume']).chaikin_money_flow()
        
        return data
    
    @staticmethod
    def add_trend_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based indicators."""
        data = data.copy()
        
        # Parabolic SAR
        data['psar'] = ta.trend.PSARIndicator(data['high'], data['low'], data['close']).psar()
        
        # Average Directional Index
        adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'])
        data['adx'] = adx.adx()
        data['adx_pos'] = adx.adx_pos()
        data['adx_neg'] = adx.adx_neg()
        
        # Aroon
        aroon = ta.trend.AroonIndicator(data['high'], data['low'])
        data['aroon_up'] = aroon.aroon_up()
        data['aroon_down'] = aroon.aroon_down()
        data['aroon_indicator'] = aroon.aroon_indicator()
        
        # Ichimoku
        ichimoku = ta.trend.IchimokuIndicator(data['high'], data['low'])
        data['ichimoku_a'] = ichimoku.ichimoku_a()
        data['ichimoku_b'] = ichimoku.ichimoku_b()
        data['ichimoku_base'] = ichimoku.ichimoku_base_line()
        data['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        return data


class FeatureEngineer:
    """Feature engineering for stock price data."""
    
    def __init__(self, config):
        """
        Initialize feature engineer.
        
        Args:
            config: Data configuration object
        """
        self.config = config
        self.scalers = {}
        
    def add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        data = data.copy()
        
        # Price changes and returns
        data['price_change'] = data['close'].diff()
        data['price_change_pct'] = data['close'].pct_change()
        data['high_low_pct'] = (data['high'] - data['low']) / data['close']
        data['open_close_pct'] = (data['close'] - data['open']) / data['open']
        
        # Log returns
        data['log_return'] = np.log(data['close'] / data['close'].shift(1))
        
        # Price position within the day's range
        data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
        
        # Gap analysis
        data['gap'] = data['open'] - data['close'].shift(1)
        data['gap_pct'] = data['gap'] / data['close'].shift(1)
        
        # Price acceleration
        data['price_acceleration'] = data['price_change'].diff()
        
        return data
    
    def add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        data = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Time components
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['day_of_month'] = data['timestamp'].dt.day
        data['week_of_year'] = data['timestamp'].dt.isocalendar().week
        data['month'] = data['timestamp'].dt.month
        data['quarter'] = data['timestamp'].dt.quarter
        data['year'] = data['timestamp'].dt.year
        
        # Cyclical encoding for time features
        data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Market session indicators (assuming US market hours)
        data['hour'] = data['timestamp'].dt.hour
        data['is_market_open'] = ((data['hour'] >= 9) & (data['hour'] <= 16) & 
                                 (data['day_of_week'] < 5)).astype(int)
        
        return data
    
    def add_lag_features(self, data: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Add lagged features."""
        data = data.copy()
        
        if lags is None:
            lags = [1, 2, 3, 5, 10, 20]
        
        base_features = ['close', 'volume', 'price_change_pct', 'log_return']
        
        for feature in base_features:
            if feature in data.columns:
                for lag in lags:
                    data[f'{feature}_lag_{lag}'] = data[feature].shift(lag)
        
        return data
    
    def add_rolling_features(self, data: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Add rolling window features."""
        data = data.copy()
        
        if windows is None:
            windows = [5, 10, 20, 50]
        
        base_features = ['close', 'volume', 'price_change_pct', 'log_return']
        
        for feature in base_features:
            if feature in data.columns:
                for window in windows:
                    data[f'{feature}_mean_{window}'] = data[feature].rolling(window).mean()
                    data[f'{feature}_std_{window}'] = data[feature].rolling(window).std()
                    data[f'{feature}_min_{window}'] = data[feature].rolling(window).min()
                    data[f'{feature}_max_{window}'] = data[feature].rolling(window).max()
                    data[f'{feature}_median_{window}'] = data[feature].rolling(window).median()
        
        return data
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators based on configuration."""
        data = data.copy()
        
        if not self.config.include_technical_indicators:
            return data
        
        indicators = TechnicalIndicators()
        
        # Add different types of indicators
        data = indicators.add_moving_averages(data)
        data = indicators.add_momentum_indicators(data)
        data = indicators.add_volatility_indicators(data)
        data = indicators.add_volume_indicators(data)
        data = indicators.add_trend_indicators(data)
        
        # Filter to only include specified indicators
        if self.config.technical_indicators:
            available_indicators = [col for col in data.columns 
                                  if col in self.config.technical_indicators]
            missing_indicators = [ind for ind in self.config.technical_indicators 
                                if ind not in data.columns]
            
            if missing_indicators:
                logger.warning(f"Missing technical indicators: {missing_indicators}")
        
        return data
    
    def normalize_features(self, train_data: pd.DataFrame, 
                          val_data: pd.DataFrame = None,
                          test_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, ...]:
        """
        Normalize features using specified method.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            test_data: Test data (optional)
            
        Returns:
            Tuple of normalized datasets
        """
        # Select features to normalize (exclude timestamp and symbol)
        exclude_cols = ['timestamp', 'symbol']
        feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        # Choose scaler based on configuration
        if self.config.normalization_method == 'minmax':
            scaler = MinMaxScaler()
        elif self.config.normalization_method == 'standard':
            scaler = StandardScaler()
        elif self.config.normalization_method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.config.normalization_method}")
        
        # Fit scaler on training data
        train_features = train_data[feature_cols].values
        scaler.fit(train_features)
        self.scalers['features'] = scaler
        
        # Transform data
        train_normalized = train_data.copy()
        train_normalized[feature_cols] = scaler.transform(train_features)
        
        results = [train_normalized]
        
        if val_data is not None:
            val_normalized = val_data.copy()
            val_normalized[feature_cols] = scaler.transform(val_data[feature_cols].values)
            results.append(val_normalized)
        
        if test_data is not None:
            test_normalized = test_data.copy()
            test_normalized[feature_cols] = scaler.transform(test_data[feature_cols].values)
            results.append(test_normalized)
        
        return tuple(results)
    
    def create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (sequences, targets)
        """
        # Select feature columns
        feature_cols = [col for col in self.config.features if col in data.columns]
        target_col = self.config.target_column
        
        if not feature_cols:
            raise ValueError("No valid feature columns found in data")
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        features = data[feature_cols].values
        targets = data[target_col].values
        
        sequences = []
        sequence_targets = []
        
        for i in range(len(data) - self.config.sequence_length - self.config.prediction_horizon + 1):
            # Input sequence
            seq = features[i:i + self.config.sequence_length]
            
            # Target (future value)
            target_idx = i + self.config.sequence_length + self.config.prediction_horizon - 1
            target = targets[target_idx]
            
            sequences.append(seq)
            sequence_targets.append(target)
        
        return np.array(sequences), np.array(sequence_targets)
    
    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n_samples = len(data)
        
        # Calculate split indices
        train_end = int(n_samples * self.config.train_split)
        val_end = int(n_samples * (self.config.train_split + self.config.val_split))
        
        # Split data
        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end:].copy()
        
        logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete data processing pipeline.
        
        Args:
            data: Raw stock data
            
        Returns:
            Dictionary containing processed data and metadata
        """
        logger.info("Starting data processing pipeline")
        
        # Add all features
        data = self.add_price_features(data)
        data = self.add_time_features(data)
        data = self.add_lag_features(data)
        data = self.add_rolling_features(data)
        data = self.add_technical_indicators(data)
        
        # Remove rows with NaN values (created by indicators and lags)
        initial_length = len(data)
        data = data.dropna()
        final_length = len(data)
        
        logger.info(f"Removed {initial_length - final_length} rows with NaN values")
        
        if len(data) == 0:
            raise ValueError("No data remaining after feature engineering")
        
        # Split data
        train_data, val_data, test_data = self.split_data(data)
        
        # Normalize features
        train_norm, val_norm, test_norm = self.normalize_features(train_data, val_data, test_data)
        
        # Create sequences
        X_train, y_train = self.create_sequences(train_norm)
        X_val, y_val = self.create_sequences(val_norm)
        X_test, y_test = self.create_sequences(test_norm)
        
        logger.info(f"Created sequences - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'train_data': train_norm,
            'val_data': val_norm,
            'test_data': test_norm,
            'feature_columns': [col for col in self.config.features if col in data.columns],
            'scalers': self.scalers
        }


def load_and_process_data(symbol: str, config, data_dir: str = "data/raw") -> Dict[str, Any]:
    """
    Load and process stock data for a symbol.
    
    Args:
        symbol: Stock symbol
        config: Data configuration
        data_dir: Directory containing raw data
        
    Returns:
        Processed data dictionary
    """
    # Load data
    data_path = Path(data_dir) / f"{symbol}_latest.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_csv(data_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True)
    
    logger.info(f"Loaded {len(data)} records for {symbol}")
    
    # Process data
    feature_engineer = FeatureEngineer(config)
    processed_data = feature_engineer.process_data(data)
    
    return processed_data
