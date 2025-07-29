#!/usr/bin/env python3
"""
Stock price prediction pipeline using trained LSTM models.

This module provides comprehensive prediction functionality including
model loading, inference, visualization, and evaluation.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add the src directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import click
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.settings import (
    get_model_config, 
    list_available_models,
    list_available_symbols,
    get_device,
    get_model_save_path
)
from models.lstm_models import create_model
from data.preprocessing import load_and_process_data, FeatureEngineer
from utils.logging import get_logger

logger = get_logger(__name__)


class StockPredictor:
    """Stock price predictor using trained LSTM models."""
    
    def __init__(self, model_path: str, device: torch.device):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Computing device
        """
        self.device = device
        self.model = None
        self.model_config = None
        self.data_config = None
        self.scaler = None
        
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model from checkpoint."""
        checkpoint_path = Path(model_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # If it's a directory, look for best_model.pt inside
        if checkpoint_path.is_dir():
            best_model_path = checkpoint_path / "best_model.pt"
            if best_model_path.exists():
                checkpoint_path = best_model_path
            else:
                # Look for the latest checkpoint file
                checkpoint_files = list(checkpoint_path.glob("checkpoint-*.pt"))
                if checkpoint_files:
                    # Sort by checkpoint number and get the latest one
                    checkpoint_files.sort(key=lambda x: int(x.stem.split('-')[1]))
                    checkpoint_path = checkpoint_files[-1]
                    logger.info(f"Using latest checkpoint: {checkpoint_path.name}")
                else:
                    raise FileNotFoundError(f"No model files found in directory: {model_path}")
        
        logger.info(f"Loading model from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model_config = checkpoint['model_config']
        self.data_config = checkpoint['data_config']
        
        # Create and load model
        self.model = create_model(self.model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully: {self.model_config.name}")
    
    def predict_single(self, input_sequence: np.ndarray) -> float:
        """
        Make a single prediction.
        
        Args:
            input_sequence: Input sequence of shape (sequence_length, features)
            
        Returns:
            Predicted value
        """
        with torch.no_grad():
            # Add batch dimension
            input_tensor = torch.FloatTensor(input_sequence).unsqueeze(0).to(self.device)
            prediction = self.model(input_tensor)
            return prediction.cpu().numpy()[0, 0]
    
    def predict_sequence(self, 
                        initial_sequence: np.ndarray, 
                        num_predictions: int,
                        use_recursive: bool = True) -> np.ndarray:
        """
        Make multiple predictions.
        
        Args:
            initial_sequence: Initial sequence of shape (sequence_length, features)
            num_predictions: Number of future predictions to make
            use_recursive: Whether to use recursive prediction (feed predictions back)
            
        Returns:
            Array of predictions
        """
        predictions = []
        current_sequence = initial_sequence.copy()
        
        for _ in range(num_predictions):
            # Make prediction
            pred = self.predict_single(current_sequence)
            predictions.append(pred)
            
            if use_recursive and len(current_sequence) > 0:
                # Update sequence for next prediction
                # Shift sequence and add prediction (simplified - assumes target is close price)
                new_row = current_sequence[-1].copy()
                new_row[3] = pred  # Assuming close price is at index 3 (OHLCV)
                
                # Shift sequence
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = new_row
        
        return np.array(predictions)
    
    def evaluate_predictions(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate prediction performance.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Directional accuracy
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(true_direction == pred_direction) * 100
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }
    
    def predict_on_test_data(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make predictions on test data.
        
        Args:
            processed_data: Processed data dictionary
            
        Returns:
            Predictions and evaluation metrics
        """
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        predictions = []
        
        logger.info(f"Making predictions on {len(X_test)} test samples...")
        
        for i in range(len(X_test)):
            pred = self.predict_single(X_test[i])
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Evaluate predictions
        metrics = self.evaluate_predictions(y_test, predictions)
        
        logger.info("Test Results:")
        logger.info(f"  MAE: {metrics['mae']:.6f}")
        logger.info(f"  RMSE: {metrics['rmse']:.6f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  Directional Accuracy: {metrics['directional_accuracy']:.2f}%")
        
        return {
            'predictions': predictions,
            'actuals': y_test,
            'metrics': metrics
        }


def create_prediction_plot(timestamps: List[datetime],
                          actual_prices: np.ndarray,
                          predicted_prices: np.ndarray,
                          symbol: str,
                          future_timestamps: List[datetime] = None,
                          future_predictions: np.ndarray = None) -> go.Figure:
    """
    Create an interactive plot for predictions.
    
    Args:
        timestamps: Historical timestamps
        actual_prices: Historical actual prices
        predicted_prices: Historical predicted prices
        symbol: Stock symbol
        future_timestamps: Future timestamps for predictions
        future_predictions: Future price predictions
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{symbol} Price Prediction', 'Prediction Error'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Main price plot
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actual_prices,
            mode='lines',
            name='Actual Price',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=predicted_prices,
            mode='lines',
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Future predictions if provided
    if future_timestamps is not None and future_predictions is not None:
        fig.add_trace(
            go.Scatter(
                x=future_timestamps,
                y=future_predictions,
                mode='lines+markers',
                name='Future Predictions',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
    
    # Error plot
    error = actual_prices - predicted_prices
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=error,
            mode='lines',
            name='Prediction Error',
            line=dict(color='orange', width=1),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add zero line for error
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price Prediction Results',
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Error ($)", row=2, col=1)
    
    return fig


def create_animated_prediction_plot(timestamps: List[datetime],
                                  actual_prices: np.ndarray,
                                  predicted_prices: np.ndarray,
                                  symbol: str,
                                  start_index: int = 0,
                                  fps: int = 10) -> go.Figure:
    """
    Create an animated plot showing real vs predicted prices progressing over time.
    
    Args:
        timestamps: List of timestamps
        actual_prices: Array of actual prices
        predicted_prices: Array of predicted prices
        symbol: Stock symbol
        start_index: Starting point in the dataset (default: 0)
        fps: Frames per second for animation (default: 10)
        
    Returns:
        Plotly animated figure
    """
    # Slice data from starting point
    timestamps = timestamps[start_index:]
    actual_prices = actual_prices[start_index:]
    predicted_prices = predicted_prices[start_index:]
    
    # Create frames for animation
    frames = []
    frame_duration = int(1000 / fps)  # Duration in milliseconds
    
    for i in range(1, len(timestamps) + 1):
        # Data up to current point
        current_timestamps = timestamps[:i]
        current_actual = actual_prices[:i]
        current_predicted = predicted_prices[:i]
        
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=current_timestamps,
                    y=current_actual,
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='#1f77b4', width=3),
                    showlegend=True if i == 1 else False
                ),
                go.Scatter(
                    x=current_timestamps,
                    y=current_predicted,
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='#ff7f0e', width=2, dash='dot'),
                    opacity=0.7,
                    showlegend=True if i == 1 else False
                ),
                # Current point markers
                go.Scatter(
                    x=[current_timestamps[-1]],
                    y=[current_actual[-1]],
                    mode='markers',
                    name='Current Actual',
                    marker=dict(color='#1f77b4', size=8, symbol='circle'),
                    showlegend=True if i == 1 else False
                ),
                go.Scatter(
                    x=[current_timestamps[-1]],
                    y=[current_predicted[-1]],
                    mode='markers',
                    name='Current Predicted',
                    marker=dict(color='#ff7f0e', size=8, symbol='diamond'),
                    showlegend=True if i == 1 else False
                )
            ],
            name=f"frame_{i}",
            layout=go.Layout(
                title=f"{symbol} Price Evolution - {current_timestamps[-1].strftime('%Y-%m-%d')}",
                annotations=[
                    dict(
                        x=current_timestamps[-1],
                        y=max(current_actual[-1], current_predicted[-1]) * 1.1,
                        text=f"Actual: ${current_actual[-1]:.2f}<br>Predicted: ${current_predicted[-1]:.2f}<br>Error: ${abs(current_actual[-1] - current_predicted[-1]):.2f}",
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor="#636363",
                        ax=20,
                        ay=-30,
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor="#636363",
                        borderwidth=1
                    )
                ]
            )
        )
        frames.append(frame)
    
    # Initial figure (empty)
    fig = go.Figure(
        data=[
            go.Scatter(x=[], y=[], mode='lines', name='Actual Price', 
                      line=dict(color='#1f77b4', width=3)),
            go.Scatter(x=[], y=[], mode='lines', name='Predicted Price', 
                      line=dict(color='#ff7f0e', width=2, dash='dot'), opacity=0.7),
            go.Scatter(x=[], y=[], mode='markers', name='Current Actual',
                      marker=dict(color='#1f77b4', size=8, symbol='circle')),
            go.Scatter(x=[], y=[], mode='markers', name='Current Predicted',
                      marker=dict(color='#ff7f0e', size=8, symbol='diamond'))
        ],
        frames=frames
    )
    
    # Add animation controls
    fig.update_layout(
        title=f"{symbol} Stock Price Prediction Animation",
        xaxis=dict(
            title="Date",
            range=[timestamps[0], timestamps[-1]],
            type='date'
        ),
        yaxis=dict(
            title="Price ($)",
            range=[min(min(actual_prices), min(predicted_prices)) * 0.95,
                   max(max(actual_prices), max(predicted_prices)) * 1.15]
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"frame": {"duration": frame_duration, "redraw": True},
                               "fromcurrent": True, "transition": {"duration": 50}}],
                        label="Play",
                        method="animate"
                    ),
                    dict(
                        args=[{"frame": {"duration": 0, "redraw": True},
                               "mode": "immediate", "transition": {"duration": 0}}],
                        label="Pause",
                        method="animate"
                    ),
                    dict(
                        args=[{"frame": {"duration": frame_duration // 2, "redraw": True},
                               "fromcurrent": True, "transition": {"duration": 25}}],
                        label="Fast",
                        method="animate"
                    ),
                    dict(
                        args=[{"frame": {"duration": frame_duration * 2, "redraw": True},
                               "fromcurrent": True, "transition": {"duration": 100}}],
                        label="Slow",
                        method="animate"
                    )
                ]),
                pad={"r": 10, "t": 87},
                showactive=False,
                x=0.011,
                xanchor="right",
                y=0,
                yanchor="top"
            ),
        ],
        sliders=[
            dict(
                active=0,
                yanchor="top",
                xanchor="left",
                currentvalue={
                    "font": {"size": 20},
                    "prefix": "Date: ",
                    "visible": True,
                    "xanchor": "right"
                },
                transition={"duration": 50, "easing": "cubic-in-out"},
                pad={"b": 10, "t": 50},
                len=0.9,
                x=0.1,
                y=0,
                steps=[
                    dict(
                        args=[
                            [f"frame_{k}"],
                            {
                                "frame": {"duration": frame_duration, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 50}
                            }
                        ],
                        label=timestamps[k-1].strftime('%Y-%m-%d'),
                        method="animate"
                    )
                    for k in range(1, len(timestamps) + 1)
                ]
            )
        ],
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def interactive_prediction_mode(predictor: StockPredictor, symbol: str, data_dir: str):
    """Interactive prediction mode."""
    print(f"\nInteractive Prediction Mode for {symbol}")
    print("=" * 50)
    
    try:
        # Load and process data
        processed_data = load_and_process_data(symbol, predictor.data_config, data_dir)
        
        while True:
            print(f"\nOptions:")
            print("1. Test on recent data")
            print("2. Make future predictions")
            print("3. Evaluate model performance")
            print("4. Create animated prediction visualization")
            print("5. Switch symbol")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                # Test on recent data
                days = int(input("Enter number of recent days to test (default 30): ") or 30)
                
                if len(processed_data['X_test']) < days:
                    print(f"Only {len(processed_data['X_test'])} test samples available")
                    days = len(processed_data['X_test'])
                
                recent_X = processed_data['X_test'][-days:]
                recent_y = processed_data['y_test'][-days:]
                
                predictions = []
                for i in range(len(recent_X)):
                    pred = predictor.predict_single(recent_X[i])
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                metrics = predictor.evaluate_predictions(recent_y, predictions)
                
                print(f"\nRecent {days} days performance:")
                print(f"MAE: ${metrics['mae']:.2f}")
                print(f"RMSE: ${metrics['rmse']:.2f}")
                print(f"MAPE: {metrics['mape']:.2f}%")
                print(f"Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
                
            elif choice == "2":
                # Make future predictions
                days = int(input("Enter number of days to predict (default 7): ") or 7)
                
                # Use last sequence from test data
                last_sequence = processed_data['X_test'][-1]
                future_preds = predictor.predict_sequence(last_sequence, days)
                
                print(f"\nPredicted prices for next {days} days:")
                for i, pred in enumerate(future_preds, 1):
                    print(f"Day {i}: ${pred:.2f}")
                
            elif choice == "3":
                # Full evaluation
                results = predictor.predict_on_test_data(processed_data)
                
                # Create and save plot
                test_data = processed_data['test_data']
                timestamps = test_data['timestamp'].iloc[:len(results['actuals'])].tolist()
                
                fig = create_prediction_plot(
                    timestamps,
                    results['actuals'],
                    results['predictions'],
                    symbol
                )
                
                plot_path = f"results/{symbol}_prediction_plot.html"
                Path("results").mkdir(exist_ok=True)
                fig.write_html(plot_path)
                print(f"\nPlot saved to: {plot_path}")
                
            elif choice == "4":
                # Create animated visualization
                results = predictor.predict_on_test_data(processed_data)
                
                # Get animation parameters
                start_idx = int(input("Enter starting point in dataset (default 0): ") or 0)
                fps = int(input("Enter animation speed in FPS (default 10): ") or 10)
                
                test_data = processed_data['test_data']
                timestamps = test_data['timestamp'].iloc[:len(results['actuals'])].tolist()
                
                # Ensure start_index is within bounds
                max_start = len(timestamps) - 10  # At least 10 frames
                start_idx = min(start_idx, max(0, max_start))
                
                print(f"Creating animated visualization from index {start_idx} at {fps} FPS...")
                
                animated_fig = create_animated_prediction_plot(
                    timestamps,
                    results['actuals'],
                    results['predictions'],
                    symbol,
                    start_index=start_idx,
                    fps=fps
                )
                
                animated_path = f"results/{symbol}_animated_prediction.html"
                Path("results").mkdir(exist_ok=True)
                animated_fig.write_html(animated_path)
                print(f"Animated plot saved to: {animated_path}")
                print("Open the HTML file in a web browser to view the animation!")
                
            elif choice == "5":
                # Switch symbol
                print(f"Available symbols: {', '.join(list_available_symbols())}")
                new_symbol = input("Enter new symbol: ").strip().upper()
                
                if new_symbol in list_available_symbols():
                    symbol = new_symbol
                    try:
                        processed_data = load_and_process_data(symbol, predictor.data_config, data_dir)
                        print(f"Switched to {symbol}")
                    except Exception as e:
                        print(f"Error loading data for {symbol}: {str(e)}")
                else:
                    print("Invalid symbol")
                    
            elif choice == "6":
                print("Goodbye!")
                break
            else:
                print("Invalid choice, please try again")
                
    except Exception as e:
        logger.error(f"Error in interactive mode: {str(e)}")


@click.command()
@click.option('--model', '-m', default='lstm-small', help='Model name to use for prediction')
@click.option('--symbol', '-s', default='AAPL', help='Stock symbol to predict')
@click.option('--symbols', multiple=True, help='Multiple symbols to predict')
@click.option('--config', '-c', default='development', help='Training configuration used')
@click.option('--days', '-d', type=int, default=30, help='Number of days to predict')
@click.option('--data-dir', default='data/raw', help='Directory containing data files')
@click.option('--model-dir', default='models', help='Directory containing trained models')
@click.option('--output-dir', default='results', help='Directory to save results')
@click.option('--plot/--no-plot', default=True, help='Generate prediction plots')
@click.option('--animate', is_flag=True, help='Create animated prediction visualization')
@click.option('--start-index', type=int, default=0, help='Starting point in dataset for animation (default: 0)')
@click.option('--fps', type=int, default=10, help='Frames per second for animation (default: 10)')
@click.option('--interactive', is_flag=True, help='Run in interactive mode')
@click.option('--evaluate', is_flag=True, help='Evaluate model on test data')
@click.option('--list-models', is_flag=True, help='List available models')
@click.option('--list-symbols', is_flag=True, help='List available symbols')
def main(model, symbol, symbols, config, days, data_dir, model_dir, output_dir, 
         plot, animate, start_index, fps, interactive, evaluate, list_models, list_symbols):
    """Make predictions using trained LSTM models."""
    
    # List options
    if list_models:
        print("Available models:")
        for model_name in list_available_models():
            model_path = Path(model_dir) / get_model_save_path(model_name, "AAPL", "development")
            exists = "✓" if model_path.exists() else "✗"
            print(f"  {exists} {model_name}")
        return
    
    if list_symbols:
        print("Available symbols:")
        for sym in list_available_symbols():
            print(f"  {sym}")
        return
    
    # Setup
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine symbols to predict
    if symbols:
        symbols_to_predict = list(symbols)
    else:
        symbols_to_predict = [symbol]
    
    # Process each symbol
    for sym in symbols_to_predict:
        logger.info(f"Making predictions for symbol: {sym}")
        
        try:
            # Load model
            model_path = Path(model_dir) / get_model_save_path(model, sym, config)
            predictor = StockPredictor(str(model_path), device)
            
            if interactive:
                interactive_prediction_mode(predictor, sym, data_dir)
                return
            
            # Load and process data
            processed_data = load_and_process_data(sym, predictor.data_config, data_dir)
            
            if evaluate:
                # Evaluate on test data
                results = predictor.predict_on_test_data(processed_data)
                
                if plot:
                    # Create evaluation plot
                    test_data = processed_data['test_data']
                    timestamps = test_data['timestamp'].iloc[:len(results['actuals'])].tolist()
                    
                    fig = create_prediction_plot(
                        timestamps,
                        results['actuals'],
                        results['predictions'],
                        sym
                    )
                    
                    plot_path = Path(output_dir) / f"{sym}_evaluation_plot.html"
                    fig.write_html(plot_path)
                    logger.info(f"Evaluation plot saved to: {plot_path}")
                
                # Create animated visualization if requested
                if animate:
                    logger.info(f"Creating animated prediction visualization...")
                    test_data = processed_data['test_data']
                    timestamps = test_data['timestamp'].iloc[:len(results['actuals'])].tolist()
                    
                    # Ensure start_index is within bounds
                    max_start = len(timestamps) - 10  # At least 10 frames
                    start_idx = min(start_index, max(0, max_start))
                    
                    animated_fig = create_animated_prediction_plot(
                        timestamps,
                        results['actuals'],
                        results['predictions'],
                        sym,
                        start_index=start_idx,
                        fps=fps
                    )
                    
                    animated_path = Path(output_dir) / f"{sym}_animated_prediction.html"
                    animated_fig.write_html(animated_path)
                    logger.info(f"Animated prediction plot saved to: {animated_path}")
                    logger.info(f"Animation starts from index {start_idx} with {fps} FPS")
            
            # Make future predictions
            if days > 0:
                logger.info(f"Making {days} day predictions...")
                
                # Use last sequence from test data
                last_sequence = processed_data['X_test'][-1]
                future_predictions = predictor.predict_sequence(last_sequence, days)
                
                # Create future timestamps (business days only)
                last_date = processed_data['test_data']['timestamp'].iloc[-1]
                future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=days)
                
                # Save predictions
                predictions_df = pd.DataFrame({
                    'date': future_dates,
                    'predicted_price': future_predictions
                })
                
                predictions_path = Path(output_dir) / f"{sym}_predictions_{days}days.csv"
                predictions_df.to_csv(predictions_path, index=False)
                logger.info(f"Predictions saved to: {predictions_path}")
                
                # Display predictions
                print(f"\n{sym} Price Predictions:")
                print("=" * 30)
                for date, price in zip(future_dates, future_predictions):
                    print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
                
                if plot:
                    # Create future prediction plot
                    recent_data = processed_data['test_data'].tail(60)  # Last 60 days for context
                    
                    fig = create_prediction_plot(
                        recent_data['timestamp'].tolist(),
                        recent_data['close'].values,
                        recent_data['close'].values,  # For historical context
                        sym,
                        future_dates.tolist(),
                        future_predictions
                    )
                    
                    plot_path = Path(output_dir) / f"{sym}_future_predictions.html"
                    fig.write_html(plot_path)
                    logger.info(f"Future predictions plot saved to: {plot_path}")
            
        except Exception as e:
            logger.error(f"Error making predictions for {sym}: {str(e)}")
            continue
    
    logger.info("Prediction pipeline completed")


if __name__ == "__main__":
    main()
