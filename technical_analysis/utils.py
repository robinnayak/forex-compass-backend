import pandas as pd
from typing import Dict, Any
from .calculators import TrendCalculator, MomentumCalculator, VolatilityCalculator
from .signals.signal_generator import SignalGenerator
from .data_quality.processor import DataQualityProcessor

def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """Validate OHLCV data structure"""
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    return all(col in data.columns for col in required_columns)

def calculate_all_indicators(data: pd.DataFrame, time_frame: str) -> Dict[str, Any]:
    """
    Calculate all available indicators for given data, adapting parameters to timeframe.
    Automatically cleans and refines data before calculation.
    Adds debug output for troubleshooting.
    """
    print(f"Data testing before: {data.shape}")
    print(f"Date type: {data['Open'].dtype if 'Open' in data.columns else 'N/A'}")
    print(f"Initial DataFrame shape: {data.shape}")
    print(f"Initial columns: {data.columns.tolist()}")
    # Initial cleaning and duplicate removal
    data = DataQualityProcessor.clean_ohlcv_data(data)
    print(f"After clean_ohlcv_data: {data.shape}")
    print("NaN counts after initial cleaning:")
    for col in data.columns:
        print(f"  {col}: {data[col].isna().sum()}")
    # Refine data for quality and missing values
    data = DataQualityProcessor.refine_data(
        data,
        columns_to_clean=['Open', 'High', 'Low', 'Close'],
        missing_method='ffill',
        remove_duplicates=True,
        outlier_method='zscore',
        outlier_threshold=3.0,
        min_length=10,
        fill_volume_with_zero=True,
        verbose=True
    )
    print(f"Data testing after: {data.shape}")
    print("NaN counts after refinement:")
    for col in data.columns:
        print(f"  {col}: {data[col].isna().sum()}")
    if not validate_ohlcv_data(data):
        print("Invalid OHLCV data structure after cleaning.")
        raise ValueError("Invalid OHLCV data structure")
    results = {}
    # Trend indicators (dynamic periods)
    ma_params = TrendCalculator.get_ma_params(time_frame)
    for period in ma_params['sma_periods']:
        print(f"Calculating SMA for period {period}")
        results[f'sma_{period}'] = TrendCalculator.sma(data['Close'], period)
    for period in ma_params['ema_periods']:
        print(f"Calculating EMA for period {period}")
        results[f'ema_{period}'] = TrendCalculator.ema(data['Close'], period)
    # MACD (use default or custom periods if available)
    if hasattr(MomentumCalculator, 'macd'):
        print("Calculating MACD")
        try:
            results['macd'] = MomentumCalculator.macd(data['Close'])
        except Exception as e:
            print(f"MACD calculation error: {e}")
            results['macd'] = None
    # Momentum indicators
    if hasattr(MomentumCalculator, 'rsi'):
        print("Calculating RSI")
        try:
            results['rsi'] = MomentumCalculator.rsi(data['Close'])
        except Exception as e:
            print(f"RSI calculation error: {e}")
            results['rsi'] = None
    if hasattr(MomentumCalculator, 'stochastic'):
        print("Calculating Stochastic")
        try:
            results['stochastic'] = MomentumCalculator.stochastic(
                data['High'], data['Low'], data['Close']
            )
        except Exception as e:
            print(f"Stochastic calculation error: {e}")
            results['stochastic'] = None
    # Volatility indicators
    if hasattr(VolatilityCalculator, 'bollinger_bands'):
        print("Calculating Bollinger Bands")
        try:
            results['bollinger_bands'] = VolatilityCalculator.bollinger_bands(data['Close'])
        except Exception as e:
            print(f"Bollinger Bands calculation error: {e}")
            results['bollinger_bands'] = None
    if hasattr(VolatilityCalculator, 'atr'):
        print("Calculating ATR")
        try:
            results['atr'] = VolatilityCalculator.atr(
                data['High'], data['Low'], data['Close']
            )
        except Exception as e:
            print(f"ATR calculation error: {e}")
            results['atr'] = None
    # Signals
    if hasattr(SignalGenerator, 'generate_signals'):
        print("Generating signals")
        try:
            results['signals'] = SignalGenerator.generate_signals(data)
        except Exception as e:
            print(f"Signal generation error: {e}")
            results['signals'] = None
    results['time_frame'] = time_frame
    results['data_length'] = len(data)
    print("Indicator calculation complete.")
    return results