import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Optional, Union
from technical_analysis.data_quality.processor import clean_indicator_data

class MomentumCalculator:
    """Calculate momentum indicators with timeframe-aware parameters"""
    
    @staticmethod
    def get_rsi_params(timeframe: Optional[str] = None) -> Dict[str, Union[int, float]]:
        """
        Get RSI parameters based on timeframe
        Returns: dict with 'period', 'overbought', 'oversold' levels
        """
        timeframe = timeframe.lower() if timeframe else None
        
        if timeframe in ['1m', '2m', '3m', '5m']:
            # Binary trading - very sensitive
            return {'period': 7, 'overbought': 70, 'oversold': 30}
        elif timeframe in ['15m', '30m', '1h']:
            # Intraday - faster signals
            return {'period': 9, 'overbought': 70, 'oversold': 30}
        elif timeframe in ['4h', '6h', '8h', '12h', '1d']:
            # Swing trading - standard
            return {'period': 14, 'overbought': 70, 'oversold': 30}
        elif timeframe in ['3d', '1w']:
            # Longer term - stricter levels
            return {'period': 14, 'overbought': 65, 'oversold': 35}
        else:
            # Default
            return {'period': 14, 'overbought': 70, 'oversold': 30}
    
    @staticmethod
    def rsi(series: pd.Series, 
            period: Optional[int] = None,
            timeframe: Optional[str] = None,
            use_pandas_ta: bool = True) -> pd.Series:
        """
        Calculate RSI using pandas_ta with timeframe-aware parameters
        
        Args:
            series: Price series (usually close)
            period: Specific period to use (overrides timeframe)
            timeframe: Timeframe string ('1m', '15m', '4h', '1d', etc.)
            use_pandas_ta: Whether to use pandas_ta library (faster)
        
        Returns:
            pd.Series: RSI values
        """
        # Get parameters based on timeframe if period not provided
        if period is None:
            params = MomentumCalculator.get_rsi_params(timeframe)
            period = params['period']
        if use_pandas_ta:
            # Use pandas_ta for optimized calculation
            rsi_values = ta.rsi(series, length=period)
            # Clean NaNs and outliers
            cleaned = clean_indicator_data({'rsi': rsi_values})
            return cleaned['rsi']
        else:
            # Custom implementation (fallback)
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rs = rs.replace([np.inf, -np.inf], np.nan).fillna(0)
            rsi_values = 100 - (100 / (1 + rs))
            cleaned = clean_indicator_data({'rsi': rsi_values})
            return cleaned['rsi']
    
    @staticmethod
    def rsi_with_levels(series: pd.Series, 
                       timeframe: Optional[str] = None,
                       custom_levels: Optional[Dict[str, int]] = None) -> Dict:
        """
        Calculate RSI with overbought/oversold levels using pandas_ta
        
        Returns:
            Dict containing RSI values and signal levels
        """
        # Get parameters
        params = MomentumCalculator.get_rsi_params(timeframe)
        
        # Override with custom levels if provided
        if custom_levels:
            params.update(custom_levels)
        
        rsi_values = MomentumCalculator.rsi(series, period=params['period'])
        
        return {
            'rsi': rsi_values,
            'period': params['period'],
            'overbought_level': params['overbought'],
            'oversold_level': params['oversold'],
            'is_overbought': rsi_values > params['overbought'],
            'is_oversold': rsi_values < params['oversold']
        }
    
    @staticmethod
    def get_macd_params(timeframe: Optional[str] = None) -> Dict[str, int]:
        """
        Get MACD parameters based on timeframe
        Returns: dict with 'fast', 'slow', 'signal' periods
        """
        timeframe = timeframe.lower() if timeframe else None
        
        if timeframe in ['1m', '2m', '3m', '5m']:
            # Binary trading - very fast, for short expiry
            return {'fast': 5, 'slow': 13, 'signal': 4}
        elif timeframe in ['15m', '30m', '1h']:
            # Intraday - faster signals
            return {'fast': 8, 'slow': 21, 'signal': 5}
        elif timeframe in ['4h', '6h', '8h', '12h', '1d']:
            # Swing trading - standard
            return {'fast': 12, 'slow': 26, 'signal': 9}
        elif timeframe in ['3d', '1w', '1M']:
            # Longer term - smoother
            return {'fast': 21, 'slow': 50, 'signal': 14}
        else:
            # Default
            return {'fast': 12, 'slow': 26, 'signal': 9}
    
    @staticmethod
    def macd(series: pd.Series, 
             fast: Optional[int] = None,
             slow: Optional[int] = None, 
             signal: Optional[int] = None,
             timeframe: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate MACD with timeframe-aware parameters using pandas_ta
        
        Args:
            series: Price series (usually close)
            fast: Fast EMA period (overrides timeframe)
            slow: Slow EMA period (overrides timeframe)
            signal: Signal line period (overrides timeframe)
            timeframe: Timeframe string for automatic parameter selection
        
        Returns:
            Dict containing MACD line, signal line, histogram, and signals
        """
        # Get parameters based on timeframe if not provided
        if any(param is None for param in [fast, slow, signal]):
            params = MomentumCalculator.get_macd_params(timeframe)
            fast = fast or params['fast']
            slow = slow or params['slow']
            signal = signal or params['signal']
        
        # Use pandas_ta for optimized calculation
        macd_result = ta.macd(series, fast=fast, slow=slow, signal=signal)
        macd_line = macd_result[f'MACD_{fast}_{slow}_{signal}']
        signal_line = macd_result[f'MACDs_{fast}_{slow}_{signal}']
        histogram = macd_result[f'MACDh_{fast}_{slow}_{signal}']
        # Clean all output series including histogram
        cleaned = clean_indicator_data({
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        })
        macd_line = cleaned['macd_line']
        signal_line = cleaned['signal_line']
        histogram = cleaned['histogram']
        
        # Generate trading signals
        if len(macd_line) >= 2:
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            prev_macd = macd_line.iloc[-2]
            prev_signal = signal_line.iloc[-2]
            
            # Signal line crossover
            bullish_crossover = (prev_macd < prev_signal) and (current_macd > current_signal)
            bearish_crossover = (prev_macd > prev_signal) and (current_macd < current_signal)
            
            # Zero line crossover
            above_zero = current_macd > 0
            below_zero = current_macd < 0
            
            # Histogram momentum
            histogram_rising = histogram.iloc[-1] > histogram.iloc[-2] if len(histogram) >= 2 else False
            histogram_falling = histogram.iloc[-1] < histogram.iloc[-2] if len(histogram) >= 2 else False
        else:
            bullish_crossover = bearish_crossover = above_zero = below_zero = False
            histogram_rising = histogram_falling = False
            current_macd = current_signal = 0
        
        output = {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram,
            'fast_period': fast,
            'slow_period': slow,
            'signal_period': signal,
            'bullish_crossover': bullish_crossover,
            'bearish_crossover': bearish_crossover,
            'above_zero_line': above_zero,
            'below_zero_line': below_zero,
            'histogram_rising': histogram_rising,
            'histogram_falling': histogram_falling,
            'signal_strength': abs(current_macd - current_signal) if len(macd_line) >= 2 else 0
        }
        # Only clean pd.Series outputs, keep bools/floats as is
        output = {**clean_indicator_data({k: v for k, v in output.items() if isinstance(v, pd.Series)}),
                  **{k: v for k, v in output.items() if not isinstance(v, pd.Series)}}
        return output
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                  k_period: int = 14, d_period: int = 3,
                  timeframe: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator using pandas_ta
        """
        # Adjust parameters based on timeframe for faster/slower signals
        if timeframe and timeframe in ['1m', '2m', '3m', '5m']:
            k_period = max(7, k_period - 5)  # Faster for binary trading
        elif timeframe and timeframe in ['15m', '30m', '1h']:
            k_period = max(9, k_period - 3)  # Faster for intraday
        # Use pandas_ta for stochastic calculation
        stoch_result = ta.stoch(high, low, close, k=k_period, d=d_period)
        print(f"Stochastic columns: {stoch_result.columns.tolist()}")  # Debug output
        k_key = f'STOCHk_{k_period}_{d_period}'
        d_key = f'STOCHd_{k_period}_{d_period}'
        # Fallback to generic column names if specific keys are missing
        if k_key not in stoch_result.columns:
            k_key = 'STOCHk'
        if d_key not in stoch_result.columns:
            d_key = 'STOCHd'
        if k_key not in stoch_result.columns or d_key not in stoch_result.columns:
            raise KeyError(f"Stochastic calculation failed: columns found {stoch_result.columns.tolist()}")
        output = {
            'k': stoch_result[k_key],
            'd': stoch_result[d_key],
            'k_period': k_period,
            'd_period': d_period,
            'is_overbought': stoch_result[k_key] > 80,
            'is_oversold': stoch_result[k_key] < 20
        }
        output = clean_indicator_data({k: v for k, v in output.items() if isinstance(v, pd.Series)}) | {k: v for k, v in output.items() if not isinstance(v, pd.Series)}
        return output
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, 
            period: int = 20, timeframe: Optional[str] = None) -> pd.Series:
        """
        Calculate Commodity Channel Index using pandas_ta
        """
        # Adjust period for different timeframes
        if timeframe and timeframe in ['1m', '2m', '3m', '5m']:
            period = max(14, period - 6)  # Shorter for faster signals
        elif timeframe and timeframe in ['15m', '30m', '1h']:
            period = max(16, period - 4)  # Slightly shorter for intraday
        
        # Use pandas_ta for CCI calculation
        return ta.cci(high, low, close, length=period)
    
    @staticmethod
    def all_momentum_indicators(data: pd.DataFrame, 
                               timeframe: Optional[str] = None) -> Dict:
        """
        Calculate all momentum indicators for a DataFrame with OHLCV data using pandas_ta
        """
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            raise ValueError("DataFrame must contain OHLC columns")
        
        close = data['close']
        high = data['high']
        low = data['low']
        
        return {
            'rsi': MomentumCalculator.rsi_with_levels(close, timeframe),
            'stochastic': MomentumCalculator.stochastic(high, low, close, timeframe=timeframe),
            'cci': MomentumCalculator.cci(high, low, close, timeframe=timeframe),
            'macd': MomentumCalculator.macd(close, timeframe=timeframe)
        }