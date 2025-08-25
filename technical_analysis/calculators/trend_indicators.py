import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
from technical_analysis.data_quality.processor import DataQualityProcessor, clean_indicator_data

class TrendCalculator:
    """Calculate trend-following indicators with timeframe-aware parameters"""
    
    @staticmethod
    def get_ma_params(timeframe: Optional[str] = None) -> Dict[str, List[int]]:
        """
        Get Moving Average parameters based on timeframe
        Returns: dict with EMA and SMA periods for different timeframes
        """
        timeframe = timeframe.lower() if timeframe else None
        
        if timeframe in ['1m', '2m', '3m', '5m', '15m', '30m']:
            # Intraday: 9 EMA & 21 EMA for short-term direction
            return {'ema_periods': [9, 21], 'sma_periods': [200]}
        elif timeframe in ['1h', '4h', '6h', '8h', '12h']:
            # Swing: 20 EMA & 50 EMA for medium trend
            return {'ema_periods': [20, 50], 'sma_periods': [200]}
        elif timeframe in ['1d', '3d', '1w']:
            # Long-term: 50 EMA & 200 SMA for overall bias
            return {'ema_periods': [50], 'sma_periods': [200]}
        else:
            # Default: General purpose
            return {'ema_periods': [20, 50], 'sma_periods': [200]}
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average using pandas_ta.
        Handles edge case: returns empty series if not enough data.
        """
        if len(series) < period:
            return pd.Series([np.nan]*len(series), index=series.index)
        result = ta.sma(series, length=period)
        # pandas_ta returns NaN for first period-1 values
        # Optionally fill or drop these NaNs downstream
        result = clean_indicator_data({'sma': result})['sma']
        return result
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average using pandas_ta.
        Handles edge case: returns empty series if not enough data.
        """
        if len(series) < period:
            return pd.Series([np.nan]*len(series), index=series.index)
        result = ta.ema(series, length=period)
        # pandas_ta returns NaN for first period-1 values
        # Optionally fill or drop these NaNs downstream
        result = clean_indicator_data({'ema': result})['ema']
        return result
    
    @staticmethod
    def moving_averages(series: pd.Series, 
                       timeframe: Optional[str] = None,
                       custom_periods: Optional[Dict[str, List[int]]] = None) -> Dict[str, pd.Series]:
        """
        Calculate multiple moving averages with timeframe-aware parameters
        
        Args:
            series: Price series (usually close)
            timeframe: Timeframe for automatic parameter selection
            custom_periods: Custom periods to override automatic selection
        
        Returns:
            Dict containing all moving averages with crossover signals
        """
        # Get parameters based on timeframe if not provided
        if custom_periods is None:
            params = TrendCalculator.get_ma_params(timeframe)
            ema_periods = params['ema_periods']
            sma_periods = params['sma_periods']
        else:
            ema_periods = custom_periods.get('ema_periods', [20, 50])
            sma_periods = custom_periods.get('sma_periods', [200])
        
        results = {}
        
        # Calculate EMAs
        for period in sorted(ema_periods):
            results[f'ema_{period}'] = TrendCalculator.ema(series, period)
        
        # Calculate SMAs
        for period in sorted(sma_periods):
            results[f'sma_{period}'] = TrendCalculator.sma(series, period)
        
        # Generate crossover signals if we have multiple EMAs
        if len(ema_periods) >= 2:
            fast_ema = results[f'ema_{ema_periods[0]}']
            slow_ema = results[f'ema_{ema_periods[1]}']
            
            # EMA crossover signals
            results['ema_bullish_crossover'] = (fast_ema.shift(1) <= slow_ema.shift(1)) & (fast_ema > slow_ema)
            results['ema_bearish_crossover'] = (fast_ema.shift(1) >= slow_ema.shift(1)) & (fast_ema < slow_ema)
        
        # Price vs MA signals
        if ema_periods:
            results['price_above_ema'] = series > results[f'ema_{ema_periods[0]}']
            results['price_below_ema'] = series < results[f'ema_{ema_periods[0]}']
        
        if sma_periods:
            results['price_above_sma'] = series > results[f'sma_{sma_periods[0]}']
            results['price_below_sma'] = series < results[f'sma_{sma_periods[0]}']
        
        return results
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float, 
                             levels: List[float] = None) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels
        
        Args:
            high: Swing high price
            low: Swing low price
            levels: Custom retracement levels (default: [0.236, 0.382, 0.5, 0.618, 0.786])
        
        Returns:
            Dict with Fibonacci retracement levels
        """
        if levels is None:
            levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        price_range = high - low
        retracement_levels = {}
        
        for level in levels:
            if high >= low:  # Uptrend
                retracement_price = high - (price_range * level)
            else:  # Downtrend
                retracement_price = low + (abs(price_range) * level)
            
            retracement_levels[f'fib_{int(level*100)}'] = retracement_price
        
        # Add extreme levels
        retracement_levels['fib_0'] = high if high >= low else low
        retracement_levels['fib_100'] = low if high >= low else high
        
        return retracement_levels
    
    @staticmethod
    def fibonacci_retracement_series(high_series: pd.Series, low_series: pd.Series,
                                    swing_high: float, swing_low: float,
                                    levels: List[float] = None) -> Dict[str, pd.Series]:
        """
        Apply Fibonacci retracement to price series
        
        Args:
            high_series: High price series
            low_series: Low price series
            swing_high: Identified swing high
            swing_low: Identified swing low
            levels: Custom retracement levels
        
        Returns:
            Dict with Fibonacci levels as constant series for plotting
        """
        fib_levels = TrendCalculator.fibonacci_retracement(swing_high, swing_low, levels)
        
        # Create constant series for each Fibonacci level
        results = {}
        for level_name, price in fib_levels.items():
            results[level_name] = pd.Series(price, index=high_series.index)
        
        return results
    
    @staticmethod
    def identify_swing_points(series: pd.Series, lookback: int = 5) -> Tuple[pd.Series, pd.Series]:
        """
        Identify swing highs and lows in price series
        
        Args:
            series: Price series
            lookback: Number of periods to look back for swing points
        
        Returns:
            Tuple of (swing_highs, swing_lows) boolean series
        """
        # Find local maxima (swing highs)
        swing_highs = (series == series.rolling(window=lookback*2+1, center=True).max())
        swing_highs = swing_highs & (series > series.shift(lookback)) & (series > series.shift(-lookback))
        
        # Find local minima (swing lows)
        swing_lows = (series == series.rolling(window=lookback*2+1, center=True).min())
        swing_lows = swing_lows & (series < series.shift(lookback)) & (series < series.shift(-lookback))
        
        return swing_highs, swing_lows
    
    @staticmethod
    def trend_strength(series: pd.Series, 
                      short_period: int = 20, 
                      long_period: int = 50) -> pd.Series:
        """
        Calculate trend strength based on moving average slope
        """
        short_ma = TrendCalculator.ema(series, short_period)
        long_ma = TrendCalculator.ema(series, long_period)
        
        # Slope of moving averages
        short_slope = short_ma.diff() / short_ma.shift(1)
        long_slope = long_ma.diff() / long_ma.shift(1)
        
        # Combined trend strength
        trend_strength = (short_slope.rolling(5).mean() + long_slope.rolling(10).mean()) * 100
        
        return trend_strength
    
    @staticmethod
    def trend_analysis(high: pd.Series, low: pd.Series, close: pd.Series,
                      timeframe: Optional[str] = None) -> Dict:
        """
        Comprehensive trend analysis with moving averages and Fibonacci
        Cleans NaN values from all output series before returning.
        """
        # Moving averages
        ma_results = TrendCalculator.moving_averages(close, timeframe)
        # Clean moving averages
        ma_results = clean_indicator_data(ma_results)
        # Identify swing points
        swing_highs, swing_lows = TrendCalculator.identify_swing_points(close)
        # Clean swing points
        swing_highs = swing_highs.fillna(False)
        swing_lows = swing_lows.fillna(False)
        # Get latest significant swing high and low
        recent_highs = close[swing_highs]
        recent_lows = close[swing_lows]
        
        if len(recent_highs) > 0 and len(recent_lows) > 0:
            swing_high = recent_highs.iloc[-1]
            swing_low = recent_lows.iloc[-1]
            
            # Fibonacci retracement
            fib_levels = TrendCalculator.fibonacci_retracement_series(
                high, low, swing_high, swing_low
            )
            fib_levels = clean_indicator_data(fib_levels)
        else:
            fib_levels = {}
        
        # Trend strength
        trend_strength = TrendCalculator.trend_strength(close)
        trend_strength = clean_indicator_data({'trend_strength': trend_strength})['trend_strength']
        
        # Overall trend direction
        current_price = close.iloc[-1]
        ema_20 = ma_results.get('ema_20', pd.Series([0])).iloc[-1] if 'ema_20' in ma_results else 0
        sma_200 = ma_results.get('sma_200', pd.Series([0])).iloc[-1] if 'sma_200' in ma_results else 0
        
        if current_price > ema_20 > sma_200:
            trend_direction = 'STRONG_UPTREND'
        elif current_price > sma_200:
            trend_direction = 'UPTREND'
        elif current_price < ema_20 < sma_200:
            trend_direction = 'STRONG_DOWNTREND'
        elif current_price < sma_200:
            trend_direction = 'DOWNTREND'
        else:
            trend_direction = 'RANGING'
        
        return {
            'moving_averages': ma_results,
            'fibonacci_levels': fib_levels,
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'current_swing_high': swing_high if len(recent_highs) > 0 else None,
            'current_swing_low': swing_low if len(recent_lows) > 0 else None
        }