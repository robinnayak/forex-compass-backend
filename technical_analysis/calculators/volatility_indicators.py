import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Optional, Union
from technical_analysis.data_quality.processor import clean_indicator_data

class VolatilityCalculator:
    """Calculate volatility indicators with timeframe-aware parameters using pandas_ta"""
    
    @staticmethod
    def get_bollinger_params(timeframe: Optional[str] = None) -> Dict[str, Union[int, float]]:
        """
        Get Bollinger Bands parameters based on timeframe
        Returns: dict with 'period' and 'std_dev'
        """
        timeframe = timeframe.lower() if timeframe else None
        
        if timeframe in ['1m', '2m', '3m', '5m']:
            # Binary trading - tighter, faster reaction
            return {'period': 10, 'std_dev': 2.0}
        elif timeframe in ['15m', '30m']:
            # Scalping - faster signals
            return {'period': 14, 'std_dev': 2.5}
        elif timeframe in ['1h', '4h', '6h', '8h', '12h', '1d']:
            # Swing trading - standard
            return {'period': 20, 'std_dev': 2.0}
        elif timeframe in ['3d', '1w']:
            # Longer term - smoother
            return {'period': 20, 'std_dev': 2.0}
        else:
            # Default
            return {'period': 20, 'std_dev': 2.0}
    
    @staticmethod
    def get_atr_params(timeframe: Optional[str] = None) -> Dict[str, int]:
        """
        Get ATR parameters based on timeframe
        Returns: dict with 'period'
        """
        timeframe = timeframe.lower() if timeframe else None
        
        if timeframe in ['1m', '2m', '3m', '5m']:
            # Binary trading - faster volatility reaction
            return {'period': 7}
        elif timeframe in ['15m', '30m']:
            # Intraday scalping
            return {'period': 10}
        elif timeframe in ['1h', '4h', '6h', '8h', '12h']:
            # Intraday to swing
            return {'period': 14}
        elif timeframe in ['1d', '3d', '1w']:
            # Longer term
            return {'period': 14}
        else:
            # Default
            return {'period': 14}
    
    @staticmethod
    def bollinger_bands(series: pd.Series, 
                       period: Optional[int] = None,
                       std_dev: Optional[float] = None,
                       timeframe: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands using pandas_ta with timeframe-aware parameters
        
        Args:
            series: Price series (usually close)
            period: SMA period (overrides timeframe)
            std_dev: Standard deviation multiplier (overrides timeframe)
            timeframe: Timeframe string for automatic parameter selection
        
        Returns:
            Dict containing Bollinger Bands and trading signals
        """
        # Get parameters based on timeframe if not provided
        if period is None or std_dev is None:
            params = VolatilityCalculator.get_bollinger_params(timeframe)
            period = period or params['period']
            std_dev = std_dev or params['std_dev']
        
        # Use pandas_ta for optimized calculation
        bb_result = ta.bbands(series, length=period, std=std_dev)
        
        upper_band = bb_result[f'BBU_{period}_{std_dev}']
        middle_band = bb_result[f'BBM_{period}_{std_dev}']
        lower_band = bb_result[f'BBL_{period}_{std_dev}']
        bandwidth = bb_result[f'BBB_{period}_{std_dev}']
        
        # Clean all output series
        cleaned = clean_indicator_data({
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'bandwidth': bandwidth
        })
        upper_band = cleaned['upper_band']
        middle_band = cleaned['middle_band']
        lower_band = cleaned['lower_band']
        bandwidth = cleaned['bandwidth']
        
        # Generate trading signals
        current_price = series.iloc[-1] if len(series) > 0 else 0
        current_upper = upper_band.iloc[-1] if len(upper_band) > 0 else 0
        current_lower = lower_band.iloc[-1] if len(lower_band) > 0 else 0
        
        # Band squeeze detection (low volatility)
        bandwidth_mean = bandwidth.rolling(20).mean().ffill().bfill()
        bandwidth_std = bandwidth.rolling(20).std().replace(0, np.nan).ffill().bfill()
        # Clean bandwidth_mean and bandwidth_std to avoid NaNs
        cleaned_stats = clean_indicator_data({'bandwidth_mean': bandwidth_mean, 'bandwidth_std': bandwidth_std})
        bandwidth_mean = cleaned_stats['bandwidth_mean']
        bandwidth_std = cleaned_stats['bandwidth_std']
        is_squeeze = bandwidth < (bandwidth_mean - bandwidth_std)
        # Calculate squeeze_intensity and clean it
        squeeze_intensity = (bandwidth_mean - bandwidth) / bandwidth_std if len(bandwidth) > 20 else pd.Series([0]*len(bandwidth), index=bandwidth.index)
        squeeze_intensity = clean_indicator_data({'squeeze_intensity': squeeze_intensity})['squeeze_intensity']
        touch_upper = current_price >= current_upper * 0.995  # Within 0.5% of upper band
        touch_lower = current_price <= current_lower * 1.005  # Within 0.5% of lower band
        in_middle = (current_price > current_lower * 1.05) and (current_price < current_upper * 0.95)
        
        return {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'bandwidth': bandwidth,
            'period': period,
            'std_dev': std_dev,
            'touch_upper_band': touch_upper,
            'touch_lower_band': touch_lower,
            'in_middle_band': in_middle,
            'is_squeeze': is_squeeze.iloc[-1] if len(is_squeeze) > 0 else False,
            'squeeze_intensity': squeeze_intensity
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: Optional[int] = None,
            timeframe: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate Average True Range using pandas_ta with timeframe-aware parameters
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR period (overrides timeframe)
            timeframe: Timeframe string for automatic parameter selection
        
        Returns:
            Dict containing ATR values and stop-loss suggestions
        """
        # Get parameters based on timeframe if not provided
        if period is None:
            params = VolatilityCalculator.get_atr_params(timeframe)
            period = params['period']
        
        # Use pandas_ta for optimized calculation
        atr_values = ta.atr(high, low, close, length=period)
        
        # Clean ATR output series
        cleaned = clean_indicator_data({'atr': atr_values})
        atr_values = cleaned['atr']
        
        # Generate stop-loss suggestions
        current_atr = atr_values.iloc[-1] if len(atr_values) > 0 else 0
        
        stop_loss_suggestions = {
            'conservative': current_atr * 2.0,  # 2x ATR for wider stops
            'moderate': current_atr * 1.5,     # 1.5x ATR for balanced stops
            'aggressive': current_atr * 1.0,   # 1x ATR for tight stops
            'very_aggressive': current_atr * 0.7  # 0.7x ATR for very tight stops
        }
        
        # Volatility regime detection
        atr_mean = atr_values.rolling(20).mean()
        atr_std = atr_values.rolling(20).std()
        volatility_regime = np.where(
            atr_values > (atr_mean + atr_std),
            'HIGH_VOLATILITY',
            np.where(
                atr_values < (atr_mean - atr_std),
                'LOW_VOLATILITY',
                'NORMAL_VOLATILITY'
            )
        )
        
        return {
            'atr': atr_values,
            'period': period,
            'current_value': current_atr,
            'stop_loss_suggestions': stop_loss_suggestions,
            'volatility_regime': pd.Series(volatility_regime, index=atr_values.index),
            'is_high_volatility': atr_values > (atr_mean + atr_std),
            'is_low_volatility': atr_values < (atr_mean - atr_std)
        }
    
    @staticmethod
    def bollinger_bands_reversal_signals(data: pd.DataFrame,
                                        timeframe: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Detect Bollinger Band reversal signals with candle patterns
        """
        bb = VolatilityCalculator.bollinger_bands(data['close'], timeframe=timeframe)
        
        # Price touches bands
        touch_upper = data['high'] >= bb['upper_band'] * 0.995
        touch_lower = data['low'] <= bb['lower_band'] * 1.005
        
        # Bearish reversal signals (touch upper band + bearish candle)
        bearish_reversal = touch_upper & (
            (data['close'] < data['open']) |  # Red candle
            (data['close'] < data['open'].shift(1)) |  # Close below previous open
            ((data['high'] - data['close']) > (data['close'] - data['low']) * 2)  # Long upper wick
        )
        
        # Bullish reversal signals (touch lower band + bullish candle)
        bullish_reversal = touch_lower & (
            (data['close'] > data['open']) |  # Green candle
            (data['close'] > data['open'].shift(1)) |  # Close above previous open
            ((data['close'] - data['low']) > (data['high'] - data['close']) * 2)  # Long lower wick
        )
        
        return {
            'bearish_reversal': bearish_reversal,
            'bullish_reversal': bullish_reversal,
            'touch_upper': touch_upper,
            'touch_lower': touch_lower,
            'bollinger_bands': bb
        }
    
    @staticmethod
    def volatility_analysis(high: pd.Series, low: pd.Series, close: pd.Series,
                           timeframe: Optional[str] = None) -> Dict:
        """
        Comprehensive volatility analysis with Bollinger Bands and ATR
        """
        bb_analysis = VolatilityCalculator.bollinger_bands(close, timeframe=timeframe)
        atr_analysis = VolatilityCalculator.atr(high, low, close, timeframe=timeframe)
        reversal_signals = VolatilityCalculator.bollinger_bands_reversal_signals(
            pd.DataFrame({'high': high, 'low': low, 'close': close}),
            timeframe
        )
        
        # Combined volatility assessment
        current_volatility = 'HIGH' if atr_analysis['is_high_volatility'].iloc[-1] else \
                            'LOW' if atr_analysis['is_low_volatility'].iloc[-1] else 'NORMAL'
        
        # Clean squeeze_intensity and volatility_regime outputs
        if 'squeeze_intensity' in bb_analysis:
            cleaned_squeeze = clean_indicator_data({'squeeze_intensity': bb_analysis['squeeze_intensity']})
            bb_analysis['squeeze_intensity'] = cleaned_squeeze['squeeze_intensity']
        if 'volatility_regime' in atr_analysis:
            cleaned_volreg = clean_indicator_data({'volatility_regime': atr_analysis['volatility_regime']})
            atr_analysis['volatility_regime'] = cleaned_volreg['volatility_regime']
        
        return {
            'bollinger_bands': bb_analysis,
            'atr': atr_analysis,
            'reversal_signals': reversal_signals,
            'current_volatility_regime': current_volatility,
            'suggested_stop_loss': atr_analysis['stop_loss_suggestions']['moderate'],
            'is_breakout_environment': bb_analysis['is_squeeze'] and current_volatility == 'LOW'
        }