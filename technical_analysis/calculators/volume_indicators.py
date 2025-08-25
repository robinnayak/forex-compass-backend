import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Optional, Union
from technical_analysis.data_quality.processor import clean_indicator_data

class VolumeIndicator:
    """Calculate volume-based indicators with timeframe-aware parameters using pandas_ta"""
    
    @staticmethod
    def get_volume_params(timeframe: Optional[str] = None) -> Dict[str, int]:
        """
        Get volume indicator parameters based on timeframe
        Returns: dict with typical periods for volume indicators
        """
        timeframe = timeframe.lower() if timeframe else None
        
        if timeframe in ['1m', '2m', '3m', '5m']:
            # Binary trading - faster reaction
            return {'vmap_period': 5, 'vwap_period': 10, 'obv_ema_period': 7, 'volume_sma_period': 10}
        elif timeframe in ['15m', '30m']:
            # Intraday scalping
            return {'vmap_period': 10, 'vwap_period': 14, 'obv_ema_period': 9, 'volume_sma_period': 14}
        elif timeframe in ['1h', '4h', '6h', '8h', '12h']:
            # Swing trading
            return {'vmap_period': 14, 'vwap_period': 20, 'obv_ema_period': 14, 'volume_sma_period': 20}
        elif timeframe in ['1d', '3d', '1w']:
            # Long-term
            return {'vmap_period': 20, 'vwap_period': 30, 'obv_ema_period': 21, 'volume_sma_period': 30}
        else:
            # Default
            return {'vmap_period': 14, 'vwap_period': 20, 'obv_ema_period': 14, 'volume_sma_period': 20}
    
    @staticmethod
    def volume_sma(volume: pd.Series, 
                  period: Optional[int] = None,
                  timeframe: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate Volume SMA with timeframe-aware parameters
        
        Args:
            volume: Volume series
            period: SMA period (overrides timeframe)
            timeframe: Timeframe string for automatic parameter selection
        
        Returns:
            Dict containing volume SMA and signals
        """
        if period is None:
            params = VolumeIndicator.get_volume_params(timeframe)
            period = params['volume_sma_period']
        
        volume_sma = ta.sma(volume, length=period)
        
        # Volume signals
        volume_above_sma = volume > volume_sma
        volume_spike = volume > (volume_sma * 1.5)  # 50% above average
        
        result = {
            'volume_sma': volume_sma,
            'period': period,
            'volume_above_average': volume_above_sma,
            'volume_spike': volume_spike,
            'volume_ratio': volume / volume_sma  # Current volume relative to average
        }
        return clean_indicator_data(result)
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series,
            ema_period: Optional[int] = None,
            timeframe: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate On-Balance Volume (OBV) with EMA smoothing
        
        Args:
            close: Close price series
            volume: Volume series
            ema_period: EMA period for smoothing (overrides timeframe)
            timeframe: Timeframe string for automatic parameter selection
        
        Returns:
            Dict containing OBV and signals
        """
        # Calculate OBV using pandas_ta
        obv_values = ta.obv(close, volume)
        
        # Get EMA period for smoothing
        if ema_period is None:
            params = VolumeIndicator.get_volume_params(timeframe)
            ema_period = params['obv_ema_period']
        
        obv_ema = ta.ema(obv_values, length=ema_period)
        
        # OBV signals
        obv_rising = obv_values > obv_values.shift(1)
        obv_above_ema = obv_values > obv_ema
        obv_trend = obv_values.diff(5) > 0  # Positive 5-period trend
        
        # Divergence detection (simplified)
        price_trend = close.diff(5) > 0
        bullish_divergence = (~price_trend) & obv_trend  # Price down, OBV up
        bearish_divergence = price_trend & (~obv_trend)  # Price up, OBV down
        
        result = {
            'obv': obv_values,
            'obv_ema': obv_ema,
            'ema_period': ema_period,
            'obv_rising': obv_rising,
            'obv_above_ema': obv_above_ema,
            'obv_trend': obv_trend,
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence,
            'obv_momentum': obv_values.diff(3)  # 3-period momentum
        }
        return clean_indicator_data(result)
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
             period: Optional[int] = None,
             timeframe: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate Volume Weighted Average Price (VWAP)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            volume: Volume series
            period: VWAP period (overrides timeframe)
            timeframe: Timeframe string for automatic parameter selection
        
        Returns:
            Dict containing VWAP and trading signals
        """
        if period is None:
            params = VolumeIndicator.get_volume_params(timeframe)
            period = params['vwap_period']
        
        # Calculate VWAP using pandas_ta
        vwap_values = ta.vwap(high, low, close, volume, length=period)
        
        # VWAP signals
        price_above_vwap = close > vwap_values
        price_below_vwap = close < vwap_values
        vwap_support = (close.shift(1) < vwap_values.shift(1)) & (close > vwap_values)
        vwap_resistance = (close.shift(1) > vwap_values.shift(1)) & (close < vwap_values)
        
        result = {
            'vwap': vwap_values,
            'period': period,
            'price_above_vwap': price_above_vwap,
            'price_below_vwap': price_below_vwap,
            'vwap_support': vwap_support,
            'vwap_resistance': vwap_resistance,
            'price_vwap_distance': (close - vwap_values) / vwap_values * 100  # Percentage distance
        }
        return clean_indicator_data(result)
    
    @staticmethod
    def volume_profile(high: pd.Series, low: pd.Series, volume: pd.Series,
                      period: Optional[int] = None,
                      timeframe: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate Volume Profile (simplified)
        
        Args:
            high: High price series
            low: Low price series
            volume: Volume series
            period: Profile period (overrides timeframe)
            timeframe: Timeframe string for automatic parameter selection
        
        Returns:
            Dict containing volume profile signals
        """
        if period is None:
            params = VolumeIndicator.get_volume_params(timeframe)
            period = params['vmap_period']
        
        # Calculate typical price for volume distribution
        typical_price = (high + low + (high + low) / 2) / 3  # Modified typical price
        
        # Simple volume-at-price approximation
        volume_ema = ta.ema(volume, length=period)
        high_volume_zones = volume > volume_ema * 1.2
        
        # Price action relative to high volume zones
        in_high_volume_zone = (typical_price >= typical_price[high_volume_zones].min()) & \
                             (typical_price <= typical_price[high_volume_zones].max())
        
        result = {
            'volume_ema': volume_ema,
            'high_volume_zones': high_volume_zones,
            'in_high_volume_zone': in_high_volume_zone,
            'typical_price': typical_price,
            'volume_above_average': volume > volume_ema
        }
        return clean_indicator_data(result)
    
    @staticmethod
    def accumulation_distribution(high: pd.Series, low: pd.Series, 
                                 close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate Accumulation/Distribution Line
        
        Returns:
            Dict containing A/D line and signals
        """
        # Calculate A/D using pandas_ta
        ad_line = ta.ad(high, low, close, volume)
        
        # A/D signals
        ad_rising = ad_line > ad_line.shift(1)
        ad_falling = ad_line < ad_line.shift(1)
        ad_momentum = ad_line.diff(5)
        
        result = {
            'ad_line': ad_line,
            'ad_rising': ad_rising,
            'ad_falling': ad_falling,
            'ad_momentum': ad_momentum,
            'ad_trend': ad_line > ta.ema(ad_line, length=14)
        }
        return clean_indicator_data(result)
    
    @staticmethod
    def volume_analysis(high: pd.Series, low: pd.Series, close: pd.Series, 
                       volume: pd.Series, timeframe: Optional[str] = None) -> Dict:
        """
        Comprehensive volume analysis with multiple indicators
        
        Returns:
            Dict containing all volume indicators and combined signals
        """
        volume_sma_analysis = VolumeIndicator.volume_sma(volume, timeframe=timeframe)
        obv_analysis = VolumeIndicator.obv(close, volume, timeframe=timeframe)
        vwap_analysis = VolumeIndicator.vwap(high, low, close, volume, timeframe=timeframe)
        volume_profile_analysis = VolumeIndicator.volume_profile(high, low, volume, timeframe=timeframe)
        ad_analysis = VolumeIndicator.accumulation_distribution(high, low, close, volume)
        
        # Combined volume signals
        current_volume = volume.iloc[-1] if len(volume) > 0 else 0
        volume_sma_current = volume_sma_analysis['volume_sma'].iloc[-1] if len(volume_sma_analysis['volume_sma']) > 0 else 0
        
        volume_strength = 'HIGH' if current_volume > volume_sma_current * 1.5 else \
                         'ABOVE_AVERAGE' if current_volume > volume_sma_current else 'BELOW_AVERAGE'
        
        # Volume confirmation signals
        price_up_volume_up = (close.diff() > 0) & (volume > volume_sma_analysis['volume_sma'])
        price_down_volume_up = (close.diff() < 0) & (volume > volume_sma_analysis['volume_sma'])
        
        return {
            'volume_sma': volume_sma_analysis,
            'obv': obv_analysis,
            'vwap': vwap_analysis,
            'volume_profile': volume_profile_analysis,
            'accumulation_distribution': ad_analysis,
            'volume_strength': volume_strength,
            'price_up_volume_up': price_up_volume_up,
            'price_down_volume_up': price_down_volume_up,
            'volume_confirmation': (price_up_volume_up | price_down_volume_up),
            'volume_divergence': obv_analysis['bullish_divergence'] | obv_analysis['bearish_divergence']
        }
    
    @staticmethod
    def volume_based_stop_loss(close: pd.Series, volume: pd.Series, 
                              atr_value: float, timeframe: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate volume-based stop loss levels
        
        Args:
            close: Close price series
            volume: Volume series
            atr_value: Current ATR value
            timeframe: Timeframe for parameter selection
        
        Returns:
            Dict containing stop loss suggestions
        """
        volume_analysis = VolumeIndicator.volume_sma(volume, timeframe=timeframe)
        volume_ratio = volume_analysis['volume_ratio'].iloc[-1] if len(volume_analysis['volume_ratio']) > 0 else 1.0
        
        # Adjust stop loss based on volume
        # High volume → tighter stops (more conviction)
        # Low volume → wider stops (less conviction)
        if volume_ratio > 2.0:
            # Very high volume - tight stop
            stop_multiplier = 0.8
        elif volume_ratio > 1.5:
            # High volume - moderate stop
            stop_multiplier = 1.0
        elif volume_ratio > 1.0:
            # Above average volume
            stop_multiplier = 1.2
        else:
            # Below average volume - wider stop
            stop_multiplier = 1.5
        
        base_stop = atr_value * stop_multiplier
        
        return {
            'volume_ratio': volume_ratio,
            'stop_loss_atr_multiplier': stop_multiplier,
            'suggested_stop_loss': base_stop,
            'conservative_stop': base_stop * 1.5,
            'aggressive_stop': base_stop * 0.7,
            'volume_context': 'HIGH_VOLUME' if volume_ratio > 1.5 else \
                             'AVERAGE_VOLUME' if volume_ratio > 1.0 else 'LOW_VOLUME'
        }
    
    @staticmethod
    def get_vwap_params(timeframe: Optional[str] = None) -> Dict[str, int]:
        """
        Get VWAP parameters based on timeframe
        """
        timeframe = timeframe.lower() if timeframe else None
        
        if timeframe in ['1m', '2m', '3m']:
            return {'period': 20}  # Shorter for 1-3min
        elif timeframe in ['5m', '15m']:
            return {'period': 14}  # Standard for 5-15min
        else:
            return {'period': 20}  # Default
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
             period: Optional[int] = None, timeframe: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        Calculate Volume Weighted Average Price (VWAP)
        """
        if period is None:
            params = VolumeIndicator.get_vwap_params(timeframe)
            period = params['period']
        
        # Calculate VWAP using pandas_ta
        vwap_values = ta.vwap(high, low, close, volume, length=period)
        
        # Calculate typical price for reference
        typical_price = (high + low + close) / 3
        
        # VWAP signals
        price_above_vwap = close > vwap_values
        price_below_vwap = close < vwap_values
        
        # Distance from VWAP (percentage)
        vwap_distance = ((close - vwap_values) / vwap_values) * 100
        
        # VWAP slope (trend)
        vwap_slope = vwap_values.diff(5)  # 5-period slope
        
        result = {
            'vwap': vwap_values,
            'typical_price': typical_price,
            'period': period,
            'price_above_vwap': price_above_vwap,
            'price_below_vwap': price_below_vwap,
            'vwap_distance': vwap_distance,
            'vwap_slope': vwap_slope,
            'vwap_trend': vwap_slope > 0  # True if rising
        }
        return clean_indicator_data(result)