# strategies/breakout_flag_continuation.py
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from technical_analysis.calculators.volatility_indicators import VolatilityCalculator
from technical_analysis.calculators.momentum_indicators import MomentumCalculator
from technical_analysis.calculators.volume_indicators import VolumeCalculator
from technical_analysis.pattern_scanners.chart_patterns import ChartPatternScanner

class BreakoutFlagContinuation:
    """
    Breakout + Flag Continuation Strategy
    
    Key Rules:
    - Identify strong initial move (flag pole)
    - Wait for flag consolidation (20-50% retracement)
    - Enter on breakout of flag pattern
    - Volume confirmation on both moves
    - Target: measured move of flag pole
    """
    
    def __init__(self, account_size: float = 10000.0, risk_per_trade: float = 0.01):
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.account_size * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        return risk_amount / risk_per_share
    
    def identify_flag_pole(self, data: pd.DataFrame, lookback: int = 10) -> Dict:
        """Identify the flag pole (initial strong move)"""
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # Find significant moves
        price_change = close.pct_change(lookback)
        volume_avg = volume.rolling(lookback).mean()
        
        bull_pole = price_change > 0.03  # 3%+ move up
        bear_pole = price_change < -0.03  # 3%+ move down
        
        return {
            'bull_pole': bull_pole,
            'bear_pole': bear_pole,
            'pole_strength': abs(price_change),
            'volume_confirm': volume > volume_avg * 1.5
        }
    
    def generate_signals(self, data: pd.DataFrame, 
                        timeframe: str = '1h') -> Dict:
        """
        Generate Breakout + Flag Continuation signals
        """
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return {'signal': 'ERROR', 'reason': 'Missing OHLCV columns'}
        
        # Calculate indicators
        pole_analysis = self.identify_flag_pole(data)
        pattern_analysis = ChartPatternScanner.detect_flag_pattern(
            data['high'], data['low'], data['close'], data['volume']
        )
        volume_analysis = VolumeCalculator.volume_sma(data['volume'], timeframe=timeframe)
        
        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        volume_avg = volume_analysis['volume_sma'].iloc[-1]
        
        # Bull flag continuation
        bull_flag_conditions = (
            pole_analysis['bull_pole'].iloc[-1] and
            pole_analysis['volume_confirm'].iloc[-1] and
            pattern_analysis['bull_flag'] and
            current_volume > volume_avg * 1.2 and
            current_price > data['high'].iloc[-2]  # Breakout above consolidation
        )
        
        # Bear flag continuation
        bear_flag_conditions = (
            pole_analysis['bear_pole'].iloc[-1] and
            pole_analysis['volume_confirm'].iloc[-1] and
            pattern_analysis['bear_flag'] and
            current_volume > volume_avg * 1.2 and
            current_price < data['low'].iloc[-2]  # Breakout below consolidation
        )
        
        # Calculate measured move targets
        if bull_flag_conditions:
            # Flag pole height
            pole_high = data['high'].iloc[-10:].max()
            pole_low = data['low'].iloc[-10:].min()
            pole_height = pole_high - pole_low
            
            signal = 'BUY'
            stop_loss = data['low'].iloc[-2]  # Below consolidation
            take_profit = current_price + pole_height  # Measured move
            reason = "Bull flag continuation pattern"
            
        elif bear_flag_conditions:
            # Flag pole height
            pole_high = data['high'].iloc[-10:].max()
            pole_low = data['low'].iloc[-10:].min()
            pole_height = pole_high - pole_low
            
            signal = 'SELL'
            stop_loss = data['high'].iloc[-2]  # Above consolidation
            take_profit = current_price - pole_height  # Measured move
            reason = "Bear flag continuation pattern"
            
        else:
            signal = 'HOLD'
            stop_loss = None
            take_profit = None
            reason = "No flag pattern detected"
        
        # Calculate position size
        position_size = 0
        if signal in ['BUY', 'SELL'] and stop_loss is not None:
            position_size = self.calculate_position_size(current_price, stop_loss)
        
        return {
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'pole_strength': pole_analysis['pole_strength'].iloc[-1],
            'flag_pattern': 'BULL' if pattern_analysis['bull_flag'] else 'BEAR' if pattern_analysis['bear_flag'] else 'NONE',
            'volume_ratio': current_volume / volume_avg,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_reward_ratio': abs(take_profit - current_price) / abs(stop_loss - current_price) 
                                if stop_loss and take_profit else 0
        }