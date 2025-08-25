# strategies/bollinger_squeeze_breakout.py
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime
from technical_analysis.calculators.volatility_indicators import VolatilityCalculator
from technical_analysis.calculators.momentum_indicators import MomentumCalculator
from technical_analysis.calculators.volume_indicators import VolumeCalculator

class BollingerSqueezeBreakout:
    """
    Bollinger Band Squeeze Breakout Strategy
    
    Key Rules:
    - Wait for Bollinger Band squeeze (low volatility)
    - Enter on breakout with volume confirmation
    - Stop loss at opposite band
    - Target: 1.5x to 2x risk (ATR based)
    - Minimum squeeze duration: 5 periods
    """
    
    def __init__(self, account_size: float = 10000.0, risk_per_trade: float = 0.01):
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.min_squeeze_duration = 5
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.account_size * self.risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        return risk_amount / risk_per_share
    
    def generate_signals(self, data: pd.DataFrame, 
                        timeframe: str = '15m') -> Dict:
        """
        Generate Bollinger Band Squeeze Breakout signals
        """
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return {'signal': 'ERROR', 'reason': 'Missing OHLCV columns'}
        
        # Calculate indicators
        bb_analysis = VolatilityCalculator.bollinger_bands(data['close'], timeframe=timeframe)
        atr_analysis = VolatilityCalculator.atr(data['high'], data['low'], data['close'], timeframe=timeframe)
        volume_analysis = VolumeCalculator.volume_sma(data['volume'], timeframe=timeframe)
        
        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        volume_avg = volume_analysis['volume_sma'].iloc[-1]
        
        # Squeeze analysis
        is_squeeze = bb_analysis['is_squeeze'].iloc[-1]
        squeeze_intensity = bb_analysis['squeeze_intensity'].iloc[-1]
        
        # Check squeeze duration
        squeeze_duration = 0
        for i in range(1, 20):
            if bb_analysis['is_squeeze'].iloc[-i]:
                squeeze_duration += 1
            else:
                break
        
        # Breakout conditions
        upper_band = bb_analysis['upper_band'].iloc[-1]
        lower_band = bb_analysis['lower_band'].iloc[-1]
        middle_band = bb_analysis['middle_band'].iloc[-1]
        
        bullish_breakout = (
            is_squeeze and
            squeeze_duration >= self.min_squeeze_duration and
            current_price > upper_band and
            current_volume > volume_avg * 1.5 and
            current_price > data['close'].iloc[-2]  # Confirming breakout
        )
        
        bearish_breakout = (
            is_squeeze and
            squeeze_duration >= self.min_squeeze_duration and
            current_price < lower_band and
            current_volume > volume_avg * 1.5 and
            current_price < data['close'].iloc[-2]  # Confirming breakout
        )
        
        # Generate signals
        if bullish_breakout:
            signal = 'BUY'
            stop_loss = lower_band
            take_profit = current_price + (current_price - stop_loss) * 1.5  # 1.5:1 R/R
            reason = f"Bullish breakout after {squeeze_duration} period squeeze"
            
        elif bearish_breakout:
            signal = 'SELL'
            stop_loss = upper_band
            take_profit = current_price - (stop_loss - current_price) * 1.5  # 1.5:1 R/R
            reason = f"Bearish breakout after {squeeze_duration} period squeeze"
            
        else:
            signal = 'HOLD'
            stop_loss = None
            take_profit = None
            reason = "No breakout signal"
        
        # Calculate position size
        position_size = 0
        if signal in ['BUY', 'SELL'] and stop_loss is not None:
            position_size = self.calculate_position_size(current_price, stop_loss)
        
        return {
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': middle_band,
            'is_squeeze': is_squeeze,
            'squeeze_duration': squeeze_duration,
            'squeeze_intensity': squeeze_intensity,
            'volume_ratio': current_volume / volume_avg,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_reward_ratio': 1.5  # Fixed for this strategy
        }