# technical_analysis/pattern_scanners/chart_patterns.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class ChartPatternScanner:
    """Scan for chart patterns including flags"""
    
    @staticmethod
    def detect_flag_pattern(high: pd.Series, low: pd.Series, close: pd.Series, 
                          volume: pd.Series, lookback: int = 20) -> Dict[str, bool]:
        """
        Detect bull flag and bear flag patterns
        """
        if len(close) < lookback:
            return {'bull_flag': False, 'bear_flag': False}
        
        # Recent price action
        recent_high = high.iloc[-lookback:].max()
        recent_low = low.iloc[-lookback:].min()
        current_close = close.iloc[-1]
        
        # Volume analysis
        volume_avg = volume.rolling(lookback).mean()
        current_volume = volume.iloc[-1]
        
        # Price slope (for flag pole)
        price_slope = close.diff(5).iloc[-1] / close.iloc[-6]
        
        # Bull flag pattern: consolidation after uptrend
        bull_flag = (
            price_slope > 0.02 and  # Upward momentum
            (current_close - recent_low) / recent_low > 0.03 and  # Significant move up
            (recent_high - current_close) / recent_high < 0.02 and  # Small pullback
            current_volume > volume_avg.iloc[-1] * 0.8  # Sustained volume
        )
        
        # Bear flag pattern: consolidation after downtrend
        bear_flag = (
            price_slope < -0.02 and  # Downward momentum
            (recent_high - current_close) / recent_high > 0.03 and  # Significant move down
            (current_close - recent_low) / recent_low < 0.02 and  # Small bounce
            current_volume > volume_avg.iloc[-1] * 0.8  # Sustained volume
        )
        
        return {
            'bull_flag': bull_flag,
            'bear_flag': bear_flag,
            'flag_pole_strength': abs(price_slope),
            'consolidation_range': (recent_high - recent_low) / recent_low
        }