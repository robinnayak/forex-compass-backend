import pandas as pd
import numpy as np
from typing import Dict

class CandlestickPatternScanner:
    """Scan for candlestick patterns"""
    
    @staticmethod
    def identify_patterns(data: pd.DataFrame) -> Dict[str, pd.Series]:
        open = data['open']
        high = data['high']
        low = data['low']
        close = data['close']
        
        patterns = {}
        
        # Bullish Engulfing
        patterns['bullish_engulfing'] = (
            (close > open) & 
            (close.shift(1) < open.shift(1)) & 
            (close > open.shift(1)) & 
            (open < close.shift(1))
        )
        
        # Hammer
        body = abs(close - open)
        lower_wick = np.minimum(open, close) - low
        upper_wick = high - np.maximum(open, close)
        
        patterns['hammer'] = (
            (lower_wick > 2 * body) & 
            (upper_wick < body * 0.1) & 
            (close > open)
        )
        
        # Doji
        patterns['doji'] = (body / (high - low)) < 0.1
        
        return patterns