import pandas as pd
from typing import Dict, List
from technical_analysis.calculators import TrendCalculator, MomentumCalculator
from technical_analysis.calculators import VolatilityCalculator

class SignalGenerator:
    """Generate trading signals based on technical analysis"""
    
    @staticmethod
    def generate_signals(data: pd.DataFrame, 
                        config: Dict = None) -> Dict[str, pd.Series]:
        if config is None:
            config = {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            }
        
        signals = {}
        
        try:
            close = data['Close']
        except KeyError:
            print("SignalGenerator: 'Close' column missing in data.")
            return signals
        
        # Calculate indicators with error handling
        try:
            rsi = MomentumCalculator.rsi(close, config.get('rsi_period', 14))
        except Exception as e:
            print(f"SignalGenerator: RSI calculation error: {e}")
            rsi = pd.Series([None]*len(close), index=close.index)
        
        try:
            macd = MomentumCalculator.macd(
                close,
                config.get('macd_fast', 12),
                config.get('macd_slow', 26),
                config.get('macd_signal', 9)
            )
        except Exception as e:
            print(f"SignalGenerator: MACD calculation error: {e}")
            macd = {'macd_line': pd.Series([None]*len(close), index=close.index),
                    'signal_line': pd.Series([None]*len(close), index=close.index)}
        
        # Generate signals
        signals['rsi_oversold'] = rsi < 30
        signals['rsi_overbought'] = rsi > 70
        signals['macd_bullish'] = macd['macd_line'] > macd['signal_line']
        signals['macd_bearish'] = macd['macd_line'] < macd['signal_line']
        
        # Add candlestick patterns if available
        if hasattr(VolatilityCalculator, 'candlestick_patterns'):
            try:
                patterns = VolatilityCalculator.candlestick_patterns(data)
                signals.update(patterns)
            except Exception as e:
                print(f"SignalGenerator: Candlestick pattern error: {e}")
        
        return signals