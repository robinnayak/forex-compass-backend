# strategies/strategy_manager.py
import pandas as pd
from typing import Dict, List
# from strategies.vwap_rsi_scalping import VWAPRSIScalping
# from strategies.bollinger_squeeze_breakout import BollingerSqueezeBreakout
# from strategies.breakout_flag_continuation import BreakoutFlagContinuation

from technical_analysis.strategy.vwap_rsi_scalping import VWAPRSIScalping
from technical_analysis.strategy.bollinger_squeeze_breakout import BollingerSqueezeBreakout
from technical_analysis.strategy.breakout_flag_continuation import BreakoutFlagContinuation

class StrategyManager:
    """Manage multiple trading strategies"""
    
    def __init__(self, account_size: float = 10000.0):
        self.account_size = account_size
        self.strategies = {
            'vwap_rsi': VWAPRSIScalping(account_size),
            'bollinger_squeeze': BollingerSqueezeBreakout(account_size),
            'flag_continuation': BreakoutFlagContinuation(account_size)
        }
    
    def run_all_strategies(self, data: pd.DataFrame, 
                          timeframe: str = '15m') -> Dict[str, Dict]:
        """Run all strategies on given data"""
        results = {}
        
        for strategy_name, strategy in self.strategies.items():
            if strategy_name == 'vwap_rsi':
                results[strategy_name] = strategy.generate_signals(data, timeframe)
            else:
                results[strategy_name] = strategy.generate_signals(data, timeframe)
        
        return results
    
    def get_best_signal(self, results: Dict[str, Dict]) -> Dict:
        """Get the strongest signal from all strategies"""
        best_signal = {'signal': 'HOLD', 'confidence': 0, 'strategy': 'NONE'}
        
        for strategy_name, result in results.items():
            if result['signal'] in ['BUY', 'SELL']:
                # Calculate confidence score
                confidence = self.calculate_confidence(result, strategy_name)
                
                if confidence > best_signal['confidence']:
                    best_signal = {
                        'signal': result['signal'],
                        'confidence': confidence,
                        'strategy': strategy_name,
                        'details': result
                    }
        
        return best_signal
    
    def calculate_confidence(self, signal_data: Dict, strategy_name: str) -> float:
        """Calculate confidence score for a signal"""
        confidence = 0.0
        
        if strategy_name == 'vwap_rsi':
            # VWAP RSI confidence factors
            if signal_data['volume_ratio'] > 1.5:
                confidence += 0.3
            if 40 <= signal_data['current_rsi'] <= 60:
                confidence += 0.2
            if abs(signal_data['vwap_distance']) > 1.0:
                confidence += 0.2
        
        elif strategy_name == 'bollinger_squeeze':
            # Bollinger squeeze confidence
            if signal_data['squeeze_duration'] > 10:
                confidence += 0.3
            if signal_data['squeeze_intensity'] > 2.0:
                confidence += 0.2
            if signal_data['volume_ratio'] > 2.0:
                confidence += 0.2
        
        elif strategy_name == 'flag_continuation':
            # Flag continuation confidence
            if signal_data['pole_strength'] > 0.05:
                confidence += 0.3
            if signal_data['risk_reward_ratio'] > 2.0:
                confidence += 0.2
            if signal_data['volume_ratio'] > 1.5:
                confidence += 0.2
        
        return min(confidence, 1.0)