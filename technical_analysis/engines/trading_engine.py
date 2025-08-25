# strategies/trading_engine.py
import pandas as pd
from typing import Dict, List
from datetime import datetime
# from strategies.breakout_scalping import BreakoutScalping
# from strategies.signal_validator import SignalValidator
from technical_analysis.strategy.breakout_scalping import BreakoutScalping
from technical_analysis.signals.signal_validator import SignalValidator

class TradingEngine:
    """Real-time trading engine for Breakout Scalping"""
    
    def __init__(self, account_size: float = 10000.0):
        self.strategy = BreakoutScalping(account_size)
        self.validator = SignalValidator()
        self.symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY']
        self.current_signals = {}
        
    def process_new_data(self, symbol: str, new_data: pd.DataFrame) -> Dict:
        """Process new 1-minute data for a symbol"""
        # Generate signal
        signal = self.strategy.generate_signals(new_data, symbol)
        
        # Validate signal
        validated_signal = self.validator.validate_breakout_signal(new_data, signal)
        
        # Store current signal
        self.current_signals[symbol] = validated_signal
        
        return validated_signal
    
    def get_all_signals(self) -> Dict[str, Dict]:
        """Get signals for all trading pairs"""
        return self.current_signals
    
    def should_execute_trade(self, signal: Dict) -> bool:
        """Determine if a trade should be executed"""
        if signal['signal'] == 'HOLD':
            return False
        
        # Additional execution filters
        execution_criteria = [
            signal['validation_score'] > 0.7,
            signal['atr_pips'] > 8,
            signal['session_trades'] < self.strategy.max_trades_per_session,
            self.strategy.is_trading_session()
        ]
        
        return all(execution_criteria)