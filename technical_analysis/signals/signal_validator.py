# strategies/signal_validator.py
import pandas as pd
from typing import Dict

class SignalValidator:
    """Validate breakout signals with additional confirmation"""
    
    @staticmethod
    def validate_breakout_signal(data: pd.DataFrame, signal_data: Dict) -> Dict:
        """
        Validate breakout signal with additional criteria
        """
        if signal_data['signal'] == 'HOLD':
            return signal_data
        
        # Check volume confirmation
        current_volume = data['volume'].iloc[-1]
        volume_avg = data['volume'].rolling(20).mean().iloc[-1]
        volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
        
        # Check if breakout is sustained (not just a spike)
        previous_close = data['close'].iloc[-2]
        price_change = abs(signal_data['current_price'] - previous_close) / previous_close
        
        # Additional validation rules
        validation_rules = {
            'volume_confirm': volume_ratio > 1.2,
            'sustained_move': price_change > 0.0005,  # Minimum 0.05% move
            'not_whipsaw': abs(signal_data['current_price'] - signal_data['previous_high']) > 0.0002 if signal_data['signal'] == 'BUY' else 
                          abs(signal_data['current_price'] - signal_data['previous_low']) > 0.0002,
            'time_since_last_breakout': True  # Would check time since last signal
        }
        
        # Calculate validation score
        validation_score = sum(validation_rules.values()) / len(validation_rules)
        
        # If validation fails, revert to HOLD
        if validation_score < 0.75:
            signal_data.update({
                'signal': 'HOLD',
                'reason': f'Signal validation failed (score: {validation_score:.2f})',
                'validation_score': validation_score,
                'validation_details': validation_rules
            })
        else:
            signal_data.update({
                'validation_score': validation_score,
                'validation_details': validation_rules,
                'volume_ratio': volume_ratio
            })
        
        return signal_data