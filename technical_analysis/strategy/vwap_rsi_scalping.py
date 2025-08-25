import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, time
from technical_analysis.calculators.volume_indicators import VolumeCalculator
from technical_analysis.calculators.momentum_indicators import MomentumCalculator

class VWAPRSIScalping:
    """
    VWAP + RSI Scalping Strategy (Intraday)
    
    Key Rules:
    - Trade only between 9:30 AM - 3:45 PM (market hours)
    - Use 1-5 minute timeframes
    - RSI filter: 30-70 range only
    - VWAP as primary mean reversion level
    - Volume confirmation required
    - Daily loss limit: 2% of account
    - Max 3 trades per hour
    """
    
    def __init__(self, account_size: float = 10000.0):
        self.account_size = account_size
        self.daily_loss_limit = account_size * 0.02  # 2% daily loss limit
        self.trades_today = 0
        self.trades_this_hour = 0
        self.last_trade_hour = None
        self.daily_pnl = 0.0
        
    def reset_daily_counters(self):
        """Reset daily counters at market open"""
        current_time = datetime.now().time()
        if current_time.hour == 9 and current_time.minute >= 30:
            self.trades_today = 0
            self.daily_pnl = 0.0
            self.trades_this_hour = 0
            self.last_trade_hour = None
    
    def is_market_hours(self, current_time: Optional[datetime] = None) -> bool:
        """Check if current time is within trading hours"""
        if current_time is None:
            current_time = datetime.now()
        
        current_time = current_time.time()
        # Market hours: 9:30 AM to 3:45 PM
        start_time = time(9, 30)
        end_time = time(15, 45)
        
        return start_time <= current_time <= end_time
    
    def can_trade(self, current_time: Optional[datetime] = None) -> bool:
        """Check if trading is allowed based on rules"""
        if not self.is_market_hours(current_time):
            return False
        
        if self.daily_pnl <= -self.daily_loss_limit:
            return False  # Daily loss limit reached
        
        if current_time is None:
            current_time = datetime.now()
        
        # Check hourly trade limit
        if self.last_trade_hour == current_time.hour:
            if self.trades_this_hour >= 3:
                return False
        else:
            self.trades_this_hour = 0
            self.last_trade_hour = current_time.hour
        
        return True
    
    def calculate_position_size(self, current_price: float, stop_loss: float) -> float:
        """Calculate position size based on risk management"""
        risk_per_trade = self.account_size * 0.005  # 0.5% risk per trade
        price_risk = abs(current_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_per_trade / price_risk
        return min(position_size, self.account_size * 0.1)  # Max 10% of account
    
    def generate_signals(self, data: pd.DataFrame, 
                        timeframe: str = '5m',
                        current_time: Optional[datetime] = None) -> Dict:
        """
        Generate VWAP + RSI scalping signals
        
        Args:
            data: DataFrame with OHLCV data
            timeframe: Trading timeframe
            current_time: Current timestamp for trading hours check
        
        Returns:
            Dict with trading signals and parameters
        """
        if not self.can_trade(current_time):
            return {'signal': 'NO_TRADE', 'reason': 'Trading not allowed'}
        
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return {'signal': 'ERROR', 'reason': 'Missing OHLCV columns'}
        
        # Calculate indicators
        vwap_analysis = VolumeCalculator.vwap(
            data['high'], data['low'], data['close'], data['volume'], timeframe=timeframe
        )
        
        rsi_analysis = MomentumCalculator.rsi_with_levels(
            data['close'], timeframe=timeframe
        )
        
        current_price = data['close'].iloc[-1]
        current_vwap = vwap_analysis['vwap'].iloc[-1]
        current_rsi = rsi_analysis['rsi'].iloc[-1]
        vwap_distance = vwap_analysis['vwap_distance'].iloc[-1]
        
        # Volume analysis
        volume_avg = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
        
        # Signal conditions
        bullish_conditions = [
            current_price < current_vwap,  # Price below VWAP
            vwap_distance < -0.5,  # At least 0.5% below VWAP
            30 <= current_rsi <= 70,  # RSI in neutral range
            volume_ratio > 1.2,  # Volume above average
            vwap_analysis['vwap_trend'].iloc[-1],  # VWAP trending up
            data['close'].iloc[-1] > data['open'].iloc[-1]  # Bullish candle
        ]
        
        bearish_conditions = [
            current_price > current_vwap,  # Price above VWAP
            vwap_distance > 0.5,  # At least 0.5% above VWAP
            30 <= current_rsi <= 70,  # RSI in neutral range
            volume_ratio > 1.2,  # Volume above average
            not vwap_analysis['vwap_trend'].iloc[-1],  # VWAP trending down
            data['close'].iloc[-1] < data['open'].iloc[-1]  # Bearish candle
        ]
        
        # Generate signals
        if all(bullish_conditions):
            signal = 'BUY'
            stop_loss = current_price * 0.995  # 0.5% stop loss
            take_profit = current_vwap  # Target VWAP
            reason = "Bullish VWAP reversion with RSI confirmation"
            
        elif all(bearish_conditions):
            signal = 'SELL'
            stop_loss = current_price * 1.005  # 0.5% stop loss
            take_profit = current_vwap  # Target VWAP
            reason = "Bearish VWAP reversion with RSI confirmation"
            
        else:
            signal = 'HOLD'
            stop_loss = None
            take_profit = None
            reason = "No clear signal"
        
        # Calculate position size if we have a signal
        position_size = 0
        if signal in ['BUY', 'SELL'] and stop_loss is not None:
            position_size = self.calculate_position_size(current_price, stop_loss)
        
        return {
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'current_vwap': current_vwap,
            'current_rsi': current_rsi,
            'vwap_distance': vwap_distance,
            'volume_ratio': volume_ratio,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_reward_ratio': abs(take_profit - current_price) / abs(stop_loss - current_price) 
                                if stop_loss and take_profit else 0,
            'timestamp': current_time or datetime.now()
        }
    
    def execute_strategy(self, data: pd.DataFrame, 
                        timeframe: str = '5m',
                        current_time: Optional[datetime] = None) -> Dict:
        """
        Execute the complete VWAP + RSI scalping strategy
        """
        # Reset daily counters if needed
        self.reset_daily_counters()
        
        # Generate signals
        signal_data = self.generate_signals(data, timeframe, current_time)
        
        # If we have a valid trade signal
        if signal_data['signal'] in ['BUY', 'SELL']:
            # Update trade counters
            self.trades_today += 1
            self.trades_this_hour += 1
            
            # Calculate expected P&L (for simulation)
            entry_price = signal_data['current_price']
            stop_loss = signal_data['stop_loss']
            take_profit = signal_data['take_profit']
            position_size = signal_data['position_size']
            
            if signal_data['signal'] == 'BUY':
                potential_loss = (entry_price - stop_loss) * position_size
                potential_gain = (take_profit - entry_price) * position_size
            else:  # SELL
                potential_loss = (stop_loss - entry_price) * position_size
                potential_gain = (entry_price - take_profit) * position_size
            
            signal_data.update({
                'potential_gain': potential_gain,
                'potential_loss': potential_loss,
                'trades_today': self.trades_today,
                'trades_this_hour': self.trades_this_hour,
                'daily_pnl_remaining': self.daily_loss_limit + self.daily_pnl
            })
        
        return signal_data
    
    def update_pnl(self, pnl: float):
        """Update daily P&L"""
        self.daily_pnl += pnl
    
    def get_strategy_stats(self) -> Dict:
        """Get current strategy statistics"""
        return {
            'account_size': self.account_size,
            'daily_loss_limit': self.daily_loss_limit,
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today,
            'trades_this_hour': self.trades_this_hour,
            'pnl_remaining': self.daily_loss_limit + self.daily_pnl,
            'max_trades_per_hour': 3,
            'trading_hours': '9:30 AM - 3:45 PM'
        }