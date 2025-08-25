# strategies/breakout_scalping.py
import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, time, timedelta

class BreakoutScalping:
    """
    Breakout Scalping Strategy (M1 Timeframe)
    
    Key Rules:
    - Timeframe: M1 (1-minute charts)
    - Session: 8:00-10:00 GMT (Any session Open)
    - Pairs: EUR/USD, GBP/USD, USD/JPY
    - Indicators: EMA 5 & EMA 13, RSI (14), ATR (14)
    - Risk: 0.5% per trade, 8-10 pip SL, 6-8 pip TP
    - Max 30 trades per session
    """
    
    def __init__(self, account_size: float = 10000.0, risk_per_trade: float = 0.005):
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
        self.max_trades_per_session = 30
        self.trades_today = 0
        self.session_trades = 0
        self.last_trade_time = None
        self.session_start = None
        
        # Strategy parameters
        self.ema_fast_period = 5
        self.ema_slow_period = 13
        self.rsi_period = 14
        self.atr_period = 14
        self.min_atr = 0.0008  # 8 pips for major pairs
        self.rsi_long_range = (45, 65)
        self.rsi_short_range = (35, 55)
        
    def is_trading_session(self, current_time: Optional[datetime] = None) -> bool:
        """Check if current time is within trading session (8:00-10:00 GMT)"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        # Convert to GMT if needed
        current_time_gmt = current_time
        
        session_start = time(8, 0)   # 8:00 GMT
        session_end = time(10, 0)    # 10:00 GMT
        
        return session_start <= current_time_gmt.time() <= session_end
    
    def reset_session_counters(self):
        """Reset session counters at 8:00 GMT"""
        current_time = datetime.utcnow()
        if current_time.hour == 8 and current_time.minute == 0:
            self.session_trades = 0
            self.session_start = current_time
    
    def can_trade(self) -> bool:
        """Check if trading is allowed based on rules"""
        if not self.is_trading_session():
            return False
        
        if self.session_trades >= self.max_trades_per_session:
            return False
            
        return True
    
    def calculate_pip_value(self, symbol: str, price: float) -> float:
        """Calculate pip value for different currency pairs"""
        pip_values = {
            'EUR/USD': 0.0001,
            'GBP/USD': 0.0001,
            'USD/JPY': 0.01
        }
        return pip_values.get(symbol, 0.0001)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, symbol: str) -> float:
        """Calculate position size based on risk in pips"""
        risk_amount = self.account_size * self.risk_per_trade
        pip_value = self.calculate_pip_value(symbol, entry_price)
        
        # Calculate stop loss in pips
        stop_loss_pips = abs(entry_price - stop_loss) / pip_value
        
        if stop_loss_pips == 0:
            return 0
        
        # Position size in units
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        return position_size
    
    def calculate_previous_high_low(self, data: pd.DataFrame, minutes: int = 15) -> Dict[str, float]:
        """Calculate previous 15-minute high and low"""
        if len(data) < minutes:
            return {'high': 0, 'low': 0}
        
        recent_data = data.iloc[-minutes:]
        return {
            'high': recent_data['high'].max(),
            'low': recent_data['low'].min()
        }
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all required indicators"""
        close = data['close']
        
        # EMAs
        ema_fast = ta.ema(close, length=self.ema_fast_period)
        ema_slow = ta.ema(close, length=self.ema_slow_period)
        
        # RSI
        rsi = ta.rsi(close, length=self.rsi_period)
        
        # ATR
        atr = ta.atr(data['high'], data['low'], close, length=self.atr_period)
        
        return {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'rsi': rsi,
            'atr': atr,
            'ema_bullish': ema_fast > ema_slow,
            'ema_bearish': ema_fast < ema_slow
        }
    
    def generate_signals(self, data: pd.DataFrame, symbol: str = 'EUR/USD') -> Dict:
        """
        Generate Breakout Scalping signals
        
        Args:
            data: DataFrame with OHLCV data (1-minute)
            symbol: Trading symbol for pip calculation
        
        Returns:
            Dict with trading signals and parameters
        """
        if not self.can_trade():
            return {'signal': 'NO_TRADE', 'reason': 'Outside trading session or max trades reached'}
        
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return {'signal': 'ERROR', 'reason': 'Missing OHLCV columns'}
        
        if len(data) < 20:  # Need enough data for indicators
            return {'signal': 'HOLD', 'reason': 'Insufficient data'}
        
        # Calculate indicators
        indicators = self.calculate_indicators(data)
        prev_high_low = self.calculate_previous_high_low(data, 15)
        
        current_price = data['close'].iloc[-1]
        current_rsi = indicators['rsi'].iloc[-1]
        current_atr = indicators['atr'].iloc[-1]
        ema_bullish = indicators['ema_bullish'].iloc[-1]
        ema_bearish = indicators['ema_bearish'].iloc[-1]
        
        previous_high = prev_high_low['high']
        previous_low = prev_high_low['low']
        
        # Entry conditions
        long_conditions = [
            current_price > previous_high,  # Break above previous 15min high
            ema_bullish,  # EMA 5 > EMA 13
            self.rsi_long_range[0] <= current_rsi <= self.rsi_long_range[1],  # RSI in range
            current_atr > self.min_atr,  # Sufficient volatility
            current_price > data['open'].iloc[-1]  # Bullish candle
        ]
        
        short_conditions = [
            current_price < previous_low,  # Break below previous 15min low
            ema_bearish,  # EMA 5 < EMA 13
            self.rsi_short_range[0] <= current_rsi <= self.rsi_short_range[1],  # RSI in range
            current_atr > self.min_atr,  # Sufficient volatility
            current_price < data['open'].iloc[-1]  # Bearish candle
        ]
        
        # Generate signals
        if all(long_conditions):
            signal = 'BUY'
            stop_loss = current_price - (10 * self.calculate_pip_value(symbol, current_price))
            take_profit = current_price + (8 * self.calculate_pip_value(symbol, current_price))
            reason = "Bullish breakout above 15min high with EMA/RSI confirmation"
            
        elif all(short_conditions):
            signal = 'SELL'
            stop_loss = current_price + (10 * self.calculate_pip_value(symbol, current_price))
            take_profit = current_price - (8 * self.calculate_pip_value(symbol, current_price))
            reason = "Bearish breakout below 15min low with EMA/RSI confirmation"
            
        else:
            signal = 'HOLD'
            stop_loss = None
            take_profit = None
            reason = "No breakout signal with confirmation"
        
        # Calculate position size
        position_size = 0
        if signal in ['BUY', 'SELL'] and stop_loss is not None:
            position_size = self.calculate_position_size(current_price, stop_loss, symbol)
            
            # Update trade counters
            self.session_trades += 1
            self.trades_today += 1
            self.last_trade_time = datetime.now()
        
        return {
            'signal': signal,
            'reason': reason,
            'current_price': current_price,
            'previous_high': previous_high,
            'previous_low': previous_low,
            'ema_fast': indicators['ema_fast'].iloc[-1],
            'ema_slow': indicators['ema_slow'].iloc[-1],
            'rsi': current_rsi,
            'atr': current_atr,
            'atr_pips': current_atr / self.calculate_pip_value(symbol, current_price),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_reward_ratio': 0.8,  # Fixed 0.8:1
            'session_trades': self.session_trades,
            'max_trades': self.max_trades_per_session,
            'timestamp': datetime.now()
        }
    
    def get_strategy_stats(self) -> Dict:
        """Get current strategy statistics"""
        current_time = datetime.utcnow()
        in_session = self.is_trading_session(current_time)
        
        return {
            'account_size': self.account_size,
            'risk_per_trade': self.risk_per_trade * 100,  # Percentage
            'trades_today': self.trades_today,
            'session_trades': self.session_trades,
            'max_session_trades': self.max_trades_per_session,
            'in_trading_session': in_session,
            'session_remaining': 120 - (current_time.hour * 60 + current_time.minute - 8 * 60) if in_session else 0,
            'last_trade_time': self.last_trade_time,
            'trading_pairs': ['EUR/USD', 'GBP/USD', 'USD/JPY']
        }