"""
LIVE PAPER TRADING WITH ALL STRATEGIES
Monitors live option chain and executes paper trades
Tests: Conservative, Aggressive, Strong_Signals, Bullish_Only, Bearish_Only, Scalping, Swing, ADAPTIVE
"""

from colorama import init, Fore, Style
import pandas as pd
from datetime import datetime, timedelta
import time
from api_helper import ShoonyaApiPy
import pyotp
import re
import threading
import json

init(autoreset=True)

# Import your existing code's key functions
from maket_direction import (
    NiftyOptionChainLTP,
    calculate_direction_signal,
    format_number
)


class TradingStrategy:
    """Base class for all trading strategies"""

    def __init__(self, name, params, lot_size=50):
        self.name = name
        self.params = params
        self.lot_size = lot_size
        self.in_trade = False
        self.entry_price = 0
        self.entry_nifty_price = 0
        self.entry_strike = 0
        self.entry_option_type = None
        self.entry_time = None
        self.entry_direction = None
        self.entry_minutes = 0
        self.trades = []
        self.daily_pnl = 0

        # OI exit tracking - SIMPLIFIED
        self.last_oi_exit_time = None
        self.last_oi_exit_score = None
        self.last_oi_exit_direction = None
        self.oi_exit_cooldown_min = 15  # 15 minute cooldown period

    def check_entry(self, spot_price, direction, score, timestamp, oi_check_func=None, get_option_ltp_func=None):
        """
        Simplified entry logic:
        - During cooldown: Only allow if score is STRONGER than exit
        - After cooldown: Reset tracking, let strategy decide
        - Always check OI barrier distance
        """
        if self.in_trade:
            return False, None

        # Calculate time since last OI exit
        cooldown_elapsed = 0
        if self.last_oi_exit_time:
            cooldown_elapsed = (timestamp - self.last_oi_exit_time).total_seconds() / 60

            # Reset score tracking after cooldown period
            if cooldown_elapsed >= self.oi_exit_cooldown_min:
                self.last_oi_exit_score = None
                self.last_oi_exit_direction = None
                self.last_oi_exit_time = None

        # BEARISH ENTRY
        if 'BEARISH' in self.params['trade_signals']:
            if direction in ['BEARISH', 'STRONG_BEARISH'] and abs(score) >= self.params['entry_score_bearish']:

                # Check score strength if still in cooldown period
                if self.last_oi_exit_score is not None:
                    if self.last_oi_exit_direction == 'BEARISH':
                        # For bearish: current score must be MORE negative
                        if score > self.last_oi_exit_score:
                            print(
                                f"  ⚠️ {self.name} - Cooldown: Score weakened ({self.last_oi_exit_score:.2f} → {score:.2f})")
                            return False, None
                        else:
                            print(
                                f"  ✅ {self.name} - Cooldown: Score strengthened ({self.last_oi_exit_score:.2f} → {score:.2f})")

                # Check OI barrier at current price
                if oi_check_func:
                    should_avoid, strike, ratio = oi_check_func(spot_price, 'BEARISH', range_points=20)
                    if should_avoid:
                        print(f"  ⚠️ {self.name} - AT OI BARRIER {strike} (PE/CE: {ratio:.2f})")
                        return False, None

                    # Check distance to nearest support
                    nearest_support = self._find_nearest_oi_barrier(spot_price, 'BEARISH', oi_check_func)
                    if nearest_support:
                        distance = spot_price - nearest_support
                        if distance < 50:
                            print(f"  ⚠️ {self.name} - Too close to support {nearest_support} ({distance:.0f} pts)")
                            return False, None

                # Get ATM Put option
                atm_strike = int(round(spot_price / 50, 0)) * 50
                option_ltp = 0
                if get_option_ltp_func:
                    option_ltp = get_option_ltp_func(atm_strike, 'PE')

                if option_ltp == 0:
                    print(f"  ⚠️ {self.name} - No option price for {atm_strike} PE")
                    return False, None

                # Enter trade
                self.in_trade = True
                self.entry_price = option_ltp
                self.entry_nifty_price = spot_price
                self.entry_strike = atm_strike
                self.entry_option_type = 'PE'
                self.entry_time = timestamp
                self.entry_direction = 'BEARISH'
                self.entry_minutes = 0

                cooldown_msg = f"(Cooldown {cooldown_elapsed:.0f}m)" if self.last_oi_exit_score else ""
                print(f"  ✅ {self.name} - BUY {atm_strike} PE @ ₹{option_ltp:.2f} (Score: {score:.2f}) {cooldown_msg}")
                return True, 'BEARISH'

        # BULLISH ENTRY
        if 'BULLISH' in self.params['trade_signals']:
            if direction in ['BULLISH', 'STRONG_BULLISH'] and score >= self.params['entry_score_bullish']:

                # Check score strength if still in cooldown period
                if self.last_oi_exit_score is not None:
                    if self.last_oi_exit_direction == 'BULLISH':
                        # For bullish: current score must be MORE positive
                        if score < self.last_oi_exit_score:
                            print(
                                f"  ⚠️ {self.name} - Cooldown: Score weakened ({self.last_oi_exit_score:.2f} → {score:.2f})")
                            return False, None
                        else:
                            print(
                                f"  ✅ {self.name} - Cooldown: Score strengthened ({self.last_oi_exit_score:.2f} → {score:.2f})")

                # Check OI barrier at current price
                if oi_check_func:
                    should_avoid, strike, ratio = oi_check_func(spot_price, 'BULLISH', range_points=20)
                    if should_avoid:
                        print(f"  ⚠️ {self.name} - AT OI BARRIER {strike} (CE/PE: {ratio:.2f})")
                        return False, None

                    # Check distance to nearest resistance
                    nearest_resistance = self._find_nearest_oi_barrier(spot_price, 'BULLISH', oi_check_func)
                    if nearest_resistance:
                        distance = nearest_resistance - spot_price
                        if distance < 50:
                            print(
                                f"  ⚠️ {self.name} - Too close to resistance {nearest_resistance} ({distance:.0f} pts)")
                            return False, None

                # Get ATM Call option
                atm_strike = int(round(spot_price / 50, 0)) * 50
                option_ltp = 0
                if get_option_ltp_func:
                    option_ltp = get_option_ltp_func(atm_strike, 'CE')

                if option_ltp == 0:
                    print(f"  ⚠️ {self.name} - No option price for {atm_strike} CE")
                    return False, None

                # Enter trade
                self.in_trade = True
                self.entry_price = option_ltp
                self.entry_nifty_price = spot_price
                self.entry_strike = atm_strike
                self.entry_option_type = 'CE'
                self.entry_time = timestamp
                self.entry_direction = 'BULLISH'
                self.entry_minutes = 0

                cooldown_msg = f"(Cooldown {cooldown_elapsed:.0f}m)" if self.last_oi_exit_score else ""
                print(f"  ✅ {self.name} - BUY {atm_strike} CE @ ₹{option_ltp:.2f} (Score: {score:.2f}) {cooldown_msg}")
                return True, 'BULLISH'

        return False, None


    def check_exit(self, spot_price, direction, score, timestamp, oi_check_func=None, get_option_ltp_func=None):
        """Check if should exit trade"""
        if not self.in_trade:
            return False, None, 0

        # Calculate holding time from timestamps
        holding_minutes = (timestamp - self.entry_time).total_seconds() / 60

        # Don't exit before minimum holding time
        if holding_minutes < self.params['min_holding_min']:
            return False, None, 0

        # Get current option LTP
        current_option_ltp = 0
        if get_option_ltp_func:
            current_option_ltp = get_option_ltp_func(self.entry_strike, self.entry_option_type)

        if current_option_ltp == 0:
            return False, None, 0  # Skip if can't get option price
        # **Check OI barrier exit FIRST (highest priority)**
        if oi_check_func:
            should_exit_oi, strike, ratio = oi_check_func(spot_price, self.entry_direction)
            if should_exit_oi:
                profit_points = current_option_ltp - self.entry_price
                profit_inr = profit_points * self.lot_size

                trade = {
                    'strategy': self.name,
                    'entry_time': self.entry_time,
                    'entry_price': self.entry_price,
                    'exit_price': current_option_ltp,
                    'entry_strike': self.entry_strike,
                    'option_type': self.entry_option_type,
                    'entry_direction': self.entry_direction,
                    'exit_time': timestamp,
                    'points': profit_points,
                    'profit_inr': profit_inr,
                    'exit_reason': f"OI_BARRIER@{strike}",
                    'holding_min': holding_minutes
                }

                self.trades.append(trade)
                self.daily_pnl += profit_inr
                self.in_trade = False

                # Record exit details for cooldown tracking - IMPORTANT!
                self.last_oi_exit_time = timestamp
                self.last_oi_exit_score = score
                self.last_oi_exit_direction = self.entry_direction

                print(
                    f"  🚧 OI BARRIER EXIT at {strike} - SELL {self.entry_strike} {self.entry_option_type} @ ₹{current_option_ltp:.2f} | Score: {score:.2f}")
                return True, f"OI_BARRIER@{strike}", profit_inr

        # Calculate P&L based on option premium change
        premium_change = current_option_ltp - self.entry_price
        premium_change_pct = (premium_change / self.entry_price) * 100

        # Initialize exit_reason
        exit_reason = None

        # Modified Stop Loss: 30% loss on option premium OR points-based on Nifty
        sl_percentage = 30  # 30% loss triggers SL
        if premium_change_pct <= -sl_percentage:
            exit_reason = 'STOP_LOSS'

        # Target: Check if option gained enough (convert target_points to premium)
        # Rule: For every 50 points Nifty move, option moves ~40-60 Rs (approximate)
        # More accurate: use percentage gain
        target_percentage = (self.params['target_points'] / 50) * 100  # Rough conversion
        if premium_change_pct >= target_percentage:
            exit_reason = 'TARGET'

        # Score reversal
        if self.entry_direction == 'BULLISH':
            if score <= -self.params['exit_score_threshold']:
                exit_reason = 'SCORE_REVERSAL'
            elif direction in ['BEARISH', 'STRONG_BEARISH']:
                exit_reason = 'DIRECTION_CHANGE'
        else:  # BEARISH
            if score >= self.params['exit_score_threshold']:
                exit_reason = 'SCORE_REVERSAL'
            elif direction in ['BULLISH', 'STRONG_BULLISH']:
                exit_reason = 'DIRECTION_CHANGE'

        # Max time
        if holding_minutes >= self.params['max_holding_min']:
            exit_reason = 'MAX_TIME'

        # Execute exit if condition met
        # Execute exit if condition met
        if exit_reason:
            profit_points = premium_change
            profit_inr = profit_points * self.lot_size

            trade = {
                'strategy': self.name,
                'entry_time': self.entry_time,
                'entry_price': self.entry_price,
                'exit_price': current_option_ltp,
                'entry_strike': self.entry_strike,
                'option_type': self.entry_option_type,
                'entry_direction': self.entry_direction,
                'exit_time': timestamp,
                'points': profit_points,
                'profit_inr': profit_inr,
                'exit_reason': exit_reason,
                'holding_min': holding_minutes
            }

            self.trades.append(trade)
            self.daily_pnl += profit_inr
            self.in_trade = False

            # Only record cooldown for regular exits (not OI exits - those are recorded above)
            # DON'T record cooldown for regular exits - only for OI exits

            print(
                f"  🔴 EXIT - SELL {self.entry_strike} {self.entry_option_type} @ ₹{current_option_ltp:.2f} | {exit_reason} | P&L: ₹{profit_inr:+,.0f}")
            return True, exit_reason, profit_inr

        return False, None, 0  # IMPORTANT: Add this line at the end

    def _find_nearest_oi_barrier(self, spot_price, direction, oi_check_func, max_range=300):
        """
        Find nearest OI barrier in trade direction - ROUND NUMBERS ONLY (100 intervals)
        Returns: Strike price of nearest barrier or None

        Example: 25900, 26000, 26100 (ignores 25950, 26050, etc.)
        """
        if not oi_check_func:
            return None

        # Round current price to nearest 100
        current_rounded = int(round(spot_price / 100, 0)) * 100

        if direction == 'BEARISH':
            # Check strikes BELOW current price (supports) - only round numbers
            for offset in [100, 200, 300]:  # Check 100, 200, 300 points below
                test_strike = current_rounded - offset

                # Skip if strike is above current price
                if test_strike >= spot_price:
                    continue

                # Check if this round strike is an OI barrier
                should_avoid, strike, ratio = oi_check_func(test_strike, direction, range_points=10)
                if should_avoid:
                    return test_strike  # Found nearest round number support

        elif direction == 'BULLISH':
            # Check strikes ABOVE current price (resistances) - only round numbers
            for offset in [100, 200, 300]:  # Check 100, 200, 300 points above
                test_strike = current_rounded + offset

                # Skip if strike is below current price
                if test_strike <= spot_price:
                    continue

                # Check if this round strike is an OI barrier
                should_avoid, strike, ratio = oi_check_func(test_strike, direction, range_points=10)
                if should_avoid:
                    return test_strike  # Found nearest round number resistance

        return None  # No barrier found within range


class AdaptiveStrategy(TradingStrategy):
    """Adaptive strategy that switches between sub-strategies based on market conditions"""

    def __init__(self, lot_size=50):
        # Define all sub-strategies
        self.sub_strategies = {
            'Bearish_Only': {'entry_score_bullish': 999, 'entry_score_bearish': 1.5, 'exit_score_threshold': 0.5,
                             'stop_loss_points': 50, 'target_points': 100, 'min_holding_min': 5, 'max_holding_min': 60,
                             'trade_signals': ['BEARISH']},
            'Scalping': {'entry_score_bullish': 1.5, 'entry_score_bearish': 1.5, 'exit_score_threshold': 0.3,
                         'stop_loss_points': 30, 'target_points': 50, 'min_holding_min': 2, 'max_holding_min': 15,
                         'trade_signals': ['BULLISH', 'BEARISH']},
            'Swing': {'entry_score_bullish': 2.0, 'entry_score_bearish': 2.0, 'exit_score_threshold': 1.0,
                      'stop_loss_points': 100, 'target_points': 250, 'min_holding_min': 15, 'max_holding_min': 120,
                      'trade_signals': ['BULLISH', 'BEARISH']},
            'Aggressive': {'entry_score_bullish': 1.5, 'entry_score_bearish': 1.5, 'exit_score_threshold': 0.5,
                           'stop_loss_points': 75, 'target_points': 100, 'min_holding_min': 3, 'max_holding_min': 45,
                           'trade_signals': ['BULLISH', 'BEARISH']}
        }

        super().__init__('ADAPTIVE', None, lot_size)
        self.current_sub_strategy = None
        self.market_open_price = 0
        self.market_high = 0
        self.market_low = 0

    def detect_market_conditions(self, spot_price):
        """Detect current market conditions"""
        if self.market_open_price == 0:
            self.market_open_price = spot_price
            self.market_high = spot_price
            self.market_low = spot_price
            return 'RANGE_BOUND', 'LOW'

        self.market_high = max(self.market_high, spot_price)
        self.market_low = min(self.market_low, spot_price)

        daily_change_pct = ((spot_price - self.market_open_price) / self.market_open_price) * 100
        range_pts = self.market_high - self.market_low

        # Market type
        if daily_change_pct > 0.3:
            market_type = 'TRENDING_UP'
        elif daily_change_pct < -0.3:
            market_type = 'TRENDING_DOWN'
        else:
            market_type = 'RANGE_BOUND'

        # Volatility
        if range_pts > 150:
            volatility = 'HIGH'
        elif range_pts > 100:
            volatility = 'MEDIUM'
        else:
            volatility = 'LOW'

        return market_type, volatility

    def select_sub_strategy(self, market_type, volatility):
        """Select best sub-strategy based on market conditions"""
        # Rule: TRENDING_DOWN + HIGH = Swing
        if market_type == 'TRENDING_DOWN' and volatility == 'HIGH':
            return 'Swing', self.sub_strategies['Swing']

        # Rule: TRENDING_DOWN + MEDIUM/LOW = Bearish_Only
        if market_type == 'TRENDING_DOWN':
            return 'Bearish_Only', self.sub_strategies['Bearish_Only']

        # Rule: RANGE_BOUND + HIGH = Bearish_Only
        if market_type == 'RANGE_BOUND' and volatility == 'HIGH':
            return 'Bearish_Only', self.sub_strategies['Bearish_Only']

        # Rule: RANGE_BOUND + MEDIUM/LOW = Scalping
        if market_type == 'RANGE_BOUND':
            return 'Scalping', self.sub_strategies['Scalping']

        # Rule: TRENDING_UP = Bearish_Only (counterintuitive but works in data!)
        if market_type == 'TRENDING_UP':
            return 'Bearish_Only', self.sub_strategies['Bearish_Only']

        # Default
        return 'Bearish_Only', self.sub_strategies['Bearish_Only']

    def check_entry(self, spot_price, direction, score, timestamp, oi_check_func=None, get_option_ltp_func=None):
        """Adaptive entry logic"""
        if self.in_trade:
            return False, None

        # Detect market conditions and select strategy
        market_type, volatility = self.detect_market_conditions(spot_price)
        sub_name, sub_params = self.select_sub_strategy(market_type, volatility)
        self.current_sub_strategy = sub_name
        self.params = sub_params

        # Use parent class entry logic (with OI check AND option LTP function)
        entered, direction = super().check_entry(spot_price, direction, score, timestamp, oi_check_func,
                                                 get_option_ltp_func)

        if entered:
            return True, f"{direction} ({sub_name})"

        return False, None


class LivePaperTradingSystem(NiftyOptionChainLTP):
    """Extended live trading system with all strategies"""

    def __init__(self):
        super().__init__()

        # Initialize all strategies
        self.strategies = {
            'Conservative': TradingStrategy('Conservative', {'entry_score_bullish': 2.5, 'entry_score_bearish': 2.5,
                                                             'exit_score_threshold': 0.5, 'stop_loss_points': 50,
                                                             'target_points': 150, 'min_holding_min': 5,
                                                             'max_holding_min': 60,
                                                             'trade_signals': ['BULLISH', 'BEARISH']}),
            'Aggressive': TradingStrategy('Aggressive', {'entry_score_bullish': 1.5, 'entry_score_bearish': 1.5,
                                                         'exit_score_threshold': 0.5, 'stop_loss_points': 75,
                                                         'target_points': 100, 'min_holding_min': 3,
                                                         'max_holding_min': 45,
                                                         'trade_signals': ['BULLISH', 'BEARISH']}),
            'Strong_Signals_Only': TradingStrategy('Strong_Signals_Only',
                                                   {'entry_score_bullish': 3.0, 'entry_score_bearish': 3.0,
                                                    'exit_score_threshold': 1.0, 'stop_loss_points': 40,
                                                    'target_points': 200, 'min_holding_min': 10, 'max_holding_min': 90,
                                                    'trade_signals': ['BULLISH', 'BEARISH']}),
            'Bullish_Only': TradingStrategy('Bullish_Only', {'entry_score_bullish': 1.5, 'entry_score_bearish': 999,
                                                             'exit_score_threshold': 0.5, 'stop_loss_points': 50,
                                                             'target_points': 100, 'min_holding_min': 5,
                                                             'max_holding_min': 60, 'trade_signals': ['BULLISH']}),
            'Bearish_Only': TradingStrategy('Bearish_Only', {'entry_score_bullish': 999, 'entry_score_bearish': 1.5,
                                                             'exit_score_threshold': 0.5, 'stop_loss_points': 50,
                                                             'target_points': 100, 'min_holding_min': 5,
                                                             'max_holding_min': 60, 'trade_signals': ['BEARISH']}),
            'Scalping': TradingStrategy('Scalping', {'entry_score_bullish': 1.5, 'entry_score_bearish': 1.5,
                                                     'exit_score_threshold': 0.3, 'stop_loss_points': 30,
                                                     'target_points': 50, 'min_holding_min': 2, 'max_holding_min': 15,
                                                     'trade_signals': ['BULLISH', 'BEARISH']}),
            'Swing': TradingStrategy('Swing', {'entry_score_bullish': 2.0, 'entry_score_bearish': 2.0,
                                               'exit_score_threshold': 1.0, 'stop_loss_points': 100,
                                               'target_points': 250, 'min_holding_min': 15, 'max_holding_min': 120,
                                               'trade_signals': ['BULLISH', 'BEARISH']}),
            'ADAPTIVE': AdaptiveStrategy()
        }

        self.minute_counter = 0

    def process_trading_signals(self):
        """Process trading signals for all strategies"""
        analysis = self.analyze_market_direction()

        if not analysis:
            return

        timestamp = datetime.now()
        spot_price = self.current_ltp
        direction = analysis['direction']
        score = analysis['score']

        # Increment minute counter
        self.minute_counter += 1

        # Check all strategies
        for name, strategy in self.strategies.items():
            # Check for exit first
            if strategy.in_trade:
                exited, exit_reason, profit = strategy.check_exit(spot_price, direction, score, timestamp)

                if exited:
                    profit_color = Fore.GREEN if profit > 0 else Fore.RED
                    print(f"\n{Fore.YELLOW}[{name}] EXIT {strategy.entry_direction} @ {spot_price:.2f} | "
                          f"{exit_reason} | {profit_color}{profit:+,.0f} INR{Style.RESET_ALL} | "
                          f"Hold: {strategy.entry_minutes} min")

            # Check for entry
            else:
                entered, entry_dir = strategy.check_entry(spot_price, direction, score, timestamp)

                if entered:
                    print(f"\n{Fore.CYAN}[{name}] ENTER {entry_dir} @ {spot_price:.2f} | "
                          f"Score: {score:.2f} | Time: {timestamp.strftime('%H:%M:%S')}{Style.RESET_ALL}")

    def display_trading_dashboard(self):
        """Display comprehensive trading dashboard"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'=' * 140}{Style.RESET_ALL}")
        print(
            f"{Fore.CYAN}{Style.BRIGHT}LIVE PAPER TRADING DASHBOARD - {datetime.now().strftime('%H:%M:%S')}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 140}{Style.RESET_ALL}\n")

        # Market overview
        analysis = self.analyze_market_direction()
        if analysis:
            print(f"{analysis['color']}Market: {analysis['direction']} | Score: {analysis['score']:.2f} | "
                  f"Spot: {self.current_ltp:.2f}{Style.RESET_ALL}\n")

        # Strategy performance
        print(
            f"{Style.BRIGHT}{'Strategy':<20} {'Status':<15} {'Trades':>7} {'P&L Today':>12} {'In Trade':<30}{Style.RESET_ALL}")
        print(f"{'-' * 100}")

        for name, strategy in sorted(self.strategies.items(), key=lambda x: x[1].daily_pnl, reverse=True):
            status_color = Fore.GREEN if strategy.in_trade else Fore.WHITE
            pnl_color = Fore.GREEN if strategy.daily_pnl > 0 else (Fore.RED if strategy.daily_pnl < 0 else Fore.WHITE)

            if strategy.in_trade:
                current_pnl = (
                                          self.current_ltp - strategy.entry_price) * strategy.lot_size if strategy.entry_direction == 'BULLISH' else (
                                                                                                                                                                 strategy.entry_price - self.current_ltp) * strategy.lot_size
                status_str = f"🔴 {strategy.entry_direction} @ {strategy.entry_price:.2f}"
                trade_info = f"Live: {current_pnl:+,.0f} INR | {strategy.entry_minutes} min"
            else:
                status_str = "Monitoring"
                trade_info = "-"

            print(f"{name:<20} {status_color}{status_str:<15}{Style.RESET_ALL} "
                  f"{len(strategy.trades):>7} {pnl_color}{strategy.daily_pnl:>12,.0f}{Style.RESET_ALL} "
                  f"{trade_info:<30}")

        print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 140}{Style.RESET_ALL}\n")

    def save_trades_log(self):
        """Save all trades to JSON file"""
        all_trades = []

        for name, strategy in self.strategies.items():
            for trade in strategy.trades:
                trade['strategy'] = name
                all_trades.append(trade)

        filename = f"paper_trades_{datetime.now().strftime('%Y%m%d')}.json"
        with open(filename, 'w') as f:
            json.dump(all_trades, f, indent=2, default=str)

        print(f"{Fore.GREEN}Trades saved to {filename}{Style.RESET_ALL}")


def main():
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 140}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}LIVE PAPER TRADING SYSTEM - ALL STRATEGIES{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'=' * 140}{Style.RESET_ALL}\n")

    trading_system = LivePaperTradingSystem()

    if trading_system.login():
        trading_system.get_option_chain_tokens()

        print("\nWaiting for live data...")
        time.sleep(10)

        try:
            iteration = 0

            while True:
                # Process trading signals every minute
                trading_system.process_trading_signals()

                # Display dashboard every 5 minutes
                if iteration % 2 == 0:
                    trading_system.display_trading_dashboard()

                # Display option chain
                trading_system.display_option_chain()

                iteration += 1
                time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            print("\n\nStopping paper trading...")
            trading_system.stop_depth_updates()
            trading_system.save_trades_log()

            # Final summary
            print(f"\n{Fore.CYAN}{Style.BRIGHT}FINAL SUMMARY{Style.RESET_ALL}\n")
            trading_system.display_trading_dashboard()


if __name__ == "__main__":
    main()
