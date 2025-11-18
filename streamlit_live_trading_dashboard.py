"""
STREAMLIT LIVE TRADING DASHBOARD - COMPLETE FIXED VERSION
Properly inherits from market_direction.py and uses backtest score calculation
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import glob


# Import the complete market direction system
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maket_direction import NiftyOptionChainLTP
from live_paper_trading_all_strategies import TradingStrategy, AdaptiveStrategy

# Page config
st.set_page_config(
    page_title="Live Trading Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    .stDataFrame {font-size: 11px;}
    div[data-testid="stMetricValue"] {font-size: 20px;}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trading_system' not in st.session_state:
    st.session_state.trading_system = None
    st.session_state.running = False
    st.session_state.last_update = None

# Data Logger
class DataLogger:
    def __init__(self, base_path="trading_data"):
        self.base_path = base_path
        self.today = datetime.now().strftime('%Y%m%d')
        os.makedirs(f"{base_path}/{self.today}", exist_ok=True)

        # Files for today
        self.files = {
            'trades': f"{base_path}/{self.today}/trades.csv",
            'strategy_pnl': f"{base_path}/{self.today}/strategy_pnl.csv"
        }

        # Initialize CSV with headers if doesn't exist
        if not os.path.exists(self.files['trades']):
            header_df = pd.DataFrame(columns=[
                'timestamp', 'strategy', 'entry_time', 'exit_time',
                'direction', 'entry_price', 'exit_price', 'points',
                'profit_inr', 'exit_reason', 'holding_min'
            ])
            header_df.to_csv(self.files['trades'], index=False)

    def append_trade(self, trade):
        """Append completed trade with full details"""
        df = pd.DataFrame([{
            'timestamp': datetime.now(),
            'strategy': trade.get('strategy', 'Unknown'),
            'entry_time': trade['entry_time'],
            'exit_time': trade['exit_time'],
            'direction': trade['entry_direction'],
            'entry_price': trade['entry_price'],
            'exit_price': trade['exit_price'],
            'points': trade['points'],
            'profit_inr': trade['profit_inr'],
            'exit_reason': trade['exit_reason'],
            'holding_min': trade['holding_min']
        }])

        # Always append (header already exists)
        df.to_csv(self.files['trades'], mode='a', header=False, index=False)

    def load_all_historical_trades(self):
        """Load ALL historical trades from all dates"""
        all_trades = []

        # Get all date folders
        if os.path.exists(self.base_path):
            date_folders = glob.glob(f"{self.base_path}/*/")

            for folder in sorted(date_folders, reverse=True):  # Most recent first
                trades_file = os.path.join(folder, "trades.csv")
                if os.path.exists(trades_file):
                    try:
                        df = pd.read_csv(trades_file)
                        if not df.empty:
                            all_trades.append(df)
                    except Exception as e:
                        print(f"Error reading {trades_file}: {e}")

        # Combine all trades
        if all_trades:
            combined_df = pd.concat(all_trades, ignore_index=True)
            # Convert timestamp columns to datetime
            for col in ['timestamp', 'entry_time', 'exit_time']:
                if col in combined_df.columns:
                    combined_df[col] = pd.to_datetime(combined_df[col], errors='coerce')
            # Sort by most recent first
            combined_df = combined_df.sort_values('exit_time', ascending=False).reset_index(drop=True)
            return combined_df

        return pd.DataFrame()

    def save_strategy_pnl(self, strategies_dict):
        data = []
        for name, strategy in strategies_dict.items():
            data.append({
                'timestamp': datetime.now(),
                'strategy': name,
                'trades': len(strategy.trades),
                'daily_pnl': strategy.daily_pnl,
                'in_trade': strategy.in_trade
            })
        df = pd.DataFrame(data)
        df.to_csv(self.files['strategy_pnl'], index=False)

# Complete Trading System for Streamlit
class StreamlitTradingSystem(NiftyOptionChainLTP):
    """Properly inherits from NiftyOptionChainLTP with all strategies"""

    def __init__(self):
        super().__init__()
        self.logger = DataLogger()
        self.last_signal_process = None  # Track last signal processing time

        # Initialize all trading strategies
        self.strategies = {
            'Conservative': TradingStrategy('Conservative', {'entry_score_bullish': 2.5, 'entry_score_bearish': 2.5, 'exit_score_threshold': 0.5, 'stop_loss_points': 50, 'target_points': 150, 'min_holding_min': 5, 'max_holding_min': 60, 'trade_signals': ['BULLISH', 'BEARISH']}),
            'Aggressive': TradingStrategy('Aggressive', {'entry_score_bullish': 1.5, 'entry_score_bearish': 1.5, 'exit_score_threshold': 0.5, 'stop_loss_points': 75, 'target_points': 100, 'min_holding_min': 3, 'max_holding_min': 45, 'trade_signals': ['BULLISH', 'BEARISH']}),
            'Strong_Signals_Only': TradingStrategy('Strong_Signals_Only', {'entry_score_bullish': 3.0, 'entry_score_bearish': 3.0, 'exit_score_threshold': 1.0, 'stop_loss_points': 40, 'target_points': 200, 'min_holding_min': 10, 'max_holding_min': 90, 'trade_signals': ['BULLISH', 'BEARISH']}),
            'Bullish_Only': TradingStrategy('Bullish_Only', {'entry_score_bullish': 1.5, 'entry_score_bearish': 999, 'exit_score_threshold': 0.5, 'stop_loss_points': 50, 'target_points': 100, 'min_holding_min': 5, 'max_holding_min': 60, 'trade_signals': ['BULLISH']}),
            'Bearish_Only': TradingStrategy('Bearish_Only', {'entry_score_bullish': 999, 'entry_score_bearish': 1.5, 'exit_score_threshold': 0.5, 'stop_loss_points': 50, 'target_points': 100, 'min_holding_min': 5, 'max_holding_min': 60, 'trade_signals': ['BEARISH']}),
            'Scalping': TradingStrategy('Scalping', {'entry_score_bullish': 1.5, 'entry_score_bearish': 1.5, 'exit_score_threshold': 0.3, 'stop_loss_points': 30, 'target_points': 50, 'min_holding_min': 2, 'max_holding_min': 15, 'trade_signals': ['BULLISH', 'BEARISH']}),
            'Swing': TradingStrategy('Swing', {'entry_score_bullish': 2.0, 'entry_score_bearish': 2.0, 'exit_score_threshold': 1.0, 'stop_loss_points': 100, 'target_points': 250, 'min_holding_min': 15, 'max_holding_min': 120, 'trade_signals': ['BULLISH', 'BEARISH']}),
            'ADAPTIVE': AdaptiveStrategy()
        }
        if hasattr(self, 'start_depth_updates'):
            self.start_depth_updates()
            print("✅ Started bid-ask depth updates")

    def process_trading_signals(self):
        """Process signals - ONLY ONCE PER MINUTE"""

        # Check if we should process (once per minute)
        now = datetime.now()

        if self.last_signal_process:
            time_diff = (now - self.last_signal_process).total_seconds()
            if time_diff < 60:  # Less than 1 minute since last process
                return  # Skip this call

        self.last_signal_process = now

        # Now process signals
        analysis = self.analyze_market_direction()

        if not analysis:
            return

        timestamp = now
        spot_price = self.current_ltp
        direction = analysis['direction']
        score = analysis['score']

        for name, strategy in self.strategies.items():
            if strategy.in_trade:
                exited, exit_reason, profit = strategy.check_exit(
                    spot_price, direction, score, timestamp,
                    oi_check_func=self.check_oi_support_resistance,
                    get_option_ltp_func=self.get_option_ltp  # Add this
                )

                if exited:
                    # Get the last trade that was just added
                    last_trade = strategy.trades[-1]

                    # Save to CSV immediately
                    self.logger.append_trade(last_trade)

                    print(f"[{timestamp.strftime('%H:%M:%S')}] {name} EXIT: {exit_reason} | "
                          f"P&L: ₹{profit:+,.0f} | Points: {last_trade['points']:+.2f}")
            # Check for entry
            else:
                entered, entry_dir = strategy.check_entry(
                    spot_price, direction, score, timestamp,
                    oi_check_func=self.check_oi_support_resistance,
                    get_option_ltp_func=self.get_option_ltp  # Add this
                )

                if entered:
                    print(
                        f"[{timestamp.strftime('%H:%M:%S')}] {name} ENTER {entry_dir} @ {spot_price:.2f} | Score: {score:.2f}")

    def get_basic_option_chain_formatted(self):
        """Format option chain with strike score"""
        if not hasattr(self, 'option_data') or not self.option_data:
            return pd.DataFrame()

        # Get ATM consistently
        analysis = self.analyze_market_direction()
        atm = analysis['atm_strike'] if analysis else int(round(self.current_ltp / 50, 0)) * 50

        # Build strikes dict
        strikes_dict = {}
        for token, data in self.option_data.items():
            strike = int(data.get('strike', 0))
            if strike not in strikes_dict:
                strikes_dict[strike] = {
                    'Strike': strike,
                    '_is_atm': (strike == atm)
                }

            if data.get('type') == 'CE':
                strikes_dict[strike]['CE B-A'] = data.get('qtydiff', 0)
                strikes_dict[strike]['CE LTP'] = f"{data.get('ltp', 0):.2f}"
                strikes_dict[strike]['CE OI'] = f"{data.get('oi', 0):,}"
                strikes_dict[strike]['CE OI Chg'] = f"{data.get('oi_chg_day', 0):,}"
                strikes_dict[strike]['CE Volume'] = f"{data.get('volume', 0):,}"

            elif data.get('type') == 'PE':
                strikes_dict[strike]['PE LTP'] = f"{data.get('ltp', 0):.2f}"
                strikes_dict[strike]['PE OI'] = f"{data.get('oi', 0):,}"
                strikes_dict[strike]['PE OI Chg'] = f"{data.get('oi_chg_day', 0):,}"
                strikes_dict[strike]['PE Volume'] = f"{data.get('volume', 0):,}"
                strikes_dict[strike]['PE B-A'] = data.get('qtydiff', 0)

        if not strikes_dict:
            return pd.DataFrame()

        # Calculate score for each strike
        for strike in strikes_dict.keys():
            strikes_dict[strike]['Score'] = self.calculate_strike_score(strike)

        # Create DataFrame
        df = pd.DataFrame(list(strikes_dict.values()))
        df = df.sort_values('Strike').reset_index(drop=True)

        # Column order (Score added between Strike and PE LTP)
        cols = [
            'CE B-A', 'CE OI', 'CE OI Chg', 'CE Volume', 'CE LTP',
            'Strike', 'Score',
            'PE LTP', 'PE Volume', 'PE OI Chg', 'PE OI', 'PE B-A'
        ]

        # Fill missing columns
        for col in cols:
            if col not in df.columns:
                if col == 'Score':
                    df[col] = 0.0
                elif 'B-A' in col:
                    df[col] = 0
                else:
                    df[col] = '0'

        df = df[cols + ['_is_atm']]

        return df

    def highlight_extremes(self, df):
        """Apply highlighting with Score column"""
        df_numeric = df.replace(',', '', regex=True)

        # Convert numeric columns
        for col in ['CE OI', 'CE OI Chg', 'PE OI', 'PE OI Chg', 'CE Volume', 'PE Volume']:
            df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')

        # Get sorted values for OI/Volume highlighting
        ce_oi_sorted = df_numeric['CE OI'].sort_values(ascending=False).unique()
        ce_oichg_sorted = df_numeric['CE OI Chg'].sort_values(ascending=False).unique()
        pe_oi_sorted = df_numeric['PE OI'].sort_values(ascending=False).unique()
        pe_oichg_sorted = df_numeric['PE OI Chg'].sort_values(ascending=False).unique()
        ce_vol_sorted = df_numeric['CE Volume'].sort_values(ascending=False).unique()
        pe_vol_sorted = df_numeric['PE Volume'].sort_values(ascending=False).unique()

        def style_func(val, col):
            # Score column styling
            if col == 'Score':
                try:
                    score = float(val)
                    if score >= 3.0:
                        return 'background-color:#28a745;color:white;font-weight:bold'  # Strong bullish
                    elif score >= 1.5:
                        return 'background-color:#7bc96f;color:black'  # Bullish
                    elif score <= -3.0:
                        return 'background-color:#dc3545;color:white;font-weight:bold'  # Strong bearish
                    elif score <= -1.5:
                        return 'background-color:#f66;color:white'  # Bearish
                    else:
                        return 'background-color:#ffc107;color:black'  # Neutral
                except:
                    return ''

            # Existing styling for other columns
            try:
                val_num = float(val.replace(',', ''))
            except:
                return ''

            if col == 'CE OI':
                if val_num == ce_oi_sorted[0]:
                    return 'background-color:#ff5c5c;color:white;font-weight:bold'
                elif len(ce_oi_sorted) > 1 and val_num == ce_oi_sorted[1]:
                    return 'background-color:#ffb3b3;color:black'

            if col == 'CE OI Chg':
                if val_num == ce_oichg_sorted[0]:
                    return 'background-color:#b90b0b;color:white;font-weight:bold'
                elif len(ce_oichg_sorted) > 1 and val_num == ce_oichg_sorted[1]:
                    return 'background-color:#ffb3b3;color:black'

            if col == 'PE OI':
                if val_num == pe_oi_sorted[0]:
                    return 'background-color:#13ad13;color:white;font-weight:bold'
                elif len(pe_oi_sorted) > 1 and val_num == pe_oi_sorted[1]:
                    return 'background-color:#b3ffb3;color:black'

            if col == 'PE OI Chg':
                if val_num == pe_oichg_sorted[0]:
                    return 'background-color:#0b793b;color:white;font-weight:bold'
                elif len(pe_oichg_sorted) > 1 and val_num == pe_oichg_sorted[1]:
                    return 'background-color:#b3ffb3;color:black'

            if col == 'CE Volume':
                if val_num == ce_vol_sorted[0]:
                    return 'background-color:#6dfc6d;color:black;font-weight:bold'
                elif len(ce_vol_sorted) > 1 and val_num == ce_vol_sorted[1]:
                    return 'background-color:#cefaad;color:black'

            if col == 'PE Volume':
                if val_num == pe_vol_sorted[0]:
                    return 'background-color:#fd9b9b;color:black;font-weight:bold'
                elif len(pe_vol_sorted) > 1 and val_num == pe_vol_sorted[1]:
                    return 'background-color:#ffd6d6;color:black'

            return ''

        styled = df.style \
            .applymap(lambda v: style_func(v, 'Score'), subset=['Score']) \
            .applymap(lambda v: style_func(v, 'CE OI'), subset=['CE OI']) \
            .applymap(lambda v: style_func(v, 'PE OI'), subset=['PE OI']) \
            .applymap(lambda v: style_func(v, 'CE OI Chg'), subset=['CE OI Chg']) \
            .applymap(lambda v: style_func(v, 'PE OI Chg'), subset=['PE OI Chg']) \
            .applymap(lambda v: style_func(v, 'CE Volume'), subset=['CE Volume']) \
            .applymap(lambda v: style_func(v, 'PE Volume'), subset=['PE Volume'])

        return styled

    def get_oi_analysis_enhanced(self):
        """Enhanced OI analysis with proper ATM detection"""

        if not hasattr(self, 'option_data') or not self.option_data:
            return pd.DataFrame()

        # Get ATM from analyze_market_direction (consistent calculation)
        analysis = self.analyze_market_direction()
        atm = analysis['atm_strike'] if analysis else int(round(self.current_ltp / 50, 0)) * 50

        # Build data by strike
        strikes_data = {}

        for token, opt_data in self.option_data.items():
            strike = int(opt_data.get('strike', 0))
            opt_type = opt_data.get('type', '')

            if strike not in strikes_data:
                # Proper ATM/ITM/OTM classification
                if strike == atm:
                    strike_type = 'ATM'
                elif strike < atm:
                    strike_type = 'ITM'
                else:
                    strike_type = 'OTM'

                strikes_data[strike] = {
                    'Strike': strike,
                    'Type': strike_type
                }

            prefix = 'CE' if opt_type == 'CE' else 'PE'

            strikes_data[strike][f'{prefix} OI'] = opt_data.get('oi', 0)
            strikes_data[strike][f'{prefix} Δ5m'] = opt_data.get('oi_chg_5m', 0)
            strikes_data[strike][f'{prefix} Δ15m'] = opt_data.get('oi_chg_15m', 0)
            strikes_data[strike][f'{prefix} Δ1h'] = opt_data.get('oi_chg_1h', 0)
            strikes_data[strike][f'{prefix} Vol'] = opt_data.get('volume', 0)
            strikes_data[strike][f'{prefix} LTP'] = opt_data.get('ltp', 0)

        if not strikes_data:
            return pd.DataFrame()

        df = pd.DataFrame(list(strikes_data.values()))
        df = df.sort_values('Strike').reset_index(drop=True)
        df = df.fillna(0)

        # Reorder columns
        col_order = [
            'CE OI', 'CE Δ5m', 'CE Δ15m', 'CE Δ1h', 'CE Vol', 'CE LTP',
            'Strike', 'Type',
            'PE LTP', 'PE Vol', 'PE Δ1h', 'PE Δ15m', 'PE Δ5m', 'PE OI'
        ]

        for col in col_order:
            if col not in df.columns:
                df[col] = 0

        df = df[col_order]

        return df

    def get_option_ltp(self, strike, option_type):
        """
        Get LTP of specific option strike
        Args:
            strike: Strike price (e.g., 26000)
            option_type: 'CE' or 'PE'
        Returns:
            LTP (float) or 0 if not found
        """
        if not hasattr(self, 'option_data') or not self.option_data:
            return 0

        for token, opt_data in self.option_data.items():
            if int(opt_data.get('strike', 0)) == strike and opt_data.get('type') == option_type:
                return opt_data.get('ltp', 0)

        return 0

    def analyze_market_direction(self):
        """
        EXACT SAME score calculation as backtesting
        Multi-timeframe OI/Volume pressure analysis
        """

        if not hasattr(self, 'option_data') or not self.option_data:
            return None

        # Get ATM strike and surrounding strikes
        spot = self.current_ltp
        atm_strike = int(round(spot / 50, 0)) * 50
        strike_diff = 50
        STRIKE_RANGE = 6

        # Get ATM focus strikes (±2 strikes from ATM)
        atm_focus_strikes = []
        for offset in range(-2, 3):
            strike = atm_strike + (offset * strike_diff)
            atm_focus_strikes.append(strike)

        # Initialize timeframe data (EXACTLY as backtesting)
        timeframe_data = {
            '5m': {'ce_vol': 0, 'pe_vol': 0, 'ce_oi': 0, 'pe_oi': 0, 'weight': 0.4},
            '15m': {'ce_vol': 0, 'pe_vol': 0, 'ce_oi': 0, 'pe_oi': 0, 'weight': 0.3},
            '1h': {'ce_vol': 0, 'pe_vol': 0, 'ce_oi': 0, 'pe_oi': 0, 'weight': 0.2},
            'day': {'ce_vol': 0, 'pe_vol': 0, 'ce_oi': 0, 'pe_oi': 0, 'weight': 0.1}
        }

        # Collect OI/Volume changes from option_data
        for token, opt_data in self.option_data.items():
            strike = int(opt_data.get('strike', 0))
            opt_type = opt_data.get('type', '')

            # Focus on ATM strikes
            if strike in atm_focus_strikes:
                # Get changes
                vol_5m = opt_data.get('vol_chg_5m', 0)
                oi_5m = opt_data.get('oi_chg_5m', 0)
                vol_15m = opt_data.get('vol_chg_15m', 0)
                oi_15m = opt_data.get('oi_chg_15m', 0)
                vol_1h = opt_data.get('vol_chg_1h', 0)
                oi_1h = opt_data.get('oi_chg_1h', 0)
                oi_day = opt_data.get('oi_chg_day', 0)

                if opt_type == 'CE':
                    timeframe_data['5m']['ce_vol'] += abs(vol_5m)
                    timeframe_data['5m']['ce_oi'] += oi_5m
                    timeframe_data['15m']['ce_vol'] += abs(vol_15m)
                    timeframe_data['15m']['ce_oi'] += oi_15m
                    timeframe_data['1h']['ce_vol'] += abs(vol_1h)
                    timeframe_data['1h']['ce_oi'] += oi_1h
                    timeframe_data['day']['ce_oi'] += oi_day

                elif opt_type == 'PE':
                    timeframe_data['5m']['pe_vol'] += abs(vol_5m)
                    timeframe_data['5m']['pe_oi'] += oi_5m
                    timeframe_data['15m']['pe_vol'] += abs(vol_15m)
                    timeframe_data['15m']['pe_oi'] += oi_15m
                    timeframe_data['1h']['pe_vol'] += abs(vol_1h)
                    timeframe_data['1h']['pe_oi'] += oi_1h
                    timeframe_data['day']['pe_oi'] += oi_day

        # Calculate weighted score (EXACTLY as backtesting)
        total_score = 0

        for tf_name, tf_data in timeframe_data.items():
            tf_score = 0

            # Volume analysis
            if tf_name != 'day':  # Day doesn't use volume
                total_vol = tf_data['ce_vol'] + tf_data['pe_vol']
                if total_vol > 0:
                    ce_vol_pct = (tf_data['ce_vol'] / total_vol) * 100

                    # Scoring based on volume distribution
                    if ce_vol_pct >= 75:
                        tf_score += 2  # Strong bullish
                    elif ce_vol_pct >= 60:
                        tf_score += 1  # Bullish
                    elif ce_vol_pct <= 25:
                        tf_score -= 2  # Strong bearish
                    elif ce_vol_pct <= 40:
                        tf_score -= 1  # Bearish

            # OI analysis
            if tf_data['ce_oi'] > 0 and tf_data['pe_oi'] < 0:
                tf_score -= 1.5  # CE building, PE unwinding = Bearish
            elif tf_data['ce_oi'] < 0 and tf_data['pe_oi'] > 0:
                tf_score += 1.5  # CE unwinding, PE building = Bullish

            # Apply timeframe weight
            total_score += tf_score * tf_data['weight']

        # Determine direction based on score
        if total_score >= 3:
            direction = "STRONG_BULLISH"
        elif total_score >= 1.5:
            direction = "BULLISH"
        elif total_score <= -3:
            direction = "STRONG_BEARISH"
        elif total_score <= -1.5:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            'direction': direction,
            'score': total_score,
            'timeframe_data': timeframe_data,
            'atm_strike': atm_strike,
            'spot': spot
        }

    def get_all_historical_trades(self):
        """Get ALL historical trades from CSV files"""
        return self.logger.load_all_historical_trades()

    def get_strategy_summary(self):
        """Get strategy performance"""
        data = []

        for name, strategy in self.strategies.items():
            current_pnl = 0
            if strategy.in_trade and self.current_ltp > 0:
                if strategy.entry_direction == 'BULLISH':
                    current_pnl = (self.current_ltp - strategy.entry_price) * strategy.lot_size
                else:
                    current_pnl = (strategy.entry_price - self.current_ltp) * strategy.lot_size

            data.append({
                'Strategy': name,
                'Status': f"🔴 {strategy.entry_direction} @ {strategy.entry_price:.2f}" if strategy.in_trade else "⚪ Monitoring",
                'Current P&L': f"{current_pnl:+.0f}" if strategy.in_trade else "-",
                'Trades': len(strategy.trades),
                'Daily P&L': f"{strategy.daily_pnl:+,.0f}",
                'Win %': f"{(len([t for t in strategy.trades if t['profit_inr'] > 0]) / len(strategy.trades) * 100):.1f}" if strategy.trades else "0.0"
            })

        df = pd.DataFrame(data)
        return df.sort_values('Daily P&L', ascending=False) if not df.empty else df

def main():
    st.title("🚀 Live Trading Dashboard")

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        if st.button("▶️ Start" if not st.session_state.running else "⏸️ Pause"):
            st.session_state.running = not st.session_state.running

            if st.session_state.running and st.session_state.trading_system is None:
                with st.spinner("Connecting..."):
                    system = StreamlitTradingSystem()

                    if system.login():
                        system.get_option_chain_tokens()
                        system.start_depth_updates()
                        time.sleep(3)
                        st.session_state.trading_system = system
                        st.success("✅ Connected!")
                    else:
                        st.error("❌ Login failed")
                        st.session_state.running = False

        refresh_rate = st.slider("Refresh (sec)", 1, 60, 30)

    # Main display
    if st.session_state.running and st.session_state.trading_system:
        system = st.session_state.trading_system

        # Process signals
        system.process_trading_signals()
        # Header metrics - COMPACT
        analysis = system.analyze_market_direction()
        if analysis:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.metric("NIFTY", f"{system.current_ltp:.2f}", label_visibility="visible")
            with col2:
                dir_emoji = "🟢" if "BULLISH" in analysis['direction'] else "🔴" if "BEARISH" in analysis[
                    'direction'] else "⚪"
                st.metric("Signal", f"{dir_emoji} {analysis['direction'][:4]}")  # Shortened
            with col3:
                st.metric("Score", f"{analysis['score']:.2f}")
            with col4:
                total_pnl = sum([s.daily_pnl for s in system.strategies.values()])
                pnl_color = "normal" if total_pnl >= 0 else "inverse"
                st.metric("Daily P&L", f"₹{total_pnl:,.0f}", delta=f"{total_pnl:+,.0f}", delta_color=pnl_color)
            with col5:
                active_strategies = sum([1 for s in system.strategies.values() if s.in_trade])
                st.metric("Active", f"{active_strategies}/8")
            with col6:
                st.metric("Time", datetime.now().strftime("%H:%M:%S"))

        # Three tabs - NO DIVIDER
        tab1, tab2, tab3 = st.tabs(["📊 Chain", "📈 OI", "💼 Trades"])

        with tab1:
            st.subheader("📊 Option Chain")
            if st.button("🔍 Debug Bid-Ask Data"):
                st.write("Sample option_data for debugging:")
                sample_tokens = list(system.option_data.keys())[:3]
                for token in sample_tokens:
                    data = system.option_data[token]
                    st.write(f"Strike {data.get('strike')}, Type {data.get('type')}:")
                    st.write(f"  - bidqty: {data.get('bidqty', 'NOT FOUND')}")
                    st.write(f"  - askqty: {data.get('askqty', 'NOT FOUND')}")
                    st.write(f"  - qtydiff: {data.get('qtydiff', 'NOT FOUND')}")

            df_display = system.get_basic_option_chain_formatted()

            if not df_display.empty:
                # Remove internal flag for display
                df_show = df_display.drop(columns=['_is_atm']).copy()

                # Get ATM row index
                atm_idx = df_display[df_display['_is_atm']].index.tolist()

                # Apply max OI/Volume highlighting first
                styled_df = system.highlight_extremes(df_show)

                # Then override with ATM row highlighting
                if atm_idx:
                    styled_df = styled_df.apply(
                        lambda row: ['background-color: #fff3cd; font-weight: bold;'] * len(
                            row) if row.name in atm_idx else [''] * len(row),
                        axis=1
                    )

                st.dataframe(styled_df, use_container_width=True, height=600)

                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    total_ce_oi = df_show['CE OI'].str.replace(',', '').astype(float).sum()
                    st.metric("Total CE OI", f"{total_ce_oi / 100000:.2f}L")
                with col2:
                    total_pe_oi = df_show['PE OI'].str.replace(',', '').astype(float).sum()
                    st.metric("Total PE OI", f"{total_pe_oi / 100000:.2f}L")
                with col3:
                    pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
                    st.metric("PCR", f"{pcr:.3f}")
            else:
                st.warning("⏳ Loading...")

        with tab2:
            st.subheader("🔍 OI Analysis - Multi-Timeframe Changes")

            oi_df = system.get_oi_analysis_enhanced()

            if not oi_df.empty:
                # Color functions
                def color_change(val):
                    try:
                        val_num = float(val)
                        if val_num > 1000:
                            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                        elif val_num > 0:
                            return 'background-color: #d4edda; color: #155724;'
                        elif val_num < -1000:
                            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                        elif val_num < 0:
                            return 'background-color: #f8d7da; color: #721c24;'
                    except:
                        pass
                    return ''

                def highlight_atm(row):
                    if row['Type'] == 'ATM':
                        return ['background-color: #fff3cd; font-weight: bold;'] * len(row)
                    return [''] * len(row)

                # Apply styling
                styled = oi_df.style \
                    .apply(highlight_atm, axis=1) \
                    .applymap(color_change, subset=['CE Δ5m', 'CE Δ15m', 'CE Δ1h', 'PE Δ5m', 'PE Δ15m', 'PE Δ1h']) \
                    .format({
                    'CE OI': '{:,.0f}',
                    'CE Δ5m': '{:+,.0f}',
                    'CE Δ15m': '{:+,.0f}',
                    'CE Δ1h': '{:+,.0f}',
                    'CE Vol': '{:,.0f}',
                    'CE LTP': '₹{:.2f}',
                    'PE LTP': '₹{:.2f}',
                    'PE Vol': '{:,.0f}',
                    'PE Δ1h': '{:+,.0f}',
                    'PE Δ15m': '{:+,.0f}',
                    'PE Δ5m': '{:+,.0f}',
                    'PE OI': '{:,.0f}'
                })

                st.dataframe(styled, use_container_width=True, height=500)

                # Rest of your summary code...
            with tab3:
                st.subheader("💼 Strategy Performance & Trade History")

                # Current Status Table
                st.write("### Current Positions")
                strategy_df = system.get_strategy_summary()

                if not strategy_df.empty:
                    st.dataframe(strategy_df, use_container_width=True, height=300)

                    # Summary metrics
                    col1, col2, col3 = st.columns(3)

                    total_pnl = sum([s.daily_pnl for s in system.strategies.values()])
                    total_trades = sum([len(s.trades) for s in system.strategies.values()])
                    active_strategies = sum([1 for s in system.strategies.values() if s.in_trade])

                    with col1:
                        st.metric("Total P&L", f"₹{total_pnl:,.0f}", delta=f"{total_pnl:+.0f}")
                    with col2:
                        st.metric("Total Completed", total_trades)
                    with col3:
                        st.metric("Active Now", f"{active_strategies}/8")

                st.divider()

                # Trade History Table
                st.write("📋 All Completed Trades (Historical)")
                all_trades = system.get_all_historical_trades()  # NEW METHOD
                if not all_trades.empty:
                    # Rename columns for display
                    all_trades_display = all_trades.rename(columns={
                        'strategy': 'Strategy',
                        'entry_time': 'Entry Time',
                        'exit_time': 'Exit Time',
                        'direction': 'Direction',
                        'entry_price': 'Entry Price',
                        'exit_price': 'Exit Price',
                        'points': 'Points',
                        'profit_inr': 'Profit INR',
                        'exit_reason': 'Exit Reason',
                        'holding_min': 'Holding'
                    })

                    # Color code P&L
                    def color_pnl(val):
                        try:
                            if ('+' in str(val)) or (isinstance(val, (int, float)) and val > 0):
                                return 'background-color:#d4edda;color:#155724;'
                            elif ('-' in str(val)) or (isinstance(val, (int, float)) and val < 0):
                                return 'background-color:#f8d7da;color:#721c24;'
                        except:
                            pass
                        return ''

                    styled_trades = all_trades_display.style.applymap(
                        color_pnl,
                        subset=['Profit INR', 'Points']
                    ).format({
                        'Entry Time': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, pd.Timestamp) else x,
                        'Exit Time': lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if isinstance(x, pd.Timestamp) else x,
                        'Entry Price': '{:.2f}',
                        'Exit Price': '{:.2f}',
                        'Points': '{:.2f}',
                        'Profit INR': '{:,.0f}',
                        'Holding': '{:.0f} min'
                    })

                    st.dataframe(styled_trades, use_container_width=True, height=400)

                    # Download button
                    csv = all_trades.to_csv(index=False)
                    st.download_button(
                        "📥 Download All Trades",
                        csv,
                        f"trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        "text/csv",
                        key='download-trades'
                    )
                else:
                    st.info("No historical trades found. Waiting for signals...")

            strategy_df = system.get_strategy_summary()

            if not strategy_df.empty:
                st.dataframe(strategy_df, use_container_width=True, height=400)
            else:
                st.info("Loading strategies...")

        # Auto-refresh
        time.sleep(refresh_rate)
        st.rerun()

    else:
        st.info("👈 Click Start in sidebar")

if __name__ == "__main__":
    main()
