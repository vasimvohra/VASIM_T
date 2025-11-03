from colorama import init, Fore, Back, Style
import pandas as pd
from datetime import datetime, timedelta
import time
from api_helper import ShoonyaApiPy
import pyotp
import re
import threading

init(autoreset=True)


def format_number(num):
    """Convert large numbers to compact format"""
    if num >= 10000000:
        return f"{num / 10000000:.2f}Cr"
    elif num >= 100000:
        return f"{num / 100000:.2f}L"
    elif num >= 1000:
        return f"{num / 1000:.2f}K"
    else:
        return f"{num}"


def download_today_full_data(api, option_data):
    """Download complete intraday data from 9:15 AM to current time"""
    today = datetime.now()
    dt_start = datetime(today.year, today.month, today.day, 9, 15)
    dt_end = datetime.now()

    intraday_data = {}
    tokens_list = list(option_data.keys())

    print(f"Downloading full intraday data for {len(tokens_list)} tokens...")

    for i, token in enumerate(tokens_list):
        try:
            raw_hist = api.get_time_price_series(
                exchange='NFO',
                token=token,
                starttime=dt_start.timestamp(),
                endtime=dt_end.timestamp(),
                interval=1
            )

            if raw_hist:
                df = pd.DataFrame(raw_hist)

                if 'time' not in df.columns:
                    df['time'] = df.apply(lambda row: row.get('time', ''), axis=1)

                df = df.sort_values(by='time').reset_index(drop=True)
                df['time'] = pd.to_datetime(df['time'], format='%d-%m-%Y %H:%M:%S')

                if 'oi' not in df.columns:
                    df['oi'] = df.apply(lambda row: row.get('oi', 0), axis=1)
                if 'v' not in df.columns:
                    df['v'] = df.apply(lambda row: row.get('v', 0), axis=1)
                if 'c' not in df.columns:
                    df['c'] = df.apply(lambda row: row.get('c', 0), axis=1)

                df['oi'] = pd.to_numeric(df['oi'], errors='coerce').fillna(0).astype(int)
                df['v'] = pd.to_numeric(df['v'], errors='coerce').fillna(0).astype(int)
                df['c'] = pd.to_numeric(df['c'], errors='coerce').fillna(0).astype(float)

                intraday_data[token] = df
            else:
                intraday_data[token] = pd.DataFrame()

        except Exception as e:
            print(f"Error downloading data for token {token}: {e}")
            intraday_data[token] = pd.DataFrame()

        if (i + 1) % 10 == 0 or (i + 1) == len(tokens_list):
            print(f"Progress: {i + 1}/{len(tokens_list)}")

    print("Download complete!")
    return intraday_data


def calculate_changes(token, current_oi, current_vol, intraday_df):
    """Calculate OI and Volume changes for different timeframes"""
    changes = {
        'oi_chg_5m': 0, 'vol_chg_5m': 0,
        'oi_chg_15m': 0, 'vol_chg_15m': 0,
        'oi_chg_1h': 0, 'vol_chg_1h': 0,
        'oi_chg_day': 0, 'vol_chg_day': 0
    }

    if intraday_df.empty:
        return changes

    now = datetime.now()

    opening_row = intraday_df.iloc[0]
    opening_oi = int(opening_row['oi'])
    opening_vol = int(opening_row['v'])

    changes['oi_chg_day'] = current_oi - opening_oi
    changes['vol_chg_day'] = current_vol - opening_vol

    time_5m_ago = now - timedelta(minutes=5)
    df_5m = intraday_df[intraday_df['time'] <= time_5m_ago]
    if not df_5m.empty:
        row_5m = df_5m.iloc[-1]
        changes['oi_chg_5m'] = current_oi - int(row_5m['oi'])
        changes['vol_chg_5m'] = current_vol - int(row_5m['v'])

    time_15m_ago = now - timedelta(minutes=15)
    df_15m = intraday_df[intraday_df['time'] <= time_15m_ago]
    if not df_15m.empty:
        row_15m = df_15m.iloc[-1]
        changes['oi_chg_15m'] = current_oi - int(row_15m['oi'])
        changes['vol_chg_15m'] = current_vol - int(row_15m['v'])

    time_1h_ago = now - timedelta(hours=1)
    df_1h = intraday_df[intraday_df['time'] <= time_1h_ago]
    if not df_1h.empty:
        row_1h = df_1h.iloc[-1]
        changes['oi_chg_1h'] = current_oi - int(row_1h['oi'])
        changes['vol_chg_1h'] = current_vol - int(row_1h['v'])

    return changes


def get_top_values(data_list):
    """Get top 2 unique values from a list"""
    unique_sorted = sorted(set(data_list), reverse=True)
    top1 = unique_sorted[0] if len(unique_sorted) > 0 else None
    top2 = unique_sorted[1] if len(unique_sorted) > 1 else None
    return top1, top2


def format_with_highlight(value, top1, top2, is_call=True, is_compact=False):
    """Format value with color highlighting"""
    if is_compact:
        display_val = format_number(value)
        width = 10
    else:
        display_val = f"{value:,}"
        width = 12

    if value == top1:
        if is_call:
            return f"{Back.RED}{Fore.WHITE}{display_val:>{width}}{Style.RESET_ALL}"
        else:
            return f"{Back.GREEN}{Fore.WHITE}{display_val:>{width}}{Style.RESET_ALL}"
    elif value == top2:
        if is_call:
            return f"{Back.LIGHTRED_EX}{Fore.BLACK}{display_val:>{width}}{Style.RESET_ALL}"
        else:
            return f"{Back.LIGHTGREEN_EX}{Fore.BLACK}{display_val:>{width}}{Style.RESET_ALL}"
    else:
        return f"{display_val:>{width}}"


def calculate_direction_signal(ce_oi_chg, pe_oi_chg, ce_vol_chg, pe_vol_chg):
    """
    Calculate directional signal for a specific timeframe
    Returns: (signal, confidence)
    signal: -1 (bearish) to +1 (bullish)
    confidence: 0 to 1
    """
    signal = 0.0
    confidence = 0.0

    # OI Analysis
    if ce_oi_chg > 0 and pe_oi_chg < 0:
        # CE building, PE unwinding = Bearish
        signal -= 1.0
        confidence += 1.0
    elif ce_oi_chg < 0 and pe_oi_chg > 0:
        # CE unwinding, PE building = Bullish
        signal += 1.0
        confidence += 1.0
    elif ce_oi_chg > 0 and pe_oi_chg > 0:
        # Both building = Neutral/Volatile
        signal += 0.0
        confidence += 0.5
    elif ce_oi_chg < 0 and pe_oi_chg < 0:
        # Both unwinding = Neutral
        signal += 0.0
        confidence += 0.3

    # Volume Analysis (Higher priority)
    ce_vol_abs = abs(ce_vol_chg)
    pe_vol_abs = abs(pe_vol_chg)

    if pe_vol_abs > 0 and ce_vol_abs > 0:
        vol_ratio = pe_vol_abs / ce_vol_abs

        if vol_ratio > 1.5:
            # PE volume much higher = Bearish
            signal -= 1.5
            confidence += 2.0
        elif vol_ratio < 0.67:
            # CE volume much higher = Bullish
            signal += 1.5
            confidence += 2.0
        else:
            # Balanced volume
            confidence += 0.5

    # Normalize signal
    if confidence > 0:
        normalized_signal = max(-1.0, min(1.0, signal / confidence))
    else:
        normalized_signal = 0.0

    return normalized_signal, confidence


class NiftyOptionChainLTP:
    def __init__(self):
        self.api = ShoonyaApiPy()
        self.feed_opened = False
        self.current_ltp = 0
        self.nifty_prev_close = 0
        self.strike = 0
        self.option_data = {}
        self.intraday_data = {}
        self.strike_diff = 50
        self.instrument = "NIFTY"
        self.nifty_token = "26000"

        # Configurable strike range
        self.STRIKE_RANGE = 6

        # Background thread control
        self.depth_update_running = False
        self.depth_thread = None

        # Market direction analysis
        self.direction_analysis = {}

        self.token_login = '3LODN5H53437L6J7TE6657CX276E7PDK'
        self.user = 'FA18555'
        self.pwd = 'Thankyou@9'
        self.vc = 'FA18555_U'
        self.app_key = 'f76a17b942afafea1278eb8bf5b1bcf6'
        self.imei = 'abc1234'

    def login(self):
        otp = pyotp.TOTP(self.token_login).now()
        ret = self.api.login(
            userid=self.user,
            password=self.pwd,
            twoFA=otp,
            vendor_code=self.vc,
            api_secret=self.app_key,
            imei=self.imei
        )
        if ret['stat'] == 'Ok':
            print("Login successful!")
            self.start_websocket()
            return True
        else:
            print(f"Login failed: {ret['emsg']}")
            return False

    def start_websocket(self):
        self.api.start_websocket(
            order_update_callback=self.order_update_callback,
            subscribe_callback=self.feed_update_callback,
            socket_open_callback=self.socket_open_callback
        )
        timeout = 0
        while not self.feed_opened and timeout < 80:
            time.sleep(0.1)
            timeout += 1
        print("Websocket connected!")

    def socket_open_callback(self):
        self.feed_opened = True

    def order_update_callback(self, order_update):
        pass

    def feed_update_callback(self, tick_data):
        if 'tk' in tick_data:
            token = tick_data['tk']

            if token == self.nifty_token and 'lp' in tick_data:
                self.current_ltp = float(tick_data['lp'])

            if token in self.option_data:
                if 'lp' in tick_data:
                    self.option_data[token]['ltp'] = float(tick_data['lp'])
                if 'oi' in tick_data:
                    self.option_data[token]['oi'] = int(tick_data['oi'])
                if 'v' in tick_data:
                    self.option_data[token]['volume'] = int(tick_data['v'])

                current_oi = self.option_data[token]['oi']
                current_vol = self.option_data[token]['volume']
                intraday_df = self.intraday_data.get(token, pd.DataFrame())

                changes = calculate_changes(token, current_oi, current_vol, intraday_df)
                self.option_data[token].update(changes)

    def update_market_depth(self, token):
        """Fetch market depth and calculate bid-ask quantity difference"""
        try:
            quotes = self.api.get_quotes(exchange='NFO', token=token)

            if quotes and 'stat' in quotes and quotes['stat'] == 'Ok':
                bid_qty = int(quotes.get('bq1', 0))
                ask_qty = int(quotes.get('sq1', 0))
                qty_diff = bid_qty - ask_qty

                self.option_data[token]['bid_qty'] = bid_qty
                self.option_data[token]['ask_qty'] = ask_qty
                self.option_data[token]['qty_diff'] = qty_diff

                for i in range(1, 6):
                    bid_qty = int(quotes.get(f'bq{i}', 0))
                    bid_price = float(quotes.get(f'bp{i}', 0))
                    self.option_data[token][f'bid_qty_{i}'] = bid_qty
                    self.option_data[token][f'bid_price_{i}'] = bid_price

                    ask_qty = int(quotes.get(f'sq{i}', 0))
                    ask_price = float(quotes.get(f'sp{i}', 0))
                    self.option_data[token][f'ask_qty_{i}'] = ask_qty
                    self.option_data[token][f'ask_price_{i}'] = ask_price

        except Exception as e:
            pass

    def update_all_depths_background(self):
        """Background thread to update depth for all strikes periodically"""
        while self.depth_update_running:
            try:
                tokens_list = list(self.option_data.keys())
                for token in tokens_list:
                    if not self.depth_update_running:
                        break
                    self.update_market_depth(token)
                    time.sleep(0.1)

                for _ in range(50):
                    if not self.depth_update_running:
                        break
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error in depth update: {e}")
                time.sleep(1)

    def start_depth_updates(self):
        """Start background thread for continuous depth updates"""
        self.depth_update_running = True
        self.depth_thread = threading.Thread(target=self.update_all_depths_background, daemon=True)
        self.depth_thread.start()
        print("Started background depth updates")

    def stop_depth_updates(self):
        """Stop background depth updates"""
        self.depth_update_running = False
        if self.depth_thread:
            self.depth_thread.join(timeout=2)

    def get_expiry_date(self):
        try:
            ret = self.api.get_quotes(exchange='NSE', token=self.nifty_token)
            ltp = float(ret.get('lp'))
            self.current_ltp = ltp
            self.nifty_prev_close = float(ret.get('c', ltp))

            sd = self.api.searchscrip('NFO', self.instrument)
            if 'values' in sd and sd['values']:
                expiry_dates = set()
                for option in sd['values']:
                    symbol = option['tsym']
                    date_match = re.search(r'NIFTY(\d{2}[A-Z]{3}\d{2})[CP]', symbol)
                    if date_match:
                        expiry_dates.add(date_match.group(1))

                if expiry_dates:
                    nearest_expiry = sorted(list(expiry_dates))[0]
                    print(f"Found expiry: {nearest_expiry}")
                    return nearest_expiry
            return None
        except Exception as e:
            print(f"Error getting expiry date: {e}")
            return None

    def get_atm_strike(self):
        ret = self.api.get_quotes(exchange='NSE', token=self.nifty_token)
        ltp = float(ret['lp'])
        self.nifty_prev_close = float(ret.get('c', ltp))
        self.current_ltp = ltp
        return int(round(ltp / self.strike_diff, 0)) * self.strike_diff

    def get_option_chain_tokens(self):
        expiry_date = self.get_expiry_date()
        if not expiry_date:
            print("Could not get expiry date")
            return

        print(f"Using expiry: {expiry_date}")
        atm_strike = self.get_atm_strike()
        self.strike = atm_strike
        print(f"ATM Strike: {atm_strike}")

        strikes = []
        for i in range(-self.STRIKE_RANGE, self.STRIKE_RANGE + 1):
            strikes.append(atm_strike + (i * self.strike_diff))

        print(f"Strike range: {min(strikes)} to {max(strikes)} ({len(strikes) * 2} contracts)")

        for_token = f"{self.instrument}{expiry_date}P{atm_strike}"

        try:
            option_chain = self.api.get_option_chain('NFO', for_token, atm_strike, (self.STRIKE_RANGE * 2) + 1)

            if not option_chain or 'values' not in option_chain:
                print("No option chain data available")
                return

            option_chain_sym = option_chain['values']
            tokens_to_subscribe = [f"NSE|{self.nifty_token}"]

            for option in option_chain_sym:
                strike_price = float(option['strprc'])
                if strike_price in strikes:
                    token = option['token']
                    symbol = option['tsym']
                    option_type = option['optt']

                    self.option_data[token] = {
                        'symbol': symbol,
                        'strike': strike_price,
                        'type': option_type,
                        'ltp': 0.0,
                        'oi': 0,
                        'volume': 0,
                        'oi_chg_5m': 0, 'vol_chg_5m': 0,
                        'oi_chg_15m': 0, 'vol_chg_15m': 0,
                        'oi_chg_1h': 0, 'vol_chg_1h': 0,
                        'oi_chg_day': 0, 'vol_chg_day': 0,
                        'bid_qty': 0,
                        'ask_qty': 0,
                        'qty_diff': 0
                    }

                    for i in range(1, 6):
                        self.option_data[token][f'bid_qty_{i}'] = 0
                        self.option_data[token][f'bid_price_{i}'] = 0.0
                        self.option_data[token][f'ask_qty_{i}'] = 0
                        self.option_data[token][f'ask_price_{i}'] = 0.0

                    tokens_to_subscribe.append(f"NFO|{token}")

            print(f"Found {len(self.option_data)} option contracts")

            print("\nDownloading intraday historical data...")
            self.intraday_data = download_today_full_data(self.api, self.option_data)

            self.api.subscribe(tokens_to_subscribe)
            print(f"Subscribed to {len(tokens_to_subscribe)} instruments")

            self.start_depth_updates()

        except Exception as e:
            print(f"Error getting option chain: {e}")
            import traceback
            traceback.print_exc()

    def analyze_market_direction(self):
        """
        Comprehensive market direction analysis using multi-timeframe OI and Volume
        """
        if not self.option_data:
            return None

        # Organize data by strikes
        strikes = {}
        for token, data in self.option_data.items():
            strike = data['strike']
            if strike not in strikes:
                strikes[strike] = {'CE': {}, 'PE': {}}
            strikes[strike][data['type']] = data

        # Focus on ATM ± 2 strikes
        atm_strikes = []
        for i in range(-2, 3):
            target_strike = self.strike + (i * self.strike_diff)
            if target_strike in strikes:
                atm_strikes.append(target_strike)

        # Calculate signals for each timeframe
        timeframes = {
            '5min': {'oi_key': 'oi_chg_5m', 'vol_key': 'vol_chg_5m', 'weight': 0.4},
            '15min': {'oi_key': 'oi_chg_15m', 'vol_key': 'vol_chg_15m', 'weight': 0.3},
            '1hr': {'oi_key': 'oi_chg_1h', 'vol_key': 'vol_chg_1h', 'weight': 0.2},
            'day': {'oi_key': 'oi_chg_day', 'vol_key': 'vol_chg_day', 'weight': 0.1}
        }

        timeframe_signals = {}

        for tf_name, tf_config in timeframes.items():
            total_signal = 0.0
            total_confidence = 0.0

            for strike in atm_strikes:
                ce_data = strikes[strike].get('CE', {})
                pe_data = strikes[strike].get('PE', {})

                ce_oi_chg = ce_data.get(tf_config['oi_key'], 0)
                pe_oi_chg = pe_data.get(tf_config['oi_key'], 0)
                ce_vol_chg = abs(ce_data.get(tf_config['vol_key'], 0))
                pe_vol_chg = abs(pe_data.get(tf_config['vol_key'], 0))

                sig, conf = calculate_direction_signal(ce_oi_chg, pe_oi_chg, ce_vol_chg, pe_vol_chg)

                # ATM gets higher weight
                weight = 2.0 if strike == self.strike else 1.0

                total_signal += sig * weight * conf
                total_confidence += weight * conf

            if total_confidence > 0:
                timeframe_signals[tf_name] = {
                    'signal': total_signal / total_confidence,
                    'confidence': min(1.0, total_confidence / 10.0)
                }
            else:
                timeframe_signals[tf_name] = {'signal': 0.0, 'confidence': 0.0}

        # Weighted aggregate score
        final_score = 0.0
        for tf_name, tf_config in timeframes.items():
            tf_signal = timeframe_signals[tf_name]['signal']
            final_score += tf_signal * tf_config['weight']

        # Calculate PCR (Put-Call Ratio)
        total_ce_oi = sum(strikes[s]['CE'].get('oi', 0) for s in strikes if 'CE' in strikes[s])
        total_pe_oi = sum(strikes[s]['PE'].get('oi', 0) for s in strikes if 'PE' in strikes[s])
        total_ce_vol = sum(strikes[s]['CE'].get('volume', 0) for s in strikes if 'CE' in strikes[s])
        total_pe_vol = sum(strikes[s]['PE'].get('volume', 0) for s in strikes if 'PE' in strikes[s])

        pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 1.0
        pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 1.0

        # PCR adjustment (contrarian)
        if pcr_oi > 1.3:
            final_score += 0.15  # Too bearish -> contrarian bullish
        elif pcr_oi < 0.7:
            final_score -= 0.15  # Too bullish -> contrarian bearish

        # Bid-Ask pressure
        total_ce_bidask = sum(strikes[s]['CE'].get('qty_diff', 0) for s in atm_strikes if 'CE' in strikes[s])
        total_pe_bidask = sum(strikes[s]['PE'].get('qty_diff', 0) for s in atm_strikes if 'PE' in strikes[s])

        if total_ce_bidask > 10000:
            final_score += 0.1
        elif total_ce_bidask < -10000:
            final_score -= 0.05

        if total_pe_bidask > 10000:
            final_score -= 0.1
        elif total_pe_bidask < -10000:
            final_score += 0.05

        # Determine direction
        if final_score > 0.6:
            direction = "STRONG BULLISH"
            color = Fore.GREEN + Style.BRIGHT
        elif final_score > 0.3:
            direction = "BULLISH"
            color = Fore.GREEN
        elif final_score > -0.3:
            direction = "NEUTRAL/RANGE"
            color = Fore.YELLOW
        elif final_score > -0.6:
            direction = "BEARISH"
            color = Fore.RED
        else:
            direction = "STRONG BEARISH"
            color = Fore.RED + Style.BRIGHT

        # Find support and resistance
        max_pe_oi_strike = max(strikes.keys(), key=lambda s: strikes[s].get('PE', {}).get('oi', 0))
        max_ce_oi_strike = max(strikes.keys(), key=lambda s: strikes[s].get('CE', {}).get('oi', 0))

        self.direction_analysis = {
            'direction': direction,
            'score': final_score,
            'confidence': abs(final_score) * 100,
            'color': color,
            'timeframe_signals': timeframe_signals,
            'pcr_oi': pcr_oi,
            'pcr_vol': pcr_vol,
            'support': max_pe_oi_strike,
            'resistance': max_ce_oi_strike,
            'ce_bidask': total_ce_bidask,
            'pe_bidask': total_pe_bidask
        }

        return self.direction_analysis

    def analyze_pressure_buildup(self):
        """
        Analyze market pressure buildup for breakdown/breakout detection
        Returns pressure analysis with breakdown/breakout probability
        """
        if not self.option_data:
            return None

        # Organize data by strikes
        strikes = {}
        for token, data in self.option_data.items():
            strike = data['strike']
            if strike not in strikes:
                strikes[strike] = {'CE': {}, 'PE': {}}
            strikes[strike][data['type']] = data

        current_spot = self.current_ltp
        atm = self.strike

        # Classify strikes (ITM, ATM, OTM)
        itm_ce_strikes = [s for s in strikes.keys() if s < atm - self.strike_diff]  # Below ATM
        atm_ce_strike = atm
        otm_ce_strikes = [s for s in strikes.keys() if s > atm]  # Above ATM

        itm_pe_strikes = [s for s in strikes.keys() if s > atm + self.strike_diff]  # Above ATM
        atm_pe_strike = atm
        otm_pe_strikes = [s for s in strikes.keys() if s < atm]  # Below ATM

        # Near OTM (±2 strikes)
        near_otm_ce = [s for s in otm_ce_strikes if s <= atm + (2 * self.strike_diff)]
        far_otm_ce = [s for s in otm_ce_strikes if s > atm + (2 * self.strike_diff)]

        near_otm_pe = [s for s in otm_pe_strikes if s >= atm - (2 * self.strike_diff)]
        far_otm_pe = [s for s in otm_pe_strikes if s < atm - (2 * self.strike_diff)]

        # Calculate OI and Volume for each zone
        def sum_metric(strike_list, option_type, metric, timeframe='5m'):
            total = 0
            for strike in strike_list:
                if option_type in strikes.get(strike, {}):
                    if timeframe == 'current':
                        total += strikes[strike][option_type].get(metric, 0)
                    else:
                        total += strikes[strike][option_type].get(f'{metric}_chg_{timeframe}', 0)
            return total

        # OTM CE Analysis (Resistance)
        otm_ce_oi_current = sum_metric(near_otm_ce, 'CE', 'oi', 'current')
        otm_ce_oi_5m = sum_metric(near_otm_ce, 'CE', 'oi', '5m')
        otm_ce_oi_15m = sum_metric(near_otm_ce, 'CE', 'oi', '15m')
        otm_ce_vol = sum_metric(near_otm_ce, 'CE', 'vol', '5m')

        # OTM PE Analysis (at same level for comparison)
        otm_ce_level_pe_oi = sum_metric(near_otm_ce, 'PE', 'oi', 'current')

        # OTM PE Analysis (Support)
        otm_pe_oi_current = sum_metric(near_otm_pe, 'PE', 'oi', 'current')
        otm_pe_oi_5m = sum_metric(near_otm_pe, 'PE', 'oi', '5m')
        otm_pe_oi_15m = sum_metric(near_otm_pe, 'PE', 'oi', '15m')
        otm_pe_vol = sum_metric(near_otm_pe, 'PE', 'vol', '5m')

        # OTM CE Analysis (at same level for comparison)
        otm_pe_level_ce_oi = sum_metric(near_otm_pe, 'CE', 'oi', 'current')

        # ITM Analysis
        itm_ce_oi_5m = sum_metric(itm_ce_strikes[:2] if len(itm_ce_strikes) >= 2 else itm_ce_strikes, 'CE', 'oi', '5m')
        itm_pe_oi_5m = sum_metric(itm_pe_strikes[:2] if len(itm_pe_strikes) >= 2 else itm_pe_strikes, 'PE', 'oi', '5m')

        # Far OTM Analysis (Panic indicator)
        far_otm_pe_oi_5m = sum_metric(far_otm_pe, 'PE', 'oi', '5m')
        far_otm_ce_oi_5m = sum_metric(far_otm_ce, 'CE', 'oi', '5m')

        # Volume Analysis
        total_ce_vol_5m = sum_metric(list(strikes.keys()), 'CE', 'vol', '5m')
        total_pe_vol_5m = sum_metric(list(strikes.keys()), 'PE', 'vol', '5m')

        # Find max OI strikes (resistance/support)
        max_ce_oi_strike = max(strikes.keys(), key=lambda s: strikes[s].get('CE', {}).get('oi', 0))
        max_ce_oi_value = strikes[max_ce_oi_strike].get('CE', {}).get('oi', 0)

        max_pe_oi_strike = max(strikes.keys(), key=lambda s: strikes[s].get('PE', {}).get('oi', 0))
        max_pe_oi_value = strikes[max_pe_oi_strike].get('PE', {}).get('oi', 0)

        # Pressure Score Calculation
        pressure_score = 0
        signals = []

        # 1. Resistance Wall Detection (Bearish Setup)
        if otm_ce_oi_current > otm_ce_level_pe_oi * 1.5 and otm_ce_oi_5m > 0:
            pressure_score -= 2
            signals.append(('bearish',
                            f"Strong resistance wall at {near_otm_ce[0] if near_otm_ce else 'N/A'}+ (CE OI: {format_number(otm_ce_oi_current)} >> PE OI: {format_number(otm_ce_level_pe_oi)})"))

        # 2. Support Floor Detection (Bullish Setup)
        if otm_pe_oi_current > otm_pe_level_ce_oi * 1.5 and otm_pe_oi_5m > 0:
            pressure_score += 2
            signals.append(('bullish',
                            f"Strong support floor at {near_otm_pe[-1] if near_otm_pe else 'N/A'}- (PE OI: {format_number(otm_pe_oi_current)} >> CE OI: {format_number(otm_pe_level_ce_oi)})"))

        # 3. Volume Confirmation
        vol_ratio = total_pe_vol_5m / (total_ce_vol_5m + 1)
        if vol_ratio > 1.5:
            pressure_score -= 1.5
            signals.append(('bearish',
                            f"High PE volume: {format_number(abs(total_pe_vol_5m))} vs CE: {format_number(abs(total_ce_vol_5m))} (Ratio: {vol_ratio:.2f})"))
        elif vol_ratio < 0.67:
            pressure_score += 1.5
            signals.append(('bullish',
                            f"High CE volume: {format_number(abs(total_ce_vol_5m))} vs PE: {format_number(abs(total_pe_vol_5m))} (Ratio: {1 / vol_ratio:.2f})"))

        # 4. ITM Activity (Hedging indicator)
        if itm_ce_oi_5m > 0 and itm_pe_oi_5m < 0:
            pressure_score -= 1
            signals.append(('bearish',
                            f"ITM CE buying (+{format_number(itm_ce_oi_5m)}) with ITM PE unwinding ({format_number(itm_pe_oi_5m)})"))
        elif itm_pe_oi_5m > 0 and itm_ce_oi_5m < 0:
            pressure_score += 1
            signals.append(('bullish',
                            f"ITM PE buying (+{format_number(itm_pe_oi_5m)}) with ITM CE unwinding ({format_number(itm_ce_oi_5m)})"))

        # 5. Far OTM Analysis (Panic/No Panic)
        if far_otm_pe_oi_5m <= 0 and pressure_score < 0:
            signals.append(('bearish',
                            f"Far OTM PE flat/negative ({format_number(far_otm_pe_oi_5m)}) - No panic, controlled fall expected"))
        elif far_otm_ce_oi_5m <= 0 and pressure_score > 0:
            signals.append(('bullish',
                            f"Far OTM CE flat/negative ({format_number(far_otm_ce_oi_5m)}) - No excessive optimism, healthy rise"))

        # 6. Fresh OI Building (Recent timeframe check)
        if otm_ce_oi_5m > 0 and otm_ce_oi_15m > 0:
            signals.append(('bearish',
                            f"Fresh CE writing: 5m: +{format_number(otm_ce_oi_5m)}, 15m: +{format_number(otm_ce_oi_15m)}"))

        if otm_pe_oi_5m > 0 and otm_pe_oi_15m > 0:
            signals.append(('bullish',
                            f"Fresh PE writing: 5m: +{format_number(otm_pe_oi_5m)}, 15m: +{format_number(otm_pe_oi_15m)}"))

        # Determine pressure type and probability
        if pressure_score < -3:
            pressure_type = "STRONG BREAKDOWN PRESSURE"
            probability = min(95, 60 + abs(pressure_score) * 10)
            color = Fore.RED + Style.BRIGHT
        elif pressure_score < -1.5:
            pressure_type = "BREAKDOWN PRESSURE"
            probability = min(75, 50 + abs(pressure_score) * 10)
            color = Fore.RED
        elif pressure_score > 3:
            pressure_type = "STRONG BREAKOUT PRESSURE"
            probability = min(95, 60 + abs(pressure_score) * 10)
            color = Fore.GREEN + Style.BRIGHT
        elif pressure_score > 1.5:
            pressure_type = "BREAKOUT PRESSURE"
            probability = min(75, 50 + abs(pressure_score) * 10)
            color = Fore.GREEN
        else:
            pressure_type = "BALANCED/RANGE"
            probability = 50
            color = Fore.YELLOW

        # Calculate target levels
        if pressure_score < -1.5:  # Bearish
            immediate_support = max_pe_oi_strike
            next_support = immediate_support - self.strike_diff
            target_1 = immediate_support
            target_2 = next_support
            stop_loss = max_ce_oi_strike
        elif pressure_score > 1.5:  # Bullish
            immediate_resistance = max_ce_oi_strike
            next_resistance = immediate_resistance + self.strike_diff
            target_1 = immediate_resistance
            target_2 = next_resistance
            stop_loss = max_pe_oi_strike
        else:  # Neutral
            target_1 = max_ce_oi_strike
            target_2 = max_pe_oi_strike
            stop_loss = None

        return {
            'pressure_type': pressure_type,
            'pressure_score': pressure_score,
            'probability': probability,
            'color': color,
            'signals': signals,
            'resistance': max_ce_oi_strike,
            'resistance_oi': max_ce_oi_value,
            'support': max_pe_oi_strike,
            'support_oi': max_pe_oi_value,
            'target_1': target_1,
            'target_2': target_2,
            'stop_loss': stop_loss,
            'otm_ce_oi': otm_ce_oi_current,
            'otm_pe_oi': otm_pe_oi_current,
            'vol_ratio': vol_ratio
        }

    def check_oi_support_resistance(self, spot_price, direction, range_points=20):
        """
        Check if spot is approaching a strong OI support/resistance level
        Returns: (should_avoid_or_exit, strike_level, oi_ratio)
        """
        if not hasattr(self, 'option_data') or not self.option_data:
            return False, None, 0

        # Check strikes within range_points of spot
        for token, opt_data in self.option_data.items():
            strike = int(opt_data.get('strike', 0))
            distance = abs(spot_price - strike)

            # Only check strikes within range
            if distance <= range_points:
                # Get CE and PE OI at this strike
                ce_oi = 0
                pe_oi = 0

                # Find corresponding CE/PE at same strike
                for token2, opt_data2 in self.option_data.items():
                    if int(opt_data2.get('strike', 0)) == strike:
                        if opt_data2.get('type') == 'CE':
                            ce_oi = opt_data2.get('oi', 0)
                        elif opt_data2.get('type') == 'PE':
                            pe_oi = opt_data2.get('oi', 0)

                # Check for OI imbalance
                if ce_oi > 0 and pe_oi > 0:
                    if direction == 'BEARISH':
                        # High PUT OI = Support (avoid bearish trades)
                        put_call_ratio = pe_oi / ce_oi
                        if put_call_ratio >= 1.5:
                            return True, strike, put_call_ratio

                    elif direction == 'BULLISH':
                        # High CALL OI = Resistance (avoid bullish trades)
                        call_put_ratio = ce_oi / pe_oi
                        if call_put_ratio >= 1.5:
                            return True, strike, call_put_ratio

        return False, None, 0

    def display_pressure_analysis(self):
        """Display pressure buildup analysis"""
        pressure = self.analyze_pressure_buildup()

        if not pressure:
            return

        print("\n" + "=" * 140)
        print(
            f"{Fore.CYAN}{Style.BRIGHT}PRESSURE BUILDUP ANALYSIS - {datetime.now().strftime('%H:%M:%S')}{Style.RESET_ALL}")
        print("=" * 140)

        # Main pressure type
        print(
            f"\n{pressure['color']}{pressure['pressure_type']} (Score: {pressure['pressure_score']:+.1f} | Probability: {pressure['probability']:.0f}%){Style.RESET_ALL}")

        # Key Levels
        print(f"\n{Style.BRIGHT}Key Levels:{Style.RESET_ALL}")
        print(
            f"  {Fore.RED}Resistance: {pressure['resistance']:.0f} (CE OI: {format_number(pressure['resistance_oi'])}){Style.RESET_ALL}")
        print(f"  {Fore.CYAN}Current:    {self.current_ltp:.2f}{Style.RESET_ALL}")
        print(
            f"  {Fore.GREEN}Support:    {pressure['support']:.0f} (PE OI: {format_number(pressure['support_oi'])}){Style.RESET_ALL}")

        # OTM OI Distribution
        print(f"\n{Style.BRIGHT}OTM OI Distribution:{Style.RESET_ALL}")
        print(f"  Near OTM CE (Above): {format_number(pressure['otm_ce_oi'])}")
        print(f"  Near OTM PE (Below): {format_number(pressure['otm_pe_oi'])}")
        print(f"  Volume Ratio (PE/CE): {pressure['vol_ratio']:.2f}")

        # Trading Levels
        if pressure['stop_loss']:
            print(f"\n{Style.BRIGHT}Trading Levels:{Style.RESET_ALL}")
            if pressure['pressure_score'] < -1.5:
                print(f"  {Fore.RED}Direction: BEARISH{Style.RESET_ALL}")
                print(f"  Target 1: {pressure['target_1']:.0f}")
                print(f"  Target 2: {pressure['target_2']:.0f}")
                print(f"  Stop Loss: {pressure['stop_loss']:.0f} (if closes above)")
            else:
                print(f"  {Fore.GREEN}Direction: BULLISH{Style.RESET_ALL}")
                print(f"  Target 1: {pressure['target_1']:.0f}")
                print(f"  Target 2: {pressure['target_2']:.0f}")
                print(f"  Stop Loss: {pressure['stop_loss']:.0f} (if closes below)")

        # Pressure Signals
        print(f"\n{Style.BRIGHT}Pressure Signals:{Style.RESET_ALL}")
        for signal_type, signal_msg in pressure['signals']:
            if signal_type == 'bearish':
                icon = f"{Fore.RED}▼{Style.RESET_ALL}"
            elif signal_type == 'bullish':
                icon = f"{Fore.GREEN}▲{Style.RESET_ALL}"
            else:
                icon = f"{Fore.YELLOW}●{Style.RESET_ALL}"
            print(f"  {icon} {signal_msg}")

        print("=" * 140 + "\n")

    def display_direction_analysis(self):
        """Display market direction analysis with detailed reasoning"""
        analysis = self.analyze_market_direction()

        if not analysis:
            return

        print("\n" + "=" * 140)
        print(
            f"{Fore.CYAN}{Style.BRIGHT}MARKET DIRECTION ANALYSIS - {datetime.now().strftime('%H:%M:%S')}{Style.RESET_ALL}")
        print("=" * 140)

        # Main direction with reasoning
        print(
            f"\n{analysis['color']}Direction: {analysis['direction']} (Score: {analysis['score']:.3f} | Confidence: {analysis['confidence']:.1f}%){Style.RESET_ALL}")

        # Generate reasoning
        print(f"\n{Style.BRIGHT}Why this direction?{Style.RESET_ALL}")
        reasons = []

        # Analyze timeframe consensus
        tf_signals = analysis['timeframe_signals']
        bullish_tfs = sum(1 for tf in tf_signals.values() if tf['signal'] > 0.3)
        bearish_tfs = sum(1 for tf in tf_signals.values() if tf['signal'] < -0.3)
        neutral_tfs = 4 - bullish_tfs - bearish_tfs

        if bearish_tfs >= 3:
            reasons.append(f"{Fore.RED}✗ {bearish_tfs}/4 timeframes are bearish{Style.RESET_ALL}")
        elif bullish_tfs >= 3:
            reasons.append(f"{Fore.GREEN}✓ {bullish_tfs}/4 timeframes are bullish{Style.RESET_ALL}")
        elif neutral_tfs >= 2:
            reasons.append(f"{Fore.YELLOW}○ {neutral_tfs}/4 timeframes are neutral (mixed signals){Style.RESET_ALL}")

        # Most recent timeframe priority
        recent_signal = tf_signals['5min']['signal']
        if recent_signal > 0.3:
            reasons.append(f"{Fore.GREEN}✓ Recent 5min trend is bullish ({recent_signal:+.2f}){Style.RESET_ALL}")
        elif recent_signal < -0.3:
            reasons.append(f"{Fore.RED}✗ Recent 5min trend is bearish ({recent_signal:+.2f}){Style.RESET_ALL}")
        else:
            reasons.append(f"{Fore.YELLOW}○ Recent 5min trend is neutral ({recent_signal:+.2f}){Style.RESET_ALL}")

        # PCR Analysis
        pcr_oi = analysis['pcr_oi']
        if pcr_oi > 1.3:
            reasons.append(
                f"{Fore.GREEN}✓ PCR (OI) high at {pcr_oi:.2f} → Oversold, contrarian bullish{Style.RESET_ALL}")
        elif pcr_oi < 0.7:
            reasons.append(
                f"{Fore.RED}✗ PCR (OI) low at {pcr_oi:.2f} → Overbought, contrarian bearish{Style.RESET_ALL}")
        else:
            if pcr_oi > 1.0:
                reasons.append(f"{Fore.YELLOW}○ PCR (OI) at {pcr_oi:.2f} → Slight put dominance{Style.RESET_ALL}")
            else:
                reasons.append(f"{Fore.YELLOW}○ PCR (OI) at {pcr_oi:.2f} → Slight call dominance{Style.RESET_ALL}")

        # Bid-Ask Pressure
        ce_bidask = analysis['ce_bidask']
        pe_bidask = analysis['pe_bidask']

        if ce_bidask > 50000:
            reasons.append(f"{Fore.GREEN}✓ Strong CE buying pressure: {ce_bidask:+,} (bullish){Style.RESET_ALL}")
        elif ce_bidask < -50000:
            reasons.append(f"{Fore.RED}✗ Strong CE selling pressure: {ce_bidask:+,} (bearish){Style.RESET_ALL}")

        if pe_bidask > 50000:
            reasons.append(f"{Fore.RED}✗ Strong PE buying pressure: {pe_bidask:+,} (bearish){Style.RESET_ALL}")
        elif pe_bidask < -50000:
            reasons.append(f"{Fore.GREEN}✓ Strong PE selling pressure: {pe_bidask:+,} (bullish){Style.RESET_ALL}")

        # Support/Resistance position
        support = analysis['support']
        resistance = analysis['resistance']
        current = self.current_ltp

        if current < support:
            reasons.append(f"{Fore.RED}✗ Price {current:.2f} below support {support:.0f} (weak){Style.RESET_ALL}")
        elif current > resistance:
            reasons.append(
                f"{Fore.GREEN}✓ Price {current:.2f} above resistance {resistance:.0f} (strong){Style.RESET_ALL}")
        else:
            range_pct = ((current - support) / (resistance - support)) * 100 if resistance != support else 50
            if range_pct < 33:
                reasons.append(
                    f"{Fore.YELLOW}○ Price near support {support:.0f} ({range_pct:.0f}% in range){Style.RESET_ALL}")
            elif range_pct > 67:
                reasons.append(
                    f"{Fore.YELLOW}○ Price near resistance {resistance:.0f} ({range_pct:.0f}% in range){Style.RESET_ALL}")
            else:
                reasons.append(
                    f"{Fore.YELLOW}○ Price mid-range between {support:.0f}-{resistance:.0f}{Style.RESET_ALL}")

        # Volume vs OI signals
        pcr_vol = analysis['pcr_vol']
        if pcr_vol > 1.5 and pcr_oi < 1.0:
            reasons.append(
                f"{Fore.RED}✗ High PE volume ({pcr_vol:.2f}) vs low PE OI → Active bearish positions{Style.RESET_ALL}")
        elif pcr_vol < 0.67 and pcr_oi > 1.0:
            reasons.append(
                f"{Fore.GREEN}✓ High CE volume ({1 / pcr_vol:.2f}) vs low CE OI → Active bullish positions{Style.RESET_ALL}")

        # Print all reasons
        for reason in reasons:
            print(f"  {reason}")

        # Timeframe breakdown
        print(f"\n{Style.BRIGHT}Timeframe Breakdown:{Style.RESET_ALL}")
        print(f"  {'Timeframe':<10} {'Signal':<12} {'Value':<10} {'Confidence':<12}")
        print("  " + "-" * 50)

        for tf_name in ['5min', '15min', '1hr', 'day']:
            tf_data = analysis['timeframe_signals'][tf_name]
            signal = tf_data['signal']
            conf = tf_data['confidence']

            if signal > 0.3:
                tf_color = Fore.GREEN
                tf_dir = "Bullish"
            elif signal < -0.3:
                tf_color = Fore.RED
                tf_dir = "Bearish"
            else:
                tf_color = Fore.YELLOW
                tf_dir = "Neutral"

            print(f"  {tf_name:<10} {tf_color}{tf_dir:<12}{Style.RESET_ALL} {signal:>+6.2f}     {conf:>5.0%}")

        # Market Metrics Summary
        print(f"\n{Style.BRIGHT}Market Metrics:{Style.RESET_ALL}")
        print(f"  PCR (OI):  {analysis['pcr_oi']:.2f}  |  PCR (Vol): {analysis['pcr_vol']:.2f}")
        print(
            f"  Support:   {Fore.GREEN}{analysis['support']:.0f}{Style.RESET_ALL}  |  Spot: {Fore.CYAN}{self.current_ltp:.2f}{Style.RESET_ALL}  |  Resistance: {Fore.RED}{analysis['resistance']:.0f}{Style.RESET_ALL}")
        print(f"  CE Bid-Ask: {ce_bidask:>+10,}  |  PE Bid-Ask: {pe_bidask:>+10,}")

        print("=" * 140 + "\n")

    def display_option_chain(self):
        """Display option chain with embedded pressure analysis using color shading"""
        if not self.option_data:
            print("No option data available")
            return

        strikes = {}
        for token, data in self.option_data.items():
            strike = data['strike']
            if strike not in strikes:
                strikes[strike] = {'CE': {}, 'PE': {}}
            strikes[strike][data['type']] = data

        # Collect all values for top highlighting
        ce_oi_5m, ce_oi_15m, ce_oi_1h, ce_oi_day = [], [], [], []
        ce_vol_5m, ce_vol_15m, ce_vol_1h, ce_vol = [], [], [], []
        ce_oi = []
        pe_oi_5m, pe_oi_15m, pe_oi_1h, pe_oi_day = [], [], [], []
        pe_vol_5m, pe_vol_15m, pe_vol_1h, pe_vol = [], [], [], []
        pe_oi = []

        for strike in strikes.keys():
            ce = strikes[strike].get('CE', {})
            pe = strikes[strike].get('PE', {})

            if ce:
                ce_oi.append(ce.get('oi', 0))
                ce_oi_5m.append(abs(ce.get('oi_chg_5m', 0)))
                ce_oi_15m.append(abs(ce.get('oi_chg_15m', 0)))
                ce_oi_1h.append(abs(ce.get('oi_chg_1h', 0)))
                ce_oi_day.append(abs(ce.get('oi_chg_day', 0)))
                ce_vol.append(ce.get('volume', 0))
                ce_vol_5m.append(abs(ce.get('vol_chg_5m', 0)))
                ce_vol_15m.append(abs(ce.get('vol_chg_15m', 0)))
                ce_vol_1h.append(abs(ce.get('vol_chg_1h', 0)))

            if pe:
                pe_oi.append(pe.get('oi', 0))
                pe_oi_5m.append(abs(pe.get('oi_chg_5m', 0)))
                pe_oi_15m.append(abs(pe.get('oi_chg_15m', 0)))
                pe_oi_1h.append(abs(pe.get('oi_chg_1h', 0)))
                pe_oi_day.append(abs(pe.get('oi_chg_day', 0)))
                pe_vol.append(pe.get('volume', 0))
                pe_vol_5m.append(abs(pe.get('vol_chg_5m', 0)))
                pe_vol_15m.append(abs(pe.get('vol_chg_15m', 0)))
                pe_vol_1h.append(abs(pe.get('vol_chg_1h', 0)))

        ce_oi_top = get_top_values(ce_oi)
        ce_oi_5m_top = get_top_values(ce_oi_5m)
        ce_oi_15m_top = get_top_values(ce_oi_15m)
        ce_oi_1h_top = get_top_values(ce_oi_1h)
        ce_oi_day_top = get_top_values(ce_oi_day)
        ce_vol_top = get_top_values(ce_vol)
        ce_vol_5m_top = get_top_values(ce_vol_5m)
        ce_vol_15m_top = get_top_values(ce_vol_15m)
        ce_vol_1h_top = get_top_values(ce_vol_1h)

        pe_oi_top = get_top_values(pe_oi)
        pe_oi_5m_top = get_top_values(pe_oi_5m)
        pe_oi_15m_top = get_top_values(pe_oi_15m)
        pe_oi_1h_top = get_top_values(pe_oi_1h)
        pe_oi_day_top = get_top_values(pe_oi_day)
        pe_vol_top = get_top_values(pe_vol)
        pe_vol_5m_top = get_top_values(pe_vol_5m)
        pe_vol_15m_top = get_top_values(pe_vol_15m)
        pe_vol_1h_top = get_top_values(pe_vol_1h)

        print("\n" + "=" * 240)
        print(
            f"{Fore.CYAN}{Style.BRIGHT}NIFTY Option Chain with Pressure Analysis - {datetime.now().strftime('%H:%M:%S')}{Style.RESET_ALL}")
        print(
            f"{Fore.YELLOW}Spot: {self.current_ltp:.2f} | ATM: {self.strike} | Range: ATM ± {self.STRIKE_RANGE}{Style.RESET_ALL}")
        print("=" * 240)

        header = (
            f"{'CEVol5m':>10} {'CEVol15m':>10} {'CEVol1h':>10} "
            f"{'CEOI5m':>10} {'CEOI15m':>10} {'CEOI1h':>10} {'CEOIDay':>10} "
            f"{'CE_OI':>10} {'CE_Vol':>10} {'CE_LTP':>8} {'CEBidAsk':>10} "
            f"{'STRIKE':^10} {'Vol':^8} "
            f"{'PEBidAsk':>10} {'PE_LTP':>8} {'PE_Vol':>10} {'PE_OI':>10} "
            f"{'PEOIDay':>10} {'PEOI1h':>10} {'PEOI15m':>10} {'PEOI5m':>10} "
            f"{'PEVol5m':>10} {'PEVol15m':>10} {'PEVol1h':>10}"
        )
        print(f"{Fore.WHITE}{Style.BRIGHT}{header}{Style.RESET_ALL}")
        print("-" * 240)

        # Helper function to add pressure indicator
        def get_vol_pressure_indicator(ce_vol, pe_vol):
            """Returns pressure symbol and color based on CE vs PE volume"""
            if ce_vol == 0 and pe_vol == 0:
                return "", Fore.WHITE

            total = ce_vol + pe_vol
            if total == 0:
                return "", Fore.WHITE

            ce_pct = (ce_vol / total) * 100

            if ce_pct >= 75:
                return "↑↑", Fore.GREEN + Style.BRIGHT  # Strong bullish
            elif ce_pct >= 60:
                return "↑", Fore.GREEN  # Bullish
            elif ce_pct >= 40:
                return "↔", Fore.YELLOW  # Neutral
            elif ce_pct >= 25:
                return "↓", Fore.RED  # Bearish
            else:
                return "↓↓", Fore.RED + Style.BRIGHT  # Strong bearish

        def get_oi_pressure_background(ce_oi_chg, pe_oi_chg):
            """Returns background color based on OI change pattern"""
            # CE building + PE unwinding = Bearish (resistance forming)
            if ce_oi_chg > 0 and pe_oi_chg < 0:
                return Back.RED
            # CE unwinding + PE building = Bullish (support forming)
            elif ce_oi_chg < 0 and pe_oi_chg > 0:
                return Back.GREEN
            # Both building = Volatile
            elif ce_oi_chg > 0 and pe_oi_chg > 0:
                return Back.YELLOW
            else:
                return ""

        # Data rows with pressure indicators
        for strike in sorted(strikes.keys()):
            ce_data = strikes[strike].get('CE', {})
            pe_data = strikes[strike].get('PE', {})

            # Strike label
            if strike == self.strike:
                strike_str = f"{Fore.MAGENTA}{Style.BRIGHT}*{strike:.0f}*{Style.RESET_ALL}"
            else:
                strike_str = f"{strike:.0f}"

            # Get values
            ce_vol_5m_val = abs(ce_data.get('vol_chg_5m', 0))
            ce_vol_15m_val = abs(ce_data.get('vol_chg_15m', 0))
            ce_vol_1h_val = abs(ce_data.get('vol_chg_1h', 0))
            ce_oi_5m_val = abs(ce_data.get('oi_chg_5m', 0))
            ce_oi_15m_val = abs(ce_data.get('oi_chg_15m', 0))
            ce_oi_1h_val = abs(ce_data.get('oi_chg_1h', 0))
            ce_oi_day_val = abs(ce_data.get('oi_chg_day', 0))
            ce_oi_val = ce_data.get('oi', 0)
            ce_vol_val = ce_data.get('volume', 0)
            ce_qty_diff = ce_data.get('qty_diff', 0)

            pe_vol_5m_val = abs(pe_data.get('vol_chg_5m', 0))
            pe_vol_15m_val = abs(pe_data.get('vol_chg_15m', 0))
            pe_vol_1h_val = abs(pe_data.get('vol_chg_1h', 0))
            pe_oi_5m_val = abs(pe_data.get('oi_chg_5m', 0))
            pe_oi_15m_val = abs(pe_data.get('oi_chg_15m', 0))
            pe_oi_1h_val = abs(pe_data.get('oi_chg_1h', 0))
            pe_oi_day_val = abs(pe_data.get('oi_chg_day', 0))
            pe_oi_val = pe_data.get('oi', 0)
            pe_vol_val = pe_data.get('volume', 0)
            pe_qty_diff = pe_data.get('qty_diff', 0)

            # Volume pressure indicators
            vol_5m_symbol, vol_5m_color = get_vol_pressure_indicator(ce_vol_5m_val, pe_vol_5m_val)

            # OI pressure background (for 5min - most recent)
            ce_oi_5m_raw = ce_data.get('oi_chg_5m', 0)
            pe_oi_5m_raw = pe_data.get('oi_chg_5m', 0)
            oi_bg = get_oi_pressure_background(ce_oi_5m_raw, pe_oi_5m_raw)

            # Format bid-ask
            ce_qty_str = f"{ce_qty_diff:+,}" if ce_qty_diff != 0 else "0"
            pe_qty_str = f"{pe_qty_diff:+,}" if pe_qty_diff != 0 else "0"

            # Build row with pressure indicators
            row = (
                f"{format_with_highlight(ce_vol_5m_val, ce_vol_5m_top[0], ce_vol_5m_top[1], True, True)} "
                f"{format_with_highlight(ce_vol_15m_val, ce_vol_15m_top[0], ce_vol_15m_top[1], True, True)} "
                f"{format_with_highlight(ce_vol_1h_val, ce_vol_1h_top[0], ce_vol_1h_top[1], True, True)} "
                f"{oi_bg}{format_with_highlight(ce_oi_5m_val, ce_oi_5m_top[0], ce_oi_5m_top[1], True, True)}{Style.RESET_ALL} "
                f"{format_with_highlight(ce_oi_15m_val, ce_oi_15m_top[0], ce_oi_15m_top[1], True, True)} "
                f"{format_with_highlight(ce_oi_1h_val, ce_oi_1h_top[0], ce_oi_1h_top[1], True, True)} "
                f"{format_with_highlight(ce_oi_day_val, ce_oi_day_top[0], ce_oi_day_top[1], True, True)} "
                f"{format_with_highlight(ce_oi_val, ce_oi_top[0], ce_oi_top[1], True, True)} "
                f"{format_with_highlight(ce_vol_val, ce_vol_top[0], ce_vol_top[1], True, True)} "
                f"{ce_data.get('ltp', 0.0):>8.2f} "
                f"{ce_qty_str:>10} "
                f"{strike_str:^10} "
                f"{vol_5m_color}{vol_5m_symbol:^8}{Style.RESET_ALL} "
                f"{pe_qty_str:>10} "
                f"{pe_data.get('ltp', 0.0):>8.2f} "
                f"{format_with_highlight(pe_vol_val, pe_vol_top[0], pe_vol_top[1], False, True)} "
                f"{format_with_highlight(pe_oi_val, pe_oi_top[0], pe_oi_top[1], False, True)} "
                f"{format_with_highlight(pe_oi_day_val, pe_oi_day_top[0], pe_oi_day_top[1], False, True)} "
                f"{format_with_highlight(pe_oi_1h_val, pe_oi_1h_top[0], pe_oi_1h_top[1], False, True)} "
                f"{format_with_highlight(pe_oi_15m_val, pe_oi_15m_top[0], pe_oi_15m_top[1], False, True)} "
                f"{oi_bg}{format_with_highlight(pe_oi_5m_val, pe_oi_5m_top[0], pe_oi_5m_top[1], False, True)}{Style.RESET_ALL} "
                f"{format_with_highlight(pe_vol_5m_val, pe_vol_5m_top[0], pe_vol_5m_top[1], False, True)} "
                f"{format_with_highlight(pe_vol_15m_val, pe_vol_15m_top[0], pe_vol_15m_top[1], False, True)} "
                f"{format_with_highlight(pe_vol_1h_val, pe_vol_1h_top[0], pe_vol_1h_top[1], False, True)}"
            )
            print(row)

        # Legend
        print("=" * 240)
        print(f"\n{Style.BRIGHT}Pressure Indicators:{Style.RESET_ALL}")
        print(
            f"  Vol Column: {Fore.GREEN}↑↑{Style.RESET_ALL} Strong Bullish (CE>75%) | {Fore.GREEN}↑{Style.RESET_ALL} Bullish (CE>60%) | {Fore.YELLOW}↔{Style.RESET_ALL} Neutral | {Fore.RED}↓{Style.RESET_ALL} Bearish (PE>60%) | {Fore.RED}↓↓{Style.RESET_ALL} Strong Bearish (PE>75%)")
        print(
            f"  OI 5m BG: {Back.RED}{Fore.WHITE}Red{Style.RESET_ALL} = CE Build+PE Unwind (Resistance) | {Back.GREEN}{Fore.WHITE}Green{Style.RESET_ALL} = CE Unwind+PE Build (Support) | {Back.YELLOW}{Fore.BLACK}Yellow{Style.RESET_ALL} = Both Building (Volatile)")
        print("=" * 240 + "\n")

    def display_bid_ask_depth(self, strike_to_show):
        """Display detailed bid-ask depth for a specific strike"""
        if not self.option_data:
            print("No option data available")
            return

        ce_token, pe_token = None, None
        for token, data in self.option_data.items():
            if data['strike'] == strike_to_show:
                if data['type'] == 'CE':
                    ce_token = token
                elif data['type'] == 'PE':
                    pe_token = token

        print("\n" + "=" * 120)
        print(
            f"{Fore.CYAN}{Style.BRIGHT}Bid-Ask Depth for Strike {strike_to_show} - {datetime.now().strftime('%H:%M:%S')}{Style.RESET_ALL}")
        print("=" * 120)

        if ce_token:
            ce_data = self.option_data[ce_token]
            print(f"\n{Fore.RED}{Style.BRIGHT}CALL (CE) - {ce_data['symbol']}{Style.RESET_ALL}")
            print(
                f"LTP: {ce_data['ltp']:.2f} | OI: {format_number(ce_data['oi'])} | Volume: {format_number(ce_data['volume'])} | Bid-Ask Diff: {ce_data['qty_diff']:+,}")
            print(f"\n{'Level':>5} {'Bid Qty':>12} {'Bid Price':>12}   {'Ask Price':>12} {'Ask Qty':>12}")
            print("-" * 60)
            for i in range(1, 6):
                bid_qty = ce_data.get(f'bid_qty_{i}', 0)
                bid_price = ce_data.get(f'bid_price_{i}', 0.0)
                ask_qty = ce_data.get(f'ask_qty_{i}', 0)
                ask_price = ce_data.get(f'ask_price_{i}', 0.0)

                if bid_qty > 0 or ask_qty > 0:
                    print(
                        f"{Fore.GREEN}{i:>5} {bid_qty:>12,} {bid_price:>12.2f}   {ask_price:>12.2f} {ask_qty:>12,}{Style.RESET_ALL}")
                else:
                    print(f"{i:>5} {bid_qty:>12,} {bid_price:>12.2f}   {ask_price:>12.2f} {ask_qty:>12,}")

        if pe_token:
            pe_data = self.option_data[pe_token]
            print(f"\n{Fore.GREEN}{Style.BRIGHT}PUT (PE) - {pe_data['symbol']}{Style.RESET_ALL}")
            print(
                f"LTP: {pe_data['ltp']:.2f} | OI: {format_number(pe_data['oi'])} | Volume: {format_number(pe_data['volume'])} | Bid-Ask Diff: {pe_data['qty_diff']:+,}")
            print(f"\n{'Level':>5} {'Bid Qty':>12} {'Bid Price':>12}   {'Ask Price':>12} {'Ask Qty':>12}")
            print("-" * 60)
            for i in range(1, 6):
                bid_qty = pe_data.get(f'bid_qty_{i}', 0)
                bid_price = pe_data.get(f'bid_price_{i}', 0.0)
                ask_qty = pe_data.get(f'ask_qty_{i}', 0)
                ask_price = pe_data.get(f'ask_price_{i}', 0.0)

                if bid_qty > 0 or ask_qty > 0:
                    print(
                        f"{Fore.GREEN}{i:>5} {bid_qty:>12,} {bid_price:>12.2f}   {ask_price:>12.2f} {ask_qty:>12,}{Style.RESET_ALL}")
                else:
                    print(f"{i:>5} {bid_qty:>12,} {bid_price:>12.2f}   {ask_price:>12.2f} {ask_qty:>12,}")

        print("=" * 120 + "\n")


    def display_strike_wise_analysis(self):
        """Display detailed strike-wise pressure analysis with CE vs PE volume comparison"""
        if not self.option_data:
            return

        # Organize data by strikes
        strikes = {}
        for token, data in self.option_data.items():
            strike = data['strike']
            if strike not in strikes:
                strikes[strike] = {'CE': {}, 'PE': {}}
            strikes[strike][data['type']] = data

        print("\n" + "=" * 200)
        print(
            f"{Fore.CYAN}{Style.BRIGHT}STRIKE-WISE PRESSURE ANALYSIS - {datetime.now().strftime('%H:%M:%S')}{Style.RESET_ALL}")
        print("=" * 200)

        # Function to classify OI activity
        def classify_oi_activity(value, thresholds):
            """Classify OI change as Heavy/Mid/Light and Bullish/Bearish"""
            abs_val = abs(value)

            if abs_val >= thresholds[2]:
                intensity = "Heavy"
            elif abs_val >= thresholds[1]:
                intensity = "Mid"
            elif abs_val >= thresholds[0]:
                intensity = "Light"
            else:
                intensity = "None"

            if value > 0:
                direction = "Build"  # OI increasing
                color = Fore.YELLOW
            elif value < 0:
                direction = "Unwind"  # OI decreasing
                color = Fore.CYAN
            else:
                direction = "Flat"
                color = Fore.WHITE

            return intensity, direction, color

        def classify_total_oi(value, thresholds):
            """Classify total OI as Heavy/Mid/Light"""
            if value >= thresholds[2]:
                intensity = "Heavy"
                color = Fore.RED
            elif value >= thresholds[1]:
                intensity = "Mid"
                color = Fore.YELLOW
            elif value >= thresholds[0]:
                intensity = "Light"
                color = Fore.GREEN
            else:
                intensity = "None"
                color = Fore.WHITE
            return intensity, color

        # Calculate thresholds based on all strikes
        import numpy as np

        all_oi_5m = []
        all_oi_15m = []
        all_oi_1h = []
        all_oi_day = []
        all_oi_total = []

        for s in strikes:
            for opt_type in ['CE', 'PE']:
                if opt_type in strikes[s]:
                    all_oi_5m.append(abs(strikes[s][opt_type].get('oi_chg_5m', 0)))
                    all_oi_15m.append(abs(strikes[s][opt_type].get('oi_chg_15m', 0)))
                    all_oi_1h.append(abs(strikes[s][opt_type].get('oi_chg_1h', 0)))
                    all_oi_day.append(abs(strikes[s][opt_type].get('oi_chg_day', 0)))
                    all_oi_total.append(strikes[s][opt_type].get('oi', 0))

        oi_5m_thresh = (np.percentile(all_oi_5m, 33), np.percentile(all_oi_5m, 66), np.percentile(all_oi_5m, 85))
        oi_15m_thresh = (np.percentile(all_oi_15m, 33), np.percentile(all_oi_15m, 66), np.percentile(all_oi_15m, 85))
        oi_1h_thresh = (np.percentile(all_oi_1h, 33), np.percentile(all_oi_1h, 66), np.percentile(all_oi_1h, 85))
        oi_day_thresh = (np.percentile(all_oi_day, 33), np.percentile(all_oi_day, 66), np.percentile(all_oi_day, 85))
        oi_total_thresh = (
        np.percentile(all_oi_total, 33), np.percentile(all_oi_total, 66), np.percentile(all_oi_total, 85))

        # Header
        print(
            f"\n{Style.BRIGHT}{'Strike':<8} {'Total OI':<20} {'5m OI':<25} {'15m OI':<25} {'1h OI':<25} {'Day OI':<25} {'Vol Pressure':<30}{Style.RESET_ALL}")
        print("-" * 200)

        # Display each strike
        for strike in sorted(strikes.keys()):
            ce_data = strikes[strike].get('CE', {})
            pe_data = strikes[strike].get('PE', {})

            # Mark ATM
            if strike == self.strike:
                strike_label = f"{Fore.MAGENTA}{Style.BRIGHT}*{strike:.0f}*{Style.RESET_ALL}"
            else:
                strike_label = f"{strike:.0f}"

            # Get CE data
            ce_oi_total = ce_data.get('oi', 0) if ce_data else 0
            ce_oi_5m = ce_data.get('oi_chg_5m', 0) if ce_data else 0
            ce_oi_15m = ce_data.get('oi_chg_15m', 0) if ce_data else 0
            ce_oi_1h = ce_data.get('oi_chg_1h', 0) if ce_data else 0
            ce_oi_day = ce_data.get('oi_chg_day', 0) if ce_data else 0
            ce_vol_5m = abs(ce_data.get('vol_chg_5m', 0)) if ce_data else 0
            ce_vol_15m = abs(ce_data.get('vol_chg_15m', 0)) if ce_data else 0
            ce_vol_1h = abs(ce_data.get('vol_chg_1h', 0)) if ce_data else 0

            # Get PE data
            pe_oi_total = pe_data.get('oi', 0) if pe_data else 0
            pe_oi_5m = pe_data.get('oi_chg_5m', 0) if pe_data else 0
            pe_oi_15m = pe_data.get('oi_chg_15m', 0) if pe_data else 0
            pe_oi_1h = pe_data.get('oi_chg_1h', 0) if pe_data else 0
            pe_oi_day = pe_data.get('oi_chg_day', 0) if pe_data else 0
            pe_vol_5m = abs(pe_data.get('vol_chg_5m', 0)) if pe_data else 0
            pe_vol_15m = abs(pe_data.get('vol_chg_15m', 0)) if pe_data else 0
            pe_vol_1h = abs(pe_data.get('vol_chg_1h', 0)) if pe_data else 0

            # Classify CE OI
            ce_oi_total_int, ce_oi_total_col = classify_total_oi(ce_oi_total, oi_total_thresh)
            ce_oi_5m_int, ce_oi_5m_dir, ce_oi_5m_col = classify_oi_activity(ce_oi_5m, oi_5m_thresh)
            ce_oi_15m_int, ce_oi_15m_dir, ce_oi_15m_col = classify_oi_activity(ce_oi_15m, oi_15m_thresh)
            ce_oi_1h_int, ce_oi_1h_dir, ce_oi_1h_col = classify_oi_activity(ce_oi_1h, oi_1h_thresh)
            ce_oi_day_int, ce_oi_day_dir, ce_oi_day_col = classify_oi_activity(ce_oi_day, oi_day_thresh)

            # Classify PE OI
            pe_oi_total_int, pe_oi_total_col = classify_total_oi(pe_oi_total, oi_total_thresh)
            pe_oi_5m_int, pe_oi_5m_dir, pe_oi_5m_col = classify_oi_activity(pe_oi_5m, oi_5m_thresh)
            pe_oi_15m_int, pe_oi_15m_dir, pe_oi_15m_col = classify_oi_activity(pe_oi_15m, oi_15m_thresh)
            pe_oi_1h_int, pe_oi_1h_dir, pe_oi_1h_col = classify_oi_activity(pe_oi_1h, oi_1h_thresh)
            pe_oi_day_int, pe_oi_day_dir, pe_oi_day_col = classify_oi_activity(pe_oi_day, oi_day_thresh)

            # Volume pressure analysis (CE vs PE comparison)
            def analyze_volume_pressure(ce_vol, pe_vol):
                if ce_vol == 0 and pe_vol == 0:
                    return "No Activity", Fore.WHITE

                total_vol = ce_vol + pe_vol
                if total_vol == 0:
                    return "No Activity", Fore.WHITE

                ce_pct = (ce_vol / total_vol) * 100
                pe_pct = (pe_vol / total_vol) * 100

                if ce_pct > 65:
                    return f"CE Dom {ce_pct:.0f}% (Bullish)", Fore.GREEN
                elif pe_pct > 65:
                    return f"PE Dom {pe_pct:.0f}% (Bearish)", Fore.RED
                else:
                    return f"Balanced (CE:{ce_pct:.0f}%)", Fore.YELLOW

            vol_5m_pressure, vol_5m_col = analyze_volume_pressure(ce_vol_5m, pe_vol_5m)
            vol_15m_pressure, vol_15m_col = analyze_volume_pressure(ce_vol_15m, pe_vol_15m)
            vol_1h_pressure, vol_1h_col = analyze_volume_pressure(ce_vol_1h, pe_vol_1h)

            # Print CE row
            print(f"{strike_label:<8} "
                  f"{Fore.RED}CE{Style.RESET_ALL} {ce_oi_total_col}{ce_oi_total_int:<6}{format_number(ce_oi_total):>8}{Style.RESET_ALL}  "
                  f"{ce_oi_5m_col}{ce_oi_5m_int:<6}{ce_oi_5m_dir:<8}{format_number(abs(ce_oi_5m)):>8}{Style.RESET_ALL}  "
                  f"{ce_oi_15m_col}{ce_oi_15m_int:<6}{ce_oi_15m_dir:<8}{format_number(abs(ce_oi_15m)):>8}{Style.RESET_ALL}  "
                  f"{ce_oi_1h_col}{ce_oi_1h_int:<6}{ce_oi_1h_dir:<8}{format_number(abs(ce_oi_1h)):>8}{Style.RESET_ALL}  "
                  f"{ce_oi_day_col}{ce_oi_day_int:<6}{ce_oi_day_dir:<8}{format_number(abs(ce_oi_day)):>8}{Style.RESET_ALL}  "
                  f"{vol_5m_col}{vol_5m_pressure:<28}{Style.RESET_ALL}")

            # Print PE row
            print(f"{'':8} "
                  f"{Fore.GREEN}PE{Style.RESET_ALL} {pe_oi_total_col}{pe_oi_total_int:<6}{format_number(pe_oi_total):>8}{Style.RESET_ALL}  "
                  f"{pe_oi_5m_col}{pe_oi_5m_int:<6}{pe_oi_5m_dir:<8}{format_number(abs(pe_oi_5m)):>8}{Style.RESET_ALL}  "
                  f"{pe_oi_15m_col}{pe_oi_15m_int:<6}{pe_oi_15m_dir:<8}{format_number(abs(pe_oi_15m)):>8}{Style.RESET_ALL}  "
                  f"{pe_oi_1h_col}{pe_oi_1h_int:<6}{pe_oi_1h_dir:<8}{format_number(abs(pe_oi_1h)):>8}{Style.RESET_ALL}  "
                  f"{pe_oi_day_col}{pe_oi_day_int:<6}{pe_oi_day_dir:<8}{format_number(abs(pe_oi_day)):>8}{Style.RESET_ALL}  "
                  f"{vol_15m_col}15m: {vol_15m_pressure:<20}{Style.RESET_ALL}")

            # Print volume 1h row
            print(f"{'':8} {'':21} {'':26} {'':26} {'':26} {'':26} "
                  f"{vol_1h_col}1h: {vol_1h_pressure:<21}{Style.RESET_ALL}")

            print()  # Blank line between strikes

        # Legend
        print("=" * 200)
        print(f"\n{Style.BRIGHT}Legend:{Style.RESET_ALL}")
        print(f"  {Fore.RED}Total OI Intensity:{Style.RESET_ALL} Heavy (Top 15%) | Mid (33-66%) | Light (Bottom 33%)")
        print(f"  {Fore.YELLOW}OI Change:{Style.RESET_ALL} Build (Increasing) | Unwind (Decreasing) | Flat")
        print(
            f"  {Fore.CYAN}Volume Pressure:{Style.RESET_ALL} CE Dom (>65% CE = Bullish) | PE Dom (>65% PE = Bearish) | Balanced")
        print(
            f"\n  {Fore.RED}CE OI Building{Style.RESET_ALL} = Resistance forming (Bearish) | {Fore.GREEN}PE OI Building{Style.RESET_ALL} = Support forming (Bullish)")
        print(
            f"  {Fore.GREEN}CE Volume > PE{Style.RESET_ALL} = Bullish sentiment | {Fore.RED}PE Volume > CE{Style.RESET_ALL} = Bearish sentiment")
        print("=" * 200 + "\n")


def main():
    option_chain = NiftyOptionChainLTP()

    if option_chain.login():
        option_chain.get_option_chain_tokens()

        print("\nWaiting for live feed data and depth updates...")
        time.sleep(8)

        try:
            counter = 0
            while True:
                # Display pressure analysis first
                #option_chain.display_pressure_analysis()
                # Display strike-wise detailed analysis
                #option_chain.display_strike_wise_analysis()
                # Display direction analysis
             #   option_chain.display_direction_analysis()

                # Display option chain
                option_chain.display_option_chain()


                counter += 1
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopping...")
            option_chain.stop_depth_updates()


if __name__ == "__main__":
    main()
