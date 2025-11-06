"""
NIFTY 50 OPTIONS BOT - FIXED VERSION V4
✅ FIXED: OI Logic (CE high = Bullish, PE high = Bearish)
✅ FIXED: OI Change Tracking (Buildup vs Unwinding)
✅ FIXED: Volume Window Optimization
✅ FIXED: Signal Validation with Distance Check
✅ Enhanced Error Handling
✅ NEW: Timestamped Folder Structure for Logs & Trades
"""

import requests
import pandas as pd
import numpy as np
import datetime as dt
import time
import csv
import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# ==================== CONFIGURATION ====================
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1NUJBOVgiLCJqdGkiOiI2OTBjMTY3YTE0MmVhMzBjZDViYzI2MGIiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc2MjM5OTg2NiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzYyNDY2NDAwfQ.fJuD3SwFjuP6bEqmJzPTimFVg_klQRwsjSPAKnNzHso"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1412386951474057299/Jgft_nxzGxcfWOhoLbSWMde-_bwapvqx8l3VQGQwEoR7_8n4b9Q9zN242kMoXsVbLdvG"

NIFTY_SYMBOL = "NSE_INDEX|Nifty 50"
SIGNAL_COOLDOWN = 300

LOT_SIZE = 75
TAKE_PROFIT = 1500
STOP_LOSS = 2000
TRAILING_STOP = 500
VOLUME_THRESHOLD = 1.00
MIN_CPR_DISTANCE = 10
# =======================================================

last_signal_time = None
current_expiry_date = None
contracts_cache = []
open_position = None
previous_oi_data = {"ce": 0, "pe": 0, "timestamp": None}

# Session variables
session_id = None
session_dir = None
trade_dir = None
log_dir = None
terminal_dir = None
csv_file_path = None
logger = None


# ==================== FOLDER STRUCTURE SETUP ====================

class TeeLogger:
    """Dual output: Console + File for terminal logs"""
    def __init__(self, terminal_file):
        self.terminal = sys.stdout
        self.log_file = open(terminal_file, 'a', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def create_session_folders():
    """Create timestamped folder structure for this bot session"""
    global session_id, session_dir, trade_dir, log_dir, terminal_dir, csv_file_path
    
    session_id = dt.datetime.now().strftime('%Y-%m-%d_%H%M%S')
    
    base_dir = "nifty_bot_runs"
    session_dir = os.path.join(base_dir, f"run_{session_id}")
    trade_dir = os.path.join(session_dir, "trade_details")
    log_dir = os.path.join(session_dir, "log_reports")
    terminal_dir = os.path.join(session_dir, "terminal_logs")
    
    os.makedirs(trade_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(terminal_dir, exist_ok=True)
    
    csv_file_path = os.path.join(trade_dir, "trades.csv")
    
    print(f"\n{'=' * 80}")
    print(f"📁 SESSION FOLDER CREATED")
    print(f"{'=' * 80}")
    print(f"Session ID: {session_id}")
    print(f"Location: {session_dir}")
    print(f"  └── trade_details/trades.csv")
    print(f"  └── log_reports/bot.log")
    print(f"  └── terminal_logs/terminal.txt")
    print(f"{'=' * 80}\n")
    
    return session_dir


def setup_logging():
    """Setup structured logging to log_reports/bot.log"""
    global logger
    
    logger = logging.getLogger('NiftyBot')
    logger.setLevel(logging.DEBUG)
    
    log_file = os.path.join(log_dir, "bot.log")
    
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-25s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("BOT SESSION STARTED")
    logger.info(f"Session ID: {session_id}")
    logger.info(f"Lot Size: {LOT_SIZE}")
    logger.info(f"Take Profit: ₹{TAKE_PROFIT}")
    logger.info(f"Stop Loss: ₹{STOP_LOSS}")
    logger.info(f"Trailing Stop: ₹{TRAILING_STOP}")
    logger.info("=" * 80)


def setup_terminal_logging():
    """Redirect stdout to both console and terminal_logs/terminal.txt"""
    terminal_file = os.path.join(terminal_dir, "terminal.txt")
    sys.stdout = TeeLogger(terminal_file)


# ==================== POSITION TRACKING ====================

class Position:
    def __init__(self, signal_type, strike, entry_premium, instrument_key, timestamp):
        self.signal_type = signal_type
        self.strike = strike
        self.entry_premium = entry_premium
        self.instrument_key = instrument_key
        self.timestamp = timestamp
        self.lot_size = LOT_SIZE
        self.highest_pnl = 0
        self.trailing_stop_active = False
        self.trailing_stop_price = None
        
        logger.info(f"NEW POSITION OPENED: {signal_type} {strike} @ ₹{entry_premium}")
    
    def calculate_pnl(self, current_premium):
        premium_diff = current_premium - self.entry_premium
        pnl = premium_diff * self.lot_size
        
        if pnl > self.highest_pnl:
            self.highest_pnl = pnl
        
        return pnl, premium_diff
    
    def check_exit(self, current_premium):
        pnl, premium_diff = self.calculate_pnl(current_premium)
        
        if pnl <= -STOP_LOSS:
            logger.warning(f"STOP LOSS HIT: P&L = ₹{pnl:.2f}")
            return True, f"STOP LOSS (Loss: ₹{abs(pnl):.2f})", pnl, premium_diff
        
        if pnl >= TAKE_PROFIT:
            if not self.trailing_stop_active:
                self.trailing_stop_active = True
                self.trailing_stop_price = current_premium - (TRAILING_STOP / self.lot_size)
                print(f"  🎯 Take Profit reached! Trailing stop: ₹{self.trailing_stop_price:.2f}")
                logger.info(f"TRAILING STOP ACTIVATED at ₹{self.trailing_stop_price:.2f}")
        
        if self.trailing_stop_active:
            if current_premium <= self.trailing_stop_price:
                logger.info(f"TRAILING STOP TRIGGERED: P&L = ₹{pnl:.2f}")
                return True, f"TRAILING STOP (Profit: ₹{pnl:.2f})", pnl, premium_diff
            
            new_trailing_stop = current_premium - (TRAILING_STOP / self.lot_size)
            if new_trailing_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_trailing_stop
                print(f"  📈 Trailing stop updated: ₹{self.trailing_stop_price:.2f}")
                logger.debug(f"Trailing stop updated to ₹{self.trailing_stop_price:.2f}")
        
        return False, None, pnl, premium_diff


# ==================== DISCORD ====================

def send_discord_alert(title, description, color=0x00ff00, fields=None):
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "footer": {"text": f"Session: {session_id} | Lot: {LOT_SIZE}"}
    }
    
    if fields:
        embed["fields"] = fields
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=10)
        if response.status_code == 204:
            print("  ✅ Discord alert sent")
            logger.info(f"Discord alert sent: {title}")
    except Exception as e:
        print(f"  ⚠️  Discord error: {e}")
        logger.error(f"Discord error: {e}")


# ==================== TOKEN VALIDATION ====================

def validate_token():
    url = "https://api.upstox.com/v2/user/profile"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            profile = response.json()
            username = profile.get('data', {}).get('user_name', 'N/A')
            print(f"✅ Token Valid | User: {username}")
            logger.info(f"Token validated successfully for user: {username}")
            return True
        else:
            print(f"❌ TOKEN EXPIRED (Status: {response.status_code})")
            logger.error(f"Token validation failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Token validation error: {e}")
        logger.error(f"Token validation exception: {e}")
        return False


# ==================== GET EXPIRY DATE ====================

def get_next_tuesday_expiry():
    today = dt.datetime.now()
    
    if today.weekday() == 1:
        if today.hour < 15 or (today.hour == 15 and today.minute < 30):
            expiry = today
        else:
            expiry = today + dt.timedelta(days=7)
    else:
        days_ahead = (1 - today.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        expiry = today + dt.timedelta(days=days_ahead)
    
    expiry_date = expiry.strftime('%Y-%m-%d')
    print(f"  ✅ Next Expiry: {expiry_date} ({expiry.strftime('%A')})")
    logger.info(f"Next expiry date calculated: {expiry_date}")
    return expiry_date


# ==================== GET OPTION INSTRUMENTS ====================

def get_option_instruments():
    global current_expiry_date, contracts_cache
    
    current_expiry_date = get_next_tuesday_expiry()
    
    encoded_symbol = "NSE_INDEX%7CNifty%2050"
    url = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_symbol}&expiry_date={current_expiry_date}"
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"  ⚠️  API returned status {response.status_code}")
            logger.warning(f"Option instruments API returned status {response.status_code}")
            return []
        
        data = response.json()
        
        if "data" not in data or data["data"] is None or len(data["data"]) == 0:
            url_no_expiry = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_symbol}"
            response2 = requests.get(url_no_expiry, headers=headers, timeout=10)
            
            if response2.status_code == 200:
                data2 = response2.json()
                all_contracts = data2.get("data", [])
                
                if len(all_contracts) > 0:
                    expiries = sorted(set([c["expiry"] for c in all_contracts]))
                    nearest_expiry = expiries[0]
                    current_expiry_date = nearest_expiry
                    contracts_cache = [c for c in all_contracts if c["expiry"] == nearest_expiry]
                    logger.info(f"Using nearest expiry: {nearest_expiry}")
            else:
                return []
        else:
            contracts_cache = data["data"]
        
        if len(contracts_cache) == 0:
            print("  ⚠️  No contracts found")
            logger.warning("No option contracts found")
            return []
        
        spot_price = get_spot_price()
        
        if spot_price:
            print(f"  ✅ Nifty Spot: {spot_price:.2f}")
            filtered = [c["instrument_key"] for c in contracts_cache if abs(c["strike_price"] - spot_price) <= 500]
            print(f"  ✅ Selected {len(filtered)} contracts")
            logger.info(f"Fetched {len(filtered)} option contracts near spot price {spot_price:.2f}")
            return filtered
        else:
            return [c["instrument_key"] for c in contracts_cache[:50]]
        
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        logger.error(f"Exception in get_option_instruments: {e}")
        return []


def get_spot_price():
    try:
        encoded_symbol = NIFTY_SYMBOL.replace("|", "%7C").replace(" ", "%20")
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_symbol}"
        headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and NIFTY_SYMBOL in data["data"]:
                spot = data["data"][NIFTY_SYMBOL]["last_price"]
                logger.debug(f"Spot price fetched: {spot:.2f}")
                return spot
        
        return None
    except Exception as e:
        logger.error(f"Spot price fetch error: {e}")
        return None


# ==================== GET LIVE OI WITH CHANGE TRACKING ====================

def get_live_oi_from_quotes(instrument_keys):
    global previous_oi_data
    
    if not instrument_keys:
        return None, 0, 0, 0, 0
    
    ce_oi_total = 0
    pe_oi_total = 0
    
    for i in range(0, len(instrument_keys), 100):
        batch = instrument_keys[i:i+100]
        instrument_param = ",".join(batch)
        
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={instrument_param}"
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                continue
            
            data = response.json()
            
            if "data" in data:
                for instrument_key, quote_data in data["data"].items():
                    if "oi" in quote_data:
                        oi_value = quote_data["oi"]
                        
                        if "CE" in instrument_key:
                            ce_oi_total += oi_value
                        elif "PE" in instrument_key:
                            pe_oi_total += oi_value
        
        except Exception as e:
            logger.error(f"OI fetch error in batch: {e}")
            continue
    
    if ce_oi_total == 0 and pe_oi_total == 0:
        return None, 0, 0, 0, 0
    
    ce_oi_change = ce_oi_total - previous_oi_data["ce"]
    pe_oi_change = pe_oi_total - previous_oi_data["pe"]
    
    if ce_oi_total > pe_oi_total * 1.05:
        trend = "Bullish"
    elif pe_oi_total > ce_oi_total * 1.05:
        trend = "Bearish"
    else:
        trend = "Sideways"
    
    oi_action = "Neutral"
    if ce_oi_change > 0 and pe_oi_change > 0:
        oi_action = "Both Building"
    elif ce_oi_change > pe_oi_change * 2:
        oi_action = "CE Buildup"
    elif pe_oi_change > ce_oi_change * 2:
        oi_action = "PE Buildup"
    elif ce_oi_change < 0 and pe_oi_change < 0:
        oi_action = "Both Unwinding"
    
    print(f"  ✅ Live OI: CE={ce_oi_total:,.0f} ({ce_oi_change:+,.0f}) | PE={pe_oi_total:,.0f} ({pe_oi_change:+,.0f})")
    print(f"  📊 OI Trend: {trend} | Action: {oi_action}")
    
    logger.info(f"OI Data - CE: {ce_oi_total:,.0f} ({ce_oi_change:+,.0f}), PE: {pe_oi_total:,.0f} ({pe_oi_change:+,.0f}), Trend: {trend}, Action: {oi_action}")
    
    previous_oi_data = {
        "ce": ce_oi_total,
        "pe": pe_oi_total,
        "timestamp": dt.datetime.now()
    }
    
    return trend, ce_oi_total, pe_oi_total, ce_oi_change, pe_oi_change


# ==================== LIVE DATA FETCHING ====================

def get_live_candles(symbol):
    encoded_symbol = symbol.replace("|", "%7C").replace(" ", "%20")
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_symbol}/1minute"
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"  ⚠️  Candles API returned {response.status_code}")
            logger.warning(f"Candles API returned status {response.status_code}")
            return None
        
        data = response.json()
        
        if "data" not in data or "candles" not in data["data"]:
            print("  ⚠️  No candle data in response")
            return None
        
        candles = data["data"]["candles"]
        
        if len(candles) == 0:
            print("  ⚠️  Empty candles list")
            return None
        
        df = pd.DataFrame(candles, columns=["time","open","high","low","close","volume","oi"])
        df["time"] = pd.to_datetime(df["time"])
        df["volume"] = df["volume"].replace(0, 1)
        df = df.sort_values("time").reset_index(drop=True)
        
        df.set_index("time", inplace=True)
        df_5min = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        df_5min.reset_index(inplace=True)
        
        print(f"  ✅ Candles: {len(candles)} 1-min → {len(df_5min)} 5-min")
        logger.debug(f"Candles resampled: {len(candles)} 1-min → {len(df_5min)} 5-min")
        return df_5min
        
    except Exception as e:
        print(f"  ❌ Candle error: {e}")
        logger.error(f"Candle fetch error: {e}")
        return None


# ==================== CPR CALCULATION ====================

def calculate_cpr_from_previous_day():
    try:
        today = dt.datetime.now()
        
        prev_day = today - dt.timedelta(days=1)
        while prev_day.weekday() >= 5:
            prev_day = prev_day - dt.timedelta(days=1)
        
        prev_date = prev_day.strftime('%Y-%m-%d')
        
        encoded_symbol = NIFTY_SYMBOL.replace("|", "%7C").replace(" ", "%20")
        url = f"https://api.upstox.com/v2/historical-candle/{encoded_symbol}/day/{prev_date}"
        
        headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and "candles" in data["data"] and len(data["data"]["candles"]) > 0:
                candle = data["data"]["candles"][0]
                
                prev_high = candle[2]
                prev_low = candle[3]
                prev_close = candle[4]
                
                pivot = (prev_high + prev_low + prev_close) / 3
                bc = (prev_high + prev_low) / 2
                tc = (pivot - bc) + pivot
                
                print(f"  ✅ CPR from {prev_date}: TC={tc:.2f} | Pivot={pivot:.2f} | BC={bc:.2f}")
                logger.info(f"CPR calculated from {prev_date}: TC={tc:.2f}, Pivot={pivot:.2f}, BC={bc:.2f}")
                return tc, pivot, bc
        
        print(f"  ⚠️  Could not fetch previous day data")
        logger.warning(f"Could not fetch previous day data for CPR")
        return None, None, None
        
    except Exception as e:
        print(f"  ⚠️  CPR calculation error: {e}")
        logger.error(f"CPR calculation error: {e}")
        return None, None, None


def calculate_cpr_from_session(df):
    if len(df) < 20:
        prev_high = df['high'].max()
        prev_low = df['low'].min()
        prev_close = df.iloc[-1]['close']
    else:
        prev_day_df = df.iloc[:20]
        prev_high = prev_day_df['high'].max()
        prev_low = prev_day_df['low'].min()
        prev_close = prev_day_df.iloc[-1]['close']
    
    pivot = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2
    tc = (pivot - bc) + pivot
    
    print(f"  ✅ Session CPR: TC={tc:.2f} | Pivot={pivot:.2f} | BC={bc:.2f}")
    logger.info(f"Session CPR calculated: TC={tc:.2f}, Pivot={pivot:.2f}, BC={bc:.2f}")
    return tc, pivot, bc


# ==================== INDICATORS ====================

def calculate_indicators(df):
    df["TP"] = (df["high"] + df["low"] + df["close"]) / 3
    df["TPV"] = df["TP"] * df["volume"]
    df["Cumulative_TPV"] = df["TPV"].cumsum()
    df["Cumulative_Volume"] = df["volume"].cumsum()
    df["VWAP"] = df["Cumulative_TPV"] / df["Cumulative_Volume"]
    
    df["Avg_Volume"] = df["volume"].rolling(window=10, min_periods=1).mean()
    df["Volume_Ratio"] = df["volume"] / df["Avg_Volume"]
    
    df["VWAP"] = df["VWAP"].fillna(df["close"])
    df["Avg_Volume"] = df["Avg_Volume"].fillna(df["volume"])
    df["Volume_Ratio"] = df["Volume_Ratio"].fillna(1)
    
    return df


# ==================== STRIKE & PREMIUM ====================

def get_current_premium(instrument_key):
    quote_url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={instrument_key}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    try:
        response = requests.get(quote_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            quote_data = response.json()
            
            if "data" in quote_data:
                for key in quote_data["data"]:
                    data_item = quote_data["data"][key]
                    premium = data_item.get("last_price", 0)
                    if premium == 0:
                        premium = data_item.get("ltp", 0)
                    return premium
        
        return None
    except Exception as e:
        logger.error(f"Premium fetch error: {e}")
        return None


def find_atm_strike_with_live_premium(spot_price, option_type):
    global contracts_cache
    
    try:
        strikes = [c for c in contracts_cache if c.get("instrument_type") == option_type]
        
        if not strikes:
            return None, None, None
        
        atm_contract = min(strikes, key=lambda x: abs(x["strike_price"] - spot_price))
        atm_strike = atm_contract["strike_price"]
        instrument_key = atm_contract["instrument_key"]
        
        premium = get_current_premium(instrument_key)
        
        if premium:
            print(f"  ✅ Premium: {option_type} {atm_strike} = ₹{premium}")
            logger.info(f"ATM Strike found: {option_type} {atm_strike} @ ₹{premium}")
            return atm_strike, premium, instrument_key
        
        return atm_strike, 0, instrument_key
        
    except Exception as e:
        logger.error(f"Strike finding error: {e}")
        return None, None, None


# ==================== SIGNAL LOGIC ====================

def evaluate_signal(spot, tc, bc, vwap, volume_ratio, oi_trend, ce_oi_change, pe_oi_change):
    if oi_trend is None or tc is None or bc is None:
        return None, None
    
    distance_from_tc = abs(spot - tc)
    distance_from_bc = abs(spot - bc)
    
    conditions = {
        "CE": {
            "price_above_tc": spot > tc,
            "sufficient_distance": distance_from_tc >= MIN_CPR_DISTANCE,
            "price_above_vwap": spot > vwap,
            "volume_high": volume_ratio > VOLUME_THRESHOLD,
            "oi_bullish": oi_trend == "Bullish",
            "ce_oi_building": ce_oi_change > 0
        },
        "PE": {
            "price_below_bc": spot < bc,
            "sufficient_distance": distance_from_bc >= MIN_CPR_DISTANCE,
            "price_below_vwap": spot < vwap,
            "volume_high": volume_ratio > VOLUME_THRESHOLD,
            "oi_bearish": oi_trend == "Bearish",
            "pe_oi_building": pe_oi_change > 0
        }
    }
    
    if all(conditions["CE"].values()):
        logger.info("SIGNAL GENERATED: BUY CE - All conditions met")
        return "BUY CE", conditions
    
    if all(conditions["PE"].values()):
        logger.info("SIGNAL GENERATED: BUY PE - All conditions met")
        return "BUY PE", conditions
    
    return None, conditions


def print_signal_evaluation(conditions, tc, bc, spot):
    print(f"\n🔍 SIGNAL EVALUATION (All ✅ required for trade)")
    print("-" * 95)
    
    ce = conditions["CE"]
    pe = conditions["PE"]
    
    ce_result = "🔔 TRIGGER!" if all(ce.values()) else "❌ NO"
    pe_result = "🔔 TRIGGER!" if all(pe.values()) else "❌ NO"
    
    dist_tc = abs(spot - tc)
    dist_bc = abs(spot - bc)
    
    print(f"  CALL: {'✅' if ce['price_above_tc'] else '❌'} Above TC({tc:.2f})  "
          f"{'✅' if ce['sufficient_distance'] else '❌'} Dist:{dist_tc:.1f}  "
          f"{'✅' if ce['price_above_vwap'] else '❌'} VWAP  "
          f"{'✅' if ce['volume_high'] else '❌'} Vol  "
          f"{'✅' if ce['oi_bullish'] else '❌'} OI-Bull  "
          f"{'✅' if ce['ce_oi_building'] else '❌'} CE-Build  →  {ce_result}")
    
    print(f"  PUT:  {'✅' if pe['price_below_bc'] else '❌'} Below BC({bc:.2f})  "
          f"{'✅' if pe['sufficient_distance'] else '❌'} Dist:{dist_bc:.1f}  "
          f"{'✅' if pe['price_below_vwap'] else '❌'} VWAP  "
          f"{'✅' if pe['volume_high'] else '❌'} Vol  "
          f"{'✅' if pe['oi_bearish'] else '❌'} OI-Bear  "
          f"{'✅' if pe['pe_oi_building'] else '❌'} PE-Build  →  {pe_result}")


# ==================== LOGGING ====================

def log_signal(timestamp, signal, strike, premium, spot, vwap, volume_ratio, tc, pivot, bc, oi_trend, exit_reason=None, pnl=None, premium_diff=None):
    with open(csv_file_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, signal, strike, premium,
            round(spot, 2) if spot else 0,
            round(vwap, 2) if vwap else 0,
            round(volume_ratio, 2) if volume_ratio else 0,
            round(tc, 2) if tc else 0,
            round(pivot, 2) if pivot else 0,
            round(bc, 2) if bc else 0,
            oi_trend,
            exit_reason if exit_reason else "",
            round(pnl, 2) if pnl else "",
            round(premium_diff, 2) if premium_diff else ""
        ])
    
    if pnl:
        logger.info(f"Trade logged: {signal} {strike} | Exit: {exit_reason} | P&L: ₹{pnl:.2f}")
    else:
        logger.info(f"Trade logged: {signal} {strike} @ ₹{premium}")


# ==================== MAIN LOOP ====================

def main():
    global last_signal_time, open_position
    
    create_session_folders()
    setup_logging()
    setup_terminal_logging()
    
    print("\n" + "=" * 80)
    print("🚀 NIFTY OPTIONS BOT - FIXED VERSION V4")
    print("=" * 80)
    
    if not validate_token():
        print("\n❌ Invalid token. Exiting...")
        logger.critical("Bot stopped: Invalid token")
        return
    
    print(f"✅ Strategy: CPR + VWAP + Volume + OI (CORRECTED)")
    print(f"✅ Lot Size: {LOT_SIZE}")
    print(f"✅ Take Profit: ₹{TAKE_PROFIT}")
    print(f"✅ Stop Loss: ₹{STOP_LOSS}")
    print(f"✅ Trailing Stop: ₹{TRAILING_STOP}")
    print(f"✅ Volume Threshold: {VOLUME_THRESHOLD}x")
    print(f"✅ Min CPR Distance: {MIN_CPR_DISTANCE} points")
    print("=" * 80 + "\n")
    
    with open(csv_file_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "DateTime","Signal","Strike","Premium","Spot","VWAP","Volume_Ratio",
            "TC","Pivot","BC","OI_Trend","Exit_Reason","PnL","Premium_Diff"
        ])
    
    print("📥 Fetching option contracts...")
    option_instruments = get_option_instruments()
    
    if len(option_instruments) == 0:
        print("\n❌ No option instruments found")
        logger.critical("Bot stopped: No option instruments found")
        return
    
    print(f"\n✅ Ready | Expiry: {current_expiry_date}\n")
    
    tc, pivot, bc = calculate_cpr_from_previous_day()
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            now = dt.datetime.now()
            
            timestamp_full = now.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"\n{'=' * 80}")
            print(f"⏰ [{timestamp_full}] Check #{iteration}")
            print("=" * 80)
            
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                print("⏸  Market not open yet (Opens 9:15 AM)")
                time.sleep(60)
                continue
            
            if (now.hour == 15 and now.minute > 30) or now.hour > 15:
                print("⏸  Market Closed (Closes 3:30 PM)")
                logger.info("Market closed")
                
                if open_position:
                    current_premium = get_current_premium(open_position.instrument_key)
                    if current_premium:
                        pnl, premium_diff = open_position.calculate_pnl(current_premium)
                        
                        log_signal(timestamp_full, f"EXIT {open_position.signal_type}", 
                                 open_position.strike, current_premium, 0, 0, 0, 0, 0, 0,
                                 "", "MARKET CLOSE", pnl, premium_diff)
                        
                        send_discord_alert(
                            "🔔 Position Closed - Market Close",
                            f"**{open_position.signal_type}** | Strike: {open_position.strike}",
                            0xffff00,
                            [{"name": "P&L", "value": f"₹{pnl:.2f}", "inline": False}]
                        )
                        
                        open_position = None
                
                time.sleep(60)
                continue
            
            print("\n📥 Fetching data...")
            
            if open_position:
                print(f"\n💼 POSITION: {open_position.signal_type} {open_position.strike}")
                
                current_premium = get_current_premium(open_position.instrument_key)
                
                if current_premium:
                    pnl, premium_diff = open_position.calculate_pnl(current_premium)
                    
                    print(f"   Entry: ₹{open_position.entry_premium:.2f} | Current: ₹{current_premium:.2f}")
                    print(f"   P&L: ₹{pnl:.2f}")
                    
                    if open_position.trailing_stop_active:
                        print(f"   🎯 Trailing: ₹{open_position.trailing_stop_price:.2f}")
                    
                    should_exit, exit_reason, final_pnl, final_premium_diff = open_position.check_exit(current_premium)
                    
                    if should_exit:
                        log_signal(timestamp_full, f"EXIT {open_position.signal_type}", 
                                 open_position.strike, current_premium, 0, 0, 0, 0, 0, 0,
                                 "", exit_reason, final_pnl, final_premium_diff)
                        
                        color = 0x00ff00 if final_pnl > 0 else 0xff0000
                        send_discord_alert(
                            f"🔔 {exit_reason}",
                            f"**{open_position.signal_type}** | Strike: {open_position.strike}",
                            color,
                            [{"name": "P&L", "value": f"₹{final_pnl:.2f}", "inline": False}]
                        )
                        
                        open_position = None
                        last_signal_time = now
                
                time.sleep(60)
                continue
            
            df = get_live_candles(NIFTY_SYMBOL)
            if df is None or len(df) == 0:
                print("❌ Candles unavailable")
                time.sleep(60)
                continue
            
            df = calculate_indicators(df)
            
            if tc is None or bc is None:
                tc, pivot, bc = calculate_cpr_from_session(df)
            
            latest = df.iloc[-1]
            spot = latest["close"]
            vwap = latest["VWAP"]
            volume_ratio = latest["Volume_Ratio"]
            
            oi_trend, oi_ce, oi_pe, ce_oi_change, pe_oi_change = get_live_oi_from_quotes(option_instruments)
            
            if oi_trend is None:
                print("⏳ OI unavailable")
                time.sleep(60)
                continue
            
            print(f"\n📊 Spot: {spot:.2f} | VWAP: {vwap:.2f} | Vol: {volume_ratio:.2f}x")
            print(f"📊 CPR: TC={tc:.2f} | Pivot={pivot:.2f} | BC={bc:.2f}")
            
            if last_signal_time and (now - last_signal_time).seconds < SIGNAL_COOLDOWN:
                remaining = SIGNAL_COOLDOWN - (now - last_signal_time).seconds
                print(f"⏳ Cooldown: {remaining}s")
                time.sleep(60)
                continue
            
            signal, conditions = evaluate_signal(spot, tc, bc, vwap, volume_ratio, oi_trend, ce_oi_change, pe_oi_change)
            
            print_signal_evaluation(conditions, tc, bc, spot)
            
            if signal:
                option_type = "CE" if signal == "BUY CE" else "PE"
                strike, premium, instrument_key = find_atm_strike_with_live_premium(spot, option_type)
                
                if strike and premium and instrument_key:
                    open_position = Position(signal, strike, premium, instrument_key, timestamp_full)
                    
                    log_signal(timestamp_full, signal, strike, premium, spot, vwap, volume_ratio, tc, pivot, bc, oi_trend)
                    
                    send_discord_alert(
                        f"🚀 NEW SIGNAL - {signal}",
                        f"Strike: {strike} | Lot: {LOT_SIZE}",
                        0x00ff00,
                        [
                            {"name": "Premium", "value": f"₹{premium:.2f}", "inline": True},
                            {"name": "Spot", "value": f"{spot:.2f}", "inline": True},
                            {"name": "Investment", "value": f"₹{premium * LOT_SIZE:.2f}", "inline": False}
                        ]
                    )
                    
                    last_signal_time = now
            else:
                print("\n⏸  NO SIGNAL - Waiting for confluence...")
            
            time.sleep(60)
    
    except KeyboardInterrupt:
        print(f"\n\n⏹  STOPPED | Logs saved in: {session_dir}")
        logger.info("Bot stopped by user (KeyboardInterrupt)")
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {e}")
        logger.critical(f"Bot crashed with exception: {e}", exc_info=True)
        send_discord_alert("❌ Bot Crashed", str(e), 0xff0000)
    finally:
        if hasattr(sys.stdout, 'close'):
            sys.stdout.close()


if __name__ == "__main__":

    main()
