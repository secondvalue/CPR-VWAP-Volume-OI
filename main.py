"""
NIFTY 50 OPTIONS BOT - FIXED VERSION V3.0
✅ All logic issues corrected
✅ No fallback data - only trade with valid data
✅ Proper VWAP daily reset
✅ Fixed trailing stop logic
✅ Correct volume threshold
✅ Enhanced OI analysis
"""

import requests
import pandas as pd
import numpy as np
import datetime as dt
import time
import csv

# ==================== CONFIGURATION ====================
ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI1NUJBOVgiLCJqdGkiOiI2OTA5NzUxMGM5YzYzZDU4ZWViZjgwZDkiLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc2MjIyNzQ3MiwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzYyMjkzNjAwfQ.ZqO_xW_7ShNalpapEdzocZy6sdRlqZdeLPUhTWXDYG8"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1412386951474057299/Jgft_nxzGxcfWOhoLbSWMde-_bwapvqx8l3VQGQwEoR7_8n4b9Q9zN242kMoXsVbLdvG"

NIFTY_SYMBOL = "NSE_INDEX|Nifty 50"
CSV_FILE = "nifty_cpr_trades_fixed.csv"
SIGNAL_COOLDOWN = 300

LOT_SIZE = 75
TAKE_PROFIT = 1500
STOP_LOSS = 2000
TRAILING_STOP = 500
VOLUME_THRESHOLD = 1.5  # Fixed: Requires 1.5x average volume
OI_THRESHOLD = 1.15  # Fixed: Requires 15% difference
MIN_PREMIUM = 20  # Fixed: Minimum premium for entry
MARKET_START_DELAY = 15  # Fixed: Wait 15 min after open (9:30 start)
MIN_VOLUME_LOOKBACK = 10  # Fixed: Minimum candles for volume calc
# =======================================================

last_signal_time = None
current_expiry_date = None
contracts_cache = []
open_position = None
cpr_validated = False
daily_vwap_reset_time = None

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
        self.highest_premium = entry_premium
        self.trailing_stop_active = False
        self.trailing_stop_pnl = None
    
    def calculate_pnl(self, current_premium):
        """Calculate current P&L"""
        premium_diff = current_premium - self.entry_premium
        pnl = premium_diff * self.lot_size
        
        # Track highest PnL and premium
        if pnl > self.highest_pnl:
            self.highest_pnl = pnl
            self.highest_premium = current_premium
        
        return pnl, premium_diff
    
    def check_exit(self, current_premium):
        """
        FIXED: Trailing stop tracks from highest PnL, not current premium
        """
        pnl, premium_diff = self.calculate_pnl(current_premium)
        
        # Stop Loss check
        if pnl <= -STOP_LOSS:
            return True, f"STOP LOSS (Loss: ₹{abs(pnl):.2f})", pnl, premium_diff
        
        # Take Profit reached - activate trailing stop
        if self.highest_pnl >= TAKE_PROFIT:
            if not self.trailing_stop_active:
                self.trailing_stop_active = True
                # FIXED: Trailing stop tracks from highest PnL
                self.trailing_stop_pnl = self.highest_pnl - TRAILING_STOP
                print(f"  🎯 Take Profit reached! Trailing from ₹{self.highest_pnl:.2f}")
                print(f"     Trailing Stop PnL: ₹{self.trailing_stop_pnl:.2f}")
        
        # Check trailing stop
        if self.trailing_stop_active:
            if pnl <= self.trailing_stop_pnl:
                return True, f"TRAILING STOP (Profit: ₹{pnl:.2f})", pnl, premium_diff
            
            # Update trailing stop if PnL increased
            new_trailing_stop = self.highest_pnl - TRAILING_STOP
            if new_trailing_stop > self.trailing_stop_pnl:
                self.trailing_stop_pnl = new_trailing_stop
                print(f"  📈 Trailing updated: Highest ₹{self.highest_pnl:.2f} → Stop ₹{self.trailing_stop_pnl:.2f}")
        
        return False, None, pnl, premium_diff

# ==================== DISCORD ====================

def send_discord_alert(title, description, color=0x00ff00, fields=None):
    """Send Discord notification"""
    embed = {
        "title": title,
        "description": description,
        "color": color,
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "footer": {"text": f"Fixed Bot V3.0 | Lot: {LOT_SIZE}"}
    }
    
    if fields:
        embed["fields"] = fields
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"embeds": [embed]}, timeout=10)
        if response.status_code == 204:
            print("  ✅ Discord alert sent")
    except Exception as e:
        print(f"  ⚠️  Discord error: {e}")

# ==================== TOKEN VALIDATION ====================

def validate_token():
    """Validate Upstox access token"""
    url = "https://api.upstox.com/v2/user/profile"
    headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            profile = response.json()
            print(f"✅ Token Valid | User: {profile.get('data', {}).get('user_name', 'N/A')}")
            return True
        else:
            print(f"❌ TOKEN EXPIRED")
            return False
    except Exception as e:
        print(f"❌ Token validation error: {e}")
        return False

# ==================== GET EXPIRY DATE ====================

def get_next_tuesday_expiry():
    """Get next Tuesday expiry date"""
    today = dt.datetime.now()
    
    if today.weekday() == 1:  # Tuesday
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
    return expiry_date

# ==================== GET OPTION INSTRUMENTS ====================

def get_option_instruments():
    """
    FIXED: Fetch option instruments with enhanced error handling
    Tries multiple approaches to get valid contracts
    """
    global current_expiry_date, contracts_cache
    
    current_expiry_date = get_next_tuesday_expiry()
    
    encoded_symbol = "NSE_INDEX%7CNifty%2050"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    
    print(f"  🔍 Attempting to fetch contracts for expiry: {current_expiry_date}")
    
    try:
        # Approach 1: Try with specific expiry date
        url = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_symbol}&expiry_date={current_expiry_date}"
        
        print(f"  📡 Calling: /option/contract with expiry={current_expiry_date}")
        response = requests.get(url, headers=headers, timeout=10)
        
        print(f"  📊 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"  📦 Response Keys: {list(data.keys())}")
            
            if "data" in data and data["data"] is not None and len(data["data"]) > 0:
                contracts_cache = data["data"]
                print(f"  ✅ Got {len(contracts_cache)} contracts for {current_expiry_date}")
            else:
                print(f"  ⚠️  No contracts for {current_expiry_date}, trying without expiry filter...")
                
                # Approach 2: Get all contracts and filter
                url_all = f"https://api.upstox.com/v2/option/contract?instrument_key={encoded_symbol}"
                print(f"  📡 Calling: /option/contract without expiry filter")
                
                response2 = requests.get(url_all, headers=headers, timeout=10)
                print(f"  📊 Response Status: {response2.status_code}")
                
                if response2.status_code == 200:
                    data2 = response2.json()
                    all_contracts = data2.get("data", [])
                    
                    if len(all_contracts) > 0:
                        print(f"  📦 Total contracts available: {len(all_contracts)}")
                        
                        # Get unique expiry dates
                        expiries = sorted(set([c["expiry"] for c in all_contracts]))
                        print(f"  📅 Available expiries: {expiries[:5]}")  # Show first 5
                        
                        if len(expiries) > 0:
                            nearest_expiry = expiries[0]
                            current_expiry_date = nearest_expiry
                            contracts_cache = [c for c in all_contracts if c["expiry"] == nearest_expiry]
                            print(f"  ✅ Using nearest expiry: {nearest_expiry} ({len(contracts_cache)} contracts)")
                        else:
                            print(f"  ❌ No expiries found in contracts")
                            return []
                    else:
                        print(f"  ❌ No contracts returned from API")
                        return []
                else:
                    print(f"  ❌ API call failed: {response2.status_code}")
                    if response2.status_code == 401:
                        print(f"  ❌ Token expired or invalid")
                    return []
        else:
            print(f"  ❌ API call failed: {response.status_code}")
            if response.status_code == 401:
                print(f"  ❌ Token expired or invalid")
            elif response.status_code == 400:
                print(f"  ❌ Bad request - check instrument key or date format")
            return []
        
        # Validate contracts_cache
        if len(contracts_cache) == 0:
            print(f"  ❌ No contracts in cache after processing")
            return []
        
        # Get spot price
        spot_price = get_spot_price()
        
        if spot_price is None:
            print(f"  ⚠️  Could not fetch spot price, using all contracts")
            filtered = [c["instrument_key"] for c in contracts_cache]
            print(f"  ✅ Selected {len(filtered)} contracts (all available)")
            return filtered
        
        print(f"  ✅ Nifty Spot: {spot_price:.2f}")
        
        # Filter contracts within ±500 points
        filtered = [c["instrument_key"] for c in contracts_cache 
                   if abs(c["strike_price"] - spot_price) <= 500]
        
        print(f"  ✅ Selected {len(filtered)} contracts within ±500 points of spot")
        
        if len(filtered) == 0:
            print(f"  ⚠️  No contracts within ±500 points, expanding range to ±1000")
            filtered = [c["instrument_key"] for c in contracts_cache 
                       if abs(c["strike_price"] - spot_price) <= 1000]
            print(f"  ✅ Selected {len(filtered)} contracts within ±1000 points")
        
        return filtered
        
    except Exception as e:
        print(f"  ❌ Exception in get_option_instruments: {e}")
        import traceback
        print(f"  ❌ Traceback: {traceback.format_exc()}")
        return []

def get_spot_price():
    """Get current Nifty spot price with enhanced error handling"""
    try:
        encoded_symbol = NIFTY_SYMBOL.replace("|", "%7C").replace(" ", "%20")
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={encoded_symbol}"
        headers = {"accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data and NIFTY_SYMBOL in data["data"]:
                spot = data["data"][NIFTY_SYMBOL]["last_price"]
                if spot and spot > 0:
                    return spot
                else:
                    print(f"  ⚠️  Spot price is zero or invalid")
            else:
                print(f"  ⚠️  Spot data not found in response")
                print(f"  📦 Response keys: {list(data.keys())}")
        else:
            print(f"  ⚠️  Spot price API failed: {response.status_code}")
        
        return None
    except Exception as e:
        print(f"  ⚠️  Spot price error: {e}")
        return None

# ==================== GET LIVE OI ====================

def get_live_oi_from_quotes(instrument_keys):
    """
    FIXED: Fetch live OI with proper error handling and validation
    No fallback - returns None if data unavailable
    """
    if not instrument_keys:
        print("  ⚠️  No instrument keys provided")
        return None, 0, 0
    
    ce_oi_total = 0
    pe_oi_total = 0
    ce_count = 0
    pe_count = 0
    
    # FIXED: Reduce batch size to 50 to avoid URL length issues
    for i in range(0, len(instrument_keys), 50):
        batch = instrument_keys[i:i+50]
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
                    if "oi" in quote_data and quote_data["oi"] > 0:
                        oi_value = quote_data["oi"]
                        
                        if "CE" in instrument_key:
                            ce_oi_total += oi_value
                            ce_count += 1
                        elif "PE" in instrument_key:
                            pe_oi_total += oi_value
                            pe_count += 1
        
        except Exception as e:
            print(f"  ⚠️  OI batch error: {e}")
            continue
    
    # FIXED: Validate we got meaningful data
    if ce_count == 0 or pe_count == 0:
        print("  ❌ OI data incomplete or unavailable")
        return None, 0, 0
    
    if ce_oi_total == 0 and pe_oi_total == 0:
        print("  ❌ OI totals are zero")
        return None, 0, 0
    
    # FIXED: Use 15% threshold instead of 5%
    if pe_oi_total > ce_oi_total * OI_THRESHOLD:
        trend = "Bullish"
    elif ce_oi_total > pe_oi_total * OI_THRESHOLD:
        trend = "Bearish"
    else:
        trend = "Neutral"
    
    print(f"  ✅ Live OI: CE={ce_oi_total:,.0f} ({ce_count}) | PE={pe_oi_total:,.0f} ({pe_count}) | {trend}")
    
    return trend, ce_oi_total, pe_oi_total

# ==================== LIVE DATA FETCHING ====================

def get_live_candles(symbol):
    """Fetch and aggregate 1-min candles to 5-min"""
    encoded_symbol = symbol.replace("|", "%7C").replace(" ", "%20")
    url = f"https://api.upstox.com/v2/historical-candle/intraday/{encoded_symbol}/1minute"
    
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            print(f"  ❌ Candle fetch failed: {response.status_code}")
            return None
        
        data = response.json()
        
        if "data" not in data or "candles" not in data["data"]:
            print("  ❌ No candle data in response")
            return None
        
        candles = data["data"]["candles"]
        
        if len(candles) == 0:
            print("  ❌ Empty candle list")
            return None
        
        df = pd.DataFrame(candles, columns=["time","open","high","low","close","volume","oi"])
        df["time"] = pd.to_datetime(df["time"])
        
        # FIXED: Filter out zero volume candles instead of replacing
        df = df[df["volume"] > 0].copy()
        
        if len(df) == 0:
            print("  ❌ No valid volume candles")
            return None
        
        df = df.sort_values("time").reset_index(drop=True)
        
        # Aggregate to 5-min candles
        df.set_index("time", inplace=True)
        df_5min = df.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        df_5min.reset_index(inplace=True)
        
        # FIXED: Filter today's candles only
        today = dt.datetime.now().date()
        df_5min = df_5min[df_5min['time'].dt.date == today].copy()
        
        if len(df_5min) == 0:
            print("  ❌ No today's candles after filtering")
            return None
        
        print(f"  ✅ Candles: {len(candles)} 1-min → {len(df_5min)} 5-min (today)")
        return df_5min
        
    except Exception as e:
        print(f"  ❌ Candle error: {e}")
        return None

# ==================== CPR CALCULATION ====================

def calculate_cpr_from_previous_day():
    """
    FIXED: Calculate CPR using last completed trading day's data
    No fallback - returns None if data unavailable
    """
    try:
        today = dt.datetime.now()
        
        # FIXED: Always get last completed trading day
        if today.weekday() == 0:  # Monday
            prev_day = today - dt.timedelta(days=3)  # Friday
        elif today.weekday() == 6:  # Sunday
            prev_day = today - dt.timedelta(days=2)  # Friday
        else:
            prev_day = today - dt.timedelta(days=1)
            while prev_day.weekday() >= 5:  # Skip weekend
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
                return tc, pivot, bc, prev_date
        
        print(f"  ❌ Could not fetch previous day data for {prev_date}")
        return None, None, None, None
        
    except Exception as e:
        print(f"  ❌ CPR calculation error: {e}")
        return None, None, None, None

# ==================== INDICATORS ====================

def calculate_indicators(df):
    """
    FIXED: Calculate VWAP with daily reset and proper volume analysis
    """
    if len(df) == 0:
        return None
    
    # FIXED: Ensure we're working with today's data only
    today = dt.datetime.now().date()
    df = df[df['time'].dt.date == today].copy()
    
    if len(df) == 0:
        return None
    
    # VWAP Calculation - Daily cumulative
    df["TP"] = (df["high"] + df["low"] + df["close"]) / 3
    df["TPV"] = df["TP"] * df["volume"]
    df["Cumulative_TPV"] = df["TPV"].cumsum()
    df["Cumulative_Volume"] = df["volume"].cumsum()
    df["VWAP"] = df["Cumulative_TPV"] / df["Cumulative_Volume"]
    
    # FIXED: Volume analysis - use rolling average excluding current candle
    if len(df) >= MIN_VOLUME_LOOKBACK:
        # Calculate rolling average of previous candles (excluding current)
        df["Avg_Volume"] = df["volume"].shift(1).rolling(window=MIN_VOLUME_LOOKBACK, min_periods=5).mean()
        df["Volume_Ratio"] = df["volume"] / df["Avg_Volume"]
    else:
        # Not enough data - mark as invalid
        df["Avg_Volume"] = np.nan
        df["Volume_Ratio"] = np.nan
    
    # Fill NaN values
    df["VWAP"] = df["VWAP"].fillna(df["close"])
    
    return df

# ==================== STRIKE & PREMIUM ====================

def get_current_premium(instrument_key):
    """Get current option premium"""
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
    except:
        return None

def find_atm_strike_with_live_premium(spot_price, option_type):
    """
    FIXED: Find ATM strike with minimum premium filter
    """
    global contracts_cache
    
    try:
        strikes = [c for c in contracts_cache if c.get("instrument_type") == option_type]
        
        if not strikes:
            print(f"  ❌ No {option_type} contracts found")
            return None, None, None
        
        atm_contract = min(strikes, key=lambda x: abs(x["strike_price"] - spot_price))
        atm_strike = atm_contract["strike_price"]
        instrument_key = atm_contract["instrument_key"]
        
        premium = get_current_premium(instrument_key)
        
        if premium is None:
            print(f"  ❌ Could not fetch premium for {option_type} {atm_strike}")
            return None, None, None
        
        # FIXED: Check minimum premium requirement
        if premium < MIN_PREMIUM:
            print(f"  ❌ Premium too low: ₹{premium} < ₹{MIN_PREMIUM} (minimum)")
            return None, None, None
        
        print(f"  ✅ Premium: {option_type} {atm_strike} = ₹{premium:.2f}")
        return atm_strike, premium, instrument_key
        
    except Exception as e:
        print(f"  ❌ Strike selection error: {e}")
        return None, None, None

# ==================== SIGNAL LOGIC ====================

def evaluate_signal(spot, tc, bc, vwap, volume_ratio, oi_trend):
    """
    FIXED: Evaluate signals with proper thresholds and validation
    Returns None if any data is invalid
    """
    # FIXED: Strict validation - no trading with incomplete data
    if oi_trend is None or oi_trend == "Neutral":
        return None, None
    
    if tc is None or bc is None or vwap is None:
        return None, None
    
    if pd.isna(volume_ratio):
        return None, None
    
    conditions = {
        "CE": {
            "price_above_tc": spot > tc,
            "price_above_vwap": spot > vwap,
            "volume_high": volume_ratio > VOLUME_THRESHOLD,  # FIXED: Now 1.5x
            "oi_bullish": oi_trend == "Bullish"
        },
        "PE": {
            "price_below_bc": spot < bc,
            "price_below_vwap": spot < vwap,
            "volume_high": volume_ratio > VOLUME_THRESHOLD,  # FIXED: Now 1.5x
            "oi_bearish": oi_trend == "Bearish"
        }
    }
    
    if all(conditions["CE"].values()):
        return "BUY CE", conditions
    
    if all(conditions["PE"].values()):
        return "BUY PE", conditions
    
    return None, conditions

def print_signal_evaluation(conditions, tc, bc, vwap, volume_ratio):
    """Print detailed signal evaluation"""
    print(f"\n🔍 SIGNAL EVALUATION (All ✅ required for trade)")
    print("-" * 90)
    
    ce = conditions["CE"]
    pe = conditions["PE"]
    
    ce_result = "🔔 TRIGGER!" if all(ce.values()) else "❌ NO"
    pe_result = "🔔 TRIGGER!" if all(pe.values()) else "❌ NO"
    
    vol_display = f"{volume_ratio:.2f}x" if not pd.isna(volume_ratio) else "N/A"
    
    print(f"  CALL: {'✅' if ce['price_above_tc'] else '❌'} Above TC({tc:.2f})  "
          f"{'✅' if ce['price_above_vwap'] else '❌'} Above VWAP({vwap:.2f})  "
          f"{'✅' if ce['volume_high'] else '❌'} Vol>{VOLUME_THRESHOLD}x({vol_display})  "
          f"{'✅' if ce['oi_bullish'] else '❌'} OI-Bull  →  {ce_result}")
    
    print(f"  PUT:  {'✅' if pe['price_below_bc'] else '❌'} Below BC({bc:.2f})  "
          f"{'✅' if pe['price_below_vwap'] else '❌'} Below VWAP({vwap:.2f})  "
          f"{'✅' if pe['volume_high'] else '❌'} Vol>{VOLUME_THRESHOLD}x({vol_display})  "
          f"{'✅' if pe['oi_bearish'] else '❌'} OI-Bear  →  {pe_result}")

# ==================== LOGGING ====================

def log_signal(timestamp, signal, strike, premium, spot, vwap, volume_ratio, tc, pivot, bc, oi_trend, cpr_date, exit_reason=None, pnl=None, premium_diff=None):
    """Log trade to CSV with validation"""
    with open(CSV_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, signal, strike, premium,
            round(spot, 2) if spot else 0,
            round(vwap, 2) if vwap else 0,
            round(volume_ratio, 2) if not pd.isna(volume_ratio) else 0,
            round(tc, 2) if tc else 0,
            round(pivot, 2) if pivot else 0,
            round(bc, 2) if bc else 0,
            oi_trend if oi_trend else "",
            cpr_date if cpr_date else "",
            exit_reason if exit_reason else "",
            round(pnl, 2) if pnl else "",
            round(premium_diff, 2) if premium_diff else ""
        ])

# ==================== MAIN LOOP ====================

def main():
    global last_signal_time, open_position, cpr_validated, daily_vwap_reset_time
    
    print("\n" + "=" * 90)
    print("🚀 NIFTY OPTIONS BOT - FIXED VERSION V3.0")
    print("=" * 90)
    
    if not validate_token():
        print("\n❌ Invalid token. Exiting...")
        return
    
    print(f"✅ Strategy: CPR + VWAP + Volume + OI")
    print(f"✅ Lot Size: {LOT_SIZE}")
    print(f"✅ Take Profit: ₹{TAKE_PROFIT} | Stop Loss: ₹{STOP_LOSS} | Trailing: ₹{TRAILING_STOP}")
    print(f"✅ Volume Threshold: {VOLUME_THRESHOLD}x (FIXED)")
    print(f"✅ OI Threshold: {OI_THRESHOLD}x ({int((OI_THRESHOLD-1)*100)}% difference)")
    print(f"✅ Min Premium: ₹{MIN_PREMIUM}")
    print(f"✅ Market Start Delay: {MARKET_START_DELAY} min (Trading from 9:30 AM)")
    print("=" * 90 + "\n")
    
    # Initialize CSV
    with open(CSV_FILE, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "DateTime","Signal","Strike","Premium","Spot","VWAP","Volume_Ratio",
            "TC","Pivot","BC","OI_Trend","CPR_Date","Exit_Reason","PnL","Premium_Diff"
        ])
    
    print("📥 Fetching option contracts...")
    print("=" * 90)
    option_instruments = get_option_instruments()
    print("=" * 90)
    
    if len(option_instruments) == 0:
        print("\n❌ CRITICAL: No option instruments found")
        print("\n🔍 Diagnostic Information:")
        print(f"  - Token Status: Valid")
        print(f"  - Calculated Expiry: {current_expiry_date}")
        print(f"  - Contracts in Cache: {len(contracts_cache)}")
        
        # Try to get spot price
        spot = get_spot_price()
        if spot:
            print(f"  - Spot Price: {spot:.2f}")
        else:
            print(f"  - Spot Price: Unable to fetch")
        
        print("\n💡 Possible Issues:")
        print("  1. Market is closed (Options data not available)")
        print("  2. Expiry date format issue")
        print("  3. API endpoint changed")
        print("  4. Rate limiting")
        
        print("\n🔧 Suggested Actions:")
        print("  1. Check if market is open (Mon-Fri 9:15 AM - 3:30 PM)")
        print("  2. Try running during market hours")
        print("  3. Verify token hasn't expired")
        print("  4. Check Upstox API status")
        
        return
    
    print(f"\n✅ Ready | Expiry: {current_expiry_date} | Contracts: {len(option_instruments)}\n")
    
    # FIXED: Get CPR - will NOT use fallback
    tc, pivot, bc, cpr_date = calculate_cpr_from_previous_day()
    
    if tc is None:
        print("\n❌ CRITICAL: Could not fetch valid CPR data")
        print("❌ Bot will NOT trade without valid CPR")
        print("❌ Please retry when market data is available\n")
        return
    
    cpr_validated = True
    print(f"✅ CPR Validated from {cpr_date}\n")
    
    iteration = 0
    
    try:
        while True:
            iteration += 1
            now = dt.datetime.now()
            
            timestamp_full = now.strftime('%Y-%m-%d %H:%M:%S')
            
            # FIXED: Check if new day - reset VWAP tracking
            if daily_vwap_reset_time is None or now.date() > daily_vwap_reset_time:
                daily_vwap_reset_time = now.date()
                print(f"\n🔄 New trading day - VWAP will reset with fresh data")
            
            print(f"\n{'=' * 90}")
            print(f"⏰ [{timestamp_full}] Check #{iteration}")
            print("=" * 90)
            
            # FIXED: Market hours check
            if now.hour < 9 or (now.hour == 9 and now.minute < 15):
                print("⏸  Market not open yet (Opens 9:15 AM)")
                time.sleep(60)
                continue
            
            # FIXED: Trading start delay
            if now.hour == 9 and now.minute < (15 + MARKET_START_DELAY):
                remaining = (15 + MARKET_START_DELAY) - now.minute
                print(f"⏸  Waiting for market to stabilize ({remaining} min until 9:30 AM)")
                time.sleep(60)
                continue
            
            if (now.hour == 15 and now.minute > 30) or now.hour > 15:
                print("⏸  Market Closed (Closes 3:30 PM)")
                
                if open_position:
                    current_premium = get_current_premium(open_position.instrument_key)
                    if current_premium:
                        pnl, premium_diff = open_position.calculate_pnl(current_premium)
                        
                        log_signal(timestamp_full, f"EXIT {open_position.signal_type}", 
                                 open_position.strike, current_premium, 0, 0, 0, 0, 0, 0,
                                 "", "", "MARKET CLOSE", pnl, premium_diff)
                        
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
            
            # POSITION MONITORING
            if open_position:
                print(f"\n💼 POSITION: {open_position.signal_type} {open_position.strike}")
                
                current_premium = get_current_premium(open_position.instrument_key)
                
                if current_premium:
                    pnl, premium_diff = open_position.calculate_pnl(current_premium)
                    
                    print(f"   Entry: ₹{open_position.entry_premium:.2f} | Current: ₹{current_premium:.2f}")
                    print(f"   Current P&L: ₹{pnl:.2f} | Highest: ₹{open_position.highest_pnl:.2f}")
                    
                    if open_position.trailing_stop_active:
                        print(f"   🎯 Trailing Active | Stop PnL: ₹{open_position.trailing_stop_pnl:.2f}")
                    
                    should_exit, exit_reason, final_pnl, final_premium_diff = open_position.check_exit(current_premium)
                    
                    if should_exit:
                        log_signal(timestamp_full, f"EXIT {open_position.signal_type}", 
                                 open_position.strike, current_premium, 0, 0, 0, 0, 0, 0,
                                 "", "", exit_reason, final_pnl, final_premium_diff)
                        
                        color = 0x00ff00 if final_pnl > 0 else 0xff0000
                        send_discord_alert(
                            f"🔔 {exit_reason}",
                            f"**{open_position.signal_type}** | Strike: {open_position.strike}",
                            color,
                            [{"name": "P&L", "value": f"₹{final_pnl:.2f}", "inline": False}]
                        )
                        
                        open_position = None
                        # FIXED: Don't set cooldown on exit, only on entry
                
                time.sleep(60)
                continue
            
            # FETCH CANDLES
            df = get_live_candles(NIFTY_SYMBOL)
            if df is None or len(df) == 0:
                print("❌ Candles unavailable")
                time.sleep(60)
                continue
            
            # CALCULATE INDICATORS
            df = calculate_indicators(df)
            if df is None or len(df) == 0:
                print("❌ Indicator calculation failed")
                time.sleep(60)
                continue
            
            latest = df.iloc[-1]
            spot = latest["close"]
            vwap = latest["VWAP"]
            volume_ratio = latest["Volume_Ratio"]
            
            # FIXED: Validate volume ratio
            if pd.isna(volume_ratio):
                print(f"⏳ Volume analysis pending (need {MIN_VOLUME_LOOKBACK} candles)")
                time.sleep(60)
                continue
            
            # FETCH OI
            oi_trend, oi_ce, oi_pe = get_live_oi_from_quotes(option_instruments)
            
            if oi_trend is None:
                print("❌ OI data unavailable - cannot trade without OI confirmation")
                time.sleep(60)
                continue
            
            print(f"\n📊 Spot: {spot:.2f} | VWAP: {vwap:.2f} | Vol: {volume_ratio:.2f}x")
            print(f"📊 CPR: TC={tc:.2f} | Pivot={pivot:.2f} | BC={bc:.2f} (from {cpr_date})")
            
            # COOLDOWN CHECK
            if last_signal_time and (now - last_signal_time).seconds < SIGNAL_COOLDOWN:
                remaining = SIGNAL_COOLDOWN - (now - last_signal_time).seconds
                print(f"⏳ Cooldown: {remaining}s")
                time.sleep(60)
                continue
            
            # EVALUATE SIGNAL
            signal, conditions = evaluate_signal(spot, tc, bc, vwap, volume_ratio, oi_trend)
            
            print_signal_evaluation(conditions, tc, bc, vwap, volume_ratio)
            
            if signal:
                option_type = "CE" if signal == "BUY CE" else "PE"
                strike, premium, instrument_key = find_atm_strike_with_live_premium(spot, option_type)
                
                if strike and premium and instrument_key:
                    open_position = Position(signal, strike, premium, instrument_key, timestamp_full)
                    
                    log_signal(timestamp_full, signal, strike, premium, spot, vwap, 
                             volume_ratio, tc, pivot, bc, oi_trend, cpr_date)
                    
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
                    
                    # FIXED: Set cooldown only on entry
                    last_signal_time = now
                else:
                    print("\n❌ Strike selection failed - signal cancelled")
            else:
                print("\n⏸  NO SIGNAL - Waiting for confluence...")
            
            time.sleep(60)
    
    except KeyboardInterrupt:
        print(f"\n\n⏹  STOPPED | Trades: {CSV_FILE}")

if __name__ == "__main__":
    main()