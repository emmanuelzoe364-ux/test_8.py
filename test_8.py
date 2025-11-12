import os
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging 

# Configure basic logging for simulated alerts
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration & Setup ---
st.set_page_config(page_title="BTC/ETH Combined Index Signal Tracker", layout="wide")

# Initialize session state for persistent signal tracking (for simulating alerts)
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = pd.Timestamp.min.tz_localize(None)

# -----------------------
# Helper function for external alert (SIMULATION)
# -----------------------

def send_external_alert(signal_type: str, message: str, email: str, phone: str, conviction_score: int, lead_asset: str):
    """
    Simulates sending an external alert via Email/SMS API.
    Includes the Conviction Score and Lead Asset for immediate quality assessment.
    """
    if email or phone:
        logging.info(f"*** EXTERNAL ALERT SENT (Simulated) ***")
        if email:
            logging.info(f"EMAIL To: {email}")
        if phone:
            logging.info(f"SMS To: {phone}")
        logging.info(f"CONTENT (Score {conviction_score}, Lead: {lead_asset}): {message.replace('\n', ' | ')}")
    else:
        logging.info("External alert skipped: No email or phone recipient configured.")


# -----------------------
# Helper functions for calculations
# -----------------------

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Calculates the Exponential Moving Average-based Relative Strength Index (EMA-RSI).
    """
    delta = series.diff().dropna()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    
    # Use EWM (Exponential Moving Average) for smoothing (EMA-RSI)
    ma_up = up.ewm(span=length, adjust=False).mean()
    ma_down = down.ewm(span=length, adjust=False).mean()
    
    # Avoid division by zero
    rs = ma_up / ma_down.replace(0, 1e-10) 
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

# --- ADX Calculation ---
def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculates the Average Directional Index (ADX) along with +DI and -DI.
    Uses true OHLC data for the combined index's True Range calculation.
    """
    
    # True Range components
    tr1 = high - low
    tr2 = np.abs(high - close.shift(1))
    tr3 = np.abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    hd = high.diff()
    ld = low.diff()
    
    # +DM and -DM
    pdm = np.where((hd > 0) & (hd > ld), hd, 0)
    ndm = np.where((ld < 0) & (ld < hd), -ld, 0)
    
    # Convert numpy arrays back to pandas Series with the correct index
    pdm = pd.Series(pdm, index=high.index)
    ndm = pd.Series(ndm, index=high.index)

    # EWM Smoothing for TR, +DM, -DM
    tr_smooth = tr.ewm(span=length, adjust=False).mean()
    pdm_smooth = pdm.ewm(span=length, adjust=False).mean()
    ndm_smooth = ndm.ewm(span=length, adjust=False).mean()

    # Directional Index (+DI and -DI)
    # Avoid division by zero
    tr_smooth_safe = tr_smooth.replace(0, 1e-10) 
    pdi = (pdm_smooth / tr_smooth_safe) * 100
    ndi = (ndm_smooth / tr_smooth_safe) * 100

    # Directional Movement Index (DX)
    di_sum_safe = (pdi + ndi).replace(0, 1e-10)
    dx = np.abs(pdi - ndi) / di_sum_safe * 100
    
    # Average Directional Index (ADX)
    adx = dx.ewm(span=length, adjust=False).mean()
    
    # Return ADX, +DI, -DI
    return adx, pdi, ndi

def get_asset_stacking_score(ema_f: float, ema_m: float, ema_s: float) -> tuple[int, str]:
    """
    Calculates the EMA stacking score for a single asset (BTC or ETH).
    Score: +2 (Strong Bullish Stack) to -2 (Strong Bearish Stack).
    
    Returns: (score, description)
    """
    score = 0
    desc = "Mixed/Consolidation"
    
    # Perfect Bullish Stack: S > M > F
    if ema_s > ema_m and ema_m > ema_f:
        score = 2
        desc = "Perfect Bullish Stack"
    # Perfect Bearish Stack: S < M < F
    elif ema_s < ema_m and ema_m < ema_f:
        score = -2
        desc = "Perfect Bearish Stack"
    # Developing Bullish Bias (S>M or M>F, but not perfect stack)
    elif (ema_s > ema_m and ema_m < ema_f) or (ema_s < ema_m and ema_m > ema_f):
        score = 1
        desc = "Developing Bullish Bias (Partial Alignment)"
    # Developing Bearish Bias (S<M or M<F, but not perfect stack)
    elif (ema_s < ema_m and ema_m > ema_f) or (ema_s > ema_m and ema_m < ema_f):
        score = -1
        desc = "Developing Bearish Bias (Partial Alignment)"
        
    return score, desc


# -----------------------
# Sidebar / user inputs
# -----------------------
st.sidebar.header("Index & Indicator Settings")
# Fixed Tickers: BTC-USD and ETH-USD
TICKERS = ["BTC-USD", "ETH-USD"]

# Intraday limit enforcement (FIX from earlier conversation)
MAX_INTRADAY_DAYS = 60
period_days = st.sidebar.number_input("Fetch period (days)", min_value=7, max_value=365, value=7) 
interval = st.sidebar.selectbox("Interval", options=["15m","30m","1h","1d"], index=2)

if interval in ["15m", "30m", "1h"] and period_days > MAX_INTRADAY_DAYS:
    period_days = MAX_INTRADAY_DAYS
    st.sidebar.warning(f"Intraday period capped at {MAX_INTRADAY_DAYS} days.")

st.sidebar.markdown("---")
rsi_length = st.sidebar.number_input("RSI length", min_value=7, max_value=30, value=14)
# --- NEW ADX INPUT ---
adx_length = st.sidebar.number_input("ADX length", min_value=7, max_value=30, value=14)
# --- END NEW ADX INPUT ---
index_ema_span = st.sidebar.number_input("Cumulative Index EMA Span (Smoother)", min_value=1, max_value=10, value=5)
st.sidebar.markdown("_The EMA span above is for smoothing the index before RSI/ADX calculation._")
st.sidebar.markdown("---")

# EMA inputs with new EMA 72
ema_short = st.sidebar.number_input("EMA Short (14)", min_value=5, max_value=25, value=14)
ema_long = st.sidebar.number_input("EMA Medium (30)", min_value=26, max_value=50, value=30)
ema_very_long = st.sidebar.number_input("EMA Very Long (72)", min_value=51, max_value=365, value=72) # New 72 EMA

st.sidebar.markdown("---")
min_bars_after_cycle = st.sidebar.number_input("Max bars to look for re-alignment (0 = unlimited)", min_value=0, max_value=9999, value=0)

# Volume Filtering
volume_length = st.sidebar.number_input("Volume MA Length", min_value=1, max_value=50, value=14)
enable_volume_filter = st.sidebar.checkbox("Require Volume Confirmation", value=False)


st.sidebar.header("External Notification Settings")
recipient_email = st.sidebar.text_input("Recipient Email (for simulation)", value="")
recipient_phone = st.sidebar.text_input("Recipient Phone (for simulation, e.g., +15551234)", value="")
st.sidebar.markdown("_The external alerts are simulated via logging. To make them real, you'd integrate a service like Twilio or SendGrid._")
st.markdown("---")

# --- Main Title ---
st.title(f"🔥 BTC/ETH Combined Index (50/50) Tracker")
st.subheader(f"Triple-EMA Confirmation: {ema_short} > {ema_long} > {ema_very_long} | ADX({adx_length})") # Updated Title

st.sidebar.markdown("RSI cycle rules: **rising** = cross up 29 → later cross up 71. **falling** = cross down 71 → later cross down 29.")
st.sidebar.markdown("Signals fire only after a completed cycle + Normalized Price dip/spike + **Combined Index Triple EMA Alignment**.")


# -----------------------
# 🚀 Data Fetching and Processing (Optimized with st.cache_data)
# -----------------------

@st.cache_data(ttl=timedelta(seconds=300))
def fetch_and_process_data(tickers: list, period_days: int, interval: str, index_ema_span: int, ema_short: int, ema_long: int, ema_very_long: int, adx_length: int): # Added adx_length
    """
    Fetches data for multiple tickers, creates the normalized combined index, 
    and calculates all required indicators, including individual asset EMAs.
    """
    
    status_text = st.empty()
    status_text.info(f"Fetching {', '.join(tickers)} {interval} data for {period_days} days. Cache TTL: 300 seconds.")
    
    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=period_days)
    
    try:
        raw = yf.download(tickers, start=start_date, end=end_date, interval=interval, progress=False)
    except Exception as e:
        status_text.error(f"Failed during data fetch: {e}. Check ticker symbols or try reducing period.")
        return pd.DataFrame(), pd.DataFrame() 

    if raw.empty:
        status_text.error("Fetched data is empty. Try a different interval or period.")
        return pd.DataFrame(), pd.DataFrame()
        
    # --- 1. CLEANING AND NORMALIZATION ---
    df = pd.DataFrame()
    # Initialize ohlc_for_adx as an empty DataFrame 
    ohlc_for_adx = pd.DataFrame() 
    total_volume = 0
    
    for ticker in tickers:
        ticker_base = ticker.split("-")[0].lower()
        
        # Handle single vs. multi-ticker download structure
        if isinstance(raw['Close'], pd.DataFrame):
            close_raw = raw['Close'][ticker]
            high_raw = raw['High'][ticker]
            low_raw = raw['Low'][ticker]
            volume_raw = raw['Volume'][ticker]
        else: # Case for single ticker download
            close_raw = raw['Close']
            high_raw = raw['High']
            low_raw = raw['Low']
            volume_raw = raw['Volume']
        
        # Forward/Backward fill missing values
        close_raw = close_raw.ffill().bfill()
        high_raw = high_raw.ffill().bfill()
        low_raw = low_raw.ffill().bfill()
        volume_raw = volume_raw.ffill().bfill()
        
        # 🔑 CRITICAL FIX: Ensure all raw series are timezone-naive BEFORE joining/adding
        # This addresses the 'TypeError: Cannot join tz-naive with tz-aware DatetimeIndex'
        if close_raw.index.tz is not None:
            close_raw.index = close_raw.index.tz_localize(None)
            high_raw.index = high_raw.index.tz_localize(None)
            low_raw.index = low_raw.index.tz_localize(None)

        base_price = close_raw.iloc[0]
        if base_price == 0:
            status_text.error(f"Initial price for {ticker} is zero, cannot normalize.")
            return pd.DataFrame(), pd.DataFrame()
        
        # Normalized Cumulative Price
        df[f'{ticker_base}_cum'] = close_raw / base_price
        
        # --- Calculate individual asset EMAs for stacking and divergence analysis ---
        df[f'{ticker_base}_ema_short'] = df[f'{ticker_base}_cum'].ewm(span=ema_short, adjust=False).mean()
        df[f'{ticker_base}_ema_long'] = df[f'{ticker_base}_cum'].ewm(span=ema_long, adjust=False).mean()
        df[f'{ticker_base}_ema_very_long'] = df[f'{ticker_base}_cum'].ewm(span=ema_very_long, adjust=False).mean()
        
        # Summing OHLC for the *Combined Index* ADX (use mean of normalized High/Low/Close)
        
        # Initialize or add to the combined OHLC dataframe
        if ohlc_for_adx.empty:
            # Initialize with the first ticker's normalized data
            ohlc_for_adx['High'] = high_raw / base_price
            ohlc_for_adx['Low'] = low_raw / base_price
            ohlc_for_adx['Close'] = close_raw / base_price
        else:
            # Add to the existing normalized data
            ohlc_for_adx['High'] += high_raw / base_price
            ohlc_for_adx['Low'] += low_raw / base_price
            ohlc_for_adx['Close'] += close_raw / base_price
            
        total_volume += volume_raw

    # Average the OHLC for the combined index ADX calculation (only if more than one ticker)
    if len(tickers) > 1:
        ohlc_for_adx[['High', 'Low', 'Close']] = ohlc_for_adx[['High', 'Low', 'Close']] / len(tickers)

    # --- 2. COMBINED INDEX ---
    df['index_cum'] = df[[f'{t.split("-")[0].lower()}_cum' for t in tickers]].mean(axis=1)
    df['index_cum_smooth'] = df['index_cum'].ewm(span=index_ema_span, adjust=False).mean()
    
    # --- NEW: Short-term Momentum Divergence (BTC EMA 14 minus ETH EMA 14) ---
    df['ema_14_divergence'] = df['btc_ema_short'] - df['eth_ema_short']

    # --- 3. VOLUME AND INDICATORS ---
    df['volume'] = total_volume
    df['Volume_MA'] = df['volume'].rolling(volume_length, min_periods=1).mean()
    
    # --- 4. ADX CALCULATION ---
    # We use the averaged OHLC data for the combined index ADX
    df['ADX'], df['PDI'], df['NDI'] = calculate_adx(
        high=ohlc_for_adx['High'], 
        low=ohlc_for_adx['Low'], 
        close=df['index_cum_smooth'], 
        length=adx_length
    )
    # --- END ADX CALCULATION ---
    
    status_text.empty()
    return df, ohlc_for_adx 

# --- Execution ---
df, _ = fetch_and_process_data(TICKERS, period_days, interval, index_ema_span, ema_short, ema_long, ema_very_long, adx_length)

if df.empty:
    st.stop()

# -----------------------
# Indicators & Cycles (Calculated on the SMOOTHED Combined Index)
# -----------------------
df['EMA_short'] = df['index_cum_smooth'].ewm(span=ema_short, adjust=False).mean()
df['EMA_long'] = df['index_cum_smooth'].ewm(span=ema_long, adjust=False).mean() 
df['EMA_very_long'] = df['index_cum_smooth'].ewm(span=ema_very_long, adjust=False).mean()
df['RSI'] = rsi(df['index_cum_smooth'], length=rsi_length)


# --- RSI Cycle Detection (Original Logic) ---
cycle_id = 0
in_cycle = False
cycle_type = None
cycle_start_idx = None
cycles = [] 
rsi_series = df['RSI']

df_index_list = df.index.to_list() 

if len(df_index_list) > 1:
    prev_rsi = rsi_series.iloc[0]
    for idx in df_index_list[1:]:
        cur_rsi = rsi_series.loc[idx]
        
        # Cycle start detection
        if not in_cycle:
            if (prev_rsi <= 29) and (cur_rsi > 29):
                in_cycle = True; cycle_type = 'rising'; cycle_start_idx = idx; cycle_id += 1
            elif (prev_rsi >= 71) and (cur_rsi < 71):
                in_cycle = True; cycle_type = 'falling'; cycle_start_idx = idx; cycle_id += 1
        # Cycle end detection
        else:
            if cycle_type == 'rising' and (prev_rsi < 71) and (cur_rsi >= 71):
                cycles.append({'id': cycle_id, 'type': 'rising', 'start': cycle_start_idx, 'end': idx})
                in_cycle = False; cycle_type = None; cycle_start_idx = None
            elif cycle_type == 'falling' and (prev_rsi > 29) and (cur_rsi <= 29):
                cycles.append({'id': cycle_id, 'type': 'falling', 'start': cycle_start_idx, 'end': idx})
                in_cycle = False; cycle_type = None; cycle_start_idx = None
        prev_rsi = cur_rsi

# -----------------------
# Realignment detection and signal setting (Updated with Cross-Asset Conviction & Lead Asset)
# -----------------------
df['signal'] = 0 # 1 buy, -1 sell
df['signal_reason'] = None
df['conviction_score'] = 0
df['btc_stack_desc'] = None
df['eth_stack_desc'] = None
df['lead_contributor'] = None


# Volume check function
def check_volume_confirmation(idx):
    if not enable_volume_filter:
        return True # Filter disabled, always allow signal
    
    # Check if current volume is greater than its average (Volume_MA)
    return df.at[idx, 'volume'] > df.at[idx, 'Volume_MA']

# Function to determine which asset crossed its EMA 14/30 first in the search window
def find_lead_contributor(df_slice, direction):
    btc_cross_idx = None
    eth_cross_idx = None
    
    # Look for the 14/30 cross-over event within the preceding bars
    for t in df_slice.index.to_list():
        
        # Bullish Cross (14 > 30)
        if direction == 'rising':
            # BTC Cross
            if btc_cross_idx is None and df.at[t, 'btc_ema_short'] > df.at[t, 'btc_ema_long']:
                if t != df_slice.index[0]: # Ensure we aren't using the first bar of the search slice
                    btc_cross_idx = t
            # ETH Cross
            if eth_cross_idx is None and df.at[t, 'eth_ema_short'] > df.at[t, 'eth_ema_long']:
                if t != df_slice.index[0]:
                    eth_cross_idx = t

        # Bearish Cross (14 < 30)
        elif direction == 'falling':
            # BTC Cross
            if btc_cross_idx is None and df.at[t, 'btc_ema_short'] < df.at[t, 'btc_ema_long']:
                if t != df_slice.index[0]:
                    btc_cross_idx = t
            # ETH Cross
            if eth_cross_idx is None and df.at[t, 'eth_ema_short'] < df.at[t, 'eth_ema_long']:
                if t != df_slice.index[0]:
                    eth_cross_idx = t

    # Compare timestamps (earlier cross wins)
    if btc_cross_idx is None and eth_cross_idx is None:
        return 'Index Alignment Only'
    
    if btc_cross_idx is not None and eth_cross_idx is not None:
        if btc_cross_idx <= eth_cross_idx:
            return 'BTC Lead'
        else:
            return 'ETH Lead'
    elif btc_cross_idx is not None:
        return 'BTC Lead (ETH lagged)'
    elif eth_cross_idx is not None:
        return 'ETH Lead (BTC lagged)'
    else:
        return 'Unclear'


for c in cycles:
    end_idx = c['end']
    
    # Define the search window starting from the bar after the cycle ends
    search_idx_list = df.loc[end_idx:].index.to_list()
    if len(search_idx_list) <= 1:
        continue
    
    if min_bars_after_cycle > 0:
        search_idx_list = search_idx_list[1:min_bars_after_cycle+2] 
    else:
        search_idx_list = search_idx_list[1:]

    dipped = False; spiked = False
    
    if c['type'] == 'rising':
        dip_idx = None; reclaim_idx = None
        for t in search_idx_list:
            # Look for dip below EMA long
            if (not dipped) and (df.at[t, 'index_cum_smooth'] < df.at[t, 'EMA_long']):
                dipped = True; dip_idx = t
            # Look for reclaim above EMA long
            if dipped and (df.at[t, 'index_cum_smooth'] > df.at[t, 'EMA_long']):
                reclaim_idx = t
                
                # 1. FINAL BUY CONDITIONS (Combined Index Stacking)
                is_stacked = (df.at[reclaim_idx, 'EMA_short'] > df.at[reclaim_idx, 'EMA_long']) and \
                             (df.at[reclaim_idx, 'EMA_long'] > df.at[reclaim_idx, 'EMA_very_long'])
                             
                if is_stacked and check_volume_confirmation(reclaim_idx):
                    
                    # 2. CHECK CROSS-ASSET CONVICTION 
                    btc_score, btc_desc = get_asset_stacking_score(
                        df.at[reclaim_idx, 'btc_ema_very_long'], df.at[reclaim_idx, 'btc_ema_long'], df.at[reclaim_idx, 'btc_ema_short']
                    )
                    eth_score, eth_desc = get_asset_stacking_score(
                        df.at[reclaim_idx, 'eth_ema_very_long'], df.at[reclaim_idx, 'eth_ema_long'], df.at[reclaim_idx, 'eth_ema_short']
                    )
                    total_conviction = btc_score + eth_score
                    
                    # 3. DETERMINE LEAD CONTRIBUTOR
                    lookback_slice = df.loc[dip_idx:reclaim_idx]
                    lead_contributor = find_lead_contributor(lookback_slice, 'rising')

                    # 4. RECORD SIGNAL
                    df.at[reclaim_idx, 'signal'] = 1
                    df.at[reclaim_idx, 'conviction_score'] = total_conviction
                    df.at[reclaim_idx, 'btc_stack_desc'] = btc_desc
                    df.at[reclaim_idx, 'eth_stack_desc'] = eth_desc
                    df.at[reclaim_idx, 'lead_contributor'] = lead_contributor
                    
                    vol_note = " (Vol Confirmed)" if enable_volume_filter else ""
                    df.at[reclaim_idx, 'signal_reason'] = f"BUY | Lead: {lead_contributor} | Conviction: {total_conviction}/4{vol_note}"
                    break
                else: 
                    # If combined index is not stacked, no signal fires.
                    break
                    
    elif c['type'] == 'falling':
        spike_idx = None; drop_idx = None
        for t in search_idx_list:
            # Look for spike above EMA long
            if (not spiked) and (df.at[t, 'index_cum_smooth'] > df.at[t, 'EMA_long']):
                spiked = True; spike_idx = t
            # Look for drop below EMA long
            if spiked and (df.at[t, 'index_cum_smooth'] < df.at[t, 'EMA_long']):
                drop_idx = t
                
                # 1. FINAL SELL CONDITIONS (Combined Index Stacking)
                is_stacked = (df.at[drop_idx, 'EMA_short'] < df.at[drop_idx, 'EMA_long']) and \
                             (df.at[drop_idx, 'EMA_long'] < df.at[drop_idx, 'EMA_very_long'])
                             
                if is_stacked and check_volume_confirmation(drop_idx):
                    
                    # 2. CHECK CROSS-ASSET CONVICTION 
                    btc_score, btc_desc = get_asset_stacking_score(
                        df.at[drop_idx, 'btc_ema_very_long'], df.at[drop_idx, 'btc_ema_long'], df.at[drop_idx, 'btc_ema_short']
                    )
                    eth_score, eth_desc = get_asset_stacking_score(
                        df.at[drop_idx, 'eth_ema_very_long'], df.at[drop_idx, 'eth_ema_long'], df.at[drop_idx, 'eth_ema_short']
                    )
                    total_conviction = btc_score + eth_score
                    
                    # 3. DETERMINE LEAD CONTRIBUTOR
                    lookback_slice = df.loc[spike_idx:drop_idx]
                    lead_contributor = find_lead_contributor(lookback_slice, 'falling')
                    
                    # 4. RECORD SIGNAL
                    df.at[drop_idx, 'signal'] = -1
                    df.at[drop_idx, 'conviction_score'] = total_conviction
                    df.at[drop_idx, 'btc_stack_desc'] = btc_desc
                    df.at[drop_idx, 'eth_stack_desc'] = eth_desc
                    df.at[drop_idx, 'lead_contributor'] = lead_contributor
                    
                    vol_note = " (Vol Confirmed)" if enable_volume_filter else ""
                    df.at[drop_idx, 'signal_reason'] = f"SELL | Lead: {lead_contributor} | Conviction: {total_conviction}/4{vol_note}"
                    break
                else: 
                    # If combined index is not stacked, no signal fires.
                    break

# -----------------------
# Real-time Alerting (External + Internal)
# -----------------------
latest_signal = df[df['signal'] != 0].tail(1)

if not latest_signal.empty:
    latest_time = latest_signal.index[0] 
    signal_value = latest_signal['signal'].iloc[0]
    signal_type = "BUY" if signal_value == 1 else "SELL"
    conviction_score = latest_signal['conviction_score'].iloc[0]
    lead_contributor = latest_signal['lead_contributor'].iloc[0]
    
    if latest_time > st.session_state.last_signal_time:
        st.session_state.last_signal_time = latest_time
        
        # --- 1. Internal Alert Message (in the app) ---
        alert_message = (
            f"🔔 **NEW ALERT ({signal_type})**: Cycle Realignment Signal Fired for BTC/ETH Index!\n\n"
            f"**Time**: {latest_time.strftime('%Y-%m-%d %H:%M:%S')} ({interval})\n"
            f"**Action**: {signal_type}\n"
            f"**Lead Contributor**: **{lead_contributor}** (First to cross 14/30 EMA)\n"
            f"**Conviction Score**: {conviction_score} / 4 (Cross-Asset Confirmation)\n"
            f"**BTC Stacking**: {latest_signal['btc_stack_desc'].iloc[0]}\n"
            f"**ETH Stacking**: {latest_signal['eth_stack_desc'].iloc[0]}"
        )
        st.error(alert_message, icon="🚨") 

        # --- 2. External Alert Generation (Simulated) ---
        external_message = f"BTC/ETH Index ALERT ({interval}): {signal_type} @ {latest_time.strftime('%H:%M')}. Lead: {lead_contributor}. Score: {conviction_score}/4."
        send_external_alert(signal_type, external_message, recipient_email, recipient_phone, conviction_score, lead_contributor)


# -----------------------
# Plotting: main chart (Price/EMAs) + RSI subplot + ADX subplot (NEW)
# -----------------------
# Increased rows to 3 for the ADX subplot, adjusted row_heights
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                    row_heights=[0.6, 0.2, 0.2], # Adjusted heights
                    vertical_spacing=0.05,
                    specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]) # Added third spec for ADX

# --- ROW 1: Price Normalized Tracks & EMAs ---
# 1. Price Normalized Tracks (Faded)
fig.add_trace(go.Scatter(x=df.index, y=df['btc_cum'], mode='lines', name='BTC-USD Normalized (Raw)', 
                         line=dict(color='rgba(247, 147, 26, 0.5)', dash='dash'), opacity=0.8), row=1, col=1) 
fig.add_trace(go.Scatter(x=df.index, y=df['eth_cum'], mode='lines', name='ETH-USD Normalized (Raw)', 
                         line=dict(color='rgba(130, 130, 130, 0.5)', dash='dash'), opacity=0.8), row=1, col=1) 

# 2. Combined Index (The primary line)
fig.add_trace(go.Scatter(x=df.index, y=df['index_cum_smooth'], mode='lines', 
                         name=f'Combined Index EMA {index_ema_span}', line=dict(color='#0077c9', width=2)), row=1, col=1)

# 3. EMAs (on combined index)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_short'], mode='lines', name=f'Index EMA {ema_short}', line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_long'], mode='lines', name=f'Index EMA {ema_long}', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['EMA_very_long'], mode='lines', name=f'Index EMA {ema_very_long}', line=dict(color='purple', dash='dot')), row=1, col=1) 

fig.add_hline(y=1.0, line=dict(color='gray', dash='dash'), row=1, col=1) # Baseline for Normalized Price
fig.update_yaxes(title="Normalized Index (Base 1.0)", row=1, col=1)


# 4. Signals Markers
buys = df[df['signal'] == 1]
sells = df[df['signal'] == -1]
if not buys.empty:
    fig.add_trace(go.Scatter(x=buys.index, y=buys['index_cum_smooth'], mode='markers', marker_symbol='triangle-up',
                             marker_color='green', marker_size=12, name='BUY', marker_line_width=1), row=1, col=1)
if not sells.empty:
    fig.add_trace(go.Scatter(x=sells.index, y=sells['index_cum_smooth'], mode='markers', marker_symbol='triangle-down',
                             marker_color='red', marker_size=12, name='SELL', marker_line_width=1), row=1, col=1)

# --- ROW 2: RSI Subplot ---
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name=f'RSI({rsi_length}) (EMA-Smoothed)', line=dict(color='black')), row=2, col=1)
fig.add_hrect(y0=71, y1=100, fillcolor="red", opacity=0.1, line_width=0, row=2, col=1) # Overbought Zone
fig.add_hrect(y0=0, y1=29, fillcolor="green", opacity=0.1, line_width=0, row=2, col=1) # Oversold Zone
fig.add_hline(y=50, line=dict(color='grey', dash='dot'), row=2, col=1)
fig.update_yaxes(range=[0, 100], title="RSI", row=2, col=1) 

# --- ROW 3: ADX Subplot (NEW) ---
# ADX Line
fig.add_trace(go.Scatter(x=df.index, y=df['ADX'], mode='lines', name=f'ADX({adx_length})', line=dict(color='blue', width=2)), row=3, col=1)
# PDI and NDI
fig.add_trace(go.Scatter(x=df.index, y=df['PDI'], mode='lines', name='+DI', line=dict(color='green', dash='dot')), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['NDI'], mode='lines', name='-DI', line=dict(color='red', dash='dot')), row=3, col=1)
# Key strength lines (20: start of trend, 40: strong trend)
fig.add_hline(y=20, line=dict(color='gray', dash='dash'), row=3, col=1)
fig.add_hline(y=40, line=dict(color='gray', dash='dash'), row=3, col=1)
fig.update_yaxes(range=[0, df['ADX'].max() * 1.1 or 100], title="ADX", row=3, col=1) 
# --- END NEW ADX SUBPLOT ---

fig.update_layout(title="BTC/ETH Combined Index Momentum Dashboard",
                  xaxis=dict(rangeslider=dict(visible=False)), height=1000, hovermode="x unified") # Increased height for third panel
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Show table of signals and diagnostics
# -----------------------
st.markdown("### Signals and Diagnostics")
if df['signal'].abs().sum() == 0:
    st.info("No signals found in the selected period with current parameters. Try adjusting settings.")
else:
    # Added 'ADX' to the displayed columns
    sig_df = df[df['signal'] != 0][['index_cum_smooth','EMA_short','EMA_long','EMA_very_long','ADX','PDI','NDI','conviction_score', 'lead_contributor', 'ema_14_divergence', 'btc_stack_desc', 'eth_stack_desc', 'RSI','volume', 'Volume_MA','signal_reason','signal']].copy()
    sig_df.index = sig_df.index.strftime('%Y-%m-%d %H:%M:%S')
    
    # Rename columns for clarity in the output table
    sig_df.rename(columns={
        'index_cum_smooth': 'Index Price',
        'conviction_score': 'Total Conviction (Max 4)',
        'btc_stack_desc': 'BTC Stacking Status',
        'eth_stack_desc': 'ETH Stacking Status',
        'lead_contributor': 'Lead Contributor (14/30 Cross)',
        'ema_14_divergence': 'BTC - ETH EMA 14 Divergence'
    }, inplace=True)
    
    st.dataframe(sig_df.tail(50))

# small metrics
st.markdown("### Summary")
st.write(f"Total cycles detected: **{len(cycles)}**")
st.write(f"Total signals detected: **{int(df['signal'].abs().sum())}** (Filtered by Combined Index Triple EMA Stack)")
st.write(f"Last signal timestamp recorded: **{st.session_state.last_signal_time.strftime('%Y-%m-%d %H:%M:%S')}** (Used to prevent duplicate alerts.)")

# -----------------------
# Auto-Refresh / Manual Refresh
# -----------------------
st.markdown("---")
col_button, col_timer = st.columns([1, 4])

# Refresh button
if col_button.button(f"🔄 Refresh / Re-fetch Index Data"):
    fetch_and_process_data.clear()
    st.experimental_rerun()

# Auto-refresh timer logic
placeholder = col_timer.empty()
refresh = 300 # Fixed refresh rate for simplicity and performance
if refresh > 0:
    for i in range(refresh, 0, -1):
        with placeholder.container():
            st.markdown(f"Next auto-refresh in **{i}** seconds...")
        time.sleep(1)
    
    fetch_and_process_data.clear()
    st.experimental_rerun()
else:
    placeholder.markdown("Auto refresh is **disabled**.")