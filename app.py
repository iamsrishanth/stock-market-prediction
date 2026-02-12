import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.web_utils import load_agent, get_market_data, get_prediction

# Page configuration
st.set_page_config(
    page_title="Crypto Live Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar settings
st.sidebar.title("Configuration")
symbol = st.sidebar.text_input("Symbol", "BTCUSDT")
interval = st.sidebar.selectbox("Interval", ["1h", "15m", "5m", "1m"], index=0)
lookback = st.sidebar.number_input("Lookback Window", min_value=10, max_value=200, value=100)
refresh_rate = st.sidebar.number_input("Refresh Rate (seconds)", min_value=10, value=600)

model_path = "models"

# Main title
st.title(f"Live Trading Dashboard - {symbol}")

# Initialize session state for auto-refresh
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

# Function to load data and predict
def update_dashboard():
    with st.spinner('Fetching data and calculating predictions...'):
        # Load agent (cached if possible, but for now re-loading is safer to avoid session state issues with TF)
        # In a real app, we'd cache the agent globally/resource.
        agent = load_agent(model_path, symbol, lookback)
        
        if agent is None:
            st.error(f"Could not load model for {symbol}. Make sure 'models/{symbol}_actor_best.keras' exists.")
            return

        # Get data
        df = get_market_data(symbol, interval, lookback)
        
        if df is None or df.empty:
            st.error("Failed to fetch market data.")
            return

        # Get prediction
        probs, processed_df = get_prediction(agent, df, lookback)

        if probs is None:
            st.error("Failed to make prediction.")
            return

        # --- Display Layout ---
        
        # 1. Top Metrics (Price, Prediction)
        current_price = df['close'].iloc[-1]
        
        # Probabilities: 0=Buy, 1=Hold, 2=Sell
        # User asked for "Call" (Buy) or "Put" (Sell) probability. 
        # We can show Buy vs Sell vs Hold.
        prob_buy = probs[0]
        prob_hold = probs[1]
        prob_sell = probs[2]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}")
            
        with col2:
            st.metric("CALL (Buy) Probability", f"{prob_buy:.1%}", delta=None)
            
        with col3:
            st.metric("PUT (Sell) Probability", f"{prob_sell:.1%}", delta=None)
            
        with col4:
            # Determine overall signal
            if prob_buy > 0.4 and prob_buy > prob_sell: # Simple threshold
                signal = "BUY (CALL)"
                color = "green"
            elif prob_sell > 0.4 and prob_sell > prob_buy:
                signal = "SELL (PUT)"
                color = "red"
            else:
                signal = "HOLD"
                color = "gray"
            st.markdown(f"**Signal:** :{color}[{signal}]")

        # 2. Probability Chart
        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Action': ['Buy (Call)', 'Hold', 'Sell (Put)'],
            'Probability': [prob_buy, prob_hold, prob_sell]
        })
        st.bar_chart(prob_df.set_index('Action'))

        # 3. Main Candlestick Chart
        st.subheader(f"Price History (Last {lookback} Candles)")
        
        # Subset for display (last 100)
        display_df = df.tail(lookback)
        
        fig = go.Figure(data=[go.Candlestick(x=display_df.index,
                        open=display_df['open'],
                        high=display_df['high'],
                        low=display_df['low'],
                        close=display_df['close'])])
        
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 4. Technical Indicators (Optional but requested "stats")
        st.subheader("Technical Indicators (Processed)")
        
        # Show specific indicators from processed_df
        # Re-align processed_df index with original df if needed, or just plot tail
        # processed_df has reset index usually or matches if carefully handled.
        # Let's use the tail of processed_df
        p_tail = processed_df.tail(lookback)
        
        tab1, tab2, tab3 = st.tabs(["RSI", "ATR", "CMF"])
        
        with tab1:
            st.line_chart(p_tail['rsi'])
        with tab2:
            st.line_chart(p_tail['atr'])
        with tab3:
            st.line_chart(p_tail['cmf'])
            
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Initial Load
update_dashboard()

# Auto-refresh logic (using st.empty() or similar is complex, using simple rerun)
# Streamlit runs the script from top to bottom on every interaction/rerun.
if refresh_rate > 0:
    time.sleep(refresh_rate)
    st.rerun()
