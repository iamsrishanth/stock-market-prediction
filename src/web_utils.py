import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from binance.client import Client

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_processor import DataProcessor
from src.models.ppo_agent import PPOAgent

def load_agent(model_path, symbol, lookback_window_size=100):
    """
    Load the trained PPO agent and models.
    """
    # Initialize data processor (needed for input shape determination if not hardcoded)
    # But we know input shape is (lookback_window_size, 5) based on data_processor
    # 5 features: close_diff, rsi, atr, cmf, close_orig (but close_orig is usually dropped for training?)
    # Let's check DataProcessor.apply_difference:
    # columns_to_keep = ['close_orig', 'close_diff', 'rsi', 'atr', 'cmf']
    # And DataProcessor.prepare_data calls apply_difference.
    # Then normalize_data usually excludes 'close_orig'.
    # Wait, PPOAgent input shape depends on what's passed to it.
    # In live_trading.py: state = self._prepare_state(processed_df[['close_diff', 'rsi', 'atr', 'cmf']])
    # So 4 features.
    
    input_shape = (lookback_window_size, 4) # close_diff, rsi, atr, cmf
    action_space = 3  # Buy, Hold, Sell
    
    agent = PPOAgent(input_shape, action_space)
    
    try:
        actor_path = os.path.join(model_path, f"{symbol}_actor_best.keras")
        critic_path = os.path.join(model_path, f"{symbol}_critic_best.keras")
        
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            agent.load_models(actor_path, critic_path)
            print(f"Successfully loaded models for {symbol}")
            return agent
        else:
            print(f"Model files not found for {symbol} at {actor_path}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_market_data(symbol, interval, lookback_window_size=100):
    """
    Fetch historical data and prepare it for prediction and visualization.
    """
    data_processor = DataProcessor()
    
    # Fetch enough data for lookback + indicators calculation (approx 200 points should be safe)
    # We want to show the last 100 hours, but need more for accurate indicators
    limit = lookback_window_size + 50 
    
    # We can't easily specify limit in download_data if it takes dates.
    # So let's calculate dates.
    end_time = datetime.now()
    # Assuming interval is '1h', 150 hours ago. 
    # If interval is different, this might need adjustment, but '1h' is default.
    start_time = end_time - timedelta(hours=limit)
    
    df = data_processor.download_data(
        symbol,
        interval,
        start_time.strftime('%Y-%m-%d'),
        end_time.strftime('%Y-%m-%d')
    )
    
    return df

def get_prediction(agent, df, lookback_window_size=100):
    """
    Process data and get prediction probabilities.
    """
    data_processor = DataProcessor()
    
    # Process data
    # Note: prepare_data adds indicators, diffs, and normalizes
    processed_df = data_processor.prepare_data(df)
    
    # Check if we have enough data
    if len(processed_df) < lookback_window_size:
        print(f"Not enough data for prediction. Need {lookback_window_size}, got {len(processed_df)}")
        return None, None
        
    # Prepare state: take last 'lookback_window_size' rows and specific columns
    # Must match training features: close_diff, rsi, atr, cmf
    recent_data = processed_df.iloc[-lookback_window_size:][['close_diff', 'rsi', 'atr', 'cmf']]
    
    # Create state array
    state = recent_data.values
    
    # Add batch dimension
    state = np.expand_dims(state, axis=0)
    
    # Get action probabilities directly from actor model
    # agent.actor is a Keras model
    try:
        action_probs = agent.actor.predict(state, verbose=0)[0]
        # action_probs is [prob_buy, prob_hold, prob_sell]
        return action_probs, processed_df
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None
