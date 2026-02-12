# Cryptocurrency Trading Bot with PPO Deep Reinforcement Learning

This project implements a cryptocurrency trading bot influenced by the research paper "Automated Cryptocurrency Trading Bot Implementing DRL" by Aisha Peng, Sau Loong Ang, and Chia Yean Lim. The bot uses Proximal Policy Optimization (PPO) with a CNN-LSTM neural network architecture to learn optimal trading strategies.

## Features

- **Automated Trading**: Uses deep reinforcement learning to make trading decisions.
- **CNN-LSTM Architecture**: Extracts features from historical price data and technical indicators using CNNs and models temporal dependencies with LSTMs.
- **Technical Indicators**: Incorporates RSI, ATR, and Chaikin Money Flow (CMF).
- **Advanced Preprocessing**: Applies differencing for stationarity and normalization for optimal model performance.
- **PPO Agent**: Implements the Actor-Critic framework with Proximal Policy Optimization.
- **Live Trading Dashboard**: Interactive Streamlit web app for real-time visualization and probability analysis.
- **Live Trading Simulation**: Integration with Binance API for paper trading and real execution.
- **Backtesting Engine**: Comprehensive backtesting with detailed performance reports.

## Key Formulations

The bot implements key formulations from the research paper:

1. **PPO-CLIP Objective Function**:
   L<sup>CLIP</sup>(θ) = Ê<sub>t</sub>[min(r<sub>t</sub>(θ)Â<sub>t</sub>, clip(r<sub>t</sub>(θ), 1 − ε, 1 + ε)Â<sub>t</sub>)]

2. **Probability Ratio Calculation**:
   r<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) / π<sub>θ old</sub>(a<sub>t</sub>|s<sub>t</sub>)

3. **Trading Mechanism**:
   - Buy: Amount bought = Current net worth / Current crypto closing price
   - Sell: Amount sold = Current crypto amount held × Current crypto closing price
   - Reward: r_t = (v_t - v_{t-1}) / v_{t-1}

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── cnn_lstm_model.py  # CNN-LSTM neural network architecture
│   │   └── ppo_agent.py       # PPO reinforcement learning agent
│   ├── data/
│   │   └── ...                # Data storage
│   ├── env/
│   │   └── crypto_env.py      # Trading environment
│   ├── utils/
│   │   └── data_processor.py  # Data processing utilities
│   ├── web_utils.py           # Web app utilities
│   ├── train.py               # Training script
│   ├── backtest.py            # Backtesting implementation
│   └── live_trading.py        # Live trading implementation
├── models/                    # Saved model weights
├── results/                   # Training results and logs
├── app.py                     # Streamlit Web App Dashboard
├── train_bot.py               # Training wrapper script
├── backtest_bot.py            # Backtesting wrapper script
├── live_trade.py              # Live trading wrapper script
├── demo_data_processing.py    # Script to demonstrate data preprocessing
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository.
2. Install dependencies (Windows/Linux/Mac compatible):

   ```bash
   pip install -r requirements.txt
   ```

   *Note: This project is optimized for TensorFlow 2.16.1.*

## Usage

### 1. Data Collection & Preprocessing

The bot handles data automatically. To see the preprocessing in action:

```bash
python demo_data_processing.py
```

### 2. Training

Train the PPO agent:

```bash
python train_bot.py --symbol BTCUSDT --episodes 100 --initial-balance 1000
```

Models are saved to the `models/` directory (e.g., `BTCUSDT_actor_best.keras`).

### 3. Backtesting

Test the trained model on historical data:

```bash
python backtest_bot.py --symbol BTCUSDT --start-date 2024-01-01 --end-date 2025-01-01
```

Results are saved to `results/`.

### 4. Live Trading Dashboard (Web App)

Visualize live data and model predictions in a web interface:

```bash
streamlit run app.py
```

- **Live Chart**: Shows the last 100 hours of price action.
- **Probabilities**: Displays "Call" (Buy), "Put" (Sell), and Hold probabilities.
- **Auto-Refresh**: Updates every 10 minutes by default.
- **Configuration**: Adjust symbol, interval, and refresh rate in the sidebar.

### 5. Live Trading Script

Run the bot in **Paper Trading (Test)** mode (defaults to 10-minute intervals):

```bash
python live_trade.py --symbol BTCUSDT --test-mode
```

Run in **Real Trading** mode (Requires Binance API keys):

```bash
python live_trade.py --symbol BTCUSDT --api-key "YOUR_KEY" --api-secret "YOUR_SECRET" --test-mode False
```

## Model Architecture

The trading bot uses a hybrid CNN-LSTM architecture:

1. **CNN Layers**: Extract features from historical price data and technical indicators.
2. **LSTM Layers**: Model temporal dependencies in the time series data.
3. **PPO Agent**: Actor-critic framework that learns optimal trading policy.

## Performance Metrics

- **Total Return**: ROI compared to initial balance.
- **Sharpe Ratio**: Risk-adjusted return metric.
- **Max Drawdown**: Largest peak-to-trough decline.
- **Win Rate**: Percentage of profitable trades.
