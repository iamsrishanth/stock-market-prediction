# Cryptocurrency Trading Bot with PPO Deep Reinforcement Learning

This project implements a cryptocurrency trading bot influenced by the research paper "Automated Cryptocurrency Trading Bot Implementing DRL" by Aisha Peng, Sau Loong Ang, and Chia Yean Lim. The bot uses Proximal Policy Optimization (PPO) with a CNN-LSTM neural network architecture to learn optimal trading strategies.

## Features

- Automated cryptocurrency trading using deep reinforcement learning
- CNN-LSTM architecture for feature extraction and time series analysis
- Technical indicators including RSI, ATR, and Chaikin Money Flow
- Enhanced data preprocessing with differencing for stationarity and normalization
- PPO agent implementation with actor-critic framework
- Live trading simulation with Binance API integration
- Performance metrics and visualization
- Comprehensive backtesting functionality with detailed reports

## Key Formulations

The bot implements the following key formulations from the research paper:

1. **PPO-CLIP Objective Function**:

   L<sup>CLIP</sup>(θ) = Ê<sub>t</sub>[min(r<sub>t</sub>(θ)Â<sub>t</sub>, clip(r<sub>t</sub>(θ), 1 − ε, 1 + ε)Â<sub>t</sub>)]

2. **Probability Ratio Calculation**:

   r<sub>t</sub>(θ) = π<sub>θ</sub>(a<sub>t</sub>|s<sub>t</sub>) / π<sub>θ old</sub>(a<sub>t</sub>|s<sub>t</sub>)

3. **Trading Mechanism**:
   - Buy calculation: Amount bought = Current net worth / Current crypto closing price
   - Sell calculation: Amount sold = Current crypto amount held × Current crypto closing price
   - Reward function: r_t = (v_t - v_{t-1}) / v_{t-1}

## Project Structure

```
.
├── src/
│   ├── models/
│   │   ├── cnn_lstm_model.py  # CNN-LSTM neural network architecture
│   │   └── ppo_agent.py       # PPO reinforcement learning agent
│   ├── data/
│   │   └── ...                # Data storage (will be created)
│   ├── env/
│   │   └── crypto_env.py      # Trading environment
│   ├── utils/
│   │   └── data_processor.py  # Data processing utilities
│   ├── train.py               # Training script
│   ├── backtest.py            # Backtesting implementation
│   └── live_trading.py        # Live trading implementation
├── models/                    # Saved model weights (will be created)
├── results/                   # Training results and logs (will be created)
├── train_bot.py               # Training wrapper script
├── backtest_bot.py            # Backtesting wrapper script
├── live_trade.py              # Live trading wrapper script
├── demo_data_processing.py    # Script to demonstrate data preprocessing
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository

2. Install dependencies (Windows/Linux/Mac compatible):

   ```bash
   pip install -r requirements.txt
   ```

   *Note: This project is optimized for TensorFlow 2.16.1 and Gymnasium.*

## Verified Results (2024)

The bot has been trained and verified on BTCUSDT data:

- **Training**: Learns aggressive profitable strategies (up to +66% in episodes).
- **Backtesting (2024 Full Year)**: Achieved **+114% Return**, matching the Buy & Hold strategy in a strong bull market.
- **Strategy**: Adapts between active trading and trend following (Buy & Hold) based on market conditions.

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

- Models are saved to `models/` directory (e.g., `BTCUSDT_actor_best.keras`).

### 3. Backtesting

Test the trained model on historical data:

```bash
python backtest_bot.py --symbol BTCUSDT --start-date 2024-01-01 --end-date 2025-01-01
```

Results are saved to `results/` including equity curves and trade logs.

### 4. Live Trading

Run the bot in **Paper Trading (Test)** mode:

```bash
python live_trade.py --symbol BTCUSDT --test-mode
```

Run in **Real Trading** mode (Requires Binance API keys):

```bash
python live_trade.py --symbol BTCUSDT --api-key "YOUR_KEY" --api-secret "YOUR_SECRET" --test-mode False
```

## Model Architecture

The trading bot uses a hybrid CNN-LSTM architecture:

1. **CNN Layers**: Extract features from historical price data and technical indicators
2. **LSTM Layers**: Model temporal dependencies in the time series data
3. **PPO Agent**: Actor-critic framework that learns optimal trading policy

## Performance Metrics

- **Total Return**: ROI compared to initial balance.
- **Sharpe Ratio**: Risk-adjusted return metric.
- **Max Drawdown**: Largest peak-to-trough decline.
- **Win Rate**: Percentage of profitable trades.
