#!/usr/bin/env python3
"""
Reinforcement Learning Stock Price Predictor

Uses Q-learning to learn optimal trading strategies and predict stock movements.
The agent learns from historical data by:
- Taking actions (BUY/SELL/HOLD) based on market state
- Receiving rewards based on profit/loss
- Learning optimal policy through exploration and exploitation
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import pickle
import random
from collections import deque

import sys
import os

# Add the parent directory to the path so we can import from stock_advisor
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from stock_advisor.core.stock_data import StockDataFetcher
except ImportError:
    # Fallback to relative imports if running from within the package
    from ..core.stock_data import StockDataFetcher


class StockTradingEnvironment:
    """Trading environment for reinforcement learning"""

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_sold = 0
        self.max_net_worth = self.initial_balance
        self.done = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        if self.current_step >= len(self.data):
            return np.zeros(10)

        current = self.data.iloc[self.current_step]

        # Technical indicators state with safe division
        close_price = max(current.get("Close", 1), 0.01)  # Avoid zero
        volume = max(current.get("Volume", 0), 0)
        rsi = current.get("RSI", 50)
        macd = current.get("MACD", 0)
        sma_5 = max(current.get("SMA_5", close_price), 0.01)
        sma_10 = max(current.get("SMA_10", close_price), 0.01)
        sma_20 = max(current.get("SMA_20", close_price), 0.01)

        # Handle NaN values in technical indicators
        if pd.isna(rsi):
            rsi = 50
        if pd.isna(macd):
            macd = 0
        if pd.isna(sma_5):
            sma_5 = close_price
        if pd.isna(sma_10):
            sma_10 = close_price
        if pd.isna(sma_20):
            sma_20 = close_price

        state = [
            close_price / 1000,  # Normalized price
            volume / 1e6,  # Normalized volume
            max(min(rsi / 100, 1.0), 0.0),  # RSI (0-1) clamped
            max(min(macd, 10), -10),  # MACD clamped
            sma_5 / 1000,  # 5-day MA
            sma_10 / 1000,  # 10-day MA
            sma_20 / 1000,  # 20-day MA
            max(self.balance / self.initial_balance, 0),  # Balance ratio
            max(self.shares_held / 100, 0),  # Shares held (normalized)
            max(
                self._get_portfolio_value() / self.initial_balance, 0
            ),  # Portfolio ratio
        ]

        # Ensure no NaN or infinite values
        state_array = np.array(state, dtype=np.float32)
        state_clean = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)

        return state_clean

    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        if self.current_step >= len(self.data):
            return self.balance

        current_price = self.data.iloc[self.current_step]["Close"]
        return self.balance + (self.shares_held * current_price)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment

        Actions:
        0 = HOLD
        1 = BUY
        2 = SELL
        """
        if self.current_step >= len(self.data) - 1:
            self.done = True
            return self._get_state(), 0, True, {}

        current_price = self.data.iloc[self.current_step]["Close"]
        next_price = self.data.iloc[self.current_step + 1]["Close"]

        # Execute action
        reward = 0
        if action == 1:  # BUY
            reward = self._buy(current_price)
        elif action == 2:  # SELL
            reward = self._sell(current_price)
        # action == 0 is HOLD, no action needed

        # Move to next step
        self.current_step += 1

        # Calculate reward based on price movement and action
        price_change = (next_price - current_price) / current_price

        if action == 1 and self.shares_held > 0:  # Bought and price went up
            reward += price_change * 100
        elif action == 2 and price_change < 0:  # Sold before price drop
            reward += abs(price_change) * 100
        elif action == 0:  # Holding
            if self.shares_held > 0:
                reward += price_change * 50  # Reduced reward for holding

        # Portfolio performance reward
        current_net_worth = self._get_portfolio_value()
        if current_net_worth > self.max_net_worth:
            reward += (
                (current_net_worth - self.max_net_worth) / self.initial_balance * 100
            )
            self.max_net_worth = current_net_worth

        # Penalty for holding too long without action
        if action == 0:
            reward -= 0.1

        # Check if done
        if self.current_step >= len(self.data) - 1:
            self.done = True
            # Final portfolio value bonus/penalty
            final_return = (
                current_net_worth - self.initial_balance
            ) / self.initial_balance
            reward += final_return * 1000

        return (
            self._get_state(),
            reward,
            self.done,
            {
                "portfolio_value": current_net_worth,
                "price": next_price if not self.done else current_price,
            },
        )

    def _buy(self, price: float) -> float:
        """Execute buy action"""
        shares_to_buy = self.balance // price
        if shares_to_buy > 0:
            cost = shares_to_buy * price
            self.balance -= cost
            self.shares_held += shares_to_buy
            return 1  # Small positive reward for taking action
        return -1  # Penalty for invalid action

    def _sell(self, price: float) -> float:
        """Execute sell action"""
        if self.shares_held > 0:
            revenue = self.shares_held * price
            self.balance += revenue
            self.total_shares_sold += self.shares_held
            self.shares_held = 0
            return 1  # Small positive reward for taking action
        return -1  # Penalty for invalid action


class DQNAgent:
    """Deep Q-Network agent for stock trading"""

    def __init__(
        self, state_size: int = 10, action_size: int = 3, learning_rate: float = 0.001
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=2000)
        self.q_table = {}  # Using Q-table for simplicity instead of neural network

    def _get_state_key(self, state: np.ndarray) -> str:
        """Convert state to string key for Q-table"""
        # Handle NaN and infinite values
        state_clean = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)

        # Discretize continuous state space
        discretized = np.round(state_clean * 10).astype(int)
        return str(discretized.tolist())

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        return np.argmax(self.q_table[state_key])

    def replay(self, batch_size: int = 32):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in batch:
            state_key = self._get_state_key(state)
            next_state_key = self._get_state_key(next_state)

            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)

            target = reward
            if not done:
                target += 0.95 * np.amax(self.q_table[next_state_key])

            target_f = self.q_table[state_key].copy()
            target_f[action] = target
            self.q_table[state_key] = target_f

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filepath: str):
        """Save the Q-table to file"""
        with open(filepath, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)

    def load_model(self, filepath: str):
        """Load Q-table from file"""
        try:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                self.q_table = data["q_table"]
                self.epsilon = data["epsilon"]
            return True
        except FileNotFoundError:
            return False


class RLStockPredictor:
    """Reinforcement Learning Stock Predictor"""

    def __init__(self):
        self.stock_fetcher = StockDataFetcher()
        self.agent = DQNAgent()
        self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.model_dir, exist_ok=True)

    def train_agent(self, symbol: str, episodes: int = 100) -> Dict:
        """Train the RL agent on historical data"""
        symbol = symbol.upper()

        # Get historical data
        print(f"Fetching training data for {symbol}...")
        stock_data = self.stock_fetcher.get_stock_data(symbol, period="1y")
        if stock_data.empty:
            return {"error": f"No data available for {symbol}"}

        # Add technical indicators
        stock_data = self.stock_fetcher.calculate_technical_indicators(stock_data)

        # Create environment
        env = StockTradingEnvironment(stock_data)

        # Training loop
        scores = []
        portfolio_values = []

        print(f"Training agent for {episodes} episodes...")

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0

            while not env.done:
                action = self.agent.act(state)
                next_state, reward, done, info = env.step(action)
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(self.agent.memory) > 32:
                    self.agent.replay(32)

            scores.append(total_reward)
            portfolio_values.append(env._get_portfolio_value())

            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:])
                avg_portfolio = np.mean(portfolio_values[-10:])
                print(
                    f"Episode {episode}/{episodes}, "
                    f"Avg Score: {avg_score:.2f}, "
                    f"Avg Portfolio: ${avg_portfolio:.2f}, "
                    f"Epsilon: {self.agent.epsilon:.3f}"
                )

        # Save trained model
        model_path = os.path.join(self.model_dir, f"{symbol}_rl_model.pkl")
        self.agent.save_model(model_path)

        return {
            "episodes_trained": episodes,
            "final_epsilon": self.agent.epsilon,
            "average_score": np.mean(scores[-10:]),
            "average_portfolio_value": np.mean(portfolio_values[-10:]),
            "model_saved": model_path,
        }

    def predict_next_3_days(self, symbol: str, auto_train: bool = True) -> Dict:
        """
        Predict next 3 days using trained RL agent

        Args:
            symbol: Stock symbol
            auto_train: Whether to train if no model exists
        """
        symbol = symbol.upper()

        # Try to load existing model
        model_path = os.path.join(self.model_dir, f"{symbol}_rl_model.pkl")
        model_loaded = self.agent.load_model(model_path)

        if not model_loaded and auto_train:
            print(f"No trained model found for {symbol}. Training new model...")
            training_result = self.train_agent(symbol, episodes=50)
            if "error" in training_result:
                return training_result
        elif not model_loaded:
            return {
                "error": f"No trained model for {symbol}. Use auto_train=True or train manually."
            }

        # Get recent data for prediction
        stock_data = self.stock_fetcher.get_stock_data(symbol, period="3mo")
        if stock_data.empty:
            return {"error": f"No data available for {symbol}"}

        stock_data = self.stock_fetcher.calculate_technical_indicators(stock_data)

        # Create environment for prediction
        env = StockTradingEnvironment(stock_data.tail(60))  # Use last 60 days
        state = env.reset()

        # Move to the end of available data
        while not env.done and env.current_step < len(env.data) - 4:
            action = self.agent.act(state)
            state, _, done, _ = env.step(action)

        # Predict next 3 days
        predictions = []
        current_price = stock_data["Close"].iloc[-1]

        for day in range(1, 4):
            # Get agent's action recommendation
            action = self.agent.act(state)
            action_names = ["HOLD", "BUY", "SELL"]

            # Translate action to price prediction
            if action == 1:  # BUY signal
                predicted_change = random.uniform(0.005, 0.02)  # 0.5% to 2% up
                confidence = 70
                direction = "UP"
                reasoning = "RL agent recommends BUY - expecting price increase"
            elif action == 2:  # SELL signal
                predicted_change = random.uniform(-0.02, -0.005)  # 0.5% to 2% down
                confidence = 70
                direction = "DOWN"
                reasoning = "RL agent recommends SELL - expecting price decrease"
            else:  # HOLD signal
                predicted_change = random.uniform(-0.005, 0.005)  # ±0.5%
                confidence = 50
                direction = "FLAT"
                reasoning = "RL agent recommends HOLD - expecting sideways movement"

            # Adjust for day (less confident further out)
            predicted_change *= 1 - (day - 1) * 0.2
            confidence *= 1 - (day - 1) * 0.15

            predicted_price = current_price * (1 + predicted_change)

            predictions.append(
                {
                    "day": day,
                    "predicted_price": predicted_price,
                    "predicted_change_pct": predicted_change * 100,
                    "direction": direction,
                    "confidence": max(30, confidence),
                    "rl_action": action_names[action],
                    "reasoning": reasoning,
                }
            )

            current_price = predicted_price

        # Overall summary
        total_change = (
            predictions[-1]["predicted_price"] - stock_data["Close"].iloc[-1]
        ) / stock_data["Close"].iloc[-1]
        overall_direction = (
            "UP" if total_change > 0 else "DOWN" if total_change < 0 else "FLAT"
        )

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": stock_data["Close"].iloc[-1],
            "model_info": {
                "model_loaded": model_loaded,
                "epsilon": self.agent.epsilon,
                "q_table_size": len(self.agent.q_table),
            },
            "daily_predictions": predictions,
            "overall_summary": {
                "direction": overall_direction,
                "total_change_pct": total_change * 100,
                "final_price": predictions[-1]["predicted_price"],
                "avg_confidence": np.mean([p["confidence"] for p in predictions]),
            },
        }

    def print_prediction_summary(self, result: Dict):
        """Print formatted prediction summary"""
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return

        print(f"\n{'='*70}")
        print(f"REINFORCEMENT LEARNING PREDICTION: {result['symbol']}")
        print(f"{'='*70}")
        print(f"Analysis Time: {result['timestamp']}")
        print(f"Current Price: ${result['current_price']:.2f}")

        # Model Info
        model_info = result["model_info"]
        print(f"\nRL MODEL INFO:")
        print(
            f"Model Status: {'Loaded' if model_info['model_loaded'] else 'Newly Trained'}"
        )
        print(f"Exploration Rate (ε): {model_info['epsilon']:.3f}")
        print(f"Q-Table Size: {model_info['q_table_size']} states learned")

        # Overall Prediction
        overall = result["overall_summary"]
        print(f"\n3-DAY RL FORECAST:")
        print(f"Overall Direction: {overall['direction']}")
        print(f"Total Change: {overall['total_change_pct']:+.2f}%")
        print(f"Target Price: ${overall['final_price']:.2f}")
        print(f"Average Confidence: {overall['avg_confidence']:.1f}%")

        # Daily Breakdown
        print(f"\nDAILY RL PREDICTIONS:")
        print(
            f"{'Day':<5} {'Action':<6} {'Direction':<10} {'Price':<12} {'Change':<10} {'Confidence':<12}"
        )
        print(f"{'-'*70}")

        for pred in result["daily_predictions"]:
            print(
                f"{pred['day']:<5} "
                f"{pred['rl_action']:<6} "
                f"{pred['direction']:<10} "
                f"${pred['predicted_price']:<11.2f} "
                f"{pred['predicted_change_pct']:+8.2f}% "
                f"{pred['confidence']:<11.1f}%"
            )

        print(f"\nRL REASONING:")
        for i, pred in enumerate(result["daily_predictions"], 1):
            print(f"Day {i}: {pred['reasoning']}")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python rl_predictor.py <SYMBOL> [--train-only] [--episodes=N]")
        print("Examples:")
        print(
            "  python rl_predictor.py AAPL                 # Predict (auto-train if needed)"
        )
        print("  python rl_predictor.py AAPL --train-only    # Only train the model")
        print("  python rl_predictor.py AAPL --episodes=100  # Train with 100 episodes")
        return

    symbol = sys.argv[1].upper()
    train_only = "--train-only" in sys.argv
    episodes = 50  # Default episodes

    # Check for custom episodes
    for arg in sys.argv:
        if arg.startswith("--episodes="):
            try:
                episodes = int(arg.split("=")[1])
            except ValueError:
                print("Invalid episodes value, using default (50)")

    print("Reinforcement Learning Stock Predictor")
    print("=====================================")
    print(f"Symbol: {symbol}")
    print(f"Method: Q-Learning with trading environment")

    predictor = RLStockPredictor()

    if train_only:
        print(f"\nTraining RL model for {symbol} with {episodes} episodes...")
        result = predictor.train_agent(symbol, episodes)
        if "error" not in result:
            print(f"\nTraining completed!")
            print(f"Episodes: {result['episodes_trained']}")
            print(f"Final exploration rate: {result['final_epsilon']:.3f}")
            print(f"Average score: {result['average_score']:.2f}")
            print(f"Model saved: {result['model_saved']}")
        else:
            print(f"Training failed: {result['error']}")
    else:
        result = predictor.predict_next_3_days(symbol, auto_train=True)
        predictor.print_prediction_summary(result)

    print(f"\n{'='*70}")
    print("DISCLAIMER: RL predictions are experimental and for educational use.")
    print("Not financial advice. RL models require significant training data.")
    print(f"{'='*70}")


# Alias for compatibility
RLPredictor = RLStockPredictor

if __name__ == "__main__":
    main()
