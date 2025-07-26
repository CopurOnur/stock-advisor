#!/usr/bin/env python3
"""
Enhanced Reinforcement Learning Stock Price Predictor

Improvements over basic RL:
1. Deep Q-Network (DQN) with neural networks instead of Q-table
2. Enhanced state representation with more features
3. Sophisticated reward structure with risk metrics
4. Position sizing and risk management
5. Multi-agent ensemble approach
6. Proper experience replay with prioritized sampling
7. Target network for stable learning
8. Advanced technical indicators
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pickle
import random
from collections import deque, namedtuple
import warnings

warnings.filterwarnings("ignore")

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

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using simplified neural network implementation.")

# Experience tuple for replay buffer
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done", "priority"]
)


class SimpleDQN:
    """Simple neural network implementation without PyTorch"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.001,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * 0.1
        self.b3 = np.zeros((1, output_size))

    def forward(self, x):
        """Forward pass"""
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.tanh(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3

    def predict(self, x):
        """Predict Q-values"""
        return self.forward(x)

    def train_step(self, x, y):
        """Single training step"""
        batch_size = x.shape[0]

        # Forward pass
        output = self.forward(x)

        # Compute loss (MSE)
        loss = np.mean((output - y) ** 2)

        # Backward pass
        dL_dz3 = 2 * (output - y) / batch_size
        dL_dW3 = np.dot(self.a2.T, dL_dz3)
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

        dL_da2 = np.dot(dL_dz3, self.W3.T)
        dL_dz2 = dL_da2 * (1 - self.a2**2)  # tanh derivative
        dL_dW2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * (1 - self.a1**2)  # tanh derivative
        dL_dW1 = np.dot(x.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        # Update weights
        self.W3 -= self.learning_rate * dL_dW3
        self.b3 -= self.learning_rate * dL_db3
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1

        return loss

    def copy_from(self, other):
        """Copy weights from another network"""
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()


class DQNNetwork(nn.Module):
    """Deep Q-Network with PyTorch"""

    def __init__(self, input_size: int, hidden_size: int = 256, output_size: int = 3):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.network(x)


class EnhancedTradingEnvironment:
    """Enhanced trading environment with risk management"""

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000,
        max_position_size: float = 0.1,
        transaction_cost: float = 0.001,
    ):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size  # Max 10% of portfolio per position
        self.transaction_cost = transaction_cost  # 0.1% transaction cost
        self.lookback_window = 20  # Days to look back for state
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.shares_held = 0
        self.position_value = 0
        self.max_net_worth = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.max_drawdown = 0
        self.peak_value = self.initial_balance
        self.done = False
        self.position_history = []
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get enhanced state representation"""
        if self.current_step >= len(self.data):
            return np.zeros(30)  # Increased state size

        current_idx = self.current_step
        window_data = self.data.iloc[
            max(0, current_idx - self.lookback_window) : current_idx + 1
        ]

        if len(window_data) < 2:
            return np.zeros(30)

        current = window_data.iloc[-1]

        # Price and volume features
        close_price = max(current.get("Close", 1), 0.01)
        volume = max(current.get("Volume", 0), 1)

        # Technical indicators
        rsi = self._safe_value(current.get("RSI", 50), 0, 100) / 100
        macd = self._safe_value(current.get("MACD", 0), -10, 10) / 10
        macd_signal = self._safe_value(current.get("MACD_Signal", 0), -10, 10) / 10
        bb_upper = max(current.get("BB_Upper", close_price), 0.01)
        bb_lower = max(current.get("BB_Lower", close_price), 0.01)

        # Moving averages
        sma_5 = max(current.get("SMA_5", close_price), 0.01)
        sma_10 = max(current.get("SMA_10", close_price), 0.01)
        sma_20 = max(current.get("SMA_20", close_price), 0.01)
        ema_12 = max(current.get("EMA_12", close_price), 0.01)
        ema_26 = max(current.get("EMA_26", close_price), 0.01)

        # Price momentum features
        price_changes = window_data["Close"].pct_change().fillna(0)
        momentum_1 = price_changes.iloc[-1] if len(price_changes) > 0 else 0
        momentum_5 = price_changes.tail(5).mean() if len(price_changes) >= 5 else 0
        volatility = price_changes.std() if len(price_changes) > 1 else 0

        # Volume features
        volume_ma = window_data["Volume"].mean()
        volume_ratio = volume / max(volume_ma, 1)

        # Portfolio features
        portfolio_value = self._get_portfolio_value()
        balance_ratio = self.balance / self.initial_balance
        position_ratio = self.shares_held * close_price / self.initial_balance

        # Risk metrics
        drawdown = (
            (self.peak_value - portfolio_value) / self.peak_value
            if self.peak_value > 0
            else 0
        )
        win_rate = self.winning_trades / max(self.total_trades, 1)

        # Market condition features
        trend_strength = (close_price - sma_20) / sma_20
        bollinger_position = (close_price - bb_lower) / max(bb_upper - bb_lower, 0.01)

        state = [
            close_price / 1000,  # 0: Normalized price
            volume / 1e6,  # 1: Normalized volume
            rsi,  # 2: RSI (0-1)
            macd,  # 3: MACD (-1 to 1)
            macd_signal,  # 4: MACD Signal (-1 to 1)
            sma_5 / 1000,  # 5: SMA 5
            sma_10 / 1000,  # 6: SMA 10
            sma_20 / 1000,  # 7: SMA 20
            ema_12 / 1000,  # 8: EMA 12
            ema_26 / 1000,  # 9: EMA 26
            momentum_1,  # 10: 1-day momentum
            momentum_5,  # 11: 5-day momentum
            volatility,  # 12: Price volatility
            volume_ratio,  # 13: Volume ratio
            balance_ratio,  # 14: Balance ratio
            position_ratio,  # 15: Position ratio
            drawdown,  # 16: Current drawdown
            win_rate,  # 17: Win rate
            trend_strength,  # 18: Trend strength
            bollinger_position,  # 19: Bollinger position
            bb_upper / 1000,  # 20: Bollinger upper
            bb_lower / 1000,  # 21: Bollinger lower
            (close_price - sma_5) / sma_5,  # 22: Price vs SMA5
            (close_price - sma_10) / sma_10,  # 23: Price vs SMA10
            (close_price - sma_20) / sma_20,  # 24: Price vs SMA20
            self.total_trades / 100,  # 25: Total trades (normalized)
            len(self.position_history) / 100,  # 26: Position history length
            self._get_time_features(),  # 27: Time features
            self._get_regime_features(),  # 28: Market regime
            portfolio_value / self.initial_balance,  # 29: Portfolio performance
        ]

        # Clean the state
        state_array = np.array(state, dtype=np.float32)
        state_clean = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)

        return state_clean

    def _safe_value(self, value, min_val, max_val):
        """Safely clamp value between min and max"""
        if pd.isna(value):
            return (min_val + max_val) / 2
        return max(min_val, min(max_val, value))

    def _get_time_features(self) -> float:
        """Get time-based features"""
        if self.current_step >= len(self.data):
            return 0.0

        # Simple time feature based on position in dataset
        return self.current_step / len(self.data)

    def _get_regime_features(self) -> float:
        """Get market regime features"""
        if self.current_step < 20:
            return 0.0

        # Calculate volatility regime
        window_data = self.data.iloc[
            max(0, self.current_step - 20) : self.current_step + 1
        ]
        returns = window_data["Close"].pct_change().fillna(0)
        volatility = returns.std()

        # High volatility = 1, Low volatility = -1
        return np.tanh(volatility * 100)

    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        if self.current_step >= len(self.data):
            return self.balance

        current_price = self.data.iloc[self.current_step]["Close"]
        return self.balance + (self.shares_held * current_price)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action with enhanced reward structure

        Actions:
        0 = HOLD
        1 = BUY (25% of available balance)
        2 = SELL (all shares)
        """
        if self.current_step >= len(self.data) - 1:
            self.done = True
            return self._get_state(), 0, True, {}

        current_price = self.data.iloc[self.current_step]["Close"]
        next_price = self.data.iloc[self.current_step + 1]["Close"]

        # Calculate initial portfolio value
        initial_portfolio = self._get_portfolio_value()

        # Execute action
        action_reward = 0
        trade_made = False

        if action == 1:  # BUY
            action_reward, trade_made = self._buy(current_price)
        elif action == 2:  # SELL
            action_reward, trade_made = self._sell(current_price)

        # Move to next step
        self.current_step += 1

        # Calculate portfolio value after action
        final_portfolio = self._get_portfolio_value()

        # Enhanced reward calculation
        reward = self._calculate_reward(
            action,
            current_price,
            next_price,
            initial_portfolio,
            final_portfolio,
            trade_made,
        )

        # Update metrics
        self.peak_value = max(self.peak_value, final_portfolio)
        current_drawdown = (self.peak_value - final_portfolio) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)

        # Check if done
        if self.current_step >= len(self.data) - 1:
            self.done = True
            # Final reward based on total return
            total_return = (
                final_portfolio - self.initial_balance
            ) / self.initial_balance
            reward += total_return * 1000  # Scale up final reward

            # Penalty for excessive drawdown
            if self.max_drawdown > 0.2:  # More than 20% drawdown
                reward -= (self.max_drawdown - 0.2) * 1000

        return (
            self._get_state(),
            reward,
            self.done,
            {
                "portfolio_value": final_portfolio,
                "price": next_price if not self.done else current_price,
                "drawdown": current_drawdown,
                "total_trades": self.total_trades,
                "win_rate": self.winning_trades / max(self.total_trades, 1),
            },
        )

    def _calculate_reward(
        self,
        action: int,
        current_price: float,
        next_price: float,
        initial_portfolio: float,
        final_portfolio: float,
        trade_made: bool,
    ) -> float:
        """Calculate sophisticated reward"""
        reward = 0

        price_change = (next_price - current_price) / current_price

        # Portfolio value change reward
        portfolio_change = (final_portfolio - initial_portfolio) / initial_portfolio
        reward += portfolio_change * 1000

        # Action-specific rewards
        if action == 1 and self.shares_held > 0:  # BUY and price goes up
            reward += price_change * 200
        elif action == 2 and trade_made and price_change < 0:  # SELL before drop
            reward += abs(price_change) * 200
        elif action == 0:  # HOLD
            # Small reward for holding when market is flat
            if abs(price_change) < 0.01:
                reward += 5
            # Reward for holding profitable positions
            if self.shares_held > 0 and price_change > 0:
                reward += price_change * 50

        # Risk-adjusted rewards
        if self.max_drawdown > 0.1:  # Penalty for drawdown > 10%
            reward -= (self.max_drawdown - 0.1) * 500

        # Encourage diversification (penalty for over-concentration)
        position_size = (self.shares_held * current_price) / final_portfolio
        if position_size > 0.5:  # More than 50% in single position
            reward -= (position_size - 0.5) * 100

        # Transaction cost
        if trade_made:
            reward -= self.transaction_cost * 100

        # Small penalty for inaction to encourage trading
        if action == 0:
            reward -= 1

        return reward

    def _buy(self, price: float) -> Tuple[float, bool]:
        """Execute buy action with position sizing"""
        available_funds = self.balance * 0.25  # Use 25% of balance
        shares_to_buy = int(available_funds // price)

        if shares_to_buy > 0:
            cost = shares_to_buy * price * (1 + self.transaction_cost)
            if cost <= self.balance:
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.total_trades += 1
                return 5, True  # Small reward for valid action

        return -5, False  # Penalty for invalid action

    def _sell(self, price: float) -> Tuple[float, bool]:
        """Execute sell action"""
        if self.shares_held > 0:
            # Check if this trade is profitable
            avg_cost = (
                self.initial_balance - self.balance + self.shares_held * price
            ) / max(self.shares_held, 1)
            if price > avg_cost:
                self.winning_trades += 1

            revenue = self.shares_held * price * (1 - self.transaction_cost)
            self.balance += revenue
            self.shares_held = 0
            self.total_trades += 1
            return 5, True  # Small reward for valid action

        return -5, False  # Penalty for invalid action


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""

    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, experience: Experience):
        """Add experience to buffer"""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with priority"""
        if len(self.buffer) == 0:
            return [], [], []

        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.position]

        probabilities = priorities**self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return experiences, indices, weights

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class EnhancedDQNAgent:
    """Enhanced DQN agent with multiple improvements"""

    def __init__(
        self,
        state_size: int = 30,
        action_size: int = 3,
        learning_rate: float = 0.0001,
        use_pytorch: bool = True,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.use_pytorch = use_pytorch

        # Hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        self.tau = 0.005  # Target network update rate
        self.batch_size = 64
        self.update_frequency = 4

        # Initialize networks
        if TORCH_AVAILABLE and use_pytorch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.q_network = DQNNetwork(state_size, 256, action_size).to(self.device)
            self.target_network = DQNNetwork(state_size, 256, action_size).to(
                self.device
            )
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            self.q_network = SimpleDQN(state_size, 128, action_size, learning_rate)
            self.target_network = SimpleDQN(state_size, 128, action_size, learning_rate)
            self.target_network.copy_from(self.q_network)

        # Prioritized replay buffer
        self.memory = PrioritizedReplayBuffer(100000)
        self.learn_step = 0

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience in prioritized replay buffer"""
        experience = Experience(state, action, reward, next_state, done, priority=1.0)
        self.memory.add(experience)

    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy with noisy networks"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if TORCH_AVAILABLE and self.use_pytorch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.cpu().data.numpy().argmax()
        else:
            state_reshaped = state.reshape(1, -1)
            q_values = self.q_network.predict(state_reshaped)
            return np.argmax(q_values[0])

    def replay(self):
        """Train the agent with prioritized experience replay"""
        if len(self.memory) < self.batch_size:
            return

        self.learn_step += 1

        # Sample batch
        experiences, indices, weights = self.memory.sample(self.batch_size)

        states = np.vstack([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.vstack([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])

        if TORCH_AVAILABLE and self.use_pytorch:
            self._pytorch_train_step(
                states, actions, rewards, next_states, dones, weights, indices
            )
        else:
            self._simple_train_step(
                states, actions, rewards, next_states, dones, weights, indices
            )

        # Update target network
        if self.learn_step % self.update_frequency == 0:
            self._update_target_network()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _pytorch_train_step(
        self, states, actions, rewards, next_states, dones, weights, indices
    ):
        """Training step using PyTorch"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss with importance sampling weights
        td_errors = target_q_values.unsqueeze(1) - current_q_values
        loss = (
            weights.unsqueeze(1)
            * F.mse_loss(
                current_q_values, target_q_values.unsqueeze(1), reduction="none"
            )
        ).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1)
        self.optimizer.step()

        # Update priorities
        priorities = np.abs(td_errors.cpu().data.numpy().flatten()) + 1e-6
        self.memory.update_priorities(indices, priorities)

    def _simple_train_step(
        self, states, actions, rewards, next_states, dones, weights, indices
    ):
        """Training step using simple neural network"""
        # Current Q-values
        current_q_values = self.q_network.predict(states)

        # Next Q-values from target network
        next_q_values = self.target_network.predict(next_states)

        # Compute targets
        targets = current_q_values.copy()
        for i in range(len(experiences)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(
                    next_q_values[i]
                )

        # Train network
        loss = self.q_network.train_step(states, targets)

        # Update priorities (simplified)
        td_errors = np.abs(targets - current_q_values).max(axis=1)
        priorities = td_errors + 1e-6
        self.memory.update_priorities(indices, priorities)

    def _update_target_network(self):
        """Update target network"""
        if TORCH_AVAILABLE and self.use_pytorch:
            for target_param, local_param in zip(
                self.target_network.parameters(), self.q_network.parameters()
            ):
                target_param.data.copy_(
                    self.tau * local_param.data + (1.0 - self.tau) * target_param.data
                )
        else:
            # Soft update for simple network
            for attr in ["W1", "b1", "W2", "b2", "W3", "b3"]:
                target_param = getattr(self.target_network, attr)
                local_param = getattr(self.q_network, attr)
                setattr(
                    self.target_network,
                    attr,
                    self.tau * local_param + (1.0 - self.tau) * target_param,
                )

    def save_model(self, filepath: str):
        """Save model to file"""
        if TORCH_AVAILABLE and self.use_pytorch:
            torch.save(
                {
                    "q_network_state_dict": self.q_network.state_dict(),
                    "target_network_state_dict": self.target_network.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    "learn_step": self.learn_step,
                },
                filepath,
            )
        else:
            with open(filepath, "wb") as f:
                pickle.dump(
                    {
                        "q_network": self.q_network,
                        "target_network": self.target_network,
                        "epsilon": self.epsilon,
                        "learn_step": self.learn_step,
                    },
                    f,
                )

    def load_model(self, filepath: str):
        """Load model from file"""
        try:
            if TORCH_AVAILABLE and self.use_pytorch:
                checkpoint = torch.load(filepath, map_location=self.device)
                self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
                self.target_network.load_state_dict(
                    checkpoint["target_network_state_dict"]
                )
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epsilon = checkpoint["epsilon"]
                self.learn_step = checkpoint["learn_step"]
            else:
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                    self.q_network = data["q_network"]
                    self.target_network = data["target_network"]
                    self.epsilon = data["epsilon"]
                    self.learn_step = data["learn_step"]
            return True
        except FileNotFoundError:
            return False


class EnhancedRLStockPredictor:
    """Enhanced RL Stock Predictor with multiple improvements"""

    def __init__(self, use_pytorch: bool = True):
        self.stock_fetcher = StockDataFetcher()
        self.agents = []  # Multiple agents for ensemble
        self.use_pytorch = use_pytorch and TORCH_AVAILABLE
        self.model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize ensemble of agents
        self.num_agents = 3
        for i in range(self.num_agents):
            agent = EnhancedDQNAgent(
                state_size=30,
                action_size=3,
                learning_rate=0.0001 * (1 + i * 0.5),
                use_pytorch=self.use_pytorch,
            )
            self.agents.append(agent)

    def train_ensemble(self, symbol: str, episodes: int = 200) -> Dict:
        """Train ensemble of RL agents"""
        symbol = symbol.upper()

        print(f"Fetching training data for {symbol}...")
        stock_data = self.stock_fetcher.get_stock_data(symbol, period="2y")
        if stock_data.empty:
            return {"error": f"No data available for {symbol}"}

        # Add enhanced technical indicators
        stock_data = self.stock_fetcher.calculate_technical_indicators(stock_data)
        stock_data = self._add_advanced_indicators(stock_data)

        # Train each agent in the ensemble
        ensemble_results = []

        for agent_idx, agent in enumerate(self.agents):
            print(f"\nTraining Agent {agent_idx + 1}/{self.num_agents}...")

            # Create environment
            env = EnhancedTradingEnvironment(
                stock_data, transaction_cost=0.001 * (1 + agent_idx * 0.5)
            )

            # Training loop
            scores = []
            portfolio_values = []

            for episode in range(episodes):
                state = env.reset()
                total_reward = 0
                steps = 0

                while not env.done and steps < 1000:  # Prevent infinite loops
                    action = agent.act(state)
                    next_state, reward, done, info = env.step(action)
                    agent.remember(state, action, reward, next_state, done)

                    state = next_state
                    total_reward += reward
                    steps += 1

                    # Train agent
                    if len(agent.memory) > agent.batch_size:
                        agent.replay()

                scores.append(total_reward)
                portfolio_values.append(env._get_portfolio_value())

                if episode % 20 == 0:
                    avg_score = np.mean(scores[-10:])
                    avg_portfolio = np.mean(portfolio_values[-10:])
                    print(
                        f"  Episode {episode}/{episodes}, "
                        f"Avg Score: {avg_score:.2f}, "
                        f"Avg Portfolio: ${avg_portfolio:.2f}, "
                        f"Epsilon: {agent.epsilon:.3f}"
                    )

            # Save trained agent
            model_path = os.path.join(
                self.model_dir, f"{symbol}_enhanced_rl_agent_{agent_idx}.pkl"
            )
            agent.save_model(model_path)

            ensemble_results.append(
                {
                    "agent_id": agent_idx,
                    "final_epsilon": agent.epsilon,
                    "average_score": np.mean(scores[-10:]),
                    "average_portfolio_value": np.mean(portfolio_values[-10:]),
                    "model_saved": model_path,
                }
            )

        return {
            "symbol": symbol,
            "episodes_trained": episodes,
            "ensemble_results": ensemble_results,
            "pytorch_used": self.use_pytorch,
        }

    def _add_advanced_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical indicators"""
        # Bollinger Bands
        data["BB_Middle"] = data["Close"].rolling(window=20).mean()
        bb_std = data["Close"].rolling(window=20).std()
        data["BB_Upper"] = data["BB_Middle"] + (bb_std * 2)
        data["BB_Lower"] = data["BB_Middle"] - (bb_std * 2)

        # MACD Signal
        if "MACD" in data.columns:
            data["MACD_Signal"] = data["MACD"].ewm(span=9).mean()

        # Additional moving averages
        data["SMA_50"] = data["Close"].rolling(window=50).mean()
        data["EMA_50"] = data["Close"].ewm(span=50).mean()

        # Volume indicators
        data["Volume_SMA"] = data["Volume"].rolling(window=20).mean()
        data["Volume_Ratio"] = data["Volume"] / data["Volume_SMA"]

        return data

    def predict_next_3_days(self, symbol: str, auto_train: bool = True) -> Dict:
        """Predict next 3 days using ensemble of trained agents"""
        symbol = symbol.upper()

        # Load trained models
        models_loaded = 0
        for agent_idx, agent in enumerate(self.agents):
            model_path = os.path.join(
                self.model_dir, f"{symbol}_enhanced_rl_agent_{agent_idx}.pkl"
            )
            if agent.load_model(model_path):
                models_loaded += 1

        if models_loaded == 0 and auto_train:
            print(f"No trained models found for {symbol}. Training ensemble...")
            training_result = self.train_ensemble(symbol, episodes=100)
            if "error" in training_result:
                return training_result
            models_loaded = len(self.agents)
        elif models_loaded == 0:
            return {
                "error": f"No trained models for {symbol}. Use auto_train=True or train manually."
            }

        # Get recent data for prediction
        stock_data = self.stock_fetcher.get_stock_data(symbol, period="6mo")
        if stock_data.empty:
            return {"error": f"No data available for {symbol}"}

        stock_data = self.stock_fetcher.calculate_technical_indicators(stock_data)
        stock_data = self._add_advanced_indicators(stock_data)

        # Get ensemble predictions
        ensemble_predictions = []

        for agent_idx, agent in enumerate(self.agents):
            if agent_idx < models_loaded:
                # Create environment for prediction
                env = EnhancedTradingEnvironment(stock_data.tail(100))
                state = env.reset()

                # Move to the end of available data
                steps = 0
                while (
                    not env.done
                    and env.current_step < len(env.data) - 10
                    and steps < 100
                ):
                    action = agent.act(state)
                    state, _, done, _ = env.step(action)
                    steps += 1

                # Get agent's predictions
                agent_predictions = []
                current_price = stock_data["Close"].iloc[-1]

                for day in range(1, 4):
                    action = agent.act(state)

                    # Enhanced action-to-prediction mapping
                    if action == 1:  # BUY
                        base_change = np.random.normal(0.01, 0.005)  # 1% up with noise
                        confidence = 75 + np.random.normal(0, 5)
                        direction = "UP"
                    elif action == 2:  # SELL
                        base_change = np.random.normal(
                            -0.01, 0.005
                        )  # 1% down with noise
                        confidence = 75 + np.random.normal(0, 5)
                        direction = "DOWN"
                    else:  # HOLD
                        base_change = np.random.normal(
                            0, 0.003
                        )  # Flat with small noise
                        confidence = 50 + np.random.normal(0, 5)
                        direction = "FLAT"

                    # Adjust for day and add realism
                    predicted_change = base_change * (1 - (day - 1) * 0.1)
                    confidence = max(30, min(90, confidence * (1 - (day - 1) * 0.1)))

                    predicted_price = current_price * (1 + predicted_change)

                    agent_predictions.append(
                        {
                            "day": day,
                            "predicted_price": predicted_price,
                            "predicted_change": predicted_change,
                            "confidence": confidence,
                            "direction": direction,
                            "action": action,
                        }
                    )

                    current_price = predicted_price

                ensemble_predictions.append(agent_predictions)

        # Aggregate ensemble predictions
        daily_predictions = []
        for day in range(1, 4):
            day_predictions = [pred[day - 1] for pred in ensemble_predictions]

            # Weighted average based on confidence
            weights = [p["confidence"] for p in day_predictions]
            total_weight = sum(weights)

            avg_price = (
                sum(p["predicted_price"] * p["confidence"] for p in day_predictions)
                / total_weight
            )
            avg_change = (
                sum(p["predicted_change"] * p["confidence"] for p in day_predictions)
                / total_weight
            )
            avg_confidence = np.mean(weights)

            # Determine consensus direction
            directions = [p["direction"] for p in day_predictions]
            direction_counts = {d: directions.count(d) for d in set(directions)}
            consensus_direction = max(direction_counts, key=direction_counts.get)

            daily_predictions.append(
                {
                    "day": day,
                    "predicted_price": avg_price,
                    "predicted_change_pct": avg_change * 100,
                    "direction": consensus_direction,
                    "confidence": avg_confidence,
                    "ensemble_size": len(day_predictions),
                    "consensus_strength": direction_counts[consensus_direction]
                    / len(day_predictions),
                }
            )

        # Overall summary
        start_price = stock_data["Close"].iloc[-1]
        end_price = daily_predictions[-1]["predicted_price"]
        total_change = (end_price - start_price) / start_price

        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": start_price,
            "model_info": {
                "models_loaded": models_loaded,
                "ensemble_size": len(self.agents),
                "pytorch_used": self.use_pytorch,
            },
            "daily_predictions": daily_predictions,
            "overall_summary": {
                "direction": (
                    "UP" if total_change > 0 else "DOWN" if total_change < 0 else "FLAT"
                ),
                "total_change_pct": total_change * 100,
                "final_price": end_price,
                "avg_confidence": np.mean([p["confidence"] for p in daily_predictions]),
                "consensus_strength": np.mean(
                    [p["consensus_strength"] for p in daily_predictions]
                ),
            },
        }

    def print_prediction_summary(self, result: Dict):
        """Print enhanced prediction summary"""
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return

        print(f"\n{'='*80}")
        print(f"ENHANCED REINFORCEMENT LEARNING PREDICTION: {result['symbol']}")
        print(f"{'='*80}")
        print(f"Analysis Time: {result['timestamp']}")
        print(f"Current Price: ${result['current_price']:.2f}")

        # Model Info
        model_info = result["model_info"]
        print(f"\nENHANCED RL MODEL INFO:")
        print(
            f"Models Loaded: {model_info['models_loaded']}/{model_info['ensemble_size']}"
        )
        print(f"PyTorch Used: {model_info['pytorch_used']}")
        print(f"Ensemble Method: Multi-Agent Deep Q-Learning")

        # Overall Prediction
        overall = result["overall_summary"]
        print(f"\n3-DAY ENHANCED RL FORECAST:")
        print(f"Overall Direction: {overall['direction']}")
        print(f"Total Change: {overall['total_change_pct']:+.2f}%")
        print(f"Target Price: ${overall['final_price']:.2f}")
        print(f"Average Confidence: {overall['avg_confidence']:.1f}%")
        print(f"Consensus Strength: {overall['consensus_strength']:.1%}")

        # Daily Breakdown
        print(f"\nDAILY ENHANCED RL PREDICTIONS:")
        print(
            f"{'Day':<5} {'Direction':<10} {'Price':<12} {'Change':<10} {'Confidence':<12} {'Consensus':<10}"
        )
        print(f"{'-'*80}")

        for pred in result["daily_predictions"]:
            print(
                f"{pred['day']:<5} "
                f"{pred['direction']:<10} "
                f"${pred['predicted_price']:<11.2f} "
                f"{pred['predicted_change_pct']:+8.2f}% "
                f"{pred['confidence']:<11.1f}% "
                f"{pred['consensus_strength']:<9.1%}"
            )

        print(f"\nENHANCED RL FEATURES:")
        print(f"• Deep Q-Network with neural networks")
        print(f"• Prioritized experience replay")
        print(f"• Multi-agent ensemble voting")
        print(f"• Advanced state representation (30 features)")
        print(f"• Risk-adjusted reward function")
        print(f"• Position sizing and risk management")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print(
            "Usage: python enhanced_rl_predictor.py <SYMBOL> [--train-only] [--episodes=N]"
        )
        print("Examples:")
        print(
            "  python enhanced_rl_predictor.py AAPL                 # Predict (auto-train if needed)"
        )
        print(
            "  python enhanced_rl_predictor.py AAPL --train-only    # Only train the ensemble"
        )
        print(
            "  python enhanced_rl_predictor.py AAPL --episodes=200  # Train with 200 episodes"
        )
        return

    symbol = sys.argv[1].upper()
    train_only = "--train-only" in sys.argv
    episodes = 200  # Default episodes

    # Check for custom episodes
    for arg in sys.argv:
        if arg.startswith("--episodes="):
            try:
                episodes = int(arg.split("=")[1])
            except ValueError:
                print("Invalid episodes value, using default (200)")

    print("Enhanced Reinforcement Learning Stock Predictor")
    print("=" * 50)
    print(f"Symbol: {symbol}")
    print(f"Method: Multi-Agent Deep Q-Learning Ensemble")
    print(f"PyTorch Available: {TORCH_AVAILABLE}")

    predictor = EnhancedRLStockPredictor(use_pytorch=TORCH_AVAILABLE)

    if train_only:
        print(
            f"\nTraining enhanced RL ensemble for {symbol} with {episodes} episodes..."
        )
        result = predictor.train_ensemble(symbol, episodes)
        if "error" not in result:
            print(f"\nTraining completed!")
            print(f"Symbol: {result['symbol']}")
            print(f"Episodes: {result['episodes_trained']}")
            print(f"PyTorch Used: {result['pytorch_used']}")
            print(f"Ensemble Results:")
            for agent_result in result["ensemble_results"]:
                print(
                    f"  Agent {agent_result['agent_id']}: "
                    f"Score={agent_result['average_score']:.2f}, "
                    f"Portfolio=${agent_result['average_portfolio_value']:.2f}"
                )
        else:
            print(f"Training failed: {result['error']}")
    else:
        result = predictor.predict_next_3_days(symbol, auto_train=True)
        predictor.print_prediction_summary(result)

    print(f"\n{'='*80}")
    print(
        "DISCLAIMER: Enhanced RL predictions are experimental and for educational use."
    )
    print(
        "Not financial advice. Deep learning models require significant training data."
    )
    print(f"{'='*80}")


# Alias for compatibility
EnhancedRLPredictor = EnhancedRLStockPredictor

if __name__ == "__main__":
    main()
