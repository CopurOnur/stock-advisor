#!/usr/bin/env python3
"""
Hybrid Stock Price Predictor

Combines three prediction approaches for enhanced accuracy:
1. Reinforcement Learning (Q-learning with extended state space)
2. Technical Analysis (traditional indicators and patterns)
3. News Sentiment Analysis (market sentiment and stock-specific news)

The hybrid approach uses ensemble voting and confidence weighting to
generate final predictions with higher reliability.
"""

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from stock_advisor.core.stock_data import StockDataFetcher
    from stock_advisor.core.news_collector import NewsCollector
except ImportError:
    # Fallback to relative imports if running from within the package
    from ..core.stock_data import StockDataFetcher
    from ..core.news_collector import NewsCollector


class EnhancedTradingEnvironment:
    """Enhanced trading environment with news sentiment integration"""
    
    def __init__(self, data: pd.DataFrame, news_sentiment: Optional[Dict] = None, 
                 initial_balance: float = 10000):
        self.data = data.reset_index(drop=True)
        self.news_sentiment = news_sentiment or {}
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
        return self._get_enhanced_state()
    
    def _get_enhanced_state(self) -> np.ndarray:
        """Get enhanced state with news sentiment"""
        if self.current_step >= len(self.data):
            return np.zeros(15)
        
        current = self.data.iloc[self.current_step]
        
        # Technical indicators state
        state = [
            current.get('Close', 0) / 1000,  # Normalized price
            current.get('Volume', 0) / 1e6,  # Normalized volume
            current.get('RSI', 50) / 100,    # RSI (0-1)
            current.get('MACD', 0),          # MACD
            current.get('SMA_5', 0) / 1000,  # 5-day MA
            current.get('SMA_10', 0) / 1000, # 10-day MA
            current.get('SMA_20', 0) / 1000, # 20-day MA
            self.balance / self.initial_balance,  # Balance ratio
            self.shares_held / 100,          # Shares held (normalized)
            self._get_portfolio_value() / self.initial_balance,  # Portfolio ratio
        ]
        
        # Add news sentiment features
        sentiment_score = self.news_sentiment.get('score', 0)
        sentiment_confidence = self.news_sentiment.get('confidence', 0.5)
        positive_keywords = self.news_sentiment.get('positive_keywords', 0)
        negative_keywords = self.news_sentiment.get('negative_keywords', 0)
        
        news_features = [
            sentiment_score,                    # News sentiment score (-1 to 1)
            sentiment_confidence,               # Confidence in sentiment
            positive_keywords / 10,             # Normalized positive keywords
            negative_keywords / 10,             # Normalized negative keywords
            1 if sentiment_score > 0.1 else 0, # Strong positive sentiment flag
        ]
        
        state.extend(news_features)
        return np.array(state, dtype=np.float32)
    
    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        if self.current_step >= len(self.data):
            return self.balance
        
        current_price = self.data.iloc[self.current_step]['Close']
        return self.balance + (self.shares_held * current_price)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action with enhanced reward system"""
        if self.current_step >= len(self.data) - 1:
            self.done = True
            return self._get_enhanced_state(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['Close']
        next_price = self.data.iloc[self.current_step + 1]['Close']
        
        # Execute action
        reward = 0
        if action == 1:  # BUY
            reward = self._buy(current_price)
        elif action == 2:  # SELL
            reward = self._sell(current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Enhanced reward calculation
        price_change = (next_price - current_price) / current_price
        sentiment_score = self.news_sentiment.get('score', 0)
        
        # Reward based on action alignment with news sentiment
        if action == 1 and sentiment_score > 0.1:  # Buy with positive sentiment
            reward += sentiment_score * 50
        elif action == 2 and sentiment_score < -0.1:  # Sell with negative sentiment
            reward += abs(sentiment_score) * 50
        
        # Standard price movement rewards
        if action == 1 and self.shares_held > 0:
            reward += price_change * 100
        elif action == 2 and price_change < 0:
            reward += abs(price_change) * 100
        elif action == 0 and self.shares_held > 0:
            reward += price_change * 50
        
        # Portfolio performance reward
        current_net_worth = self._get_portfolio_value()
        if current_net_worth > self.max_net_worth:
            reward += (current_net_worth - self.max_net_worth) / self.initial_balance * 100
            self.max_net_worth = current_net_worth
        
        # Penalty for inaction when there are strong signals
        if action == 0 and abs(sentiment_score) > 0.3:
            reward -= 5  # Penalty for ignoring strong sentiment
        
        # Check if done
        if self.current_step >= len(self.data) - 1:
            self.done = True
            final_return = (current_net_worth - self.initial_balance) / self.initial_balance
            reward += final_return * 1000
        
        return self._get_enhanced_state(), reward, self.done, {
            'portfolio_value': current_net_worth,
            'price': next_price if not self.done else current_price,
            'sentiment_used': sentiment_score
        }
    
    def _buy(self, price: float) -> float:
        """Execute buy action"""
        shares_to_buy = self.balance // price
        if shares_to_buy > 0:
            cost = shares_to_buy * price
            self.balance -= cost
            self.shares_held += shares_to_buy
            return 1
        return -1
    
    def _sell(self, price: float) -> float:
        """Execute sell action"""
        if self.shares_held > 0:
            revenue = self.shares_held * price
            self.balance += revenue
            self.total_shares_sold += self.shares_held
            self.shares_held = 0
            return 1
        return -1


class TechnicalAnalysisModule:
    """Technical analysis component"""
    
    def __init__(self):
        pass
    
    def analyze_technical_signals(self, data: pd.DataFrame) -> Dict:
        """Comprehensive technical analysis"""
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest
        
        signals = {
            'direction_votes': [],
            'confidence_scores': [],
            'signal_explanations': []
        }
        
        # Moving Average Analysis
        if 'SMA_5' in data.columns:
            if latest['Close'] > latest['SMA_5'] > latest['SMA_10'] > latest['SMA_20']:
                signals['direction_votes'].append(1)
                signals['confidence_scores'].append(0.8)
                signals['signal_explanations'].append("Strong bullish MA alignment")
            elif latest['Close'] < latest['SMA_5'] < latest['SMA_10'] < latest['SMA_20']:
                signals['direction_votes'].append(-1)
                signals['confidence_scores'].append(0.8)
                signals['signal_explanations'].append("Strong bearish MA alignment")
            
            # Golden/Death Cross
            if prev['SMA_5'] <= prev['SMA_10'] and latest['SMA_5'] > latest['SMA_10']:
                signals['direction_votes'].append(1)
                signals['confidence_scores'].append(0.9)
                signals['signal_explanations'].append("Golden cross detected")
            elif prev['SMA_5'] >= prev['SMA_10'] and latest['SMA_5'] < latest['SMA_10']:
                signals['direction_votes'].append(-1)
                signals['confidence_scores'].append(0.9)
                signals['signal_explanations'].append("Death cross detected")
        
        # RSI Analysis
        if 'RSI' in data.columns:
            rsi = latest['RSI']
            if rsi < 30:
                signals['direction_votes'].append(1)
                signals['confidence_scores'].append(0.7)
                signals['signal_explanations'].append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                signals['direction_votes'].append(-1)
                signals['confidence_scores'].append(0.7)
                signals['signal_explanations'].append(f"RSI overbought ({rsi:.1f})")
        
        # MACD Analysis
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd = latest['MACD']
            signal_line = latest['MACD_Signal']
            prev_macd = prev['MACD']
            prev_signal = prev['MACD_Signal']
            
            if prev_macd <= prev_signal and macd > signal_line:
                signals['direction_votes'].append(1)
                signals['confidence_scores'].append(0.8)
                signals['signal_explanations'].append("MACD bullish crossover")
            elif prev_macd >= prev_signal and macd < signal_line:
                signals['direction_votes'].append(-1)
                signals['confidence_scores'].append(0.8)
                signals['signal_explanations'].append("MACD bearish crossover")
        
        # Volume Analysis
        recent_volume = data['Volume'].tail(5).mean()
        longer_volume = data['Volume'].tail(20).mean()
        volume_ratio = recent_volume / longer_volume
        
        if volume_ratio > 1.5:
            # High volume confirms existing trend
            price_trend = 1 if latest['Close'] > prev['Close'] else -1
            signals['direction_votes'].append(price_trend)
            signals['confidence_scores'].append(0.4)
            signals['signal_explanations'].append("High volume confirms trend")
        
        return signals
    
    def predict_technical_direction(self, data: pd.DataFrame) -> Tuple[int, float, List[str]]:
        """Predict direction based on technical analysis"""
        signals = self.analyze_technical_signals(data)
        
        if not signals['direction_votes']:
            return 0, 0.3, ["No strong technical signals"]
        
        # Weighted voting
        weighted_sum = sum(vote * conf for vote, conf in 
                          zip(signals['direction_votes'], signals['confidence_scores']))
        total_weight = sum(signals['confidence_scores'])
        
        if total_weight > 0:
            direction_score = weighted_sum / total_weight
            confidence = min(0.9, total_weight / len(signals['direction_votes']))
            
            if direction_score > 0.2:
                return 1, confidence, signals['signal_explanations']
            elif direction_score < -0.2:
                return -1, confidence, signals['signal_explanations']
        
        return 0, 0.4, signals['signal_explanations']


class NewsAnalysisModule:
    """News sentiment analysis component"""
    
    def __init__(self):
        self.news_collector = NewsCollector()
    
    def analyze_news_sentiment(self, symbol: str, days_back: int = 7) -> Dict:
        """Analyze news sentiment for stock"""
        try:
            news_data = self.news_collector.get_comprehensive_news([symbol], days_back)
            sentiment = news_data['sentiment_summary'].get(symbol, {})
            market_sentiment = news_data['sentiment_summary'].get('MARKET', {})
            
            # Combine stock-specific and market sentiment
            stock_score = sentiment.get('score', 0)
            market_score = market_sentiment.get('score', 0)
            
            # Weight stock sentiment more heavily
            combined_score = (stock_score * 0.7) + (market_score * 0.3)
            
            return {
                'stock_sentiment': sentiment,
                'market_sentiment': market_sentiment,
                'combined_score': combined_score,
                'confidence': sentiment.get('confidence', 0.5),
                'positive_keywords': sentiment.get('positive_keywords', 0),
                'negative_keywords': sentiment.get('negative_keywords', 0),
                'news_articles_count': len(news_data['stock_specific'].get(symbol, []))
            }
        except Exception as e:
            print(f"News analysis failed: {e}")
            return {
                'combined_score': 0,
                'confidence': 0,
                'positive_keywords': 0,
                'negative_keywords': 0,
                'news_articles_count': 0
            }
    
    def predict_news_direction(self, news_analysis: Dict) -> Tuple[int, float, str]:
        """Predict direction based on news sentiment"""
        score = news_analysis.get('combined_score', 0)
        confidence = news_analysis.get('confidence', 0)
        articles_count = news_analysis.get('news_articles_count', 0)
        
        # Adjust confidence based on number of articles
        if articles_count < 3:
            confidence *= 0.5  # Lower confidence with few articles
        elif articles_count > 10:
            confidence *= 1.2  # Higher confidence with more articles
        
        confidence = min(confidence, 0.9)
        
        if score > 0.1:
            return 1, confidence, f"Positive news sentiment ({score:.3f})"
        elif score < -0.1:
            return -1, confidence, f"Negative news sentiment ({score:.3f})"
        else:
            return 0, 0.3, f"Neutral news sentiment ({score:.3f})"


class HybridDQNAgent:
    """Enhanced DQN agent with expanded state space"""
    
    def __init__(self, state_size: int = 15, action_size: int = 3, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=3000)  # Increased memory
        self.q_table = {}
        
    def _get_state_key(self, state: np.ndarray) -> str:
        """Convert state to string key with better discretization"""
        # Handle NaN and infinite values
        state_clean = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # More granular discretization for better learning
        discretized = np.round(state_clean * 20).astype(int)
        return str(discretized.tolist())
    
    def remember(self, state, action, reward, next_state, done):
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
    
    def replay(self, batch_size: int = 64):
        """Enhanced training with larger batch size"""
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
        """Save model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon
            }, f)
    
    def load_model(self, filepath: str):
        """Load model"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data['epsilon']
            return True
        except FileNotFoundError:
            return False


class HybridStockPredictor:
    """Main hybrid predictor combining RL, technical analysis, and news sentiment"""
    
    def __init__(self):
        self.stock_fetcher = StockDataFetcher()
        self.technical_module = TechnicalAnalysisModule()
        self.news_module = NewsAnalysisModule()
        self.rl_agent = HybridDQNAgent()
        self.model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def train_hybrid_agent(self, symbol: str, episodes: int = 100) -> Dict:
        """Train RL agent with enhanced environment"""
        symbol = symbol.upper()
        
        print(f"Fetching training data for {symbol}...")
        stock_data = self.stock_fetcher.get_stock_data(symbol, period="1y")
        if stock_data.empty:
            return {'error': f'No data available for {symbol}'}
        
        stock_data = self.stock_fetcher.calculate_technical_indicators(stock_data)
        
        # Get news sentiment for training period
        print("Analyzing news sentiment for training...")
        news_analysis = self.news_module.analyze_news_sentiment(symbol, days_back=30)
        
        # Create enhanced environment
        env = EnhancedTradingEnvironment(stock_data, news_analysis)
        
        scores = []
        portfolio_values = []
        
        print(f"Training hybrid agent for {episodes} episodes...")
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            while not env.done:
                action = self.rl_agent.act(state)
                next_state, reward, done, info = env.step(action)
                self.rl_agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(self.rl_agent.memory) > 64:
                    self.rl_agent.replay(64)
            
            scores.append(total_reward)
            portfolio_values.append(env._get_portfolio_value())
            
            if episode % 20 == 0:
                avg_score = np.mean(scores[-20:])
                avg_portfolio = np.mean(portfolio_values[-20:])
                print(f"Episode {episode}/{episodes}, "
                      f"Avg Score: {avg_score:.2f}, "
                      f"Avg Portfolio: ${avg_portfolio:.2f}, "
                      f"Epsilon: {self.rl_agent.epsilon:.3f}")
        
        # Save model
        model_path = os.path.join(self.model_dir, f'{symbol}_hybrid_model.pkl')
        self.rl_agent.save_model(model_path)
        
        return {
            'episodes_trained': episodes,
            'final_epsilon': self.rl_agent.epsilon,
            'average_score': np.mean(scores[-20:]),
            'model_saved': model_path
        }
    
    def predict_next_3_days(self, symbol: str, auto_train: bool = True, 
                           use_news: bool = True) -> Dict:
        """
        Hybrid prediction combining all three approaches
        
        Args:
            symbol: Stock symbol
            auto_train: Train RL model if not exists
            use_news: Include news sentiment analysis
        """
        symbol = symbol.upper()
        
        # Load or train RL model
        model_path = os.path.join(self.model_dir, f'{symbol}_hybrid_model.pkl')
        model_loaded = self.rl_agent.load_model(model_path)
        
        if not model_loaded and auto_train:
            print(f"Training new hybrid model for {symbol}...")
            training_result = self.train_hybrid_agent(symbol, episodes=75)
            if 'error' in training_result:
                return training_result
        elif not model_loaded:
            return {'error': f'No trained model for {symbol}. Use auto_train=True.'}
        
        # Get stock data
        stock_data = self.stock_fetcher.get_stock_data(symbol, period="3mo")
        if stock_data.empty:
            return {'error': f'No data available for {symbol}'}
        
        stock_data = self.stock_fetcher.calculate_technical_indicators(stock_data)
        
        # Get news analysis
        news_analysis = {}
        if use_news:
            print("Analyzing news sentiment...")
            news_analysis = self.news_module.analyze_news_sentiment(symbol, days_back=7)
        
        # Get predictions from each module
        predictions = []
        current_price = stock_data['Close'].iloc[-1]
        
        for day in range(1, 4):
            day_prediction = self._predict_single_day(
                day, current_price, stock_data, news_analysis
            )
            predictions.append(day_prediction)
            current_price = day_prediction['predicted_price']
        
        # Overall summary
        total_change = (predictions[-1]['predicted_price'] - stock_data['Close'].iloc[-1]) / stock_data['Close'].iloc[-1]
        overall_direction = 'UP' if total_change > 0 else 'DOWN' if total_change < 0 else 'FLAT'
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'current_price': stock_data['Close'].iloc[-1],
            'news_analysis_included': use_news,
            'model_info': {
                'model_loaded': model_loaded,
                'epsilon': self.rl_agent.epsilon,
                'q_table_size': len(self.rl_agent.q_table)
            },
            'news_summary': news_analysis if use_news else {},
            'daily_predictions': predictions,
            'overall_summary': {
                'direction': overall_direction,
                'total_change_pct': total_change * 100,
                'final_price': predictions[-1]['predicted_price'],
                'avg_confidence': np.mean([p['confidence'] for p in predictions])
            }
        }
    
    def _predict_single_day(self, day: int, current_price: float, 
                           stock_data: pd.DataFrame, news_analysis: Dict) -> Dict:
        """Generate single day prediction using hybrid approach"""
        
        # 1. Technical Analysis Prediction
        tech_direction, tech_confidence, tech_signals = self.technical_module.predict_technical_direction(stock_data)
        
        # 2. News Sentiment Prediction
        news_direction, news_confidence, news_explanation = self.news_module.predict_news_direction(news_analysis)
        
        # 3. RL Agent Prediction
        env = EnhancedTradingEnvironment(stock_data.tail(60), news_analysis)
        state = env.reset()
        
        # Move to end of data
        while not env.done and env.current_step < len(env.data) - 4:
            action = self.rl_agent.act(state)
            state, _, done, _ = env.step(action)
        
        rl_action = self.rl_agent.act(state)
        rl_direction = 1 if rl_action == 1 else -1 if rl_action == 2 else 0
        rl_confidence = 0.6  # Base RL confidence
        
        # 4. Ensemble Prediction
        predictions = [
            ('Technical Analysis', tech_direction, tech_confidence),
            ('News Sentiment', news_direction, news_confidence),
            ('Reinforcement Learning', rl_direction, rl_confidence)
        ]
        
        # Weighted ensemble
        total_weighted_score = 0
        total_weight = 0
        prediction_details = []
        
        for method, direction, confidence in predictions:
            weight = confidence
            total_weighted_score += direction * weight
            total_weight += weight
            
            prediction_details.append({
                'method': method,
                'direction': 'UP' if direction > 0 else 'DOWN' if direction < 0 else 'FLAT',
                'confidence': confidence * 100
            })
        
        # Final ensemble decision
        if total_weight > 0:
            ensemble_score = total_weighted_score / total_weight
            ensemble_confidence = min(90, (total_weight / 3) * 80)  # Scale confidence
        else:
            ensemble_score = 0
            ensemble_confidence = 30
        
        # Convert to price prediction
        if ensemble_score > 0.3:
            direction = 'UP'
            predicted_change = 0.005 + (ensemble_score * 0.015)  # 0.5% to 2%
        elif ensemble_score < -0.3:
            direction = 'DOWN'
            predicted_change = -0.005 + (ensemble_score * 0.015)  # -0.5% to -2%
        else:
            direction = 'FLAT'
            predicted_change = ensemble_score * 0.005  # Small movement
        
        # Adjust for day (uncertainty increases)
        predicted_change *= (1 - (day - 1) * 0.2)
        ensemble_confidence *= (1 - (day - 1) * 0.15)
        
        predicted_price = current_price * (1 + predicted_change)
        
        return {
            'day': day,
            'predicted_price': predicted_price,
            'predicted_change_pct': predicted_change * 100,
            'direction': direction,
            'confidence': max(25, ensemble_confidence),
            'ensemble_score': ensemble_score,
            'method_predictions': prediction_details,
            'technical_signals': tech_signals,
            'news_explanation': news_explanation if news_analysis else "News analysis disabled",
            'rl_action': ['HOLD', 'BUY', 'SELL'][rl_action]
        }
    
    def print_prediction_summary(self, result: Dict):
        """Print comprehensive prediction summary"""
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
        
        print(f"\n{'='*80}")
        print(f"HYBRID PREDICTION (RL + Technical + News): {result['symbol']}")
        print(f"{'='*80}")
        print(f"Analysis Time: {result['timestamp']}")
        print(f"Current Price: ${result['current_price']:.2f}")
        print(f"News Analysis: {'Included' if result['news_analysis_included'] else 'Disabled'}")
        
        # Model Info
        model_info = result['model_info']
        print(f"\nRL MODEL INFO:")
        print(f"Status: {'Loaded' if model_info['model_loaded'] else 'Newly Trained'}")
        print(f"Exploration Rate: {model_info['epsilon']:.3f}")
        print(f"States Learned: {model_info['q_table_size']}")
        
        # News Summary
        if result['news_analysis_included'] and result['news_summary']:
            news = result['news_summary']
            print(f"\nNEWS SENTIMENT:")
            print(f"Combined Score: {news.get('combined_score', 0):.3f}")
            print(f"Articles Analyzed: {news.get('news_articles_count', 0)}")
            print(f"Positive Keywords: {news.get('positive_keywords', 0)}")
            print(f"Negative Keywords: {news.get('negative_keywords', 0)}")
        
        # Overall Prediction
        overall = result['overall_summary']
        print(f"\n3-DAY HYBRID FORECAST:")
        print(f"Overall Direction: {overall['direction']}")
        print(f"Total Change: {overall['total_change_pct']:+.2f}%")
        print(f"Target Price: ${overall['final_price']:.2f}")
        print(f"Average Confidence: {overall['avg_confidence']:.1f}%")
        
        # Daily Breakdown
        print(f"\nDAILY HYBRID PREDICTIONS:")
        print(f"{'Day':<5} {'Direction':<10} {'Price':<12} {'Change':<10} {'Confidence':<12} {'Ensemble':<10}")
        print(f"{'-'*75}")
        
        for pred in result['daily_predictions']:
            print(f"{pred['day']:<5} "
                  f"{pred['direction']:<10} "
                  f"${pred['predicted_price']:<11.2f} "
                  f"{pred['predicted_change_pct']:+8.2f}% "
                  f"{pred['confidence']:<11.1f}% "
                  f"{pred['ensemble_score']:+8.3f}")
        
        # Method Breakdown
        print(f"\nMETHOD CONTRIBUTIONS:")
        for pred in result['daily_predictions']:
            print(f"\nDay {pred['day']} Details:")
            print(f"  RL Action: {pred['rl_action']}")
            print(f"  Technical: {pred['technical_signals'][:2] if pred['technical_signals'] else 'No signals'}")
            print(f"  News: {pred['news_explanation']}")
            
            print(f"  Method Predictions:")
            for method_pred in pred['method_predictions']:
                print(f"    {method_pred['method']}: {method_pred['direction']} "
                      f"({method_pred['confidence']:.1f}%)")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python hybrid_predictor.py <SYMBOL> [--no-news] [--train-only] [--episodes=N]")
        print("Examples:")
        print("  python hybrid_predictor.py AAPL                    # Full hybrid prediction")
        print("  python hybrid_predictor.py AAPL --no-news          # Without news analysis")
        print("  python hybrid_predictor.py AAPL --train-only       # Only train the model")
        print("  python hybrid_predictor.py AAPL --episodes=100     # Train with 100 episodes")
        return
    
    symbol = sys.argv[1].upper()
    use_news = '--no-news' not in sys.argv
    train_only = '--train-only' in sys.argv
    episodes = 75  # Default episodes
    
    # Check for custom episodes
    for arg in sys.argv:
        if arg.startswith('--episodes='):
            try:
                episodes = int(arg.split('=')[1])
            except ValueError:
                print("Invalid episodes value, using default (75)")
    
    print("Hybrid Stock Predictor")
    print("=====================")
    print(f"Symbol: {symbol}")
    print(f"Methods: RL + Technical Analysis + {'News Sentiment' if use_news else 'No News'}")
    
    predictor = HybridStockPredictor()
    
    if train_only:
        print(f"\nTraining hybrid model for {symbol} with {episodes} episodes...")
        result = predictor.train_hybrid_agent(symbol, episodes)
        if 'error' not in result:
            print(f"\nTraining completed!")
            print(f"Episodes: {result['episodes_trained']}")
            print(f"Final exploration rate: {result['final_epsilon']:.3f}")
            print(f"Model saved: {result['model_saved']}")
        else:
            print(f"Training failed: {result['error']}")
    else:
        result = predictor.predict_next_3_days(symbol, auto_train=True, use_news=use_news)
        predictor.print_prediction_summary(result)
    
    print(f"\n{'='*80}")
    print("DISCLAIMER: Hybrid predictions combine multiple approaches but are")
    print("still experimental. Not financial advice. Use for educational purposes only.")
    print(f"{'='*80}")


# Alias for compatibility
HybridPredictor = HybridStockPredictor

if __name__ == "__main__":
    main()