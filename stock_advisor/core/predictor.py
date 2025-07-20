import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        self.scalers = {}
        self.trained_models = {}
        self.feature_importance = {}
    
    def prepare_features(self, data: pd.DataFrame, news_sentiment: Optional[Dict] = None) -> pd.DataFrame:
        """
        Prepare features for machine learning model
        
        Args:
            data: Stock data with technical indicators
            news_sentiment: Optional sentiment data from news
        
        Returns:
            DataFrame with prepared features
        """
        df = data.copy()
        
        # Price-based features
        df['price_change'] = df['Close'].pct_change()
        df['volume_change'] = df['Volume'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_to_high'] = df['Close'] / df['High']
        df['close_to_low'] = df['Close'] / df['Low']
        
        # Volatility
        df['volatility'] = df['price_change'].rolling(window=5).std()
        
        # Volume indicators
        df['volume_sma'] = df['Volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_3'] = df['Close'] / df['Close'].shift(3) - 1
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        
        # Technical indicator ratios
        if 'SMA_5' in df.columns:
            df['price_to_sma5'] = df['Close'] / df['SMA_5']
            df['price_to_sma10'] = df['Close'] / df['SMA_10']
            df['price_to_sma20'] = df['Close'] / df['SMA_20']
        
        # Add news sentiment if available
        if news_sentiment:
            sentiment_score = news_sentiment.get('score', 0)
            df['news_sentiment'] = sentiment_score
            df['sentiment_positive'] = 1 if sentiment_score > 0 else 0
            df['sentiment_negative'] = 1 if sentiment_score < 0 else 0
        else:
            df['news_sentiment'] = 0
            df['sentiment_positive'] = 0
            df['sentiment_negative'] = 0
        
        # Target variable (next day's price change)
        df['target'] = df['Close'].shift(-1) / df['Close'] - 1
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> List[str]:
        """
        Select relevant features for training
        
        Args:
            df: DataFrame with all features
        
        Returns:
            List of selected feature names
        """
        feature_columns = [
            'price_change', 'volume_change', 'high_low_ratio', 'close_to_high', 'close_to_low',
            'volatility', 'volume_ratio', 'momentum_3', 'momentum_5', 'momentum_10',
            'news_sentiment', 'sentiment_positive', 'sentiment_negative'
        ]
        
        # Add technical indicators if they exist
        tech_indicators = ['RSI', 'MACD', 'MACD_Signal', 'price_to_sma5', 'price_to_sma10', 'price_to_sma20']
        for indicator in tech_indicators:
            if indicator in df.columns:
                feature_columns.append(indicator)
        
        # Filter out columns that actually exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        
        return available_features
    
    def train_models(self, data: pd.DataFrame, news_sentiment: Optional[Dict] = None) -> Dict:
        """
        Train prediction models
        
        Args:
            data: Stock data with technical indicators
            news_sentiment: Optional sentiment data
        
        Returns:
            Dictionary with training results
        """
        # Prepare features
        df = self.prepare_features(data, news_sentiment)
        feature_columns = self.select_features(df)
        
        # Remove rows with NaN values
        df_clean = df[feature_columns + ['target']].dropna()
        
        if len(df_clean) < 20:
            return {'error': 'Insufficient data for training (need at least 20 samples)'}
        
        X = df_clean[feature_columns]
        y = df_clean['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        results = {}
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'linear_regression':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[model_name] = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'predictions': y_pred.tolist()
                }
                
                # Store trained model
                self.trained_models[model_name] = model
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    feature_imp = dict(zip(feature_columns, model.feature_importances_))
                    self.feature_importance[model_name] = feature_imp
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        return results
    
    def predict_next_price(self, current_data: pd.DataFrame, news_sentiment: Optional[Dict] = None, 
                          model_name: str = 'random_forest', days: int = 1) -> Dict:
        """
        Predict price changes for the next N days
        
        Args:
            current_data: Recent stock data
            news_sentiment: Current news sentiment
            model_name: Which model to use for prediction
            days: Number of days to predict (1-7)
        
        Returns:
            Dictionary with prediction results
        """
        if model_name not in self.trained_models:
            return {'error': f'Model {model_name} not trained yet'}
        
        # Limit days to reasonable range
        days = min(max(days, 1), 7)
        
        try:
            # Prepare features
            df = self.prepare_features(current_data, news_sentiment)
            feature_columns = self.select_features(df)
            
            # Get the latest row
            latest_data = df[feature_columns].iloc[-1:].dropna()
            
            if latest_data.empty:
                return {'error': 'No valid data for prediction'}
            
            model = self.trained_models[model_name]
            current_price = current_data['Close'].iloc[-1]
            
            # Multi-day prediction
            predictions = []
            cumulative_change = 0
            
            for day in range(days):
                # For day 1, use actual latest data
                if day == 0:
                    input_data = latest_data
                else:
                    # For subsequent days, create synthetic data based on previous predictions
                    # This is a simplified approach - in reality, you'd want to retrain or use more sophisticated methods
                    synthetic_data = latest_data.copy()
                    
                    # Adjust some features based on predicted price movement
                    if 'price_change' in synthetic_data.columns:
                        synthetic_data['price_change'] = predictions[day-1]['predicted_change']
                    
                    # Add some uncertainty/noise for longer predictions
                    noise_factor = 0.1 * day  # Increase uncertainty with time
                    for col in synthetic_data.columns:
                        if col in ['news_sentiment', 'sentiment_positive', 'sentiment_negative']:
                            continue  # Keep sentiment stable
                        synthetic_data[col] *= (1 + np.random.normal(0, noise_factor))
                    
                    input_data = synthetic_data
                
                # Make prediction
                if model_name == 'linear_regression' and 'main' in self.scalers:
                    input_scaled = self.scalers['main'].transform(input_data)
                    daily_prediction = model.predict(input_scaled)[0]
                else:
                    daily_prediction = model.predict(input_data)[0]
                
                # Calculate cumulative effect
                cumulative_change += daily_prediction
                predicted_price = current_price * (1 + cumulative_change)
                
                # Convert to direction and confidence
                direction = 'UP' if daily_prediction > 0 else 'DOWN'
                confidence = abs(daily_prediction) * 100 * (1 - 0.1 * day)  # Decrease confidence over time
                
                predictions.append({
                    'day': day + 1,
                    'predicted_change': daily_prediction,
                    'cumulative_change': cumulative_change,
                    'predicted_price': predicted_price,
                    'direction': direction,
                    'confidence': min(max(confidence, 0), 100)  # Cap between 0-100%
                })
            
            # Overall multi-day summary
            total_change = cumulative_change
            final_price = current_price * (1 + total_change)
            overall_direction = 'UP' if total_change > 0 else 'DOWN'
            avg_confidence = np.mean([p['confidence'] for p in predictions])
            
            return {
                'current_price': current_price,
                'days_predicted': days,
                'daily_predictions': predictions,
                'overall_summary': {
                    'total_change': total_change,
                    'final_predicted_price': final_price,
                    'overall_direction': overall_direction,
                    'average_confidence': avg_confidence
                },
                'model_used': model_name
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_ensemble_prediction(self, current_data: pd.DataFrame, news_sentiment: Optional[Dict] = None, days: int = 1) -> Dict:
        """
        Get ensemble prediction from all trained models for multiple days
        
        Args:
            current_data: Recent stock data
            news_sentiment: Current news sentiment
            days: Number of days to predict
        
        Returns:
            Dictionary with ensemble prediction
        """
        individual_predictions = {}
        daily_ensemble_data = []
        
        for model_name in self.trained_models.keys():
            result = self.predict_next_price(current_data, news_sentiment, model_name, days)
            if 'error' not in result:
                individual_predictions[model_name] = result
        
        if not individual_predictions:
            return {'error': 'No models available for ensemble prediction'}
        
        # Calculate ensemble for each day
        for day in range(days):
            day_predictions = []
            day_confidences = []
            
            for model_name, prediction in individual_predictions.items():
                if 'daily_predictions' in prediction and day < len(prediction['daily_predictions']):
                    day_pred = prediction['daily_predictions'][day]
                    day_predictions.append(day_pred['predicted_change'])
                    day_confidences.append(day_pred['confidence'])
            
            if day_predictions:
                avg_change = np.mean(day_predictions)
                std_change = np.std(day_predictions)
                avg_confidence = np.mean(day_confidences)
                
                daily_ensemble_data.append({
                    'day': day + 1,
                    'predicted_change': avg_change,
                    'confidence': avg_confidence,
                    'uncertainty': std_change * 100,
                    'num_models': len(day_predictions)
                })
        
        # Calculate overall ensemble metrics
        current_price = current_data['Close'].iloc[-1]
        
        # Calculate cumulative change
        cumulative_change = 0
        ensemble_daily_predictions = []
        
        for day_data in daily_ensemble_data:
            cumulative_change += day_data['predicted_change']
            predicted_price = current_price * (1 + cumulative_change)
            direction = 'UP' if day_data['predicted_change'] > 0 else 'DOWN'
            
            ensemble_daily_predictions.append({
                'day': day_data['day'],
                'predicted_change': day_data['predicted_change'],
                'cumulative_change': cumulative_change,
                'predicted_price': predicted_price,
                'direction': direction,
                'confidence': day_data['confidence'],
                'uncertainty': day_data['uncertainty']
            })
        
        final_price = current_price * (1 + cumulative_change)
        overall_direction = 'UP' if cumulative_change > 0 else 'DOWN'
        avg_confidence = np.mean([p['confidence'] for p in ensemble_daily_predictions])
        avg_uncertainty = np.mean([p['uncertainty'] for p in ensemble_daily_predictions])
        
        return {
            'ensemble_prediction': {
                'current_price': current_price,
                'days_predicted': days,
                'daily_predictions': ensemble_daily_predictions,
                'overall_summary': {
                    'total_change': cumulative_change,
                    'final_predicted_price': final_price,
                    'overall_direction': overall_direction,
                    'average_confidence': avg_confidence,
                    'average_uncertainty': avg_uncertainty
                },
                'num_models': len(individual_predictions)
            },
            'individual_predictions': individual_predictions
        }