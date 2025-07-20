import requests
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv

load_dotenv()

class NewsCollector:
    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.rss_feeds = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
            'cnbc': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'reuters_business': 'https://feeds.reuters.com/reuters/businessNews'
        }
    
    def get_stock_news_from_api(self, symbol: str, days_back: int = 7) -> List[Dict]:
        """
        Get news for a specific stock using News API
        
        Args:
            symbol: Stock symbol
            days_back: Number of days to look back for news
        
        Returns:
            List of news articles
        """
        if not self.news_api_key:
            print("News API key not found. Please set NEWS_API_KEY in your .env file")
            return []
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f'{symbol} stock OR {symbol} earnings OR {symbol} financial',
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'sortBy': 'relevancy',
            'language': 'en',
            'apiKey': self.news_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            articles = []
            for article in data.get('articles', []):
                articles.append({
                    'title': article.get('title'),
                    'description': article.get('description'),
                    'url': article.get('url'),
                    'published_at': article.get('publishedAt'),
                    'source': article.get('source', {}).get('name'),
                    'symbol': symbol
                })
            
            return articles
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return []
    
    def get_general_market_news_rss(self, max_articles: int = 20) -> List[Dict]:
        """
        Get general market news from RSS feeds
        
        Args:
            max_articles: Maximum number of articles to return
        
        Returns:
            List of news articles
        """
        all_articles = []
        
        for source, url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_articles//len(self.rss_feeds)]:
                    article = {
                        'title': entry.get('title'),
                        'description': entry.get('summary', ''),
                        'url': entry.get('link'),
                        'published_at': entry.get('published'),
                        'source': source,
                        'symbol': 'GENERAL'
                    }
                    all_articles.append(article)
            except Exception as e:
                print(f"Error fetching RSS feed from {source}: {e}")
        
        return all_articles[:max_articles]
    
    def analyze_sentiment_llm(self, text: str) -> Dict:
        """
        LLM-based sentiment analysis using OpenAI API or local model
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment score and classification
        """
        try:
            # Try OpenAI first if API key is available
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key:
                return self._analyze_with_openai(text, openai_key)
            
            # Try Hugging Face transformers as fallback
            return self._analyze_with_transformers(text)
            
        except Exception as e:
            print(f"LLM sentiment analysis failed: {e}")
            # Fallback to simple keyword-based analysis
            return self._analyze_sentiment_simple(text)
    
    def _analyze_with_openai(self, text: str, api_key: str) -> Dict:
        """Analyze sentiment using OpenAI API"""
        try:
            import openai
            from openai import OpenAI
            
            # Initialize client with API key
            client = OpenAI(api_key=api_key)
            
            prompt = f"""
            Analyze the sentiment of this financial news text and provide a score between -1 (very negative) and 1 (very positive).
            
            Text: "{text[:1000]}"
            
            Respond in JSON format with:
            - score: float between -1 and 1
            - sentiment: "positive", "negative", or "neutral"
            - confidence: float between 0 and 1
            - reasoning: brief explanation
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = response.choices[0].message.content
        except ImportError:
            # Fallback for older OpenAI versions
            import openai
            openai.api_key = api_key
            
            prompt = f"""
            Analyze the sentiment of this financial news text and provide a score between -1 (very negative) and 1 (very positive).
            
            Text: "{text[:1000]}"
            
            Respond in JSON format with:
            - score: float between -1 and 1
            - sentiment: "positive", "negative", or "neutral" 
            - confidence: float between 0 and 1
            - reasoning: brief explanation
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            
            result = response.choices[0].message.content
        
        # Parse JSON response
        import json
        try:
            parsed = json.loads(result)
            return {
                'score': parsed.get('score', 0),
                'sentiment': parsed.get('sentiment', 'neutral'),
                'confidence': parsed.get('confidence', 0.5),
                'reasoning': parsed.get('reasoning', ''),
                'method': 'openai'
            }
        except json.JSONDecodeError:
            print("Failed to parse OpenAI response")
            return self._analyze_sentiment_simple(text)
    
    def _analyze_with_transformers(self, text: str) -> Dict:
        """Analyze sentiment using Hugging Face transformers"""
        try:
            from transformers import pipeline
            
            # Use a financial sentiment model if available, otherwise general sentiment
            try:
                # Try financial-specific model first
                classifier = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    return_all_scores=True
                )
            except:
                # Fallback to general sentiment model
                classifier = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
            
            # Limit text length for processing
            text_sample = text[:512]
            
            results = classifier(text_sample)
            
            # Process results
            if isinstance(results[0], list):
                scores = results[0]
            else:
                scores = results
            
            # Convert to our format
            sentiment_mapping = {
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative', 
                'NEUTRAL': 'neutral',
                'LABEL_0': 'negative',  # Twitter model format
                'LABEL_1': 'neutral',
                'LABEL_2': 'positive'
            }
            
            best_score = max(scores, key=lambda x: x['score'])
            sentiment = sentiment_mapping.get(best_score['label'], 'neutral')
            
            # Convert confidence to our -1 to 1 scale
            if sentiment == 'positive':
                score = best_score['score'] * 0.5  # Scale to 0 to 0.5
            elif sentiment == 'negative':
                score = -best_score['score'] * 0.5  # Scale to -0.5 to 0
            else:
                score = 0
            
            return {
                'score': score,
                'sentiment': sentiment,
                'confidence': best_score['score'],
                'method': 'transformers'
            }
            
        except ImportError:
            print("Transformers not available, using simple sentiment analysis")
            return self._analyze_sentiment_simple(text)
        except Exception as e:
            print(f"Transformers sentiment analysis failed: {e}")
            return self._analyze_sentiment_simple(text)
    
    def _analyze_sentiment_simple(self, text: str) -> Dict:
        """
        Simple sentiment analysis based on keywords (fallback)
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with sentiment score and classification
        """
        positive_words = [
            'bullish', 'positive', 'growth', 'profit', 'gain', 'rise', 'increase',
            'strong', 'beat', 'exceed', 'outperform', 'upgrade', 'buy', 'rally',
            'surge', 'soar', 'boom', 'successful', 'optimistic', 'confident',
            'breakthrough', 'milestone', 'record', 'high', 'improved', 'better'
        ]
        
        negative_words = [
            'bearish', 'negative', 'loss', 'decline', 'fall', 'decrease',
            'weak', 'miss', 'underperform', 'downgrade', 'sell', 'crash',
            'plunge', 'tumble', 'slump', 'disappointing', 'concern', 'worry',
            'risk', 'threat', 'challenge', 'problem', 'issue', 'low', 'poor'
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        
        if sentiment_score > 0.01:
            sentiment = 'positive'
        elif sentiment_score < -0.01:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'score': sentiment_score,
            'sentiment': sentiment,
            'positive_keywords': positive_count,
            'negative_keywords': negative_count,
            'method': 'simple'
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Main sentiment analysis function - tries LLM first, falls back to simple
        """
        return self.analyze_sentiment_llm(text)
    
    def get_comprehensive_news(self, symbols: List[str], days_back: int = 7) -> Dict:
        """
        Get comprehensive news data for given symbols
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back
        
        Returns:
            Dictionary with news data and sentiment analysis
        """
        news_data = {
            'stock_specific': {},
            'general_market': [],
            'sentiment_summary': {}
        }
        
        # Get stock-specific news
        for symbol in symbols:
            stock_news = self.get_stock_news_from_api(symbol, days_back)
            news_data['stock_specific'][symbol] = stock_news
            
            # Calculate sentiment for this stock
            all_text = ' '.join([
                f"{article['title']} {article['description']}" 
                for article in stock_news 
                if article['title'] and article['description']
            ])
            
            if all_text:
                sentiment = self.analyze_sentiment(all_text)
                news_data['sentiment_summary'][symbol] = sentiment
        
        # Get general market news
        news_data['general_market'] = self.get_general_market_news_rss()
        
        # Calculate overall market sentiment
        market_text = ' '.join([
            f"{article['title']} {article['description']}" 
            for article in news_data['general_market'] 
            if article['title'] and article['description']
        ])
        
        if market_text:
            market_sentiment = self.analyze_sentiment(market_text)
            news_data['sentiment_summary']['MARKET'] = market_sentiment
        
        return news_data