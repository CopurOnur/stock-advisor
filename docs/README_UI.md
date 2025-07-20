# Stock Advisor Web UI 📊

Interactive web dashboard for visualizing stock predictions from Technical Analysis, Reinforcement Learning, and Hybrid models.

## 🚀 Quick Start

### Option 1: Using the launcher script
```bash
./run_ui.sh
```

### Option 2: Manual setup
```bash
# Install UI dependencies
pip install -r requirements_ui.txt

# Run the Streamlit app
streamlit run web_ui.py
```

Then open your browser to `http://localhost:8501`

## 📱 Features

### 🎯 Interactive Dashboard
- **Stock Symbol Input**: Enter any stock symbol (AAPL, GOOGL, MSFT, etc.)
- **Method Selection**: Choose which prediction models to run
- **Real-time Analysis**: Get live predictions with confidence scores

### 📈 Visualizations

#### 1. **Price Chart & Predictions**
- Candlestick chart with historical prices
- Moving averages (SMA 5, 20)
- Volume bars
- RSI indicator with overbought/oversold levels
- Prediction lines for next 3 days

#### 2. **Method Comparison**
- Side-by-side comparison of all prediction methods
- Confidence level charts
- Daily change predictions

#### 3. **News Sentiment Gauge**
- Real-time sentiment analysis visualization
- Sentiment score from -1 (very negative) to +1 (very positive)
- Article count and keyword analysis

### 📊 Prediction Methods

#### **Technical Analysis**
- Moving average crossovers
- RSI momentum signals
- MACD indicators
- Support/resistance levels
- Volume confirmation

#### **Reinforcement Learning**
- Q-learning trading agent
- Learns from historical patterns
- Auto-training on first use
- Action recommendations (BUY/SELL/HOLD)

#### **Hybrid Model**
- Combines all three approaches
- Ensemble voting system
- Enhanced with news sentiment
- Highest accuracy predictions

### 🔧 Configuration Options

- **Auto-train RL Models**: Automatically train models if none exist
- **Include News Analysis**: Enable/disable news sentiment integration
- **Method Selection**: Run individual models or compare multiple
- **Historical Period**: Analyze different time ranges

## 📋 UI Components

### 📊 Main Dashboard
```
┌─────────────────────────────────────────────────┐
│  📈 Stock Advisor Dashboard                     │
├─────────────────────────────────────────────────┤
│  🔧 Configuration Sidebar                       │
│  ├── Stock Symbol Input                         │
│  ├── Method Selection                           │
│  ├── News Analysis Toggle                       │
│  └── 🚀 Run Analysis Button                     │
├─────────────────────────────────────────────────┤
│  📊 Current Stock Metrics                       │
│  ├── Current Price & Daily Change               │
│  ├── Volume                                     │
│  └── RSI Value                                  │
├─────────────────────────────────────────────────┤
│  📈 Interactive Price Chart                     │
│  ├── Candlestick Price Data                     │
│  ├── Moving Averages                            │
│  ├── Volume Bars                                │
│  ├── RSI Subplot                                │
│  └── Prediction Lines                           │
├─────────────────────────────────────────────────┤
│  🔄 Method Comparison (if multiple selected)    │
│  ├── Prediction Comparison Chart                │
│  └── Confidence Level Chart                     │
├─────────────────────────────────────────────────┤
│  📰 News Sentiment (if enabled)                 │
│  ├── Sentiment Gauge                            │
│  └── Article Statistics                         │
├─────────────────────────────────────────────────┤
│  📋 Detailed Predictions (Tabbed Interface)     │
│  ├── Technical Analysis Tab                     │
│  ├── Reinforcement Learning Tab                 │
│  └── Hybrid Analysis Tab                        │
└─────────────────────────────────────────────────┘
```

### 📋 Prediction Details
Each method tab shows:
- **Key Metrics**: Direction, Total Change, Target Price, Confidence
- **Daily Predictions Table**: Day-by-day breakdown
- **Method-Specific Details**:
  - Technical signals and crossovers
  - RL actions and reasoning
  - News explanations and sentiment scores

### 🎨 Visual Design
- **Clean Modern Interface**: Streamlit-powered responsive design
- **Interactive Charts**: Plotly charts with zoom, pan, and hover details
- **Color-Coded Indicators**: Green/red for up/down movements
- **Progress Indicators**: Real-time feedback during analysis
- **Responsive Layout**: Works on desktop and mobile

## 🔍 Example Usage

1. **Single Stock Analysis**:
   - Enter "AAPL" in symbol input
   - Select "Hybrid" method
   - Enable news analysis
   - Click "Run Analysis"

2. **Method Comparison**:
   - Enter "GOOGL" 
   - Select all three methods
   - Compare predictions side-by-side
   - Analyze confidence differences

3. **Technical Focus**:
   - Enter "MSFT"
   - Select only "Technical Analysis"
   - Disable news to focus on chart patterns
   - View detailed technical signals

## ⚡ Performance Tips

- **First Run**: May take 2-3 minutes due to model training
- **Subsequent Runs**: Much faster with cached models
- **Large Stocks**: Popular stocks (AAPL, GOOGL) have more news data
- **Market Hours**: More accurate during active trading hours

## 🛠️ Troubleshooting

### Common Issues:

1. **"No data available"**: Check if symbol is valid and markets are open
2. **Slow loading**: First-time model training can take time
3. **News analysis failed**: Check internet connection and API keys
4. **Missing charts**: Ensure all UI requirements are installed

### Error Messages:
- **Training failed**: Stock may not have enough historical data
- **API errors**: News API may be rate-limited or require key
- **Model errors**: Try disabling specific methods and using others

## 🔧 Advanced Configuration

### Environment Variables
```bash
# Optional: Set OpenAI API key for enhanced news analysis
export OPENAI_API_KEY="your-api-key"

# Optional: Set News API key for more news sources
export NEWS_API_KEY="your-news-api-key"

# Optional: Set Alpha Vantage key for alternative data source
export ALPHA_VANTAGE_API_KEY="your-av-key"
```

### Custom Styling
The UI supports custom Streamlit themes. Edit `run_ui.sh` to modify colors and styling.

## 📊 Data Sources

- **Stock Data**: Yahoo Finance (primary), IEX Cloud (500K calls/month), Alpha Vantage (fallback)
- **News Data**: RSS feeds from Yahoo Finance, MarketWatch, CNBC, Reuters
- **Technical Indicators**: Calculated from price/volume data
- **Sentiment Analysis**: OpenAI GPT (if available), Hugging Face FinBERT, keyword-based fallback

### 🚀 Enhanced Data Reliability

**Multi-Tier Data Fetching:**
1. **Yahoo Finance** - Unlimited, fast, primary source
2. **IEX Cloud** - 500,000 free calls/month, reliable backup
3. **Alpha Vantage** - 5 calls/minute, additional fallback
4. **Demo Data** - Offline mode for testing

**Setup IEX Cloud (Recommended):**
```bash
# Run the setup script
python setup_iex_cloud.py

# Follow instructions to get free API key from iexcloud.io
# 500,000 free calls/month = 100x more than Alpha Vantage!
```

## 🔒 Privacy & Security

- **No Data Storage**: All analysis is done locally and in real-time
- **API Keys**: Stored only in environment variables
- **No User Tracking**: No data collection or analytics
- **Open Source**: All code is transparent and auditable

---

## 🎯 Quick Demo

```bash
# Clone and setup
git clone <repository>
cd stock-advisor

# Install and run
pip install -r requirements_ui.txt
streamlit run web_ui.py

# In browser (http://localhost:8501):
# 1. Enter "AAPL"
# 2. Select "Hybrid" 
# 3. Enable news analysis
# 4. Click "Run Analysis"
# 5. Explore the interactive charts and predictions!
```

---

**Disclaimer**: This tool is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.