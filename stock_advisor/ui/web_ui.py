#!/usr/bin/env python3
"""
Stock Advisor Web UI

Interactive web interface for visualizing stock predictions from all models:
- Technical Analysis
- Reinforcement Learning  
- Enhanced RL (Deep Q-Learning with neural networks)
- Hybrid (RL + Technical + News)

Features:
- Real-time stock charts with predictions
- Interactive dashboards
- Method comparison views
- Historical performance tracking
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import numpy as np
from typing import Dict
import sys
import os

# Add the parent directory to the path so we can import from stock_advisor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from stock_advisor.core.stock_data import StockDataFetcher
    from stock_advisor.predictors.technical_predictor import TechnicalAnalysisPredictor
    from stock_advisor.predictors.rl_predictor import RLStockPredictor
    from stock_advisor.predictors.enhanced_rl_predictor import EnhancedRLStockPredictor
    from stock_advisor.predictors.hybrid_predictor import HybridStockPredictor
    from stock_advisor.utils.backtest_simulator import BacktestSimulator
except ImportError:
    # Fallback to relative imports if running from within the package
    from ..core.stock_data import StockDataFetcher
    from ..predictors.technical_predictor import TechnicalAnalysisPredictor
    from ..predictors.rl_predictor import RLStockPredictor
    from ..predictors.enhanced_rl_predictor import EnhancedRLStockPredictor
    from ..predictors.hybrid_predictor import HybridStockPredictor
    from ..utils.backtest_simulator import BacktestSimulator


class StockAdvisorUI:
    """Main UI class for stock advisor visualization"""
    
    def __init__(self):
        self.stock_fetcher = StockDataFetcher()
        self.technical_predictor = TechnicalAnalysisPredictor()
        self.rl_predictor = RLStockPredictor()
        self.enhanced_rl_predictor = EnhancedRLStockPredictor()
        self.hybrid_predictor = HybridStockPredictor()
        self.backtest_simulator = BacktestSimulator()
        
        # Configure Streamlit
        st.set_page_config(
            page_title="Stock Advisor Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def create_price_chart(self, stock_data: pd.DataFrame, predictions: dict, symbol: str) -> go.Figure:
        """Create interactive price chart with predictions"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=['Price & Predictions', 'Volume', 'RSI'],
            vertical_spacing=0.05
        )
        
        # Historical price data
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Price',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # Moving averages
        if 'SMA_5' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['SMA_5'],
                    name='SMA 5',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # Prediction lines
        last_date = stock_data.index[-1]
        current_price = stock_data['Close'].iloc[-1]
        
        # Create future dates
        future_dates = [last_date + timedelta(days=i) for i in range(1, 4)]
        
        if 'daily_predictions' in predictions:
            pred_prices = [current_price] + [p['predicted_price'] for p in predictions['daily_predictions']]
            pred_dates = [last_date] + future_dates
            
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=pred_prices,
                    name='Prediction',
                    line=dict(color='purple', width=3, dash='dash'),
                    mode='lines+markers'
                ),
                row=1, col=1
            )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name='Volume',
                marker_color='lightblue',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in stock_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['RSI'],
                    name='RSI',
                    line=dict(color='purple'),
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            title=f'{symbol} - Stock Price Analysis & Predictions',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_prediction_comparison_chart(self, results: dict) -> go.Figure:
        """Compare predictions from different methods"""
        methods = ['Technical Analysis', 'Reinforcement Learning', 'Enhanced RL', 'Hybrid']
        days = [1, 2, 3]
        
        fig = go.Figure()
        
        for method in methods:
            method_key = method.lower().replace(' ', '_')
            if method_key in results:
                method_data = results[method_key]
                if 'daily_predictions' in method_data:
                    changes = [p['predicted_change_pct'] for p in method_data['daily_predictions']]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=days,
                            y=changes,
                            name=method,
                            mode='lines+markers',
                            line=dict(width=3)
                        )
                    )
        
        fig.update_layout(
            title='Prediction Comparison - Daily Price Change %',
            xaxis_title='Day',
            yaxis_title='Predicted Change %',
            height=400
        )
        
        return fig
    
    def create_confidence_chart(self, results: dict) -> go.Figure:
        """Show confidence levels for each method"""
        methods = []
        confidences = []
        colors = []
        
        method_colors = {
            'technical_analysis': '#1f77b4',
            'reinforcement_learning': '#ff7f0e',
            'enhanced_rl': '#d62728', 
            'hybrid': '#2ca02c'
        }
        
        for method_key, method_name in [
            ('technical_analysis', 'Technical Analysis'),
            ('reinforcement_learning', 'Reinforcement Learning'),
            ('enhanced_rl', 'Enhanced RL'),
            ('hybrid', 'Hybrid')
        ]:
            if method_key in results:
                method_data = results[method_key]
                if 'overall_summary' in method_data:
                    avg_conf = method_data['overall_summary'].get('avg_confidence', 0)
                    methods.append(method_name)
                    confidences.append(avg_conf)
                    colors.append(method_colors[method_key])
        
        fig = go.Figure(data=[
            go.Bar(
                x=methods,
                y=confidences,
                marker_color=colors,
                text=[f'{c:.1f}%' for c in confidences],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Prediction Confidence by Method',
            yaxis_title='Confidence %',
            height=400,
            yaxis=dict(range=[0, 100])
        )
        
        return fig
    
    def create_news_sentiment_gauge(self, news_data: dict) -> go.Figure:
        """Create sentiment gauge chart"""
        sentiment_score = news_data.get('combined_score', 0)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = sentiment_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "News Sentiment"},
            delta = {'reference': 0},
            gauge = {
                'axis': {'range': [-1, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [-1, -0.3], 'color': "lightcoral"},
                    {'range': [-0.3, 0.3], 'color': "lightgray"},
                    {'range': [0.3, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': sentiment_score
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def create_backtest_performance_chart(self, backtest_results: Dict) -> go.Figure:
        """Create backtest performance comparison chart"""
        methods = []
        direction_accuracies = []
        avg_errors = []
        correlations = []
        
        method_names = {
            'technical': 'Technical Analysis',
            'rl': 'Reinforcement Learning', 
            'hybrid': 'Hybrid'
        }
        
        for method_key, method_name in method_names.items():
            if method_key in backtest_results['methods']:
                perf = backtest_results['methods'][method_key]['performance']
                if 'error' not in perf:
                    methods.append(method_name)
                    direction_accuracies.append(perf['direction_accuracy'])
                    avg_errors.append(perf['avg_price_error_pct'])
                    correlations.append(perf['prediction_correlation'])
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Direction Accuracy %', 'Average Error %', 'Prediction Correlation'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Direction accuracy
        fig.add_trace(
            go.Bar(x=methods, y=direction_accuracies, name='Direction Accuracy', 
                   marker_color='green', showlegend=False),
            row=1, col=1
        )
        
        # Average error
        fig.add_trace(
            go.Bar(x=methods, y=avg_errors, name='Avg Error', 
                   marker_color='red', showlegend=False),
            row=1, col=2
        )
        
        # Correlation
        fig.add_trace(
            go.Bar(x=methods, y=correlations, name='Correlation', 
                   marker_color='blue', showlegend=False),
            row=1, col=3
        )
        
        fig.update_layout(
            title='7-Day Backtest Performance Comparison',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def create_daily_backtest_chart(self, backtest_results: Dict) -> go.Figure:
        """Create daily backtest results chart"""
        fig = go.Figure()
        
        method_names = {
            'technical': 'Technical Analysis',
            'rl': 'Reinforcement Learning', 
            'hybrid': 'Hybrid'
        }
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, (method_key, method_name) in enumerate(method_names.items()):
            if method_key in backtest_results['methods']:
                daily_results = backtest_results['methods'][method_key]['daily_results']
                valid_results = [r for r in daily_results if 'error' not in r]
                
                if valid_results:
                    days = [r['day'] for r in valid_results]
                    errors = [r['price_error_pct'] for r in valid_results]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=days,
                            y=errors,
                            name=method_name,
                            mode='lines+markers',
                            line=dict(color=colors[i], width=3),
                            marker=dict(size=8)
                        )
                    )
        
        fig.update_layout(
            title='Daily Prediction Error % Over 7 Days',
            xaxis_title='Day',
            yaxis_title='Prediction Error %',
            height=400,
            xaxis=dict(tickmode='array', tickvals=list(range(1, 8)))
        )
        
        return fig
    
    def display_backtest_metrics_table(self, backtest_results: Dict):
        """Display backtest performance metrics table"""
        method_names = {
            'technical': 'Technical Analysis',
            'rl': 'Reinforcement Learning', 
            'hybrid': 'Hybrid'
        }
        
        table_data = []
        for method_key, method_name in method_names.items():
            if method_key in backtest_results['methods']:
                perf = backtest_results['methods'][method_key]['performance']
                if 'error' not in perf:
                    table_data.append({
                        'Method': method_name,
                        'Direction Accuracy': f"{perf['direction_accuracy']:.1f}%",
                        'Correct Predictions': f"{perf['direction_correct_count']}/7",
                        'Avg Error %': f"{perf['avg_price_error_pct']:.2f}%",
                        'Max Error %': f"{perf['max_price_error_pct']:.2f}%",
                        'Avg Confidence': f"{perf['avg_confidence']:.1f}%",
                        'Correlation': f"{perf['prediction_correlation']:.3f}"
                    })
                else:
                    table_data.append({
                        'Method': method_name,
                        'Direction Accuracy': 'Failed',
                        'Correct Predictions': '-',
                        'Avg Error %': '-',
                        'Max Error %': '-', 
                        'Avg Confidence': '-',
                        'Correlation': '-'
                    })
        
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
    
    def display_daily_backtest_details(self, backtest_results: Dict):
        """Display detailed daily backtest results"""
        method_names = {
            'technical': 'Technical Analysis',
            'rl': 'Reinforcement Learning', 
            'hybrid': 'Hybrid'
        }
        
        tabs = st.tabs([method_names[key] for key in method_names.keys() 
                       if key in backtest_results['methods']])
        
        tab_idx = 0
        for method_key, method_name in method_names.items():
            if method_key in backtest_results['methods']:
                with tabs[tab_idx]:
                    daily_results = backtest_results['methods'][method_key]['daily_results']
                    valid_results = [r for r in daily_results if 'error' not in r]
                    
                    if valid_results:
                        table_data = []
                        for result in valid_results:
                            table_data.append({
                                'Day': result['day'],
                                'Test Date': result['test_date'],
                                'Prediction Date': result.get('prediction_date', 'N/A'),
                                'Predicted Change': f"{result['predicted_change_pct']:+.2f}%",
                                'Actual Change': f"{result['actual_change_pct']:+.2f}%",
                                'Error %': f"{result['price_error_pct']:.2f}%",
                                'Direction': result['predicted_direction'],
                                'Correct': "✓" if result['direction_correct'] else "✗",
                                'Confidence': f"{result['confidence']:.1f}%"
                            })
                        
                        df = pd.DataFrame(table_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Summary for this method
                        perf = backtest_results['methods'][method_key]['performance']
                        st.write(f"**Summary**: {perf['direction_correct_count']}/7 correct directions "
                                f"({perf['direction_accuracy']:.1f}% accuracy)")
                    else:
                        st.error(f"No valid predictions for {method_name}")
                
                tab_idx += 1
    
    def display_prediction_metrics(self, results: dict, method: str):
        """Display prediction metrics in formatted cards"""
        if method not in results:
            st.warning(f"No results available for {method}")
            return
        
        data = results[method]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'overall_summary' in data:
                direction = data['overall_summary'].get('direction', 'N/A')
                color = 'green' if direction == 'UP' else 'red' if direction == 'DOWN' else 'gray'
                st.metric("Direction", direction)
        
        with col2:
            if 'overall_summary' in data:
                change = data['overall_summary'].get('total_change_pct', 0)
                st.metric("Total Change", f"{change:+.2f}%")
        
        with col3:
            if 'overall_summary' in data:
                final_price = data['overall_summary'].get('final_price', 0)
                st.metric("Target Price", f"${final_price:.2f}")
        
        with col4:
            if 'overall_summary' in data:
                confidence = data['overall_summary'].get('avg_confidence', 0)
                st.metric("Avg Confidence", f"{confidence:.1f}%")
    
    def display_daily_predictions_table(self, results: dict, method: str):
        """Display daily predictions in a formatted table"""
        if method not in results or 'daily_predictions' not in results[method]:
            return
        
        predictions = results[method]['daily_predictions']
        
        df_data = []
        for pred in predictions:
            df_data.append({
                'Day': pred['day'],
                'Direction': pred['direction'],
                'Price': f"${pred['predicted_price']:.2f}",
                'Change %': f"{pred['predicted_change_pct']:+.2f}%",
                'Confidence': f"{pred['confidence']:.1f}%"
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
    
    def run_dashboard(self):
        """Main dashboard interface"""
        st.title("📈 Stock Advisor Dashboard")
        st.markdown("### Interactive Stock Prediction Analysis")
        
        # Sidebar
        st.sidebar.header("🔧 Configuration")
        
        # Demo mode option
        demo_mode = st.sidebar.checkbox("🎭 Demo Mode (No API calls)", True)
        if demo_mode:
            st.sidebar.info("Demo mode uses simulated data - perfect for testing without API limits!")
        
        symbol = st.sidebar.text_input("Stock Symbol", "AAPL").upper()
        
        prediction_methods = st.sidebar.multiselect(
            "Prediction Methods",
            ["Technical Analysis", "Reinforcement Learning", "Enhanced RL", "Hybrid"],
            default=["Hybrid"]
        )
        
        use_news = st.sidebar.checkbox("Include News Analysis", not demo_mode)
        if demo_mode:
            use_news = False  # Disable news in demo mode
            
        auto_train = st.sidebar.checkbox("Auto-train RL Models", True)
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Backtesting")
        run_backtest = st.sidebar.checkbox("Include 7-Day Backtest", False)
        
        if st.sidebar.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Fetching data and running predictions..."):
                try:
                    # Get stock data (demo or real)
                    if demo_mode:
                        try:
                            from stock_advisor.core.demo_data import generate_demo_stock_data
                        except ImportError:
                            from ..core.demo_data import generate_demo_stock_data
                        stock_data = generate_demo_stock_data(symbol, period="3mo")
                        if not stock_data.empty:
                            stock_data = self.stock_fetcher.calculate_technical_indicators(stock_data)
                        st.info(f"Using demo data for {symbol}")
                    else:
                        stock_data = self.stock_fetcher.get_stock_data(symbol, period="3mo")
                        if not stock_data.empty:
                            stock_data = self.stock_fetcher.calculate_technical_indicators(stock_data)
                    
                    if stock_data.empty:
                        st.error(f"No data available for {symbol}")
                        return
                    
                    # Store results
                    results = {}
                    
                    # Run selected predictions
                    if "Technical Analysis" in prediction_methods:
                        st.info("Running Technical Analysis...")
                        if demo_mode:
                            # Override stock fetcher for demo mode
                            original_get_stock_data = self.technical_predictor.stock_fetcher.get_stock_data
                            def demo_get_stock_data(sym, period):
                                try:
                                    from stock_advisor.core.demo_data import generate_demo_stock_data
                                except ImportError:
                                    from ..core.demo_data import generate_demo_stock_data
                                return generate_demo_stock_data(sym, period)
                            self.technical_predictor.stock_fetcher.get_stock_data = demo_get_stock_data
                        
                        tech_result = self.technical_predictor.predict_next_3_days(symbol)
                        if 'error' not in tech_result:
                            results['technical_analysis'] = tech_result
                        
                        if demo_mode:
                            # Restore original function
                            self.technical_predictor.stock_fetcher.get_stock_data = original_get_stock_data
                    
                    if "Reinforcement Learning" in prediction_methods:
                        st.info("Running Reinforcement Learning...")
                        if demo_mode:
                            # Override stock fetcher for demo mode
                            original_get_stock_data = self.rl_predictor.stock_fetcher.get_stock_data
                            def demo_get_stock_data(sym, period):
                                try:
                                    from stock_advisor.core.demo_data import generate_demo_stock_data
                                except ImportError:
                                    from ..core.demo_data import generate_demo_stock_data
                                return generate_demo_stock_data(sym, period)
                            self.rl_predictor.stock_fetcher.get_stock_data = demo_get_stock_data
                        
                        rl_result = self.rl_predictor.predict_next_3_days(symbol, auto_train)
                        if 'error' not in rl_result:
                            results['reinforcement_learning'] = rl_result
                        
                        if demo_mode:
                            # Restore original function
                            self.rl_predictor.stock_fetcher.get_stock_data = original_get_stock_data
                    
                    if "Enhanced RL" in prediction_methods:
                        st.info("Running Enhanced Reinforcement Learning...")
                        if demo_mode:
                            # Override stock fetcher for demo mode
                            original_get_stock_data = self.enhanced_rl_predictor.stock_fetcher.get_stock_data
                            def demo_get_stock_data(sym, period):
                                try:
                                    from stock_advisor.core.demo_data import generate_demo_stock_data
                                except ImportError:
                                    from ..core.demo_data import generate_demo_stock_data
                                return generate_demo_stock_data(sym, period)
                            self.enhanced_rl_predictor.stock_fetcher.get_stock_data = demo_get_stock_data
                        
                        enhanced_rl_result = self.enhanced_rl_predictor.predict_next_3_days(symbol, auto_train)
                        if 'error' not in enhanced_rl_result:
                            results['enhanced_rl'] = enhanced_rl_result
                        
                        if demo_mode:
                            # Restore original function
                            self.enhanced_rl_predictor.stock_fetcher.get_stock_data = original_get_stock_data
                    
                    if "Hybrid" in prediction_methods:
                        st.info("Running Hybrid Analysis...")
                        if demo_mode:
                            # Override stock fetcher for demo mode
                            original_get_stock_data = self.hybrid_predictor.stock_fetcher.get_stock_data
                            def demo_get_stock_data(sym, period):
                                try:
                                    from stock_advisor.core.demo_data import generate_demo_stock_data
                                except ImportError:
                                    from ..core.demo_data import generate_demo_stock_data
                                return generate_demo_stock_data(sym, period)
                            self.hybrid_predictor.stock_fetcher.get_stock_data = demo_get_stock_data
                        
                        hybrid_result = self.hybrid_predictor.predict_next_3_days(
                            symbol, auto_train, use_news
                        )
                        if 'error' not in hybrid_result:
                            results['hybrid'] = hybrid_result
                        
                        if demo_mode:
                            # Restore original function
                            self.hybrid_predictor.stock_fetcher.get_stock_data = original_get_stock_data
                    
                    # Run backtest if requested
                    backtest_results = None
                    if run_backtest:
                        st.info("Running 7-day backtest simulation...")
                        if demo_mode:
                            # Override stock fetcher for demo mode
                            original_get_stock_data = self.backtest_simulator.stock_fetcher.get_stock_data
                            def demo_get_stock_data(sym, period):
                                try:
                                    from stock_advisor.core.demo_data import generate_demo_stock_data
                                except ImportError:
                                    from ..core.demo_data import generate_demo_stock_data
                                return generate_demo_stock_data(sym, period)
                            self.backtest_simulator.stock_fetcher.get_stock_data = demo_get_stock_data
                        
                        backtest_results = self.backtest_simulator.run_7_day_backtest(symbol)
                        
                        if demo_mode:
                            # Restore original function
                            self.backtest_simulator.stock_fetcher.get_stock_data = original_get_stock_data
                    
                    # Store in session state
                    st.session_state['results'] = results
                    st.session_state['stock_data'] = stock_data
                    st.session_state['symbol'] = symbol
                    st.session_state['backtest_results'] = backtest_results
                    
                except Exception as e:
                    st.error(f"Error running analysis: {str(e)}")
                    return
        
        # Display results if available
        if 'results' in st.session_state and st.session_state['results']:
            results = st.session_state['results']
            stock_data = st.session_state['stock_data']
            symbol = st.session_state['symbol']
            
            st.success("✅ Analysis Complete!")
            
            # Current stock info
            current_price = stock_data['Close'].iloc[-1]
            daily_change = ((current_price - stock_data['Close'].iloc[-2]) / 
                           stock_data['Close'].iloc[-2] * 100)
            
            # Show data source
            data_source = "Demo Data" if demo_mode else "Live Data (Yahoo Finance/IEX Cloud)"
            st.info(f"📊 Data Source: {data_source}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${current_price:.2f}", f"{daily_change:+.2f}%")
            with col2:
                st.metric("Volume", f"{stock_data['Volume'].iloc[-1]:,}")
            with col3:
                if 'RSI' in stock_data.columns:
                    st.metric("RSI", f"{stock_data['RSI'].iloc[-1]:.1f}")
            
            # Main chart
            st.subheader("📊 Price Chart & Predictions")
            if 'hybrid' in results:
                chart_data = results['hybrid']
            elif 'enhanced_rl' in results:
                chart_data = results['enhanced_rl']
            else:
                chart_data = list(results.values())[0]
            
            price_chart = self.create_price_chart(stock_data, chart_data, symbol)
            st.plotly_chart(price_chart, use_container_width=True)
            
            # Method comparison
            if len(results) > 1:
                st.subheader("🔄 Method Comparison")
                
                col1, col2 = st.columns(2)
                with col1:
                    comparison_chart = self.create_prediction_comparison_chart(results)
                    st.plotly_chart(comparison_chart, use_container_width=True)
                
                with col2:
                    confidence_chart = self.create_confidence_chart(results)
                    st.plotly_chart(confidence_chart, use_container_width=True)
            
            # News sentiment (if available)
            if 'hybrid' in results and results['hybrid'].get('news_analysis_included'):
                news_data = results['hybrid'].get('news_summary', {})
                if news_data:
                    st.subheader("📰 News Sentiment Analysis")
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        sentiment_gauge = self.create_news_sentiment_gauge(news_data)
                        st.plotly_chart(sentiment_gauge, use_container_width=True)
                    
                    with col2:
                        st.metric("Articles Analyzed", news_data.get('news_articles_count', 0))
                        st.metric("Positive Keywords", news_data.get('positive_keywords', 0))
                        st.metric("Negative Keywords", news_data.get('negative_keywords', 0))
                        st.metric("Sentiment Confidence", f"{news_data.get('confidence', 0)*100:.1f}%")
            
            # Detailed results for each method
            st.subheader("📋 Detailed Predictions")
            
            tabs = st.tabs([method.title().replace('_', ' ') for method in results.keys()])
            
            for i, (method_key, method_name) in enumerate([(k, k.replace('_', ' ').title()) for k in results.keys()]):
                with tabs[i]:
                    st.write(f"### {method_name} Results")
                    
                    # Metrics
                    self.display_prediction_metrics(results, method_key)
                    
                    # Daily predictions table
                    st.write("#### Daily Predictions")
                    self.display_daily_predictions_table(results, method_key)
                    
                    # Method-specific details
                    if method_key == 'hybrid' and 'daily_predictions' in results[method_key]:
                        st.write("#### Method Contributions")
                        for pred in results[method_key]['daily_predictions']:
                            with st.expander(f"Day {pred['day']} Details"):
                                if 'method_predictions' in pred:
                                    for method_pred in pred['method_predictions']:
                                        st.write(f"**{method_pred['method']}**: {method_pred['direction']} "
                                                f"(Confidence: {method_pred['confidence']:.1f}%)")
                                
                                if 'technical_signals' in pred and pred['technical_signals']:
                                    st.write("**Technical Signals**:")
                                    for signal in pred['technical_signals']:
                                        st.write(f"- {signal}")
                                
                                if 'news_explanation' in pred:
                                    st.write(f"**News**: {pred['news_explanation']}")
                                
                                if 'rl_action' in pred:
                                    st.write(f"**RL Action**: {pred['rl_action']}")
                    
                    elif method_key == 'enhanced_rl' and 'daily_predictions' in results[method_key]:
                        st.write("#### Enhanced RL Details")
                        
                        # Show model info
                        model_info = results[method_key].get('model_info', {})
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Models Loaded", f"{model_info.get('models_loaded', 0)}/{model_info.get('ensemble_size', 0)}")
                        with col2:
                            st.metric("PyTorch Used", "Yes" if model_info.get('pytorch_used', False) else "No")
                        with col3:
                            st.metric("Consensus Strength", f"{results[method_key]['overall_summary'].get('consensus_strength', 0):.1%}")
                        
                        # Show daily predictions with ensemble details
                        for pred in results[method_key]['daily_predictions']:
                            with st.expander(f"Day {pred['day']} Ensemble Details"):
                                st.write(f"**Direction**: {pred['direction']}")
                                st.write(f"**Confidence**: {pred['confidence']:.1f}%")
                                st.write(f"**Ensemble Size**: {pred['ensemble_size']} agents")
                                st.write(f"**Consensus Strength**: {pred['consensus_strength']:.1%}")
                                
                                # Show individual agent reasoning if available
                                if 'reasoning' in pred:
                                    st.write(f"**Reasoning**: {pred['reasoning']}")
            
            # Backtest results section
            if 'backtest_results' in st.session_state and st.session_state['backtest_results']:
                backtest_results = st.session_state['backtest_results']
                
                if 'error' not in backtest_results:
                    st.subheader("📊 7-Day Backtest Results")
                    st.markdown("*How well did each method predict the last 7 days?*")
                    
                    # Performance summary
                    st.write("#### Performance Overview")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        performance_chart = self.create_backtest_performance_chart(backtest_results)
                        st.plotly_chart(performance_chart, use_container_width=True)
                    
                    with col2:
                        daily_error_chart = self.create_daily_backtest_chart(backtest_results)
                        st.plotly_chart(daily_error_chart, use_container_width=True)
                    
                    # Performance metrics table
                    st.write("#### Detailed Metrics")
                    
                    # Show validation info
                    total_predictions = {}
                    for method_key in ['technical', 'rl', 'hybrid']:
                        if method_key in backtest_results['methods']:
                            count = len(backtest_results['methods'][method_key]['daily_results'])
                            total_predictions[method_key] = count
                    
                    if total_predictions:
                        max_predictions = max(total_predictions.values())
                        if max_predictions < 7:
                            st.warning(f"⚠️ Only {max_predictions}/7 days of predictions available. This may be due to insufficient historical data.")
                        else:
                            st.success(f"✅ All 7 days of predictions completed for testing.")
                    
                    self.display_backtest_metrics_table(backtest_results)
                    
                    # Daily breakdown
                    st.write("#### Daily Prediction Details")
                    self.display_daily_backtest_details(backtest_results)
                    
                    # Best performer highlight
                    best_method = None
                    best_accuracy = 0
                    
                    method_names = {
                        'technical': 'Technical Analysis',
                        'rl': 'Reinforcement Learning', 
                        'hybrid': 'Hybrid'
                    }
                    
                    for method_key in ['technical', 'rl', 'hybrid']:
                        if method_key in backtest_results['methods']:
                            perf = backtest_results['methods'][method_key]['performance']
                            if 'error' not in perf and perf['direction_accuracy'] > best_accuracy:
                                best_accuracy = perf['direction_accuracy']
                                best_method = method_names[method_key]
                    
                    if best_method:
                        st.success(f"🏆 **Best Performer**: {best_method} with {best_accuracy:.1f}% direction accuracy")
                
                else:
                    st.error(f"Backtest failed: {backtest_results['error']}")
            
            # Risk disclaimer
            st.warning("⚠️ **Risk Disclaimer**: These predictions are for educational purposes only and should not be considered financial advice. Always do your own research and consult with qualified financial advisors before making investment decisions.")


def main():
    """Main function to run the Streamlit app"""
    app = StockAdvisorUI()
    app.run_dashboard()


if __name__ == "__main__":
    main()