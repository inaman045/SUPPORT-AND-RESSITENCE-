# Standard library imports
import os
import time
import json
import pickle
import logging
import warnings
import threading
import queue
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party library imports
import numpy as np
import pandas as pd
import talib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Logging imports
from logging.handlers import RotatingFileHandler
# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = "stock_prediction.log"
log_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)  # 10MB per file, keep 5 backups
log_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

logger = logging.getLogger('StockPredictor')
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logger.addHandler(console_handler)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK resources: {e}")

# Set up global variables
MODEL_DIRECTORY = "models"
DATA_DIRECTORY = "data"
METRICS_DIRECTORY = "metrics"
LOGS_DIRECTORY = "logs"
BACKUP_DIRECTORY = "backup"
CONFIG_FILE = "config.json"

# Default configuration - will be overridden by config.json if present
DEFAULT_CONFIG = {
    "retries": 3,
    "backoff_factor": 0.3,
    "status_forcelist": [500, 502, 503, 504],
    "timeout": 10,
    "api_key": os.environ.get("NEWS_API_KEY", "your_api_key_here"),
    "broker_api_key": os.environ.get("BROKER_API_KEY", ""),
    "broker_api_secret": os.environ.get("BROKER_API_SECRET", ""),
    "market_hours": {
        "start_hour": 9, 
        "start_minute": 15, 
        "end_hour": 15, 
        "end_minute": 30
    },
    "trading_params": {
        "max_position_size": 1,
        "stop_loss_pct": 0.5,
        "take_profit_pct": 1.0,
        "confidence_threshold": 0.75,
        "max_trades_per_day": 5
    },
    "prediction_interval_seconds": 60,
    "model_validation_threshold": 0.75,
    "max_drawdown_pct": 2.0,
    "circuit_breaker_threshold": 0.65,
    "auto_update_model": True,
    "update_frequency_days": 3,
    "drift_detection_threshold": 0.1,
    "data_sources": {
        "price_api": "https://api.example.com/real-time-data",
        "news_api": "https://newsapi.org/v2/everything",
        "economic_calendar_api": "https://api.example.com/economic-calendar"
    },
    "fallback_mode": "conservative",
    "backup_frequency_hours": 6
}

# Global variables for monitoring
market_conditions = {
    "volatility": 0,
    "trend": 0,
    "sentiment": 0,
    "model_drift": 0,
    "system_health": 1.0  # 1.0 = healthy, 0.0 = critical
}

# Global trade tracking
trade_history = []
current_positions = {}
daily_trades_count = 0
last_trade_time = None
model_performance_metrics = {}

# Message queue for inter-thread communication
system_messages = queue.Queue()

# Create necessary directories
for directory in [MODEL_DIRECTORY, DATA_DIRECTORY, METRICS_DIRECTORY, LOGS_DIRECTORY, BACKUP_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

# Load configuration
def load_config():
    """Load configuration from config file or use defaults"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                logger.info("Configuration loaded from config.json")
                # Update with any missing default keys
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        else:
            # Save default config for future use
            with open(CONFIG_FILE, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            logger.info("Default configuration created in config.json")
            return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return DEFAULT_CONFIG

CONFIG = load_config()

# Initialize sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Create error-resilient HTTP session
def create_session():
    """Create a resilient HTTP session with retries and timeouts"""
    session = requests.Session()
    retry_strategy = Retry(
        total=CONFIG["retries"],
        backoff_factor=CONFIG["backoff_factor"],
        status_forcelist=CONFIG["status_forcelist"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# Function to fetch real-time stock data with fallback mechanisms
def fetch_real_time_data(stock_symbol, fallback_file=None, attempt=0, max_attempts=3):
    """
    Fetch real-time stock data with multiple fallback mechanisms
    Implements circuit breaking to prevent repeated failures
    """
    if attempt >= max_attempts:
        logger.critical(f"Maximum retry attempts reached for {stock_symbol}. Entering fallback mode.")
        return load_fallback_data(stock_symbol, fallback_file)
    
    session = create_session()
    try:
        # Check system health first
        if market_conditions["system_health"] < 0.5 and attempt == 0:
            logger.warning("System health below threshold, using cached data")
            return load_fallback_data(stock_symbol, fallback_file)
        
        url = f'{CONFIG["data_sources"]["price_api"]}?symbol={stock_symbol}'
        response = session.get(url, timeout=CONFIG["timeout"])
        response.raise_for_status()
        data = response.json()
        
        # Cache the data for fallback
        cache_file = f"{DATA_DIRECTORY}/{stock_symbol}_latest.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Update last successful fetch time
        with open(f"{DATA_DIRECTORY}/{stock_symbol}_last_fetch.txt", 'w') as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
        # Update system health
        market_conditions["system_health"] = 1.0
        
        # Basic data validation
        if 'prices' in data and len(data['prices']) > 0:
            return data['prices']
        else:
            logger.warning(f"Valid JSON but no price data for {stock_symbol}")
            return load_fallback_data(stock_symbol, fallback_file)
            
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout fetching data for {stock_symbol}, attempt {attempt+1}/{max_attempts}")
        # Increase timeout for next attempt
        CONFIG["timeout"] += 5
        market_conditions["system_health"] -= 0.1
        return fetch_real_time_data(stock_symbol, fallback_file, attempt + 1, max_attempts)
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching data: {e}")
        market_conditions["system_health"] -= 0.2
        time.sleep(2 * (attempt + 1))  # Progressive backoff
        return fetch_real_time_data(stock_symbol, fallback_file, attempt + 1, max_attempts)
        
    except Exception as e:
        logger.error(f"Error fetching real-time data: {e}")
        market_conditions["system_health"] -= 0.3
        return load_fallback_data(stock_symbol, fallback_file)

def load_fallback_data(stock_symbol, fallback_file=None):
    """Load fallback data when real-time data is unavailable"""
    # First try the specified fallback file
    if fallback_file and os.path.exists(fallback_file):
        logger.info(f"Using specified fallback data from {fallback_file}")
        try:
            return pd.read_csv(fallback_file)
        except Exception as e:
            logger.error(f"Error loading specified fallback file: {e}")
    
    # Then try the latest cached data
    cache_file = f"{DATA_DIRECTORY}/{stock_symbol}_latest.pkl"
    if os.path.exists(cache_file):
        logger.info(f"Using cached data for {stock_symbol}")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)['prices']
        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
    
    # Next try historical data
    historical_file = f"{DATA_DIRECTORY}/{stock_symbol}_historical.csv"
    if os.path.exists(historical_file):
        logger.info(f"Using historical data for {stock_symbol}")
        try:
            df = pd.read_csv(historical_file)
            return df.tail(30)  # Return the most recent 30 records
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    # Last resort - generate synthetic data based on previous patterns
    logger.critical("No fallback data available, generating synthetic data")
    return generate_synthetic_data(stock_symbol)

def generate_synthetic_data(stock_symbol):
    """Generate synthetic data as a last resort fallback"""
    logger.warning(f"Generating synthetic data for {stock_symbol} based on typical patterns")
    
    # Start with some reasonable default values for the stock
    base_price = 18000 if stock_symbol == 'NIFTY' else 1000  # Default values
    
    # Create date range for today and past few days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate synthetic OHLCV data with some randomness but following a pattern
    np.random.seed(42)  # For reproducibility
    
    # Create base values with some trend and volatility
    close_prices = base_price + np.cumsum(np.random.normal(0, base_price * 0.01, len(dates)))
    
    # Generate other values based on close prices
    data = {
        'date': dates,
        'open': close_prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': close_prices * (1 + abs(np.random.normal(0, 0.01, len(dates)))),
        'low': close_prices * (1 - abs(np.random.normal(0, 0.01, len(dates)))),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Make sure high is always the highest value and low is always the lowest
    for i in range(len(df)):
        values = [df.loc[i, 'open'], df.loc[i, 'close']]
        df.loc[i, 'high'] = max(df.loc[i, 'high'], max(values))
        df.loc[i, 'low'] = min(df.loc[i, 'low'], min(values))
    
    # Save this synthetic data for future reference
    df.to_csv(f"{DATA_DIRECTORY}/{stock_symbol}_synthetic.csv", index=False)
    
    logger.warning(f"Synthetic data created for {stock_symbol}. This should be replaced with real data ASAP.")
    return df

# Enhanced news sentiment analysis with caching and multiple sources
def fetch_sentiment_data(keywords=['market', 'stocks', 'finance', 'economy'], cache_hours=6):
    """
    Fetch sentiment data from news APIs with adaptive fallback mechanisms
    """
    session = create_session()
    sentiment_scores = []
    cache_valid = False
    
    # Check if we have valid cached sentiment data
    cache_file = f"{DATA_DIRECTORY}/sentiment_cache.json"
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                sentiment_cache = json.load(f)
                cache_time = datetime.fromisoformat(sentiment_cache['timestamp'])
                # Check if cache is fresh enough
                if datetime.now() - cache_time < timedelta(hours=cache_hours):
                    logger.info(f"Using cached sentiment data from {cache_time}")
                    return sentiment_cache['sentiment']
        except Exception as e:
            logger.warning(f"Error reading sentiment cache: {e}")
    
    try:
        news_sources = [
            {"name": "News API", "url": f"{CONFIG['data_sources']['news_api']}?q=KEYWORD&apiKey={CONFIG['api_key']}"},
            # Add more news sources as needed
        ]
        
        # Collect data from multiple sources with multiple keywords
        for keyword in keywords:
            keyword_scores = []
            
            for source in news_sources:
                try:
                    url = source["url"].replace("KEYWORD", keyword)
                    response = session.get(url, timeout=CONFIG["timeout"])
                    response.raise_for_status()
                    news_data = response.json()
                    
                    if 'articles' in news_data and news_data['articles']:
                        # Analyze sentiment for this source and keyword
                        sentiment = analyze_sentiment_vader(news_data['articles'])
                        keyword_scores.append(sentiment)
                        
                        # Cache the articles for offline analysis
                        today = datetime.now().strftime('%Y%m%d')
                        with open(f"{DATA_DIRECTORY}/news_{keyword}_{source['name']}_{today}.pkl", 'wb') as f:
                            pickle.dump(news_data['articles'], f)
                    else:
                        logger.warning(f"No articles found for keyword: {keyword} on {source['name']}")
                except Exception as e:
                    logger.warning(f"Error fetching sentiment from {source['name']} for {keyword}: {e}")
            
            # Average sentiment across all sources for this keyword
            if keyword_scores:
                keyword_sentiment = np.mean(keyword_scores)
                sentiment_scores.append(keyword_sentiment)
                logger.info(f"Sentiment for '{keyword}': {keyword_sentiment:.4f}")
        
        # Calculate overall sentiment and cache it
        if sentiment_scores:
            overall_sentiment = np.mean(sentiment_scores)
            
            # Cache the sentiment
            sentiment_cache = {
                'sentiment': overall_sentiment,
                'timestamp': datetime.now().isoformat(),
                'keywords': keywords
            }
            with open(cache_file, 'w') as f:
                json.dump(sentiment_cache, f)
                
            # Update market conditions
            market_conditions["sentiment"] = overall_sentiment
            
            return overall_sentiment
        else:
            logger.warning("No sentiment data available from any source")
            return 0
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis pipeline: {e}")
        
        # Try to use cached sentiment from today
        today = datetime.now().strftime('%Y%m%d')
        for keyword in keywords:
            for source_name in ["News API"]:  # Add more if you have more sources
                cache_file = f"{DATA_DIRECTORY}/news_{keyword}_{source_name}_{today}.pkl"
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'rb') as f:
                            articles = pickle.load(f)
                            return analyze_sentiment_vader(articles)
                    except Exception as inner_e:
                        logger.error(f"Error loading cached sentiment: {inner_e}")
        
        logger.warning("Using neutral sentiment as fallback")
        return 0

def analyze_sentiment_vader(articles):
    """Advanced sentiment analysis using VADER with preprocessing and weighting"""
    if not articles:
        return 0
    
    compound_scores = []
    article_weights = []  # Give more weight to recent and relevant articles
    
    # Current time for recency calculation
    current_time = datetime.now()
    
    # Preprocess and analyze each article
    for i, article in enumerate(articles):
        if 'description' in article and article['description']:
            # Calculate recency (weight more recent articles higher)
            if 'publishedAt' in article and article['publishedAt']:
                try:
                    pub_time = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
                    hours_old = (current_time - pub_time).total_seconds() / 3600
                    recency_weight = max(0.5, min(1.0, 1.0 - (hours_old / 72)))  # Linear decay over 72 hours
                except Exception:
                    recency_weight = 0.7  # Default weight if date parsing fails
            else:
                recency_weight = 0.7
            
            # Calculate relevance (weight title mentions higher)
            relevance_weight = 1.0
            if 'title' in article and article['title']:
                title_lower = article['title'].lower()
                if 'nifty' in title_lower or 'market' in title_lower:
                    relevance_weight = 1.5
            
            # Combine weights
            article_weight = recency_weight * relevance_weight
            
            # Preprocess text
            text = preprocess_text(article['description'])
            
            # Get sentiment scores with VADER
            sentiment_scores = sid.polarity_scores(text)
            compound_scores.append(sentiment_scores['compound'])
            article_weights.append(article_weight)
    
    if compound_scores:
        # Return weighted average of compound scores
        return np.average(compound_scores, weights=article_weights)
    else:
        return 0

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    try:
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
        
        # Join back into text
        processed_text = ' '.join(lemmatized_tokens)
        return processed_text
    except Exception as e:
        logger.error(f"Error in text preprocessing: {e}")
        return text

# Enhanced Technical Indicators with Advanced Support/Resistance Calculations
def add_technical_indicators(df):
    """
    Add comprehensive technical indicators and calculate support/resistance levels
    using multiple methodologies for robustness
    """
    # Basic price data must exist
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")
    
    try:
        # Momentum Indicators
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        df['MOM'] = talib.MOM(df['close'], timeperiod=10)
        df['ROC'] = talib.ROC(df['close'], timeperiod=10)
        
        # Volume Indicators
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        df['AD'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        
        # Volatility Indicators
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Trend Indicators
        df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'SMA_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'EMA_{period}'] = talib.EMA(df['close'], timeperiod=period)
        
        # Calculate Moving Average Crossovers
        df['SMA_10_20_cross'] = np.where(df['SMA_10'] > df['SMA_20'], 1, -1)
        df['EMA_10_20_cross'] = np.where(df['EMA_10'] > df['EMA_20'], 1, -1)
        
        # Bollinger Bands
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        
        # ENHANCED: Multiple timeframe support/resistance calculations
        
        # 1. Traditional Pivot Points (Daily)
        df['Pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['Support1'] = 2 * df['Pivot'] - df['high']
        df['Resistance1'] = 2 * df['Pivot'] - df['low']
        df['Support2'] = df['Pivot'] - (df['high'] - df['low'])
        df['Resistance2'] = df['Pivot'] + (df['high'] - df['low'])
        
        # 2. Calculate longer-term pivot points (weekly)
        if len(df) >= 5:
            # Group by week and calculate weekly pivots
            df['week'] = pd.to_datetime(df.index).isocalendar().week if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['date']).dt.isocalendar().week
            weekly_ohlc = df.groupby('week').agg({
                'high': 'max',
                'low': 'min',
                'close': lambda x: x.iloc[-1]
            })
            
            weekly_pivot = (weekly_ohlc['high'] + weekly_ohlc['low'] + weekly_ohlc['close']) / 3
            weekly_s1 = 2 * weekly_pivot - weekly_ohlc['high']
            weekly_r1 = 2 * weekly_pivot - weekly_ohlc['low']
            
            # Map these values back to the original dataframe
            for week in df['week'].unique():
                mask = df['week'] == week
                df.loc[mask, 'Weekly_Pivot'] = weekly_pivot.get(week, 0)
                df.loc[mask, 'Weekly_S1'] = weekly_s1.get(week, 0)
                df.loc[mask, 'Weekly_R1'] = weekly_r1.get(week, 0)
        
        # 3. Price-based Support/Resistance using recent price history
        lookback = min(100, len(df))
        recent_df = df.iloc[-lookback:].copy()
        
        # Find local maxima and minima
        price_delta = 0.0025  # 0.25% threshold for swing high/low
        df['Swing_High'] = 0
        df['Swing_Low'] = 0
        
        for i in range(2, len(df) - 2):
            # Swing high (local maximum)
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                df['Swing_High'].iloc[i] = df['high'].iloc[i]
            
            # Swing low (local minimum)
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                df['Swing_Low'].iloc[i] = df['low'].iloc[i]
        
        # Recent swing levels (non-zero values only)
        recent_swing_highs = df['Swing_High'].iloc[-lookback:][df['Swing_High'].iloc[-lookback:] > 0].tolist()
        recent_swing_lows = df['Swing_Low'].iloc[-lookback:][df['Swing_Low'].iloc[-lookback:] > 0].tolist()
        
        # Cluster nearby levels (within 0.3% of each other)
        clustered_highs = cluster_price_levels(recent_swing_highs, threshold=0.003)
        clustered_lows = cluster_price_levels(recent_swing_lows, threshold=0.003)
        
        # Store top 3 resistance and support levels based on frequency
        for i, level in enumerate(clustered_highs[:3]):
            df[f'Price_Resistance_{i+1}'] = level
            
        for i, level in enumerate(clustered_lows[:3]):
            df[f'Price_Support_{i+1}'] = level
        
        # 4. Fibonacci Retracement Levels
        highest_high = df['high'].rolling(window=20).max()
        lowest_low = df['low'].rolling(window=20).min()
        price_range = highest_high - lowest_low
        
        df['Fib_38.2'] = highest_high - 0.382 * price_range
        df['Fib_50.0'] = highest_high - 0.5 * price_range
        df['Fib_61.8'] = highest_high - 0.618 * price_range
        
        # 5. Volume Profile Based Support/Resistance (simplified)
        price_buckets = pd.cut(df['close'], bins=10)
        volume_profile = df.groupby(price_buckets)['volume'].sum()
        high_volume_levels = volume_profile.nlargest(2).index.tolist()
        
        # Get the mid-point of each bucket
        if high_volume_levels:
            df['Vol_Support'] = high_volume_levels[0].mid
            if len(high_volume_levels) > 1:
                df['Vol_Resistance'] = high_volume_levels[1].mid
        
        # 6. Dynamic Support/Resistance incorporating multiple methods
        # Assign weights to different S/R methodologies based on reliability
        weights = {
            'Pivot': 0.3,
            'Fib': 0.2,
            'Price': 0.3,
            'Vol': 0.2
        }
        
        # Calculate composite support level (using latest row)
        last_idx = len(df) - 1
        composite_support = (
            weights['Pivot'] * df['Support1'].iloc[last_idx] +
            weights['Fib'] * df['Fib_61.8'].iloc[last_idx] +
            weights['Price'] * (df['Price_Support_1'].iloc[last_idx] if 'Price_Support_1' in df.columns else df['Support1'].iloc[last_idx]) +
            weights['Vol'] * (df['Vol_Support'].iloc[last_idx] if 'Vol_Support' in df.columns else df['Support1'].iloc[last_idx])
        )
        
        # Calculate composite resistance level
        composite_resistance = (
            weights['Pivot'] * df['Resistance1'].iloc[last_idx] +
            weights['Fib'] * df['Fib_38.2'].iloc[last_idx] +
            weights['Price'] * (df['Price_Resistance_1'].iloc[last_idx] if 'Price_Resistance_1' in df.columns else df['Resistance1'].iloc[last_idx]) +
            weights['Vol'] * (df['Vol_Resistance'].iloc[last_idx] if 'Vol_Resistance' in df.columns else df['Resistance1'].iloc[last_idx])
        )
        
        # Add composite levels to DataFrame
        df['Composite_Support'] = composite_support
        df['Composite_Resistance'] = composite_resistance
        
        # Add strength indicators for support/resistance
        # Calculate how many methods confirm each level
        support_confirmations = 0
        resistance_confirmations = 0
        
        # Define confirmation threshold (0.5% from composite level)
        threshold = 0.005
        
        # Check if pivot points confirm
        if abs((df['Support1'].iloc[last_idx] / composite_support) - 1) < threshold:
            support_confirmations += 1
        if abs((df['Resistance1'].iloc[last_idx] / composite_resistance) - 1) < threshold:
            resistance_confirmations += 1
            
        # Check if Fibonacci levels confirm
        if abs((df['Fib_61.8'].iloc[last_idx] / composite_support) - 1) < threshold:
            support_confirmations += 1
        if abs((df['Fib_38.2'].iloc[last_idx] / composite_resistance) - 1) < threshold:
            resistance_confirmations += 1
            
        # Check if price-based levels confirm
        if 'Price_Support_1' in df.columns and abs((df['Price_Support_1'].iloc[last_idx] / composite_support) - 1) < threshold:
            support_confirmations += 1
        if 'Price_Resistance_1' in df.columns and abs((df['Price_Resistance_1'].iloc[last_idx] / composite_resistance) - 1) < threshold:
            resistance_confirmations += 1
            
        # Check if volume-based levels confirm
        if 'Vol_Support' in df.columns and abs((df['Vol_Support'].iloc[last_idx] / composite_support) - 1) < threshold:
            support_confirmations += 1
        if 'Vol_Resistance' in df.columns and abs((df['Vol_Resistance'].iloc[last_idx] / composite_resistance) - 1) < threshold:
            resistance_confirmations += 1
            
        # Add strength indicators (normalized from 0-1)
        df['Support_Strength'] = support_confirmations / 4
        df['Resistance_Strength'] = resistance_confirmations / 4
        
        # Update market condition indicators
        if last_idx > 0:
            # Calculate volatility based on ATR
            market_conditions["volatility"] = df['ATR'].iloc[last_idx] / df['close'].iloc[last_idx]
            
            # Calculate trend based on ADX and moving average direction
            adx_value = df['ADX'].iloc[last_idx]
            trend_direction = 1 if df['SMA_20'].iloc[last_idx] > df['SMA_50'].iloc[last_idx] else -1
            market_conditions["trend"] = trend_direction * min(adx_value / 30, 1.0)  # Normalize ADX
        
        # Fill NaN values with 0
        df.fillna(0, inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error adding technical indicators: {e}")
        # Return the original dataframe if we can't add indicators
        return df

def cluster_price_levels(price_levels, threshold=0.003):
    """
    Cluster nearby price levels and return the average of each cluster
    """
    if not price_levels:
        return []
    
    # Sort the price levels
    sorted_levels = sorted(price_levels)
    clusters = []
    current_cluster = [sorted_levels[0]]
    
    # Group into clusters
    for i in range(1, len(sorted_levels)):
        # If close to previous level, add to current cluster
        if (sorted_levels[i] - sorted_levels[i-1]) / sorted_levels[i-1] < threshold:
            current_cluster.append(sorted_levels[i])
        else:
            # Finalize current cluster and start a new one
            clusters.append(np.mean(current_cluster))
            current_cluster = [sorted_levels[i]]
    
    # Add the last cluster
    if current_cluster:
        clusters.append(np.mean(current_cluster))
    
    # Count frequency of levels in each cluster
    cluster_counts = []
    for cluster_value in clusters:
        count = sum(1 for price in price_levels if abs((price/cluster_value)-1) < threshold)
        cluster_counts.append((cluster_value, count))
    
    # Sort by frequency (most frequent first)
    sorted_clusters = [cluster for cluster, _ in sorted(cluster_counts, key=lambda x: x[1], reverse=True)]
    return sorted_clusters

# Create feature engineering pipeline
def engineer_features(df):
    """
    Create comprehensive feature set for model training/prediction
    using technical indicators, sentiment, and market conditions
    """
    try:
        # Make copy to avoid modifying original
        features_df = df.copy()
        
        # 1. Add technical indicators if not already present
        if 'RSI' not in features_df.columns:
            features_df = add_technical_indicators(features_df)
        
        # 2. Create interaction features
        features_df['RSI_MOM'] = features_df['RSI'] * features_df['MOM']
        features_df['OBV_RSI'] = features_df['OBV'] * features_df['RSI']
        features_df['BB_Width_ATR'] = features_df['BB_width'] * features_df['ATR']
        
        # 3. Create distance features
        features_df['Close_SMA20_Dist'] = (features_df['close'] - features_df['SMA_20']) / features_df['close']
        features_df['Close_SMA50_Dist'] = (features_df['close'] - features_df['SMA_50']) / features_df['close']
        features_df['Close_SMA200_Dist'] = (features_df['close'] - features_df['SMA_200']) / features_df['close']
        
        # 4. Volatility-normalized momentum
        features_df['Norm_Momentum'] = features_df['MOM'] / (features_df['ATR'] + 1e-6)
        
        # 5. Price relative to support/resistance
        if 'Composite_Support' in features_df.columns and 'Composite_Resistance' in features_df.columns:
            # Calculate where price is between support and resistance (0-1)
            price_range = features_df['Composite_Resistance'] - features_df['Composite_Support']
            features_df['SR_Position'] = (features_df['close'] - features_df['Composite_Support']) / (price_range + 1e-6)
            
            # Distance to nearest level
            features_df['Dist_To_Support'] = (features_df['close'] - features_df['Composite_Support']) / features_df['close']
            features_df['Dist_To_Resistance'] = (features_df['Composite_Resistance'] - features_df['close']) / features_df['close']
        
        # 6. Trend confirmation features
        features_df['Trend_Confirm'] = features_df['ADX'] * features_df['Close_SMA50_Dist']
        
        # 7. Volume confirmation features
        features_df['Volume_Trend'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
        
        # 8. Add external sentiment if available
        if 'sentiment' not in features_df.columns:
            # Try to get the latest sentiment or use market conditions value
            sentiment_value = market_conditions.get("sentiment", 0)
            features_df['sentiment'] = sentiment_value
        
        # 9. Calculate target variable if not already present
        # For classification: 1 if closes up X% in the next N bars, 0 otherwise
        horizon = 5
        threshold = 0.005  # 0.5%
        
        if 'target' not in features_df.columns and len(features_df) > horizon:
            # Calculate future returns
            features_df['future_return'] = features_df['close'].shift(-horizon) / features_df['close'] - 1
            
            # Create binary target - 1 for up moves, 0 for down or flat
            features_df['target'] = (features_df['future_return'] > threshold).astype(int)
            
            # Optional: Create multi-class target
            # 0 = down move, 1 = no significant move, 2 = up move
            features_df['target_multiclass'] = 1  # Default to no significant move
            features_df.loc[features_df['future_return'] < -threshold, 'target_multiclass'] = 0  # Down move
            features_df.loc[features_df['future_return'] > threshold, 'target_multiclass'] = 2  # Up move
        
        # 10. Create time-based features (optional)
        if 'date' in features_df.columns:
            features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
            features_df['hour_of_day'] = pd.to_datetime(features_df['date']).dt.hour
            
            # One-hot encode categorical time features
            for day in range(5):  # 0-4 for trading days
                features_df[f'day_{day}'] = (features_df['day_of_week'] == day).astype(int)
        
        # Fill remaining NaN values
        features_df.fillna(0, inplace=True)
        
        return features_df
    
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        return df

# Build a robust ML pipeline for stock prediction
def build_model(X_train, y_train, model_type='ensemble'):
    """
    Build and return a trained model for stock prediction
    with hyperparameter optimization and feature selection
    
    Args:
        X_train: Features dataframe
        y_train: Target series
        model_type: Type of model to build ('rf', 'gb', 'ensemble')
    
    Returns:
        Trained model object
    """
    try:
        # Define feature selection and preprocessing steps
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Start with basic preprocessing
        preprocessor = Pipeline([
            ('scaler', StandardScaler())
        ])
        
        # Define base models
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        # Create different pipelines based on model_type
        if model_type == 'rf':
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', rf_model)
            ])
            
            # Parameter grid for random forest
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [5, 10, 15, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            }
            
        elif model_type == 'gb':
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', gb_model)
            ])
            
            # Parameter grid for gradient boosting
            param_grid = {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [3, 5, 7],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            }
            
        else:  # default to ensemble
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', VotingClassifier(
                    estimators=[
                        ('rf', rf_model),
                        ('gb', gb_model)
                    ],
                    voting='soft'
                ))
            ])
            
            # Parameter grid for ensemble
            param_grid = {
                'classifier__rf__n_estimators': [100, 200],
                'classifier__rf__max_depth': [10, 15],
                'classifier__gb__n_estimators': [100, 200],
                'classifier__gb__learning_rate': [0.1, 0.2]
            }
        
        # Define time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Set up randomized search with time-series CV
        n_iter = 10  # Number of parameter combinations to try
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        
        # Fit the model
        logger.info(f"Training {model_type} model with randomized search...")
        search.fit(X_train, y_train)
        
        # Get best model
        best_model = search.best_estimator_
        
        # Log best parameters
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best cross-validation score: {search.best_score_:.4f}")
        
        # Save feature importance if applicable
        try:
            if model_type == 'rf':
                importances = best_model.named_steps['classifier'].feature_importances_
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                feature_importance.to_csv(f"{METRICS_DIRECTORY}/feature_importance_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
                logger.info(f"Top 5 important features: {feature_importance.head(5)}")
        except Exception as e:
            logger.warning(f"Couldn't save feature importance: {e}")
        
        return best_model
    
    except Exception as e:
        logger.error(f"Error building model: {e}")
        
        # Fall back to a simpler model if optimization fails
        logger.info("Falling back to basic model")
        
        simple_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        simple_pipeline.fit(X_train, y_train)
        return simple_pipeline

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model performance with various metrics
    and update monitoring statistics
    """
    try:
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Log metrics
        logger.info(f"Model Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Save metrics
        metrics_dict = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': cm.tolist(),
            'threshold': threshold
        }
        
        # Save to file
        metrics_file = f"{METRICS_DIRECTORY}/model_metrics_{datetime.now().strftime('%Y%m%d')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        
        # Update global model performance metrics
        global model_performance_metrics
        model_performance_metrics = metrics_dict
        
        # Check if model performance meets validation threshold
        if f1 < CONFIG['model_validation_threshold']:
            logger.warning(f"Model performance below threshold: {f1:.4f} < {CONFIG['model_validation_threshold']}")
            # Trigger model drift detection
            market_conditions["model_drift"] = 1.0
        else:
            market_conditions["model_drift"] = 0.0
        
        return metrics_dict
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None

# Trade management functions
def execute_trade(symbol, action, quantity, price=None):
    """
    Execute trade with broker API with fallback mechanisms
    and circuit breakers for risk management
    
    Args:
        symbol: Stock symbol
        action: 'BUY' or 'SELL'
        quantity: Number of shares/contracts
        price: Optional limit price
    
    Returns:
        Trade result dictionary
    """
    global daily_trades_count, last_trade_time, current_positions
    
    try:
        # Risk checks before executing trade
        
        # 1. Check if we've exceeded maximum trades per day
        if daily_trades_count >= CONFIG['trading_params']['max_trades_per_day']:
            logger.warning(f"Maximum daily trades ({CONFIG['trading_params']['max_trades_per_day']}) reached. Trade canceled.")
            return {'success': False, 'error': 'Max daily trades reached'}
        
        # 2. Check circuit breaker - if system health is poor, don't trade
        if market_conditions['system_health'] < CONFIG['circuit_breaker_threshold']:
            logger.warning(f"Circuit breaker triggered. System health: {market_conditions['system_health']:.2f}")
            return {'success': False, 'error': 'Circuit breaker triggered'}
        
        # 3. Check position size limits
        current_position = current_positions.get(symbol, 0)
        if action == 'BUY' and current_position + quantity > CONFIG['trading_params']['max_position_size']:
            logger.warning(f"Position size limit reached for {symbol}")
            # Adjust quantity to respect position size limit
            quantity = max(0, CONFIG['trading_params']['max_position_size'] - current_position)
            if quantity == 0:
                return {'success': False, 'error': 'Position size limit reached'}
        
        # 4. Throttle trading frequency
        current_time = datetime.now()
        if last_trade_time and (current_time - last_trade_time).total_seconds() < 300:  # 5 minutes
            logger.warning("Trade frequency too high. Enforcing cool-down period.")
            return {'success': False, 'error': 'Trading frequency too high'}
        
        # If all checks pass, proceed with trade
        session = create_session()
        
        # Construct trade request
        trade_request = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'order_type': 'MARKET'
        }
        
        if price:
            trade_request['order_type'] = 'LIMIT'
            trade_request['price'] = price
        
        # Add authentication
        headers = {
            'Authorization': f"Bearer {CONFIG['broker_api_key']}",
            'Content-Type': 'application/json'
        }
        
        # Send to broker API
        # NOTE: In a real implementation, this would be a real broker API
        # For this code, we'll simulate the response
        
        # Simulate API call
        simulated_trade_id = f"trade_{int(time.time())}_{symbol}_{action}"
        execution_price = price if price else fetch_real_time_data(symbol)[-1]['close']
        
        # Simulate successful trade
        trade_result = {
            'success': True, 
            'trade_id': simulated_trade_id,
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': execution_price,
            'timestamp': current_time.isoformat()
        }
        
        # Update position tracking
        if action == 'BUY':
            current_positions[symbol] = current_positions.get(symbol, 0) + quantity
        elif action == 'SELL':
            current_positions[symbol] = current_positions.get(symbol, 0) - quantity
        
        # Update trade history
        trade_history.append(trade_result)
        
        # Update trade counters
        daily_trades_count += 1
        last_trade_time = current_time
        
        # Save trade to file
        trade_file = f"{DATA_DIRECTORY}/trades_{datetime.now().strftime('%Y%m%d')}.json"
        with open(trade_file, 'a') as f:
            f.write(json.dumps(trade_result) + '\n')
        
        logger.info(f"Trade executed: {action} {quantity} {symbol} at {execution_price}")
        
        return trade_result
    
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        return {'success': False, 'error': str(e)}

def calculate_trade_signal(df, model, confidence_threshold=0.75):
    """
    Generate trade signals based on model predictions and market conditions
    with risk-adjusted position sizing
    """
    try:
        # Extract features for prediction
        features_df = engineer_features(df.copy())
        
        # Select only the features used in training
        model_features = get_model_features(model)
        
        # Use only the latest row for prediction
        latest_data = features_df.iloc[-1:].copy()
        
        # Filter to include only the features the model expects
        common_features = list(set(latest_data.columns) & set(model_features))
        X_pred = latest_data[common_features]
        
        # Fill any missing columns with zeros
        missing_features = list(set(model_features) - set(common_features))
        for feature in missing_features:
            X_pred[feature] = 0
            
        # Make sure columns are in the correct order
        X_pred = X_pred[model_features]
        
        # Get prediction probabilities
        pred_proba = model.predict_proba(X_pred)[0]
        
        # Extract probabilities for each class
        if len(pred_proba) == 2:  # Binary classification
            down_prob, up_prob = pred_proba
        else:  # Multi-class (down, neutral, up)
            down_prob, neutral_prob, up_prob = pred_proba
            
        # Determine signal direction
        signal = 0  # Default to no signal
        confidence = 0
        
        if up_prob > confidence_threshold:
            signal = 1  # Buy signal
            confidence = up_prob
        elif down_prob > confidence_threshold:
            signal = -1  # Sell signal
            confidence = down_prob
            
        # Adjust signal strength based on market conditions
        
        # 1. Reduce signal in high volatility
        if market_conditions['volatility'] > 0.02:  # 2% volatility is high
            signal *= 0.8
            
        # 2. Amplify signal if trend agrees with prediction
        if signal > 0 and market_conditions['trend'] > 0:
            signal *= 1.2
        elif signal < 0 and market_conditions['trend'] < 0:
            signal *= 1.2
            
        # 3. Include sentiment as a factor
        if signal > 0 and market_conditions['sentiment'] > 0.2:
            signal *= 1.1
        elif signal < 0 and market_conditions['sentiment'] < -0.2:
            signal *= 1.1
            
        # 4. Reduce signal if model drift is high
        if market_conditions['model_drift'] > 0.5:
            signal *= 0.7
            
        # 5. Calculate trade size based on confidence and market conditions
        base_quantity = CONFIG['trading_params']['max_position_size']
        adjusted_quantity = base_quantity * abs(signal) * min(confidence, 1.0)
        
        # Round to whole number
        quantity = max(1, int(adjusted_quantity))
        
        # Get current market price
        current_price = df['close'].iloc[-1]
        
        # Calculate stop loss and take profit levels
        stop_loss = current_price * (1 - CONFIG['trading_params']['stop_loss_pct'] * signal)
        take_profit = current_price * (1 + CONFIG['trading_params']['take_profit_pct'] * signal)
        
        # Generate final signal dict
        signal_dict = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'confidence': confidence,
            'quantity': quantity,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'market_conditions': market_conditions.copy()
        }
        
        return signal_dict
        
    except Exception as e:
        logger.error(f"Error calculating trade signal: {e}")
        return {'signal': 0, 'confidence': 0, 'error': str(e)}

def get_model_features(model):
    """
    Extract the feature names the model was trained on
    """
    # This is a simplified version - in reality, you'd need to 
    # extract features from your specific model type
    try:
        # Try to access feature names from various model types
        if hasattr(model, 'feature_names_in_'):
            return model.feature_names_in_
        elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            if hasattr(model.named_steps['classifier'], 'feature_names_in_'):
                return model.named_steps['classifier'].feature_names_in_
            
        # For more complex models, we might need to check from training data
        model_file = f"{MODEL_DIRECTORY}/model_features.json"
        if os.path.exists(model_file):
            with open(model_file, 'r') as f:
                return json.load(f)
                
        # Default case - return some common features
        logger.warning("Could not determine model features, using defaults")
        return ['RSI', 'MACD', 'MACD_signal', 'ATR', 'OBV', 'SMA_20', 'SMA_50', 
                'BB_width', 'ADX', 'MOM', 'sentiment']
                
    except Exception as e:
        logger.error(f"Error getting model features: {e}")
        return ['RSI', 'MACD', 'ATR', 'SMA_20', 'SMA_50']  # Minimal default set

# System monitoring and maintenance functions
def monitor_system_health():
    """
    Monitor system health and trading performance metrics
    """
    try:
        # 1. Check API connectivity
        api_status = check_api_connectivity()
        
        # 2. Check data quality
        data_quality = check_data_quality()
        
        # 3. Check model performance
        model_performance = check_model_performance()
        
        # 4. Check trading performance
        trading_performance = check_trading_performance()
        
        # Calculate overall system health score (0-1)
        health_components = [
            api_status * 0.3,
            data_quality * 0.2,
            model_performance * 0.3,
            trading_performance * 0.2
        ]
        
        overall_health = sum(health_components)
        market_conditions["system_health"] = overall_health
        
        # Log health status
        logger.info(f"System Health: {overall_health:.2f}")
        logger.info(f"Components: API={api_status:.2f}, Data={data_quality:.2f}, " +
                    f"Model={model_performance:.2f}, Trading={trading_performance:.2f}")
        
        # Check for critical conditions
        if overall_health < 0.5:
            logger.warning("System health critical. Consider maintenance or switching to fallback mode.")
            # Send alert (in a real system)
            
        # Return health metrics
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': overall_health,
            'api_status': api_status,
            'data_quality': data_quality,
            'model_performance': model_performance,
            'trading_performance': trading_performance
        }
    
    except Exception as e:
        logger.error(f"Error monitoring system health: {e}")
        market_conditions["system_health"] = 0.5  # Default to medium health on error
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 0.5,
            'error': str(e)
        }

def check_api_connectivity():
    """Check connectivity to required APIs"""
    session = create_session()
    apis_to_check = [
        CONFIG['data_sources']['price_api'],
        CONFIG['data_sources']['news_api']
    ]
    
    success_count = 0
    
    for api_url in apis_to_check:
        try:
            # Check if we can connect with a HEAD request
            response = session.head(api_url, timeout=CONFIG['timeout'])
            if response.status_code < 400:
                success_count += 1
        except Exception:
            pass
    
    return success_count / len(apis_to_check)

def check_data_quality():
    """Check quality of recent data"""
    try:
        # Check if we have recent data
        data_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith('_latest.pkl')]
        if not data_files:
            return 0.5  # Neutral if no data files
        
        quality_scores = []
        
        for file in data_files:
            try:
                with open(f"{DATA_DIRECTORY}/{file}", 'rb') as f:
                    data = pickle.load(f)
                
                # Check data freshness
                if 'timestamp' in data:
                    timestamp = data['timestamp']
                    age_hours = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds() / 3600
                    freshness_score = max(0, 1 - (age_hours / 24))  # 0 if older than 24 hours
                    quality_scores.append(freshness_score)
                
                # Check data completeness
                if 'prices' in data and len(data['prices']) > 0:
                    required_fields = ['open', 'high', 'low', 'close', 'volume']
                    if all(field in data['prices'][0] for field in required_fields):
                        quality_scores.append(1.0)
                    else:
                        quality_scores.append(0.5)
            except Exception:
                quality_scores.append(0.3)  # Penalize for errors
        
        # Calculate average quality score
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
    
    except Exception as e:
        logger.error(f"Error checking data quality: {e}")
        return 0.5

def check_trading_performance():
    """
    Check recent trading performance metrics.
    Evaluates profitability and adherence to risk limits.

    Returns:
        float: A score between 0 and 1 representing trading performance.
    """
    try:
        if not trade_history:
            logger.warning("No recent trade history available.")
            return 0.5  # Neutral if no trade history

        # Calculate total profit and average profit per trade
        total_profit = sum(trade.get('profit', 0) for trade in trade_history)
        total_trades = len(trade_history)
        avg_profit = total_profit / total_trades if total_trades > 0 else 0

        # Check adherence to risk limits (e.g., max drawdown)
        max_drawdown = max(
            (trade.get('drawdown', 0) for trade in trade_history),
            default=0
        )

        # Normalize scores
        profit_score = min(1.0, avg_profit / CONFIG['trading_params']['take_profit_pct'])
        drawdown_score = max(0.0, 1.0 - (max_drawdown / CONFIG['max_drawdown_pct']))

        # Combine scores with weights
        trading_performance = (profit_score * 0.7) + (drawdown_score * 0.3)
        logger.info(f"Trading Performance: Profit Score={profit_score:.2f}, Drawdown Score={drawdown_score:.2f}")
        return trading_performance

    except Exception as e:
        logger.error(f"Error checking trading performance: {e}")
        return 0.5  # Default to neutral score on error

def check_model_performance():
    """
    Check recent model performance metrics.
    Uses F1 score as the primary indicator of model performance.

    Returns:
        float: A score between 0 and 1 representing model performance.
    """
    try:
        global model_performance_metrics

        # If we have recent metrics
        if model_performance_metrics:
            f1_score = model_performance_metrics.get('f1', 0)
            logger.info(f"Model F1 Score: {f1_score:.4f}")
            # Normalize the F1 score to a 0-1 scale based on the validation threshold
            return min(1.0, f1_score / CONFIG['model_validation_threshold'])
        else:
            logger.warning("No recent model performance metrics available.")
            return 0.5  # Neutral if no metrics are available

    except Exception as e:
        logger.error(f"Error checking model performance: {e}")
        return 0.5  # Default to neutral score on error
