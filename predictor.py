import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from talib import talib
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import BollingerBands
import warnings
warnings.filterwarnings('ignore')

class ForexPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, df):
        """Create technical features for prediction"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Technical indicators using ta library
        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        df['rsi'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Moving averages
        df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        
        # Price position
        df['price_above_sma20'] = (df['close'] > df['sma_20']).astype(int)
        df['price_above_sma50'] = (df['close'] > df['sma_50']).astype(int)
        
        # Volatility
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Target: 1 if price will be higher in next 3 periods, 0 otherwise
        df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def train_model(self, df):
        """Train the prediction model"""
        # Create features
        df_features = self.create_features(df)
        
        # Feature columns
        feature_cols = ['rsi', 'macd', 'macd_diff', 'ema_12', 'ema_26', 
                       'sma_20', 'sma_50', 'bb_width', 'volatility',
                       'price_above_sma20', 'price_above_sma50',
                       'returns', 'high_low_ratio', 'close_open_ratio']
        
        X = df_features[feature_cols].values
        y = df_features['target'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return self.model
    
    def predict(self, df):
        """Make predictions on new data"""
        if not self.is_trained:
            self.train_model(df)
        
        df_features = self.create_features(df)
        
        feature_cols = ['rsi', 'macd', 'macd_diff', 'ema_12', 'ema_26', 
                       'sma_20', 'sma_50', 'bb_width', 'volatility',
                       'price_above_sma20', 'price_above_sma50',
                       'returns', 'high_low_ratio', 'close_open_ratio']
        
        # Ensure we have all required columns
        missing_cols = set(feature_cols) - set(df_features.columns)
        for col in missing_cols:
            df_features[col] = 0
        
        X = df_features[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        # Confidence score based on probability
        confidence = np.abs(probabilities - 0.5) * 2
        
        return predictions, confidence, probabilities
    
    def find_entry_points(self, df, predictions, confidence, threshold=0.7):
        """Identify entry points with high confidence"""
        entry_points = []
        
        for i in range(1, len(predictions) - 1):
            # High confidence signals only
            if confidence[i] > threshold:
                current_price = df.iloc[i]['close']
                prev_price = df.iloc[i-1]['close']
                
                # Buy signal (prediction = 1 and price going up)
                if predictions[i] == 1 and df.iloc[i]['close'] > df.iloc[i-1]['close']:
                    entry_points.append({
                        'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'buy',
                        'price': float(current_price),
                        'confidence': float(confidence[i]),
                        'reason': f"Bullish signal with {confidence[i]*100:.1f}% confidence"
                    })
                
                # Sell signal (prediction = 0 and price going down)
                elif predictions[i] == 0 and df.iloc[i]['close'] < df.iloc[i-1]['close']:
                    entry_points.append({
                        'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                        'type': 'sell',
                        'price': float(current_price),
                        'confidence': float(confidence[i]),
                        'reason': f"Bearish signal with {confidence[i]*100:.1f}% confidence"
                    })
        
        # Limit to top 5 entry points by confidence
        entry_points.sort(key=lambda x: x['confidence'], reverse=True)
        return entry_points[:5]
    
    def predict_with_entry_points(self, df):
        """Complete prediction pipeline"""
        predictions, confidence, probabilities = self.predict(df)
        entry_points = self.find_entry_points(df, predictions, confidence)
        
        # Format predictions for chart
        formatted_predictions = []
        for i, (pred, conf) in enumerate(zip(predictions, confidence)):
            formatted_predictions.append({
                'prediction': int(pred),
                'confidence': float(conf),
                'probability': float(probabilities[i])
            })
        
        return formatted_predictions, entry_points