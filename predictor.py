import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import talib  # Use TA-Lib instead of ta
import warnings
warnings.filterwarnings('ignore')

class ForexPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, df):
        """Create technical features for prediction using TA-Lib"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # TA-Lib indicators
        # RSI
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        macd, macd_signal, macd_hist = talib.MACD(df['close'], 
                                                  fastperiod=12, 
                                                  slowperiod=26, 
                                                  signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_diff'] = macd_hist
        
        # Moving averages
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], 
                                           timeperiod=20, 
                                           nbdevup=2, 
                                           nbdevdn=2)
        df['bb_high'] = upper
        df['bb_low'] = lower
        df['bb_middle'] = middle
        df['bb_width'] = (upper - lower) / middle
        
        # Additional indicators
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['obv'] = talib.OBV(df['close'], df['volume'] if 'volume' in df.columns else df['close'])
        
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
                       'returns', 'high_low_ratio', 'close_open_ratio',
                       'atr', 'obv']
        
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
            random_state=42
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
                       'returns', 'high_low_ratio', 'close_open_ratio',
                       'atr', 'obv']
        
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
                
                # Additional confirmation: RSI conditions
                rsi = df.iloc[i]['rsi'] if 'rsi' in df.columns else 50
                
                # Buy signal (prediction = 1 and price going up)
                if predictions[i] == 1 and df.iloc[i]['close'] > df.iloc[i-1]['close']:
                    # Additional RSI confirmation (not oversold)
                    if rsi > 30 and rsi < 70:
                        entry_points.append({
                            'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'buy',
                            'price': float(current_price),
                            'confidence': float(confidence[i]),
                            'reason': f"Bullish signal with {confidence[i]*100:.1f}% confidence (RSI: {rsi:.1f})"
                        })
                
                # Sell signal (prediction = 0 and price going down)
                elif predictions[i] == 0 and df.iloc[i]['close'] < df.iloc[i-1]['close']:
                    # Additional RSI confirmation (not overbought)
                    if rsi > 30 and rsi < 70:
                        entry_points.append({
                            'timestamp': df.index[i].strftime('%Y-%m-%d %H:%M:%S'),
                            'type': 'sell',
                            'price': float(current_price),
                            'confidence': float(confidence[i]),
                            'reason': f"Bearish signal with {confidence[i]*100:.1f}% confidence (RSI: {rsi:.1f})"
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