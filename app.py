import os
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from predictor import ForexPredictor
import json

# Load environment variables
load_dotenv()

app = Flask(__name__)

# API keys configuration
API_KEYS = [
    os.getenv('API_KEY_1', 'QSCWPKVUYLOD506J'),
    os.getenv('API_KEY_2', 'QE0TAOPZZN1VT8LH')
]
current_api_key_index = 0
api_call_count = 0
MAX_API_CALLS = 500

def get_api_key():
    """Rotate API keys when limit is reached"""
    global current_api_key_index, api_call_count
    
    if api_call_count >= MAX_API_CALLS:
        current_api_key_index = (current_api_key_index + 1) % len(API_KEYS)
        api_call_count = 0
        print(f"Switched to API key index: {current_api_key_index}")
    
    api_call_count += 1
    return API_KEYS[current_api_key_index]

def fetch_forex_data(symbol='EURUSD', interval='60min', output_size='full'):
    """Fetch forex data from Alpha Vantage"""
    api_key = get_api_key()
    url = f'https://www.alphavantage.co/query'
    
    params = {
        'function': 'FX_INTRADAY',
        'from_symbol': symbol[:3],
        'to_symbol': symbol[3:],
        'interval': interval,
        'outputsize': output_size,
        'apikey': api_key,
        'datatype': 'json'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'Time Series FX (' + interval + ')' in data:
            df = pd.DataFrame.from_dict(
                data['Time Series FX (' + interval + ')'], 
                orient='index'
            )
            df = df.astype(float)
            df.columns = ['open', 'high', 'low', 'close']
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df
        else:
            print(f"API Error: {data.get('Information', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def fetch_technical_indicators(symbol='EURUSD', interval='60min'):
    """Fetch technical indicators"""
    api_key = get_api_key()
    indicators = {}
    
    # Fetch SMA
    sma_url = f'https://www.alphavantage.co/query'
    sma_params = {
        'function': 'SMA',
        'symbol': symbol,
        'interval': interval,
        'time_period': 20,
        'series_type': 'close',
        'apikey': api_key
    }
    
    try:
        response = requests.get(sma_url, params=sma_params, timeout=10)
        sma_data = response.json()
        if 'Technical Analysis: SMA' in sma_data:
            indicators['sma'] = sma_data['Technical Analysis: SMA']
    except:
        pass
    
    # Fetch RSI
    rsi_params = {
        'function': 'RSI',
        'symbol': symbol,
        'interval': interval,
        'time_period': 14,
        'series_type': 'close',
        'apikey': api_key
    }
    
    try:
        response = requests.get(sma_url, params=rsi_params, timeout=10)
        rsi_data = response.json()
        if 'Technical Analysis: RSI' in rsi_data:
            indicators['rsi'] = rsi_data['Technical Analysis: RSI']
    except:
        pass
    
    return indicators

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['GET'])
def predict():
    """Generate predictions"""
    try:
        # Fetch data
        df = fetch_forex_data()
        if df is None or len(df) < 100:
            return jsonify({'error': 'Insufficient data or API limit reached'}), 400
        
        # Initialize predictor
        predictor = ForexPredictor()
        
        # Prepare features and predict
        predictions, entry_points = predictor.predict_with_entry_points(df)
        
        # Format response
        chart_data = {
            'timestamps': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()[-100:],
            'prices': df['close'].tolist()[-100:],
            'predictions': predictions[-100:],
            'entry_points': entry_points
        }
        
        return jsonify(chart_data)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/update', methods=['GET'])
def update_data():
    """Update data manually"""
    df = fetch_forex_data()
    if df is not None:
        return jsonify({'status': 'success', 'data_points': len(df)})
    return jsonify({'status': 'error'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)