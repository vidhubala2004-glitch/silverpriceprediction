from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import datetime
import math

import os

app = Flask(__name__)

# Load and prepare data
# Use absolute path relative to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'silver_forecast_2026.csv')
df = pd.read_csv(csv_path)
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# We fit a polynomial degree 2 to the historical data to capture the smooth trend
# This provides the "slowly increase/decrease" behavior requested by the user instead of drastic drops.
dates_ord = df['Date'].apply(lambda x: x.toordinal()).values
prices = df['Predicted_Price'].values

# Fit a simple model for a smooth, stable future trend
poly_coeffs = np.polyfit(dates_ord, prices, 2)
poly_func = np.poly1d(poly_coeffs)

hist_dates = df['Date'].dt.strftime('%Y-%m-%d').tolist()
hist_prices = prices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    target_date_str = data.get('date')
    if not target_date_str:
        return jsonify({'error': 'No date provided'}), 400
    
    try:
        target_date = datetime.datetime.strptime(target_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    
    target_ord = target_date.toordinal()
    last_hist_ord = dates_ord[-1]
    
    # We want a smooth curve that starts from the last historical point.
    # We will interpolate linearly or use a damped approach to avoid sudden drops.
    # Actually, a simple approach is just drawing a line from last price with a slight upward drift + seasonality.
    
    days_ahead = target_ord - last_hist_ord
    last_price = prices[-1]
    
    if days_ahead > 0:
        # User requested: "in future it will be increase only not decrese"
        # We use a perfectly smooth exponential curve that ONLY goes up.
        annual_growth_rate = 0.05 # 5% steady growth per year
        
        # Base value grows slowly and steadily upwards
        final_pred = last_price * ((1 + annual_growth_rate) ** (days_ahead / 365.0))
    else:
        # For historical dates, just return the polyfit value
        final_pred = poly_func(target_ord)
        
    # Generate smooth future curve
    future_dates = []
    future_prices = []
    
    if days_ahead > 0:
        step = max(1, days_ahead // 60) # Generate up to 60 points for a perfectly smooth line
        for i in range(1, days_ahead + 1, step):
            cur_ord = last_hist_ord + i
            cur_date = datetime.datetime.fromordinal(cur_ord)
            
            # Monotonically increasing curve (no sine waves, no drops)
            cur_pred = last_price * ((1 + annual_growth_rate) ** (i / 365.0))
            
            future_dates.append(cur_date.strftime('%Y-%m-%d'))
            future_prices.append(cur_pred)
            
        # Ensure the final point is exactly in the array
        if future_dates[-1] != target_date_str:
            future_dates.append(target_date_str)
            future_prices.append(final_pred)
            
    return jsonify({
        'predicted_price': round(final_pred, 2),
        'historical_dates': hist_dates[-150:],
        'historical_prices': hist_prices[-150:],
        'future_dates': future_dates,
        'future_prices': future_prices
    })

if __name__ == '__main__':
    app.run(port=8501, debug=True)
