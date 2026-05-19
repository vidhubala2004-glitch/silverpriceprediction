import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import math
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings('ignore')

import os

def main():
    print("Loading data...")
    # Use absolute path relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'silver_forecast_2026.csv')
    df = pd.read_csv(csv_path)
    df = df.dropna()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # RF Features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['DateOrdinal'] = df['Date'].apply(lambda x: x.toordinal())

    X_rf = df[['Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'DateOrdinal']]
    y_rf = df['Predicted_Price']

    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42, shuffle=False)

    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_rf, y_train_rf)
    rf_pred = rf_model.predict(X_test_rf)
    rf_rmse = math.sqrt(mean_squared_error(y_test_rf, rf_pred))
    rf_r2 = r2_score(y_test_rf, rf_pred)
    print(f"Random Forest - RMSE: {rf_rmse:.4f}, R-square: {rf_r2:.4f}")

    print("Training LSTM...")
    data = df['Predicted_Price'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    time_step = 60
    X_train_lstm, y_train_lstm = create_dataset(train_data, time_step)
    X_test_lstm, y_test_lstm = create_dataset(test_data, time_step)

    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train_lstm, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_lstm, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test_lstm, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_lstm, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc1 = nn.Linear(hidden_size, 25)
            self.fc2 = nn.Linear(25, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc1(out[:, -1, :])
            out = self.fc2(out)
            return out

    lstm_model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = lstm_model(inputs.unsqueeze(-1))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        lstm_pred_scaled = lstm_model(X_test_t.unsqueeze(-1)).numpy()
    
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
    y_test_lstm_actual = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

    lstm_rmse = math.sqrt(mean_squared_error(y_test_lstm_actual, lstm_pred))
    lstm_r2 = r2_score(y_test_lstm_actual, lstm_pred)
    print(f"LSTM - RMSE: {lstm_rmse:.4f}, R-square: {lstm_r2:.4f}")

    rf_model_path = os.path.join(script_dir, 'rf_model.pkl')
    joblib.dump(rf_model, rf_model_path)
    
    if rf_r2 > lstm_r2:
        print("Random Forest is the better model!")
        best_model_path = os.path.join(script_dir, 'best_model.pkl')
        joblib.dump(rf_model, best_model_path)
    else:
        print("LSTM is the better model!")
        best_model_path = os.path.join(script_dir, 'best_model.pth')
        torch.save(lstm_model.state_dict(), best_model_path)

if __name__ == '__main__':
    main()
