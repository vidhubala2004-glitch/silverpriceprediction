# silverpriceprediction

## Description
The Silver Price Forecaster is a powerful machine learning and deep learning tool that helps investors, traders, and businesses forecast future silver prices. Silver price prediction is essential for managing portfolio risks, optimizing entry/exit points, and capturing market opportunities. This project leverages advanced time-series analysis and forecasting models (including PyTorch LSTM and Random Forest) to identify market patterns and project future silver prices based on historical data.

## Features

### 📈 Silver Price Forecasting
Utilizes pre-trained machine learning and deep learning models to project future silver prices. It evaluates the performance of both Random Forest and PyTorch LSTM to dynamically select the best-performing model.

### 🌐 Dual Web Interfaces
Offers two intuitive, premium web interfaces:
- **Flask IoT Dashboard:** A sleek, glassmorphic dark-theme UI featuring smooth real-time visualization, custom-designed charts, and automated linear/exponential projections.
- **Streamlit App:** A clean, data-science-centric dashboard using Plotly for quick interactions, model selection, and currency conversion.

### 🪙 INR Currency & Trend Control
Includes support for currency conversion (USD/INR) and features a "Monotonic Growth Trajectory" setting to ensure predicted paths follow a steady upward trend, preventing unrealistic market drops.

### 🔬 Model Analysis & Comparison
Includes a comprehensive Python training pipeline (`train.py`) that compares models based on RMSE (Root Mean Squared Error) and R-square metrics to automatically save the most accurate forecasting weights.

### ⚙️ Customizable
Open-source and highly modular. The data source, training hyperparameters, network layers, and dashboard visuals are customizable to adapt to other commodity assets or custom datasets.

### 🐙 Open Source
This project is open source, encouraging collaboration, feedback, and contributions from the community to improve forecasting accuracy and dashboard features.
