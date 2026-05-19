import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Silver Price Predictor", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #1E1E2E;
        color: #FFFFFF;
    }
    h1 {
        color: #C0C0C0;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    .stDateInput > label {
        color: #A0A0A0;
    }
    .metric-card {
        background-color: #2D2D44;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        margin-top: 20px;
    }
    .price-value {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🪙 Silver Price Prediction")
st.markdown("<p style='text-align: center; color: #A0A0A0;'>Predict future silver prices using our trained Machine Learning models.</p>", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    rf_model = joblib.load('rf_model.pkl')
    df = pd.read_csv('silver_forecast_2026.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return rf_model, df

try:
    model, df_historical = load_models()
except Exception as e:
    st.warning("Models not found. Please train the models first.")
    st.stop()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Select Date for Prediction")
    target_date = st.date_input("Date", value=datetime.today() + timedelta(days=1))
    
    if st.button("Predict Price", use_container_width=True):
        td = pd.to_datetime(target_date)
        
        # Prepare features
        features = pd.DataFrame([{
            'Year': td.year,
            'Month': td.month,
            'Day': td.day,
            'DayOfWeek': td.dayofweek,
            'DayOfYear': td.dayofyear,
            'DateOrdinal': td.toordinal()
        }])
        
        pred_price = model.predict(features)[0]
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Price for {td.strftime('%Y-%m-%d')}</h3>
            <div class="price-value">${pred_price:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        with col2:
            st.markdown("### Price Trend & Prediction")
            
            fig = go.Figure()
            
            # Historical
            recent_hist = df_historical.tail(100) 
            fig.add_trace(go.Scatter(
                x=recent_hist['Date'], 
                y=recent_hist['Predicted_Price'],
                mode='lines',
                name='Historical Price',
                line=dict(color='#C0C0C0', width=2)
            ))
            
            # Prediction point
            fig.add_trace(go.Scatter(
                x=[td],
                y=[pred_price],
                mode='markers',
                name='Prediction',
                marker=dict(color='#4CAF50', size=15, symbol='star')
            ))
            
            last_date = recent_hist['Date'].iloc[-1]
            last_price = recent_hist['Predicted_Price'].iloc[-1]
            
            if td > last_date:
                fig.add_trace(go.Scatter(
                    x=[last_date, td],
                    y=[last_price, pred_price],
                    mode='lines',
                    line=dict(color='#4CAF50', width=2, dash='dash'),
                    showlegend=False
                ))

            fig.update_layout(
                plot_bgcolor='#1E1E2E',
                paper_bgcolor='#1E1E2E',
                font=dict(color='#FFFFFF'),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='#333344'),
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig, use_container_width=True)
