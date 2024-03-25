import streamlit as st
import pandas as pd
import pickle
import yfinance as yf
import os
from ml_models.random_forest_model import random_forest_model
from source_files.load_data import load_data

st.title("Stock Price Predictor")

tickers = [
    "ADANIPORTS.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BPCL.NS",
    "BHARTIARTL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "HDFC.NS",
    "ICICIBANK.NS",
    "ITC.NS",
    "IOC.NS",
    "INFY.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NTPC.NS",
    "NESTLEIND.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SHREECEM.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TCS.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TECHM.NS",
    "TITAN.NS",
    "UPL.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
]

stock = st.selectbox(label="Select Stock Name", options=tickers)

data = pd.read_csv(f"ticker_data/{stock}.csv")
st.write(data)

if data is not None:
    with st.form("main_form"):
        col1, col2 = st.columns(2)
        with col1:
            opn = st.number_input("Open", value=200)
            high = st.number_input("High", value=180)
            low = st.number_input("Low", value=120)
        with col2:
            prevhigh = st.number_input("Previous High", value=170)
            prevlow = st.number_input("Previous Low", value=110)
            prevclose = st.number_input("Previous Close", value=150)

        submit = st.form_submit_button("Predict", type='primary')

        input_columns = ['Open', 'High', 'Low', 'PrevHigh', 'PrevLow', 'PrevClose']
        predict_model = random_forest_model(data, input_columns, 'Close', stock)
        
        if submit:
            prediction = predict_model.predict([[opn, high, low, prevhigh, prevlow, prevclose]])
            st.write("Close Prediction : ")
            st.header(f"$ {prediction[0]: .2f}", divider="red")
