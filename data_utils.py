# data_utils.py

import streamlit as st
import yfinance as yf
from datetime import date

# Function to load tickers from the Nifty500.txt file
@st.cache_data
def load_tickers(filename="Nifty500.txt"):
    """
    Loads tickers from a file where tickers are comma-separated on a single line.
    """
    try:
        with open(filename, 'r') as f:
            content = f.read().strip()
            tickers = [ticker.strip() for ticker in content.split(',') if ticker.strip()]
            
        if not tickers:
             st.error(f"Error: The file '{filename}' is empty or contains no valid comma-separated tickers.")
             st.stop()
             
        return tickers
    except FileNotFoundError:
        st.error(f"Error: The file '{filename}' was not found in the current directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        st.stop()

# Function to get data using yf.Ticker
@st.cache_data
def get_data(ticker, start_date, end_date):
    """
    Fetches historical stock data using yf.Ticker for better reliability.
    """
    try:
        # Fetch an extra 200 days of data for indicators with long lookback periods (e.g., SMA_200)
        # We fetch history up to the 'end_date'
        buffer_start_date = start_date.replace(year=start_date.year - 1) 
        
        # 🎯 CHANGE: Using yf.Ticker().history() instead of yf.download()
        ticker_obj = yf.Ticker(ticker)
        data = ticker_obj.history(start=buffer_start_date, end=end_date, auto_adjust=False)
        
        if data.empty:
            return None
        
        # Trim the data back to the user-selected start date before returning
        return data[data.index.date >= start_date]
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None
