# chart_viewer.py (Revised - Focus on Occurence Date)

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots 

# Note: The get_chart_data function remains the same as it correctly handles strings.
@st.cache_data(ttl=600)  

def display_signal_chart(selected_row):

    if selected_row is None:
        return

    symbol = selected_row['Ticker Name']
    indicator = selected_row['Indicator']
    
    # 1. Date Extraction
    arounddate = selected_row['Occurence Date']
    
    ticker = yf.Ticker(symbol)
    if arounddate is not None:
        if isinstance(arounddate, str):
            arounddate = pd.to_datetime(arounddate)
        startdate = arounddate - pd.Timedelta(days=30)
        enddate = arounddate + pd.Timedelta(days=30)
        hist = ticker.history(start=startdate, end=enddate)
        title = f"{symbol} chart around {arounddate.date()}"
    else:
        hist = ticker.history(period=period)
        title = f"{symbol} {period} chart"
    
    if hist is None or hist.empty:
        st.warning(f"No chart data available for {symbol}")
        return None
    
    # FIX: Call reset_index() properly and assign back
    hist = hist.reset_index()  # This creates the 'Date' column
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('OHLC', 'Volume'),
        row_width=[0.2, 0.8]  # 60% height for candlesticks, 40% for volume (1.5:1 ratio)
    )
    
    # Candlestick (row 1)
    fig.add_trace(
        go.Candlestick(
            x=hist['Date'],
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # Volume bars (row 2)
    colors = ['green' if row['Close'] >= row['Open'] else 'red' 
              for _, row in hist.iterrows()]
    fig.add_trace(
        go.Bar(
            x=hist['Date'],
            y=hist['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return hist  # Return for other uses
