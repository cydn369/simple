import os
from datetime import datetime
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ======================================
# Load Nifty tickers from local files
# ======================================

def load_tickers_from_file(filename):
    """
    Read comma-separated tickers from a text file relative to the current working directory.
    Example: nifty50.txt, nifty500.txt
    """
    try:
        base_dir = os.getcwd()
        filepath = os.path.join(base_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        tickers = []
        for line in content.strip().split("\n"):
            line_tickers = [t.strip().upper() for t in line.split(",") if t.strip()]
            tickers.extend(line_tickers)
        return sorted(set(tickers))
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return []
# ======================================
# Simple trend utilities
# ======================================
def prior_trend(prices, lookback=14):
    if len(prices) < lookback:
        return "N/A"
    return "Uptrend" if prices.iloc[-1] > prices.iloc[-lookback] else "Downtrend"

def prior_volume_trend(volumes, lookback=10):
    if len(volumes) < 2 * lookback:
        return "N/A"
    recent = volumes.iloc[-lookback:].mean()
    prior = volumes.iloc[-2 * lookback:-lookback].mean()
    return "Increasing" if recent > prior else "Decreasing"
# ======================================
# Chart indicators (candlestick patterns)
# ======================================
def bullish_marubozu(hist):
    if len(hist) < 2:
        return False
    o, h, l, c = hist.iloc[-2][["Open", "High", "Low", "Close"]]
    body = c - o
    if body <= 0:
        return False
    pattern_ok = (h - c <= 0.05 * body) and (o - l <= 0.05 * body)
    confirm = hist.iloc[-1]
    confirm_ok = confirm["Close"] > confirm["Open"] and confirm["Close"] > h
    return pattern_ok and confirm_ok

def bearish_marubozu(hist):
    if len(hist) < 2:
        return False
    o, h, l, c = hist.iloc[-2][["Open", "High", "Low", "Close"]]
    body = o - c
    if body <= 0:
        return False
    pattern_ok = (h - o <= 0.05 * body) and (c - l <= 0.05 * body)
    confirm = hist.iloc[-1]
    confirm_ok = confirm["Close"] < confirm["Open"] and confirm["Close"] < l
    return pattern_ok and confirm_ok

def doji(hist):
    if len(hist) < 1:
        return False
    last = hist.iloc[-1]
    rng = last["High"] - last["Low"]
    if rng == 0:
        return False
    return abs(last["Close"] - last["Open"]) <= 0.2 * rng

def hammer(hist):
    if len(hist) < 2:
        return False
    pattern, confirm = hist.iloc[-2], hist.iloc[-1]
    body = abs(pattern["Close"] - pattern["Open"])
    upper_shadow = pattern["High"] - max(pattern["Open"], pattern["Close"])
    lower_shadow = min(pattern["Open"], pattern["Close"]) - pattern["Low"]
    pattern_ok = lower_shadow >= 1.5 * body and upper_shadow <= 1.5 * body
    confirm_ok = confirm["Close"] > confirm["Open"] and confirm["Close"] > pattern["High"]
    return pattern_ok and confirm_ok

def inverted_hammer(hist):
    if len(hist) < 2:
        return False
    pattern, confirm = hist.iloc[-2], hist.iloc[-1]
    body = abs(pattern["Close"] - pattern["Open"])
    upper_shadow = pattern["High"] - max(pattern["Open"], pattern["Close"])
    lower_shadow = min(pattern["Open"], pattern["Close"]) - pattern["Low"]
    pattern_ok = upper_shadow >= 1.5 * body and lower_shadow <= 1.5 * body
    confirm_ok = confirm["Close"] > confirm["Open"] and confirm["Close"] > pattern["High"]
    return pattern_ok and confirm_ok

def bullish_engulfing(hist):
    if len(hist) < 3:
        return False
    prev, pattern, confirm = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    pattern_ok = (
        prev["Close"] < prev["Open"] and
        pattern["Close"] > pattern["Open"] and
        pattern["Open"] <= prev["Close"] and
        pattern["Close"] >= prev["Open"]
    )
    confirm_ok = confirm["Close"] > confirm["Open"] and confirm["Close"] > pattern["High"]
    return pattern_ok and confirm_ok

def bearish_engulfing(hist):
    if len(hist) < 3:
        return False
    prev, pattern, confirm = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    pattern_ok = (
        prev["Close"] > prev["Open"] and
        pattern["Close"] < pattern["Open"] and
        pattern["Open"] >= prev["Close"] and
        pattern["Close"] <= prev["Open"]
    )
    confirm_ok = confirm["Close"] < confirm["Open"] and confirm["Close"] < pattern["Low"]
    return pattern_ok and confirm_ok

def morning_star(hist):
    if len(hist) < 4:
        return False
    a, b, c, confirm = hist.iloc[-4], hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    pattern_ok = (
        a["Close"] < a["Open"] and
        abs(b["Close"] - b["Open"]) < abs(a["Open"] - a["Close"]) * 0.5 and
        b["Close"] < a["Close"] and
        c["Close"] > c["Open"] and
        c["Close"] > (a["Open"] + a["Close"]) / 2
    )
    confirm_ok = confirm["Close"] > confirm["Open"] and confirm["Close"] > c["High"]
    return pattern_ok and confirm_ok

def evening_star(hist):
    if len(hist) < 4:
        return False
    a, b, c, confirm = hist.iloc[-4], hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    pattern_ok = (
        a["Close"] > a["Open"] and
        abs(b["Close"] - b["Open"]) < abs(a["Close"] - a["Open"]) * 0.5 and
        b["Close"] > a["Close"] and
        c["Close"] < c["Open"] and
        c["Close"] < (a["Open"] + a["Close"]) / 2
    )
    confirm_ok = confirm["Close"] < confirm["Open"] and confirm["Close"] < c["Low"]
    return pattern_ok and confirm_ok

def piercing_line(hist):
    if len(hist) < 3:
        return False
    prev, pattern, confirm = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    pattern_ok = (
        prev["Close"] < prev["Open"] and
        pattern["Open"] < prev["Close"] and
        pattern["Close"] > (prev["Open"] + prev["Close"]) / 2 and
        pattern["Close"] < prev["Open"]
    )
    confirm_ok = confirm["Close"] > confirm["Open"] and confirm["Close"] > pattern["High"]
    return pattern_ok and confirm_ok

def dark_cloud_cover(hist):
    if len(hist) < 3:
        return False
    prev, pattern, confirm = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    pattern_ok = (
        prev["Close"] > prev["Open"] and
        pattern["Open"] > prev["Close"] and
        pattern["Close"] < (prev["Open"] + prev["Close"]) / 2 and
        pattern["Close"] > prev["Open"]
    )
    confirm_ok = confirm["Close"] < confirm["Open"] and confirm["Close"] < pattern["Low"]
    return pattern_ok and confirm_ok

def spinning_top(hist):
    if len(hist) < 1:
        return False
    last = hist.iloc[-1]
    total_range = last["High"] - last["Low"]
    if total_range == 0:
        return False
    body = abs(last["Close"] - last["Open"])
    upper_shadow = last["High"] - max(last["Open"], last["Close"])
    lower_shadow = min(last["Open"], last["Close"]) - last["Low"]
    return (
        body <= 0.4 * total_range and
        upper_shadow >= 0.5 * body and
        lower_shadow >= 0.5 * body
    )

def rising_three_methods(hist):
    if len(hist) < 5:
        return False
    a, b, c, d, e = hist.iloc[-5], hist.iloc[-4], hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    cond1 = a["Close"] > a["Open"] and (a["Close"] - a["Open"]) > (a["High"] - a["Low"]) * 0.6
    cond2 = all([
        candle["High"] <= a["High"] and
        candle["Low"] >= a["Low"] and
        abs(candle["Close"] - candle["Open"]) < (a["Close"] - a["Open"]) * 0.5
        for candle in [b, c, d]
    ])
    cond3 = e["Close"] > e["Open"] and e["Close"] > a["Close"]
    return cond1 and cond2 and cond3

def abandoned_baby(hist):
    if len(hist) < 3:
        return False
    a, b, c = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    rng_b = b["High"] - b["Low"]
    if rng_b == 0:
        return False
    is_doji = abs(b["Close"] - b["Open"]) <= 0.1 * rng_b
    bullish = (
        a["Close"] < a["Open"] and
        is_doji and b["High"] < a["Low"] and
        c["Close"] > c["Open"] and c["Open"] > b["High"]
    )
    bearish = (
        a["Close"] > a["Open"] and
        is_doji and b["Low"] > a["High"] and
        c["Close"] < c["Open"] and c["Open"] < b["Low"]
    )
    return bullish or bearish

def three_inside_up(hist):
    if len(hist) < 3:
        return False
    a, b, c = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    cond1 = a["Close"] < a["Open"]
    cond2 = (
        b["Close"] > b["Open"] and
        b["Open"] >= a["Close"] and
        b["Close"] <= a["Open"]
    )
    cond3 = c["Close"] > c["Open"] and c["Close"] > a["Open"]
    return cond1 and cond2 and cond3

def three_inside_down(hist):
    if len(hist) < 3:
        return False
    a, b, c = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    cond1 = a["Close"] > a["Open"]
    cond2 = (
        b["Close"] < b["Open"] and
        b["Open"] <= a["Close"] and
        b["Close"] >= a["Open"]
    )
    cond3 = c["Close"] < c["Open"] and c["Close"] < a["Open"]
    return cond1 and cond2 and cond3

def bullish_tasuki_gap(hist):
    if len(hist) < 3:
        return False
    a, b, c = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    cond1 = a["Close"] > a["Open"]
    cond2 = (
        b["Close"] < b["Open"] and
        b["Open"] >= a["Open"] and b["Open"] <= a["Close"] and
        b["Close"] > a["Open"]
    )
    cond3 = c["Close"] > c["Open"] and c["Close"] > a["Close"]
    return cond1 and cond2 and cond3

def bearish_tasuki_gap(hist):
    if len(hist) < 3:
        return False
    a, b, c = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    cond1 = a["Close"] < a["Open"]
    cond2 = (
        b["Close"] > b["Open"] and
        b["Open"] <= a["Open"] and b["Open"] >= a["Close"] and
        b["Close"] < a["Open"]
    )
    cond3 = c["Close"] < c["Open"] and c["Close"] < a["Close"]
    return cond1 and cond2 and cond3

def mat_hold(hist):
    if len(hist) < 5:
        return False
    a, b, c, d, e = hist.iloc[-5], hist.iloc[-4], hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    cond1 = a["Close"] > a["Open"] and (a["Close"] - a["Open"]) > (a["High"] - a["Low"]) * 0.6
    cond2 = all([
        abs(candle["Close"] - candle["Open"]) < (a["Close"] - a["Open"]) * 0.5 and
        candle["Low"] > a["Open"]
        for candle in [b, c, d]
    ])
    cond3 = e["Close"] > e["Open"] and e["Close"] > a["Close"]
    return cond1 and cond2 and cond3

def kicking(hist):
    if len(hist) < 2:
        return False
    a, b = hist.iloc[-2], hist.iloc[-1]

    def is_marubozu(candle):
        body = abs(candle["Close"] - candle["Open"])
        upper_shadow = candle["High"] - max(candle["Open"], candle["Close"])
        lower_shadow = min(candle["Open"], candle["Close"]) - candle["Low"]
        return body > 0 and upper_shadow <= 0.05 * body and lower_shadow <= 0.05 * body

    cond1 = is_marubozu(a)
    cond2 = is_marubozu(b) and (
        (a["Close"] > a["Open"] and b["Close"] < b["Open"]) or
        (a["Close"] < a["Open"] and b["Close"] > b["Open"])
    )
    gap_up = (a["Close"] < a["Open"] and b["Open"] > a["High"])
    gap_down = (a["Close"] > a["Open"] and b["Open"] < a["Low"])
    return cond1 and cond2 and (gap_up or gap_down)

def three_white_soldiers(hist):
    if len(hist) < 3:
        return False
    a, b, c = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]

    def is_strong_bullish(candle):
        rng = candle["High"] - candle["Low"]
        if rng == 0:
            return False
        return candle["Close"] > candle["Open"] and (candle["Close"] - candle["Open"]) > 0.6 * rng

    cond1 = is_strong_bullish(a)
    cond2 = is_strong_bullish(b) and b["Open"] >= a["Open"] and b["Open"] <= a["Close"] and b["Close"] > a["Close"]
    cond3 = is_strong_bullish(c) and c["Open"] >= b["Open"] and c["Open"] <= b["Close"] and c["Close"] > b["Close"]
    return cond1 and cond2 and cond3

def three_black_crows(hist):
    if len(hist) < 3:
        return False
    a, b, c = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]

    def is_strong_bearish(candle):
        rng = candle["High"] - candle["Low"]
        if rng == 0:
            return False
        return candle["Close"] < candle["Open"] and (candle["Open"] - candle["Close"]) > 0.6 * rng

    cond1 = is_strong_bearish(a)
    cond2 = is_strong_bearish(b) and b["Open"] <= a["Open"] and b["Open"] >= a["Close"] and b["Close"] < a["Close"]
    cond3 = is_strong_bearish(c) and c["Open"] <= b["Open"] and c["Open"] >= b["Close"] and c["Close"] < b["Close"]
    return cond1 and cond2 and cond3

def rising_window(hist):
    if len(hist) < 2:
        return False
    a, b = hist.iloc[-2], hist.iloc[-1]
    cond1 = a["Close"] > a["Open"]
    cond2 = b["Close"] > b["Open"]
    cond3 = b["Low"] > a["High"]
    return cond1 and cond2 and cond3

def falling_window(hist):
    if len(hist) < 2:
        return False
    a, b = hist.iloc[-2], hist.iloc[-1]
    cond1 = a["Close"] < a["Open"]
    cond2 = b["Close"] < b["Open"]
    cond3 = b["High"] < a["Low"]
    return cond1 and cond2 and cond3

def bullish_separating_lines(hist):
    if len(hist) < 2:
        return False
    a, b = hist.iloc[-2], hist.iloc[-1]
    rng = a["High"] - a["Low"]
    if rng == 0:
        return False
    cond1 = a["Close"] > a["Open"]
    cond2 = b["Close"] > b["Open"]
    cond3 = abs(b["Open"] - a["Open"]) <= 0.05 * rng
    cond4 = b["Close"] > a["Close"]
    return cond1 and cond2 and cond3 and cond4

def bearish_separating_lines(hist):
    if len(hist) < 2:
        return False
    a, b = hist.iloc[-2], hist.iloc[-1]
    rng = a["High"] - a["Low"]
    if rng == 0:
        return False
    cond1 = a["Close"] < a["Open"]
    cond2 = b["Close"] < b["Open"]
    cond3 = abs(b["Open"] - a["Open"]) <= 0.05 * rng
    cond4 = b["Close"] < a["Close"]
    return cond1 and cond2 and cond3 and cond4

def upside_gap_two_crows(hist):
    if len(hist) < 3:
        return False
    a, b, c = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    cond1 = a["Close"] > a["Open"]
    cond2 = (b["Close"] < b["Open"] and b["Open"] > a["Close"] and b["Close"] > a["Close"])
    cond3 = (c["Close"] < c["Open"] and c["Open"] > b["Open"] and c["Close"] < b["Close"] and c["Close"] > a["Close"])
    return cond1 and cond2 and cond3

def on_neck(hist):
    if len(hist) < 2:
        return False
    a, b = hist.iloc[-2], hist.iloc[-1]
    rng = a["High"] - a["Low"]
    if rng == 0:
        return False
    cond1 = a["Close"] < a["Open"]
    cond2 = b["Close"] > b["Open"]
    cond3 = b["Open"] < a["Close"] and abs(b["Close"] - a["Close"]) <= 0.1 * rng
    return cond1 and cond2 and cond3

def in_neck(hist):
    if len(hist) < 2:
        return False
    a, b = hist.iloc[-2], hist.iloc[-1]
    cond1 = a["Close"] < a["Open"]
    cond2 = b["Close"] > b["Open"]
    cond3 = b["Open"] < a["Close"] and b["Close"] > a["Close"] and b["Close"] < a["Open"]
    return cond1 and cond2 and cond3

def thrusting(hist):
    if len(hist) < 2:
        return False
    a, b = hist.iloc[-2], hist.iloc[-1]
    midpoint = (a["Open"] + a["Close"]) / 2
    cond1 = a["Close"] < a["Open"]
    cond2 = b["Close"] > b["Open"]
    cond3 = b["Open"] < a["Close"] and b["Close"] > midpoint and b["Close"] < a["Open"]
    return cond1 and cond2 and cond3

def deliberation(hist):
    if len(hist) < 3:
        return False
    a, b, c = hist.iloc[-3], hist.iloc[-2], hist.iloc[-1]
    rng_a = a["High"] - a["Low"]
    if rng_a == 0:
        return False
    cond1 = a["Close"] > a["Open"] and (a["Close"] - a["Open"]) > 0.6 * rng_a
    cond2 = b["Close"] > b["Open"] and (b["Close"] - b["Open"]) < (a["Close"] - a["Open"])
    body_c = abs(c["Close"] - c["Open"])
    cond3 = (
        body_c < 0.5 * (b["Close"] - b["Open"]) and
        abs(c["Close"] - b["Close"]) <= 0.1 * (b["High"] - b["Low"])
    )
    return cond1 and cond2 and cond3
# ======================================
# Technical indicators
# ======================================
def rsi(hist, period=14):
    if len(hist) < period + 1:
        return None
    delta = hist["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    return float(rsi_series.iloc[-1])

def macd(hist, fast=12, slow=26, signal=9):
    if len(hist) < slow + signal:
        return None
    fast_ma = hist["Close"].ewm(span=fast).mean()
    slow_ma = hist["Close"].ewm(span=slow).mean()
    macd_line = fast_ma - slow_ma
    signal_line = macd_line.ewm(span=signal).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])

def golden_cross(hist):
    if len(hist) < 200:
        return False
    ma50 = hist["Close"].rolling(50).mean()
    ma200 = hist["Close"].rolling(200).mean()
    return ma50.iloc[-2] <= ma200.iloc[-2] and ma50.iloc[-1] > ma200.iloc[-1]

def death_cross(hist):
    if len(hist) < 200:
        return False
    ma50 = hist["Close"].rolling(50).mean()
    ma200 = hist["Close"].rolling(200).mean()
    return ma50.iloc[-2] >= ma200.iloc[-2] and ma50.iloc[-1] < ma200.iloc[-1]

def bollinger_breakout(hist, period=20, num_std=2):
    if len(hist) < period + 1:
        return False
    ma = hist["Close"].rolling(period).mean()
    std = hist["Close"].rolling(period).std()
    upper = ma + num_std * std
    return hist["Close"].iloc[-2] <= upper.iloc[-2] and hist["Close"].iloc[-1] > upper.iloc[-1]

def bollinger_breakdown(hist, period=20, num_std=2):
    if len(hist) < period + 1:
        return False
    ma = hist["Close"].rolling(period).mean()
    std = hist["Close"].rolling(period).std()
    lower = ma - num_std * std
    return hist["Close"].iloc[-2] >= lower.iloc[-2] and hist["Close"].iloc[-1] < lower.iloc[-1]

def volume_spike(hist, lookback=20):
    if len(hist) < lookback:
        return False
    avg_vol = hist["Volume"].rolling(lookback).mean().iloc[-1]
    return hist["Volume"].iloc[-1] > 1.5 * avg_vol
# ======================================
# Financial indicators
# ======================================
def marketcap_gt_10b(info):
    return info.get("marketCap", 0) > 10_000_000_000

def marketcap_lt_1b(info):
    return info.get("marketCap", 0) < 1_000_000_000

def pe_lt_20(info):
    return info.get("trailingPE", 999) < 20

def pe_gt_40(info):
    return info.get("trailingPE", 0) > 40

def eps_positive_growth(info):
    return info.get("earningsGrowth", 0) > 0

def eps_negative_growth(info):
    return info.get("earningsGrowth", 0) < 0

def dividend_yield_gt_2(info):
    return info.get("dividendYield", 0) and info["dividendYield"] > 0.02

def debt_equity_lt_1(info):
    return info.get("debtToEquity", 999) < 1
# ======================================
# Indicator mapping
# ======================================
INDICATOR_CHECKS = {
    # Chart patterns
    "Bullish_Marubozu": lambda hist, info: bullish_marubozu(hist),
    "Bearish_Marubozu": lambda hist, info: bearish_marubozu(hist),
    "Doji": lambda hist, info: doji(hist),
    "Hammer": lambda hist, info: hammer(hist),
    "Inverted_Hammer": lambda hist, info: inverted_hammer(hist),
    "Bullish_Engulfing": lambda hist, info: bullish_engulfing(hist),
    "Bearish_Engulfing": lambda hist, info: bearish_engulfing(hist),
    "Morning_Star": lambda hist, info: morning_star(hist),
    "Evening_Star": lambda hist, info: evening_star(hist),
    "Piercing_Line": lambda hist, info: piercing_line(hist),
    "Dark_Cloud_Cover": lambda hist, info: dark_cloud_cover(hist),
    "Spinning_Top": lambda hist, info: spinning_top(hist),
    "Rising_Three_Methods": lambda hist, info: rising_three_methods(hist),
    "Abandoned_Baby": lambda hist, info: abandoned_baby(hist),
    "Three_Inside_Up": lambda hist, info: three_inside_up(hist),
    "Three_Inside_Down": lambda hist, info: three_inside_down(hist),
    "Bullish_Tasuki_Gap": lambda hist, info: bullish_tasuki_gap(hist),
    "Bearish_Tasuki_Gap": lambda hist, info: bearish_tasuki_gap(hist),
    "Mat_Hold": lambda hist, info: mat_hold(hist),
    "Kicking": lambda hist, info: kicking(hist),
    "Three_White_Soldiers": lambda hist, info: three_white_soldiers(hist),
    "Three_Black_Crows": lambda hist, info: three_black_crows(hist),
    "Rising_Window": lambda hist, info: rising_window(hist),
    "Falling_Window": lambda hist, info: falling_window(hist),
    "Bullish_Separating_Lines": lambda hist, info: bullish_separating_lines(hist),
    "Bearish_Separating_Lines": lambda hist, info: bearish_separating_lines(hist),
    "Upside_Gap_Two_Crows": lambda hist, info: upside_gap_two_crows(hist),
    "On_Neck": lambda hist, info: on_neck(hist),
    "In_Neck": lambda hist, info: in_neck(hist),
    "Thrusting": lambda hist, info: thrusting(hist),
    "Deliberation": lambda hist, info: deliberation(hist),

    # Technical
    "RSI_Overbought": lambda hist, info: (rsi(hist) is not None) and rsi(hist) > 70,
    "RSI_Oversold": lambda hist, info: (rsi(hist) is not None) and rsi(hist) < 30,
    "MACD_Bullish": lambda hist, info: (macd(hist) is not None) and macd(hist)[0] > macd(hist)[1],
    "MACD_Bearish": lambda hist, info: (macd(hist) is not None) and macd(hist)[0] < macd(hist)[1],
    "Golden_Cross": lambda hist, info: golden_cross(hist),
    "Death_Cross": lambda hist, info: death_cross(hist),
    "Bollinger_Breakout": lambda hist, info: bollinger_breakout(hist),
    "Bollinger_Breakdown": lambda hist, info: bollinger_breakdown(hist),
    "Volume_Spike": lambda hist, info: volume_spike(hist),

    # Financial
    "MarketCap_Gt_10B": lambda hist, info: marketcap_gt_10b(info),
    "MarketCap_Lt_1B": lambda hist, info: marketcap_lt_1b(info),
    "PE_Lt_20": lambda hist, info: pe_lt_20(info),
    "PE_Gt_40": lambda hist, info: pe_gt_40(info),
    "EPS_Positive_Growth": lambda hist, info: eps_positive_growth(info),
    "EPS_Negative_Growth": lambda hist, info: eps_negative_growth(info),
    "DividendYield_Gt_2": lambda hist, info: dividend_yield_gt_2(info),
    "DebtEquity_Lt_1": lambda hist, info: debt_equity_lt_1(info),
}
# ======================================
# Data fetch and ticker parsing
# ======================================
@st.cache_data
def fetch_stock_data(ticker):
    """Fetch data once per ticker and compute all indicators."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        if not info or hist.empty:
            return None

        result = {
            "Ticker": ticker,
            "Company": info.get("longName", "N/A"),
            "Price": info.get("currentPrice", np.nan),
            "Market Cap (B)": info.get("marketCap", np.nan) / 1e9 if info.get("marketCap") else np.nan,
            "P/E Ratio": info.get("trailingPE", np.nan),
            "Dividend Yield (%)": info.get("dividendYield", np.nan) * 100 if info.get("dividendYield") else np.nan,
            "Beta": info.get("beta", np.nan),
            "Sector": info.get("sector", "N/A"),
            "Price_Trend": prior_trend(hist["Close"]),
            "Volume_Trend": prior_volume_trend(hist["Volume"]),
            "RSI": rsi(hist),
        }

        for name, fn in INDICATOR_CHECKS.items():
            result[name] = fn(hist, info)

        return result
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {e}")
        return None

def parse_ticker_file(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8")
        tickers = []
        for line in content.strip().split("\n"):
            line_tickers = [t.strip().upper() for t in line.split(",") if t.strip()]
            tickers.extend(line_tickers)
        return sorted(set(tickers))
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return []

def plotcandlestick(symbol, period="6mo", arounddate=None):
    ticker = yf.Ticker(symbol)
    if arounddate is not None:
        if isinstance(arounddate, str):
            arounddate = pd.to_datetime(arounddate)
        startdate = arounddate - pd.Timedelta(days=15)
        enddate = arounddate + pd.Timedelta(days=15)
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
    
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=('OHLC', 'Volume'),
        row_width=[0.7, 0.3]
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
# ======================================
# Simple in-memory portfolio for Streamlit
# ======================================
class StreamlitPortfolio:
    def __init__(self, cash=1000000.0):
        self.cash = cash
        # positions: {symbol: {"qty": float, "avg": float, "last_trade": str}}
        self.positions = {}
        # trades: list of dicts
        self.trades = []

    def market_order(self, symbol, side, qty, price):
        if qty <= 0 or price <= 0:
            raise ValueError("Quantity and price must be positive")
    
        cost = qty * price
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
        pos = self.positions.get(symbol, {"qty": 0.0, "avg": 0.0, "last_trade": ""})
        old_qty = pos["qty"]
    
        if side == "BUY":
            # For simplicity, allow negative cash (margin not enforced)
            self.cash -= cost
            new_qty = old_qty + qty
    
            if old_qty == 0:
                new_avg = price
            elif (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
                # Adding to existing long or existing short
                new_avg = (old_qty * pos["avg"] + (qty if new_qty > 0 else -qty) * price) / new_qty
            elif new_qty == 0:
                new_avg = 0.0
            else:
                # Reducing or flipping a short/long; keep avg as previous for remaining side
                new_avg = pos["avg"]
    
            if new_qty == 0:
                self.positions.pop(symbol, None)
            else:
                self.positions[symbol] = {
                    "qty": new_qty,
                    "avg": new_avg,
                    "last_trade": now_str,
                }
    
        elif side == "SELL":
            # Selling decreases quantity; for short, qty becomes negative
            self.cash += cost
            new_qty = old_qty - qty  # note: SELL reduces qty
    
            if old_qty == 0:
                # Opening a fresh short position
                new_avg = price
            elif (old_qty > 0 and new_qty > 0) or (old_qty < 0 and new_qty < 0):
                # Adding to existing long or existing short
                new_avg = (old_qty * pos["avg"] - (qty if new_qty < 0 else -qty) * price) / new_qty
            elif new_qty == 0:
                new_avg = 0.0
            else:
                # Reducing or flipping a long/short; keep avg as previous for remaining side
                new_avg = pos["avg"]
    
            if new_qty == 0:
                self.positions.pop(symbol, None)
            else:
                self.positions[symbol] = {
                    "qty": new_qty,
                    "avg": new_avg,
                    "last_trade": now_str,
                }
        else:
            raise ValueError("Side must be BUY or SELL")
    
        self.trades.append(
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": price,
                "time": now_str,
            }
        )

    def unrealized_pnl(self, symbol, price):
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0
        return (price - pos["avg"]) * pos["qty"]


    def to_dataframe(self):
        """
        Return a single DataFrame combining positions and trades,
        tagged by 'record_type' = 'POSITION' or 'TRADE'.
        """
        rows = []
        for s, pos in self.positions.items():
            rows.append(
                {
                    "record_type": "POSITION",
                    "symbol": s,
                    "qty": pos["qty"],
                    "avg": pos["avg"],
                    "price": None,
                    "pnl": None,
                    "last_trade": pos["last_trade"],
                    "side": None,
                    "time": None,
                    "cash": self.cash,
                }
            )
        for t in self.trades:
            rows.append(
                {
                    "record_type": "TRADE",
                    "symbol": t["symbol"],
                    "qty": t["qty"],
                    "avg": None,
                    "price": t["price"],
                    "pnl": None,
                    "last_trade": None,
                    "side": t["side"],
                    "time": t["time"],
                    "cash": self.cash,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df):
        pf = cls()
        pf.positions = {}
        pf.trades = []
        if df.empty:
            return pf

        last_cash = df["cash"].dropna().iloc[-1]
        pf.cash = float(last_cash)

        pos_df = df[df["record_type"] == "POSITION"]
        for _, row in pos_df.iterrows():
            pf.positions[row["symbol"]] = {
                "qty": float(row["qty"]),
                "avg": float(row["avg"]),
                "last_trade": row.get("last_trade") or "",
            }

        trade_df = df[df["record_type"] == "TRADE"]
        for _, row in trade_df.iterrows():
            pf.trades.append(
                {
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "qty": float(row["qty"]),
                    "price": float(row["price"]),
                    "time": row["time"],
                }
            )
        return pf
# ======================================
# Pages
# ======================================
def page_screener():
    st.title("📈 Road to Runway")

    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = None

    with st.sidebar:
        st.header("📋 Ticker Universe")
        ticker_source = st.radio(
            "Select ticker source:",
            ["Nifty50 (file)", "Nifty500 (file)", "Upload custom CSV/TXT"],
            index=0,
        )

        if ticker_source == "Nifty50 (file)":
            current_tickers = load_tickers_from_file("Nifty50.txt")
            if current_tickers:
                st.info(f"Using {len(current_tickers)} Nifty50 tickers from Nifty50.txt.")
            else:
                st.warning("nifty50.txt is empty or could not be loaded.")
            uploaded_file = None

        elif ticker_source == "Nifty500 (file)":
            current_tickers = load_tickers_from_file("Nifty500.txt")
            if current_tickers:
                st.info(f"Using {len(current_tickers)} Nifty500 tickers from Nifty500.txt.")
            else:
                st.warning("nifty500.txt is empty or could not be loaded.")
            uploaded_file = None

        else:
            uploaded_file = st.file_uploader(
                "Upload tickers file (comma-separated)",
                type=["csv", "txt"],
            )
            if uploaded_file is not None:
                current_tickers = parse_ticker_file(uploaded_file)
                st.success(f"Loaded {len(current_tickers)} unique tickers from file.")
            else:
                current_tickers = []
                st.warning("Upload a ticker file to proceed.")

        st.header("🎯 Screening Filters")
        selected_indicators = st.multiselect(
            "Select indicators (ANY match):",
            options=sorted(INDICATOR_CHECKS.keys()),
            default=[],
        )

        run_screen = st.button("🚀 Fetch & Analyze Data", type="primary")

    if run_screen and current_tickers:
        st.write(f"Fetching 1-year data for {len(current_tickers)} tickers...")
        prog = st.progress(0)

        def _fetch_with_progress(tickers):
            results = []
            total = len(tickers)
            with ThreadPoolExecutor(max_workers=20) as executor:
                for i, res in enumerate(executor.map(fetch_stock_data, tickers), 1):
                    if res is not None:
                        results.append(res)
                    prog.progress(i / total)
            return results

        results = _fetch_with_progress(current_tickers)
        df = pd.DataFrame(results)

        if df.empty:
            st.error("No valid data retrieved. Try different tickers.")
        else:
            st.success(f"Analyzed {len(df)} stocks. Screening ready.")
            st.session_state["screening_df"] = df

    if "screening_df" in st.session_state:
        df = st.session_state["screening_df"].copy()
        st.subheader("📊 Screening Results")

        if selected_indicators:
            mask = df[selected_indicators].any(axis=1)
            filtered_df = df[mask].copy()
            st.caption(
                f"Filtered by {len(selected_indicators)} indicator(s); showing stocks that match ANY of them."
            )
        else:
            filtered_df = df.copy()
            st.caption("No indicators selected; showing all analyzed stocks.")

        if filtered_df.empty:
            st.warning("No stocks match the current screening conditions.")
        else:
            st.write(f"Found **{len(filtered_df)}** matching stocks.")
            st.dataframe(filtered_df, use_container_width=True, height=400)

            tickers_in_result = filtered_df["Ticker"].tolist()
            selected_symbol = st.selectbox(
                "Click/choose a ticker to view its chart:",
                options=["(none)"] + tickers_in_result,
                index=0,
            )

            if selected_symbol != "(none)":
                st.session_state["selected_ticker"] = selected_symbol

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total analyzed", len(df))
            with c2:
                st.metric("Matches", len(filtered_df))
            with c3:
                mean_rsi = filtered_df["RSI"].mean()
                st.metric("Average RSI", f"{mean_rsi:.1f}" if not np.isnan(mean_rsi) else "N/A")

            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Download results as CSV",
                data=csv_data,
                file_name="screener_results.csv",
                mime="text/csv",
            )

            if st.session_state.get("selected_ticker"):
                st.markdown("---")
                st.subheader(f"📉 Chart for {st.session_state['selected_ticker']}")
                plotcandlestick(st.session_state["selected_ticker"])

def page_dashboard():
    st.title("📊 Dashboard")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Watchlist", 0)
    with col2:
        open_trades = len(st.session_state.get("portfolio", StreamlitPortfolio()).positions)
        st.metric("Open Paper Positions", open_trades)
    with col3:
        st.metric("Paper P&L", "0.0%")
    st.write("Use this page later to show aggregates from your screener and paper trades.")

def page_paper_trading():
    st.title("📝 Paper Trading")
    # ======================================
    # Portfolio selection - NO AUTO-LOAD
    # ======================================
    if "portfolio" not in st.session_state or st.session_state["portfolio"].cash == 0:
        st.markdown("### 🚀 **Start new or load portfolio?**")
        col_new, col_load = st.columns(2)

        with col_new:
            if st.button("💎 **New Portfolio** ₹10,00,000 cash", use_container_width=True, help="Start fresh"):
                st.session_state["portfolio"] = StreamlitPortfolio(cash=1000000.0)
                st.session_state["portfolio_name"] = "new_portfolio"
                st.rerun()

        with col_load:
            st.markdown("**📁 Load existing**")
            uploaded_file = st.file_uploader("Upload portfolio CSV", type=["csv"])
            if uploaded_file:
                try:
                    original_name = uploaded_file.name.replace('.csv', '')
                    st.session_state["portfolio_name"] = original_name

                    df_loaded = pd.read_csv(uploaded_file)
                    st.session_state["portfolio"] = StreamlitPortfolio.from_dataframe(df_loaded)
                    st.success(f"✅ Loaded **{original_name}** portfolio")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to load: {e}")

        st.stop()
    # ======================================
    # Trading UI
    # ======================================
    portfolio: StreamlitPortfolio = st.session_state["portfolio"]
    portfolio_name = st.session_state.get("portfolio_name", "portfolio")

    if "showing_history" not in st.session_state:
        st.session_state["showing_history"] = False

    # Header
    col_toggle, col_cash, col_name = st.columns([1, 1, 2])
    with col_toggle:
        if st.button("📋 " + ("**History**" if not st.session_state["showing_history"] else "**Positions**")):
            st.session_state["showing_history"] = not st.session_state["showing_history"]
            st.rerun()
    with col_cash:
        st.metric("💰 Cash", f"₹{portfolio.cash:,.0f}")
    with col_name:
        st.caption(f"📂 Active: **{portfolio_name}**")

    st.markdown("---")

    # Trading form
    with st.form("paper_trade_form", clear_on_submit=False):
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            ticker = st.text_input("Symbol", placeholder="RELIANCE.NS", help="e.g. RELIANCE.NS, TCS.NS")
        with col2:
            qty = st.text_input("Qty", value="10")
        with col3:
            side = st.selectbox("Side", ["BUY", "SELL"])
        with col4:
            use_live_price = st.checkbox("Live price", value=True)

        manual_price = st.text_input("Manual Price (optional)", value="")
        submitted = st.form_submit_button("🚀 **Submit Order**", use_container_width=True)

    if submitted:
        symbol = (ticker or "").strip().upper()
        if not symbol:
            st.error("⚠️ **Symbol required**")
        else:
            try:
                q = float(qty)
            except ValueError:
                st.error("⚠️ **Invalid quantity**")
                st.stop()

            price_val = None
            if use_live_price:
                try:
                    data = yf.Ticker(symbol).history(period="1d")
                    if not data.empty:
                        price_val = float(data["Close"].iloc[-1])
                except Exception as e:
                    st.error(f"⚠️ Failed to fetch price: {e}")

            if (not price_val) and manual_price:
                try:
                    price_val = float(manual_price)
                except ValueError:
                    st.error("⚠️ **Invalid manual price**")
                    st.stop()

            if not price_val:
                st.error(f"⚠️ **No price available for {symbol}**")
                st.stop()

            try:
                portfolio.market_order(symbol, side, q, price_val)
                st.success(f"✅ **{side} {q} {symbol} @ ₹{price_val:,.0f}**")
                st.rerun()
            except Exception as e:
                st.error(f"❌ **Trade failed: {e}**")

    # Positions or History
    if not st.session_state["showing_history"]:
        st.subheader("📊 **Open Positions**")
        rows = []
        for s, pos in portfolio.positions.items():
            try:
                data = yf.Ticker(s).history(period="1d")
                price = float(data["Close"].iloc[-1]) if not data.empty else 0.0
            except Exception:
                price = 0.0
            pnl = portfolio.unrealized_pnl(s, price)
            rows.append({
                "Symbol": s,
                "Qty": f"{pos['qty']:.0f}",
                "Avg": f"₹{pos['avg']:,.0f}",
                "Price": f"₹{price:,.0f}",
                "PnL": f"₹{pnl:,.0f}",
                "Last Trade": pos["last_trade"],
            })
        if rows:
            pos_df = pd.DataFrame(rows)
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("📭 **No open positions**")
    else:
        st.subheader("📜 **Trade History**")
        if portfolio.trades:
            hist_df = pd.DataFrame(portfolio.trades)
            hist_df = hist_df[["symbol", "side", "qty", "price", "time"]]
            hist_df.columns = ["Symbol", "Side", "Qty", "Price", "Time"]
            st.dataframe(hist_df, use_container_width=True)
        else:
            st.info("📭 **No trades yet**")
    # ======================================
    # Export section
    # ======================================
    st.markdown("---")
    st.subheader("💾 **Export Portfolio**")

    c1, c2 = st.columns(2)
    with c1:
        send_email = st.checkbox("📧 **Also email export**")
        if st.button("📥 **Save / Export Portfolio**", use_container_width=True):
            df = portfolio.to_dataframe()
            if df.empty:
                st.warning("⚠️ **Nothing to export**")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = portfolio_name
                filename = f"{base_name}_{timestamp}.csv"
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ **Download**",
                    data=csv_bytes,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
    with c2:
        st.info("💡 **Click New Portfolio or upload CSV to switch profiles**")
# ======================================
# Main app entry
# ======================================

st.set_page_config(page_title="Road to Runway", layout="wide")

page = st.sidebar.selectbox(
    "Select page",
    ["Dashboard", "Screener", "Paper Trading"],
)

if page == "Screener":
    page_screener()
elif page == "Dashboard":
    page_dashboard()
elif page == "Paper Trading":
    page_paper_trading()
