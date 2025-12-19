import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go

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

def marketcap_gt_1b(info):
    return info.get("marketCap", 0) > 1_000_000_000


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
    "Bullish_Marubozu":       lambda hist, info: bullish_marubozu(hist),
    "Bearish_Marubozu":       lambda hist, info: bearish_marubozu(hist),
    "Doji":                   lambda hist, info: doji(hist),
    "Hammer":                 lambda hist, info: hammer(hist),
    "Inverted_Hammer":        lambda hist, info: inverted_hammer(hist),
    "Bullish_Engulfing":      lambda hist, info: bullish_engulfing(hist),
    "Bearish_Engulfing":      lambda hist, info: bearish_engulfing(hist),
    "Morning_Star":           lambda hist, info: morning_star(hist),
    "Evening_Star":           lambda hist, info: evening_star(hist),
    "Piercing_Line":          lambda hist, info: piercing_line(hist),
    "Dark_Cloud_Cover":       lambda hist, info: dark_cloud_cover(hist),
    "Spinning_Top":           lambda hist, info: spinning_top(hist),
    "Rising_Three_Methods":   lambda hist, info: rising_three_methods(hist),
    "Abandoned_Baby":         lambda hist, info: abandoned_baby(hist),
    "Three_Inside_Up":        lambda hist, info: three_inside_up(hist),
    "Three_Inside_Down":      lambda hist, info: three_inside_down(hist),
    "Bullish_Tasuki_Gap":     lambda hist, info: bullish_tasuki_gap(hist),
    "Bearish_Tasuki_Gap":     lambda hist, info: bearish_tasuki_gap(hist),
    "Mat_Hold":               lambda hist, info: mat_hold(hist),
    "Kicking":                lambda hist, info: kicking(hist),
    "Three_White_Soldiers":   lambda hist, info: three_white_soldiers(hist),
    "Three_Black_Crows":      lambda hist, info: three_black_crows(hist),
    "Rising_Window":          lambda hist, info: rising_window(hist),
    "Falling_Window":         lambda hist, info: falling_window(hist),
    "Bullish_Separating_Lines": lambda hist, info: bullish_separating_lines(hist),
    "Bearish_Separating_Lines": lambda hist, info: bearish_separating_lines(hist),
    "Upside_Gap_Two_Crows":   lambda hist, info: upside_gap_two_crows(hist),
    "On_Neck":                lambda hist, info: on_neck(hist),
    "In_Neck":                lambda hist, info: in_neck(hist),
    "Thrusting":              lambda hist, info: thrusting(hist),
    "Deliberation":           lambda hist, info: deliberation(hist),

    # Technical
    "RSI_Overbought":         lambda hist, info: (rsi(hist) is not None) and rsi(hist) > 70,
    "RSI_Oversold":           lambda hist, info: (rsi(hist) is not None) and rsi(hist) < 30,
    "MACD_Bullish":           lambda hist, info: (macd(hist) is not None) and macd(hist)[0] > macd(hist)[1],
    "MACD_Bearish":           lambda hist, info: (macd(hist) is not None) and macd(hist)[0] < macd(hist)[1],
    "Golden_Cross":           lambda hist, info: golden_cross(hist),
    "Death_Cross":            lambda hist, info: death_cross(hist),
    "Bollinger_Breakout":     lambda hist, info: bollinger_breakout(hist),
    "Bollinger_Breakdown":    lambda hist, info: bollinger_breakdown(hist),
    "Volume_Spike":           lambda hist, info: volume_spike(hist),

    # Financial
    "MarketCap_Gt_1B":        lambda hist, info: marketcap_gt_1b(info),
    "MarketCap_Lt_1B":        lambda hist, info: marketcap_lt_1b(info),
    "PE_Lt_20":               lambda hist, info: pe_lt_20(info),
    "PE_Gt_40":               lambda hist, info: pe_gt_40(info),
    "EPS_Positive_Growth":    lambda hist, info: eps_positive_growth(info),
    "EPS_Negative_Growth":    lambda hist, info: eps_negative_growth(info),
    "DividendYield_Gt_2":     lambda hist, info: dividend_yield_gt_2(info),
    "DebtEquity_Lt_1":        lambda hist, info: debt_equity_lt_1(info),
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


def plot_candlestick(symbol, period="6mo", around_date=None):
    ticker = yf.Ticker(symbol)
    if around_date is not None:
        if isinstance(around_date, str):
            around_date = pd.to_datetime(around_date)
        start_date = around_date - pd.Timedelta(days=15)
        end_date = around_date + pd.Timedelta(days=15)
        hist = ticker.history(start=start_date, end=end_date)
        title = f"{symbol} chart around {around_date.date()}"
    else:
        hist = ticker.history(period=period)
        title = f"{symbol} {period} chart"

    if hist is None or hist.empty:
        st.warning(f"No chart data available for {symbol}")
        return

    hist = hist.reset_index()
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=hist["Date"],
                open=hist["Open"],
                high=hist["High"],
                low=hist["Low"],
                close=hist["Close"],
                name=symbol,
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


# ======================================
# Pages
# ======================================

def page_screener():
    st.title("📈 Road to Runway")

    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = None

    # Sidebar: ticker universe + screening filters
    with st.sidebar:
        st.header("📋 Ticker Universe")
        ticker_source = st.radio(
            "Select ticker source:",
            ["Nifty50 (file)", "Nifty500 (file)", "Upload custom CSV/TXT"],
            index=0,
        )

        if ticker_source == "Nifty50 (file)":
            current_tickers = load_tickers_from_file("nifty50.txt")
            if current_tickers:
                st.info(f"Using {len(current_tickers)} Nifty50 tickers from nifty50.txt.")
            else:
                st.warning("nifty50.txt is empty or could not be loaded.")
            uploaded_file = None

        elif ticker_source == "Nifty500 (file)":
            current_tickers = load_tickers_from_file("nifty500.txt")
            if current_tickers:
                st.info(f"Using {len(current_tickers)} Nifty500 tickers from nifty500.txt.")
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

    # Main area: screening results
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

            # ---- Ticker selection for chart ----
            tickers_in_result = filtered_df["Ticker"].tolist()
            selected_symbol = st.selectbox(
                "Click/choose a ticker to view its chart:",
                options=["(none)"] + tickers_in_result,
                index=0,
            )

            if selected_symbol != "(none)":
                st.session_state["selected_ticker"] = selected_symbol

            # Quick metrics
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

            # ---- Chart rendering below the table ----
            if st.session_state.get("selected_ticker"):
                st.markdown("---")
                st.subheader(f"📉 Chart for {st.session_state['selected_ticker']}")
                plot_candlestick(st.session_state["selected_ticker"])


def page_dashboard():
    st.title("📊 Dashboard")

    # Simple placeholders; you can wire these to real data later
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Watchlist", 0)
    with col2:
        st.metric("Open Paper Positions", len(st.session_state.get("paper_trades", [])))
    with col3:
        st.metric("Paper P&L", "0.0%")

    st.write("Use this page later to show aggregates from your screener and paper trades.")


def page_paper_trading():
    st.title("📝 Paper Trading")

    if "paper_trades" not in st.session_state:
        st.session_state["paper_trades"] = []

    with st.form("paper_trade_form"):
        ticker = st.text_input("Ticker (e.g. RELIANCE.NS)")
        side = st.selectbox("Side", ["Buy", "Sell"])
        qty = st.number_input("Quantity", min_value=1, value=1)
        price = st.number_input("Price", min_value=0.0, value=0.0, format="%.2f")
        submitted = st.form_submit_button("Add Trade")

    if submitted and ticker and price > 0:
        st.session_state["paper_trades"].append(
            {"Ticker": ticker.upper(), "Side": side, "Qty": qty, "Price": price}
        )
        st.success("Trade added to paper ledger.")

    trades = st.session_state["paper_trades"]
    if trades:
        trades_df = pd.DataFrame(trades)
        st.subheader("Paper Trade Ledger")
        st.dataframe(trades_df, use_container_width=True)


# ======================================
# Main app entry
# ======================================

st.set_page_config(page_title="Road to Runway", layout="wide")

page = st.sidebar.selectbox(
    "Select page",
    ["Screener", "Dashboard", "Paper Trading"],
)

if page == "Screener":
    page_screener()
elif page == "Dashboard":
    page_dashboard()
elif page == "Paper Trading":
    page_paper_trading()
