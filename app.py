"""
Paper Trading Simulator — load from public Google Sheet, manual-save workflow

Behavior:
- On startup the app attempts to load trades from the public Google Sheet CSV export:
  https://docs.google.com/spreadsheets/d/<SHEET_ID>/export?format=csv
  (Your sheet id: 1UV0zUkIpJ1e1BwdgOlvRQxHTVZuxsT4xbwOmw5K3Lq4)
- The app runs in single-user mode and keeps trades in session_state.
- When you press "Save changes" the app prepares an updated CSV and offers it for download.
  AFTER downloading, you must manually upload/replace the Google Sheet content (instructions shown).
- No credentials/service account required: this reads the public sheet only and relies on manual upload to persist.

Usage:
- Install deps: pip install -r requirements.txt
- Run: streamlit run app.py
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from io import StringIO, BytesIO
from typing import List, Dict, Optional
import time

st.set_page_config(page_title="Paper Trading (Google Sheet load; manual save)", layout="wide")

# Google Sheet public CSV export URL (your sheet id)
SHEET_ID = "1UV0zUkIpJ1e1BwdgOlvRQxHTVZuxsT4xbwOmw5K3Lq4"
SHEET_CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

# -----------------------
# Utilities
# -----------------------
def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def sanitize_ticker(t: str) -> str:
    return str(t).strip().upper().strip(",")

def ensure_state():
    if "cash" not in st.session_state:
        st.session_state.cash = 100000.0
    if "trades" not in st.session_state:
        st.session_state.trades: List[Dict] = []
    if "settings" not in st.session_state:
        st.session_state.settings = {"commission": 0.0}
    if "last_prices" not in st.session_state:
        st.session_state.last_prices = {}

def trade_list_to_df(trades: List[Dict]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["id","ticker","side","qty","price","commission","timestamp"])
    return pd.DataFrame(trades)

def df_to_trade_list(df: pd.DataFrame) -> List[Dict]:
    expected = ["id","ticker","side","qty","price","commission","timestamp"]
    for col in expected:
        if col not in df.columns:
            df[col] = None
    trades = []
    for _, row in df.iterrows():
        try:
            t = {
                "id": int(row["id"]) if pd.notna(row["id"]) else None,
                "ticker": sanitize_ticker(row["ticker"]),
                "side": str(row["side"]).lower(),
                "qty": float(row["qty"]) if pd.notna(row["qty"]) else 0.0,
                "price": float(row["price"]) if pd.notna(row["price"]) else 0.0,
                "commission": float(row.get("commission", 0.0) or 0.0),
                "timestamp": str(row["timestamp"]) if pd.notna(row["timestamp"]) else now_iso()
            }
            trades.append(t)
        except Exception:
            continue
    max_id = max([t["id"] for t in trades if t["id"] is not None], default=0)
    for t in trades:
        if t["id"] is None:
            max_id += 1
            t["id"] = max_id
    return trades

# -----------------------
# Google Sheets (public CSV) loader
# -----------------------
def fetch_trades_from_public_sheet(url: str, timeout: int = 15) -> (List[Dict], Optional[str]):
    """
    Fetch CSV text from a public Google Sheets export URL and parse to trade list.
    Returns (trades_list, error_string). If error_string is None, load succeeded.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        text = r.text
        df = pd.read_csv(StringIO(text), dtype=str, keep_default_na=False)
        # If csv is empty or no rows, return empty list
        if df.empty:
            return [], None
        # Try convert to trade list
        trades = df_to_trade_list(df)
        return trades, None
    except Exception as e:
        return [], f"Failed to fetch/load CSV from Google Sheets URL: {e}"

# -----------------------
# Trading logic (simple)
# -----------------------
def add_trade(ticker: str, side: str, qty: float, price: float) -> Dict:
    trades = st.session_state.trades
    next_id = max([t.get("id",0) for t in trades], default=0) + 1
    trade = {
        "id": next_id,
        "ticker": sanitize_ticker(ticker),
        "side": side.lower(),
        "qty": float(qty),
        "price": float(price),
        "commission": float(st.session_state.settings.get("commission", 0.0)),
        "timestamp": now_iso()
    }
    st.session_state.trades.append(trade)
    return trade

def compute_positions_and_pnl(trades: List[Dict]):
    positions = {}
    total_realized = 0.0
    buy_lots = {}
    for t in trades:
        tic = sanitize_ticker(t["ticker"])
        side = t["side"]
        qty = float(t["qty"])
        price = float(t["price"])
        com = float(t.get("commission", 0.0))
        if tic not in buy_lots:
            buy_lots[tic] = []
        if side == "buy":
            buy_lots[tic].append([qty, price])
        elif side == "sell":
            remaining = qty
            while remaining > 0 and buy_lots[tic]:
                lot_qty, lot_price = buy_lots[tic][0]
                if lot_qty > remaining:
                    realized = remaining * (price - lot_price) - com
                    total_realized += realized
                    buy_lots[tic][0][0] = lot_qty - remaining
                    remaining = 0
                else:
                    realized = lot_qty * (price - lot_price) - com
                    total_realized += realized
                    remaining -= lot_qty
                    buy_lots[tic].pop(0)
            if remaining > 0:
                total_realized += remaining * price - com
                remaining = 0
    for tic, lots in buy_lots.items():
        qty = sum(l[0] for l in lots)
        avg_cost = sum(l[0]*l[1] for l in lots)/qty if qty>0 else 0.0
        positions[tic] = {"qty": qty, "avg_cost": avg_cost, "lots": [(l[0], l[1]) for l in lots]}
    return positions, total_realized

def get_market_price(ticker: str) -> Optional[float]:
    ticker = sanitize_ticker(ticker)
    cached = st.session_state.last_prices.get(ticker)
    if cached is not None:
        return cached
    try:
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        if price is None:
            hist = t.history(period="1d", interval="1m")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        st.session_state.last_prices[ticker] = price
        return price
    except Exception:
        return None

# -----------------------
# App startup
# -----------------------
ensure_state()

# Load from Google Sheet at startup (if session has no trades or user forces reload)
load_msg = ""
if not st.session_state.trades:
    st.info("Attempting to load trades from the public Google Sheet...")
    trades, err = fetch_trades_from_public_sheet(SHEET_CSV_URL)
    if err:
        st.warning(err)
    else:
        if trades:
            st.session_state.trades = trades
            st.success(f"Loaded {len(trades)} trades from the Google Sheet.")
        else:
            st.info("No trades found in the Google Sheet (or sheet empty). Starting with empty trading session.")

# -----------------------
# UI
# -----------------------
st.title("Paper Trading Simulator — Load from Google Sheet / Manual Save")

left, right = st.columns([2,3])

with left:
    st.subheader("Google Sheet (load) & Export (manual save)")

    st.markdown(f"- Google Sheet (read-only load): [{SHEET_CSV_URL}]({SHEET_CSV_URL})")
    st.markdown("  *This app reads the sheet's CSV export; you must keep the sheet public for automatic loading.*")

    if st.button("Reload from Google Sheet now"):
        trades, err = fetch_trades_from_public_sheet(SHEET_CSV_URL)
        if err:
            st.error(err)
        else:
            st.session_state.trades = trades
            st.success(f"Reloaded {len(trades)} trades from the Google Sheet.")
            # clear price cache to refresh quotes
            st.session_state.last_prices = {}

    st.markdown("---")
    st.subheader("Save changes (manual upload workflow)")
    st.write("When you press 'Prepare export' the app will generate an updated CSV with current trades.")
    st.write("After downloading the CSV, open your Google Sheet, choose File → Import → Upload and select the downloaded file.")
    st.write("When importing select the option to 'Replace current sheet' (or 'Replace spreadsheet') to overwrite with the updated CSV.")
    if st.button("Prepare export (generate CSV)"):
        df_out = trade_list_to_df(st.session_state.trades)
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.session_state["_last_export"] = csv_bytes
        st.success("CSV prepared. Click the Download button below to save the file to your machine.")
        # show export timestamp
        st.write("Export prepared at:", now_iso())

    if "_last_export" in st.session_state:
        st.download_button("⬇️ Download updated trades CSV", data=st.session_state["_last_export"],
                           file_name=f"trades_export_{int(time.time())}.csv", mime="text/csv")
        st.info("After downloading, go to your Google Sheet and import this CSV to replace the sheet contents. The app will not upload automatically.")

    st.markdown("---")
    st.subheader("Upload CSV to replace in-memory trades (optional)")
    uploaded = st.file_uploader("Upload CSV to replace session trades (this does NOT modify the Google Sheet)", type=["csv"])
    if uploaded is not None:
        if st.button("Replace in-memory trades with uploaded file"):
            try:
                df = pd.read_csv(uploaded)
                trades = df_to_trade_list(df)
                st.session_state.trades = trades
                st.session_state.last_prices = {}
                st.success(f"Replaced session trades with {len(trades)} items from uploaded CSV.")
            except Exception as e:
                st.error(f"Failed to parse uploaded CSV: {e}")

    st.markdown("---")
    st.subheader("Account Controls")
    st.write(f"Cash: ${st.session_state.cash:,.2f}")
    with st.form("trade_form"):
        t_ticker = st.text_input("Ticker", value="AAPL")
        t_side = st.selectbox("Side", ("Buy","Sell"))
        t_qty = st.number_input("Quantity", min_value=1.0, value=1.0, step=1.0)
        t_price_override = st.number_input("Explicit price (0 = market)", value=0.0, format="%.4f")
        submit = st.form_submit_button("Place order")
    if submit:
        ticker = sanitize_ticker(t_ticker)
        market_price = get_market_price(ticker)
        exec_price = float(t_price_override) if float(t_price_override) > 0 else (market_price if market_price is not None else None)
        if exec_price is None:
            st.error("Unable to determine market price; enter explicit price.")
        else:
            cost = exec_price * float(t_qty) + float(st.session_state.settings.get("commission",0.0))
            positions, _ = compute_positions_and_pnl(st.session_state.trades)
            pos_qty = positions.get(ticker, {}).get("qty", 0.0)
            if t_side.lower() == "buy":
                if cost > st.session_state.cash:
                    st.error("Insufficient cash.")
                else:
                    tr = add_trade(ticker, "buy", t_qty, exec_price)
                    st.session_state.cash -= cost
                    st.success(f"Bought {t_qty} {ticker} @ {exec_price:.4f}")
            else:
                if t_qty > pos_qty:
                    st.error(f"Insufficient position to sell (you have {pos_qty}).")
                else:
                    tr = add_trade(ticker, "sell", t_qty, exec_price)
                    proceeds = exec_price * float(t_qty) - float(tr.get("commission",0.0))
                    st.session_state.cash += proceeds
                    st.success(f"Sold {t_qty} {ticker} @ {exec_price:.4f}")

    st.markdown("---")
    st.subheader("Settings")
    comm = st.number_input("Commission per trade (flat)", value=float(st.session_state.settings.get("commission",0.0)), step=0.01, format="%.2f")
    if st.button("Update commission"):
        st.session_state.settings["commission"] = float(comm)
        st.success("Commission updated.")

with right:
    st.subheader("Portfolio & Trade History")
    positions, total_realized = compute_positions_and_pnl(st.session_state.trades)
    rows = []
    total_market_value = 0.0
    for tic, v in positions.items():
        qty = v["qty"]
        avg_cost = v["avg_cost"]
        market_price = get_market_price(tic) or np.nan
        unreal = (market_price - avg_cost) * qty if (not np.isnan(market_price) and qty>0) else np.nan
        mv = (market_price * qty) if not np.isnan(market_price) else np.nan
        total_market_value += mv if not np.isnan(mv) else 0.0
        rows.append({"Ticker": tic, "Quantity": qty, "Avg Cost": avg_cost, "Market Price": market_price, "Market Value": mv, "Unrealized P/L": unreal})
    df_pos = pd.DataFrame(rows)
    st.metric("Total Realized P/L", f"${total_realized:,.2f}")
    st.metric("Portfolio Market Value", f"${total_market_value:,.2f}")
    st.metric("Total Equity", f"${st.session_state.cash + total_market_value:,.2f}")

    st.markdown("### Positions")
    if df_pos.empty:
        st.info("No positions.")
    else:
        df_display = df_pos.copy()
        df_display["Avg Cost"] = df_display["Avg Cost"].map(lambda x: f"${x:,.2f}")
        df_display["Market Price"] = df_display["Market Price"].map(lambda x: f"${x:,.2f}" if not pd.isna(x) else "N/A")
        df_display["Market Value"] = df_display["Market Value"].map(lambda x: f"${x:,.2f}" if not pd.isna(x) else "N/A")
        df_display["Unrealized P/L"] = df_display["Unrealized P/L"].map(lambda x: f"${x:,.2f}" if not pd.isna(x) else "N/A")
        st.dataframe(df_display.set_index("Ticker"))

    st.markdown("---")
    st.subheader("Trade History")
    if not st.session_state.trades:
        st.info("No trades yet.")
    else:
        df_tr = trade_list_to_df(st.session_state.trades)
        df_disp = df_tr.copy()
        df_disp["price"] = df_disp["price"].map(lambda x: f"${x:,.4f}")
        df_disp = df_disp[["id","timestamp","ticker","side","qty","price","commission"]]
        df_disp.columns = ["ID","Timestamp","Ticker","Side","Qty","Price","Commission"]
        st.dataframe(df_disp.sort_values("ID",ascending=False))
        csv_bytes = trade_list_to_df(st.session_state.trades).to_csv(index=False).encode("utf-8")
        st.download_button("Download trade history CSV", data=csv_bytes, file_name="trades_export.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("Chart")
    chart_sym = st.text_input("Ticker for chart", value="AAPL", key="chart_ticker")
    chart_period = st.selectbox("Chart period", ["1mo","3mo","6mo","1y","5y"], index=2, key="chart_period")
    if st.button("Load chart"):
        ts = sanitize_ticker(chart_sym)
        try:
            hist = yf.Ticker(ts).history(period=chart_period, auto_adjust=True)
        except Exception:
            hist = pd.DataFrame()
        if hist is None or hist.empty:
            st.error("No historical data.")
        else:
            hist = hist.reset_index()
            fig = go.Figure()
            if {"Open","High","Low","Close"}.issubset(hist.columns):
                fig = go.Figure(data=[go.Candlestick(x=hist["Date"], open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"])])
            else:
                fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Close"], mode="lines"))
            fig.update_layout(title=f"{ts} {chart_period}", height=600)
            st.plotly_chart(fig, width='stretch')
            mp = get_market_price(ts)
            if mp is not None:
                st.metric(f"Latest price ({ts})", f"${mp:,.4f}")

st.markdown("---")
st.caption("This app auto-loads trades from the public Google Sheet at startup. Use 'Prepare export' + download to export updated CSV, then manually upload to Google Sheets (File → Import → Upload → Replace).")