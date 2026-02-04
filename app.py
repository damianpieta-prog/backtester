import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Backtester Pro", layout="wide")

# --- POPRAWKA WIDOCZNOŚCI (CSS) ---
st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #a3a3a3 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #1e2227;
        border: 1px solid #3e444e;
        padding: 15px;
        border-radius: 10px;
    }
    .main {
        background-color: #0e1117;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: WYBÓR TICKERA ---
st.sidebar.header("🔍 Wybór Aktywa")

# Lista popularnych tickerów
popular_tickers = {
    "NVIDIA": "NVDA",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Tesla": "TSLA",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN",
    "S&P 500 (ETF)": "SPY",
    "CD Projekt (GPW)": "CDR.WA",
    "Orlen (GPW)": "PKN.WA",
    "Allegro (GPW)": "ALE.WA",
    "Własny ticker...": "CUSTOM"
}

selection = st.sidebar.selectbox("Wybierz instrument", list(popular_tickers.keys()), index=0)

if popular_tickers[selection] == "CUSTOM":
    ticker = st.sidebar.text_input("Wpisz ticker (np. GOOGL lub META)", value="NVDA").upper()
else:
    ticker = popular_tickers[selection]

# --- SIDEBAR: PROFILE STRATEGII ---
st.sidebar.header("🎯 Profile Strategii")
profile = st.sidebar.selectbox(
    "Wybierz profil", 
    ["Manual (Własne)", "Agresywne Momentum", "Kupowanie Dołków (-2 SD)"]
)

if profile == "Agresywne Momentum":
    d_rsi, d_sl, d_fast, d_slow = 65, 1.5, 9, 17
elif profile == "Kupowanie Dołków (-2 SD)":
    d_rsi, d_sl, d_fast, d_slow = 40, 3.0, 12, 26
else:
    d_rsi, d_sl, d_fast, d_slow = 60, 2.0, 9, 17

# --- PARAMETRY ---
st.sidebar.subheader("⚙️ Parametry Silnika")
capital = st.sidebar.number_input("Kapitał początkowy ($)", value=10000)
rsi_entry = st.sidebar.slider("RSI Próg Wejścia", 20, 80, d_rsi)
ema_fast_val = st.sidebar.slider("Szybka EMA", 5, 20, d_fast)
ema_slow_val = st.sidebar.slider("Wolna EMA", 10, 50, d_slow)
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 0.5, 10.0, d_sl) / 100
years = st.sidebar.slider("Lata wstecz", 1, 10, 2)

@st.cache_data
def load_data(symbol, yrs):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=yrs*365)
    data = yf.download(symbol, start=start_date, end=end_date, interval="1d")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data

df_raw = load_data(ticker, years)

if df_raw.empty:
    st.error(f"Brak danych dla: {ticker}. Upewnij się, że ticker jest poprawny.")
else:
    df = df_raw.copy()
    
    # --- WSKAŹNIKI ---
    df['EMA_F'] = ta.ema(df['Close'], length=ema_fast_val)
    df['EMA_S'] = ta.ema(df['Close'], length=ema_slow_val)
    df['EMA100'] = ta.ema(df['Close'], length=100)
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # --- REGRESJA ---
    log_close = np.log(df['Close'].values)
    x = np.arange(len(log_close))
    slope, intercept = np.polyfit(x, log_close, 1)
    log_reg_line = slope * x + intercept
    log_std_dev = np.std(log_close - log_reg_line)
    df['Reg_Lower'] = np.exp(log_reg_line - 2 * log_std_dev)
    df['Reg_Upper'] = np.exp(log_reg_line + 2 * log_std_dev)

    # --- CZYSZCZENIE NaN ---
    df = df.dropna(subset=['EMA_F', 'EMA_S', 'RSI', 'EMA200'])

    # --- BACKTEST ---
    df['Signal'] = 0
    cash, shares, entry_price = float(capital), 0.0, 0.0
    trade_history = []

    for i in range(len(df)):
        price = float(df['Close'].iloc[i])
        f_ema = float(df['EMA_F'].iloc[i])
        s_ema = float(df['EMA_S'].iloc[i])
        rsi = float(df['RSI'].iloc[i])
        ema100 = float(df['EMA100'].iloc[i])
        reg_low = float(df['Reg_Lower'].iloc[i])

        if shares == 0:
            mom_cond = (f_ema > s_ema and rsi >= rsi_entry)
            trend_cond = (price > ema100)
            reg_cond = True if profile != "Kupowanie Dołków (-2 SD)" else price < (reg_low * 1.10)

            if mom_cond and trend_cond and reg_cond:
                shares = cash / price
                entry_price = price
                cash = 0.0
                df.at[df.index[i], 'Signal'] = 1
                
        elif shares > 0:
            sl_price = entry_price * (1 - stop_loss_pct)
            if price <= sl_price or f_ema < s_ema:
                cash = shares * price
                pnl = (price - entry_price) / entry_price
                trade_history.append({
                    "Data": df.index[i].date(),
                    "Zysk %": round(pnl * 100, 2),
                    "Kapitał": round(cash, 2)
                })
                shares = 0.0
                df.at[df.index[i], 'Signal'] = -1

    final_val = cash if shares == 0 else shares * float(df['Close'].iloc[-1])
    profit_pct = (final_val - capital) / capital
    
    # --- WIZUALIZACJA WYNIKÓW ---
    st.title(f"📊 {ticker}: {selection}")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Wynik Strategii", f"{profit_pct:.2%}")
    c2.metric("Kapitał Końcowy", f"${final_val:,.2f}")
    c3.metric("Liczba Transakcji", len(trade_history))

    # Wykres
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Reg_Lower'], name="-2σ (Dno)", line=dict(color='cyan', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Reg_Upper'], name="+2σ (Szczyt)", line=dict(color='magenta', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Cena", line=dict(color='white')))
    
    buys = df[df['Signal'] == 1]
    sells = df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name="BUY"))
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name="SELL"))

    fig.update_layout(template="plotly_dark", height=600, yaxis_type="log", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    if trade_history:
        st.subheader("Ostatnie transakcje")
        st.dataframe(pd.DataFrame(trade_history).tail(10), use_container_width=True)
