import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Backtester Pro", layout="wide")

# --- STYLIZACJA CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: PARAMETRY ---
st.sidebar.header("🚀 Konfiguracja Backtestera")
ticker = st.sidebar.text_input("Ticker", value="NVDA").upper()
capital = st.sidebar.number_input("Kapitał początkowy ($)", value=10000)

st.sidebar.subheader("Strategia Momentum")
rsi_entry = st.sidebar.slider("RSI Próg Wejścia", 30, 80, 65)
ema_fast = st.sidebar.slider("Szybka EMA (np. 9)", 5, 20, 9)
ema_slow = st.sidebar.slider("Wolna EMA (np. 17)", 10, 50, 17)

st.sidebar.subheader("Zarządzanie Ryzykiem")
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 0.5, 5.0, 1.5) / 100

st.sidebar.subheader("Zakres Danych")
years = st.sidebar.slider("Lata wstecz", 1, 10, 2)

# --- FUNKCJA POBIERANIA DANYCH ---
@st.cache_data
def load_data(symbol, yrs):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=yrs*365)
    df = yf.download(symbol, start=start_date, end=end_date, interval="1d")
    return df

df_raw = load_data(ticker, years)

if df_raw.empty:
    st.error(f"Nie znaleziono danych dla tickera: {ticker}. Sprawdź poprawność symbolu.")
else:
    df = df_raw.copy()
    
    # --- OBLICZENIA TECHNICZNE ---
    df['EMA_F'] = ta.ema(df['Close'], length=ema_fast)
    df['EMA_S'] = ta.ema(df['Close'], length=ema_slow)
    df['EMA100'] = ta.ema(df['Close'], length=100)
    df['EMA200'] = ta.ema(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # --- REGRESJA LINIOWA LOGARYTMICZNA ---
    # Logarytmowanie ceny zamkniecia
    log_close = np.log(df['Close'].values)
    x = np.arange(len(log_close))
    
    # Wyznaczanie linii regresji (y = mx + c)
    slope, intercept = np.polyfit(x, log_close, 1)
    log_reg_line = slope * x + intercept
    
    # Odchylenie standardowe logarytmiczne
    log_std_dev = np.std(log_close - log_reg_line)
    
    # Powrót do wartości nominalnych (exp)
    df['Reg_Mid'] = np.exp(log_reg_line)
    df['Reg_Lower'] = np.exp(log_reg_line - 2 * log_std_dev) # -2 Odchylenia
    df['Reg_Upper'] = np.exp(log_reg_line + 2 * log_std_dev) # +2 Odchylenia

    # --- SILNIK BACKTESTU ---
    df['Signal'] = 0
    cash = capital
    shares = 0
    entry_price = 0
    trade_history = []

    for i in range(1, len(df)):
        current_price = df['Close'].iloc[i]
        
        # WARUNEK WEJŚCIA (BUY)
        if shares == 0:
            buy_condition = (
                df['EMA_F'].iloc[i] > df['EMA_S'].iloc[i] and
                df['RSI'].iloc[i] >= rsi_entry and
                current_price > df['EMA100'].iloc[i] and
                current_price > df['EMA200'].iloc[i]
            )
            if buy_condition:
                shares = cash / current_price
                entry_price = current_price
                cash = 0
                df.at[df.index[i], 'Signal'] = 1
                
        # WARUNEK WYJŚCIA (SELL)
        elif shares > 0:
            sl_price = entry_price * (1 - stop_loss_pct)
            sell_condition = (current_price <= sl_price) or (df['EMA_F'].iloc[i] < df['EMA_S'].iloc[i])
            
            if sell_condition:
                cash = shares * current_price
                pnl_pct = (current_price - entry_price) / entry_price
                trade_history.append({
                    "Data Wejścia": df.index[df.index.get_loc(df.index[i]) - 1], # orientacyjnie
                    "Data Wyjścia": df.index[i],
                    "Zysk/Strata %": round(pnl_pct * 100, 2),
                    "Kapitał": round(cash, 2)
                })
                shares = 0
                df.at[df.index[i], 'Signal'] = -1

    final_value = cash if shares == 0 else shares * df['Close'].iloc[-1]
    total_return = (final_value - capital) / capital
    buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]

    # --- WIZUALIZACJA WYNIKÓW ---
    st.title(f"📊 Backtester: {ticker}")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wynik Strategii", f"{total_return:.2%}", delta=f"{(total_return - buy_hold_return):.2%} vs Rynek")
    m2.metric("Buy & Hold", f"{buy_hold_return:.2%}")
    m3.metric("Kapitał Końcowy", f"${final_value:,.2f}")
    m4.metric("Ilość Transakcji", len(trade_history))

    # --- WYKRES ---
    fig = go.Figure()

    # Kanał Regresji
    fig.add_trace(go.Scatter(x=df.index, y=df['Reg_Upper'], name="+2σ (Opór)", line=dict(color='rgba(255, 0, 255, 0.4)', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Reg_Lower'], name="-2σ (Wsparcie)", line=dict(color='rgba(0, 255, 255, 0.4)', dash='dash')))
    
    # Cena
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Cena", line=dict(color='white', width=2)))
    
    # EMA
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_F'], name=f"EMA {ema_fast}", line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_S'], name=f"EMA {ema_slow}", line=dict(color='blue', width=1)))

    # Sygnały
    buys = df[df['Signal'] == 1]
    sells = df[df['Signal'] == -1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', marker=dict(symbol='triangle-up', size=12, color='#00ff00'), name="BUY"))
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', marker=dict(symbol='triangle-down', size=12, color='#ff0000'), name="SELL"))

    fig.update_layout(
        template="plotly_dark", 
        height=700, 
        yaxis_type="log", 
        title="Skala Logarytmiczna + Kanał Regresji Liniowej",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- TABELA TRANSAKCJI ---
    if trade_history:
        st.subheader("📝 Historia Transakcji")
        st.dataframe(pd.DataFrame(trade_history).sort_index(ascending=False), use_container_width=True)
    else:
        st.info("Brak zamkniętych transakcji w tym okresie przy obecnych parametrach.")
