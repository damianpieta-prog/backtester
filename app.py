import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- KONFIGURACJA ---
st.set_page_config(page_title="Backtester Pro | AI Optimizer", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; }
    div[data-testid="stMetric"] { background-color: #1e2227; border: 1px solid #3e444e; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: WYBÓR AKTYWA ---
st.sidebar.header("🔍 Wybór Instrumentu")
popular_tickers = {
    "NVIDIA (Nasdaq)": "NVDA",
    "Bitcoin (Crypto)": "BTC-USD",
    "Berkshire Hathaway": "BRK-B",
    "Apple (Nasdaq)": "AAPL",
    "Tesla": "TSLA",
    "CD Projekt (GPW)": "CDR.WA",
    "Własny ticker...": "CUSTOM"
}
selection = st.sidebar.selectbox("Instrument", list(popular_tickers.keys()))
ticker = st.sidebar.text_input("Symbol", value="NVDA").upper() if popular_tickers[selection] == "CUSTOM" else popular_tickers[selection]

# --- SIDEBAR: PROFILE ---
st.sidebar.header("🎯 Strategia")
profile = st.sidebar.selectbox(
    "Profil", 
    ["Manual", "Agresywne Momentum", "Kupuj Dołek (-2 SD) & Trzymaj", "Regresja: Reversion to Mean"]
)

# Logika profili
if profile == "Agresywne Momentum":
    d_rsi, d_sl, d_f, d_s = 65, 1.5, 9, 17
elif profile == "Kupuj Dołek (-2 SD) & Trzymaj":
    d_rsi, d_sl, d_f, d_s = 30, 10.0, 20, 50 # Luźniejszy SL dla trzymania
else:
    d_rsi, d_sl, d_f, d_s = 60, 2.0, 9, 17

# Parametry
capital = st.sidebar.number_input("Kapitał ($)", value=10000)
rsi_val = st.sidebar.slider("RSI Wejście", 20, 80, d_rsi)
ema_f = st.sidebar.slider("Szybka EMA", 5, 30, d_f)
ema_s = st.sidebar.slider("Wolna EMA", 10, 100, d_s)
sl_pct = st.sidebar.slider("Stop Loss %", 0.5, 15.0, d_sl) / 100
years = st.sidebar.slider("Lata", 1, 10, 2)

# --- SILNIK OBLICZENIOWY ---
def run_backtest(df_in, rsi_t, ef, es, sl, strat_type):
    df = df_in.copy()
    df['EMA_F'] = ta.ema(df['Close'], length=ef)
    df['EMA_S'] = ta.ema(df['Close'], length=es)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df = df.dropna()
    
    cash, shares, entry_p = float(capital), 0.0, 0.0
    for i in range(len(df)):
        p = float(df['Close'].iloc[i])
        if shares == 0:
            # Warunki wejścia zależne od strategii
            if strat_type == "Kupuj Dołek (-2 SD) & Trzymaj":
                cond = (p < df['Reg_Lower'].iloc[i] * 1.02) # Kupuj blisko -2SD
            else:
                cond = (df['EMA_F'].iloc[i] > df['EMA_S'].iloc[i] and df['RSI'].iloc[i] >= rsi_t)
            
            if cond:
                shares = cash / p
                entry_p, cash = p, 0.0
        elif shares > 0:
            # Wyjście: SL lub odwrócenie średnich
            if p <= entry_p * (1 - sl) or df['EMA_F'].iloc[i] < df['EMA_S'].iloc[i]:
                cash = shares * p
                shares = 0.0
    return cash if shares == 0 else shares * float(df['Close'].iloc[-1])

# --- DANE ---
@st.cache_data
def get_data(symbol, yrs):
    d = yf.download(symbol, start=datetime.now()-timedelta(days=yrs*365), end=datetime.now())
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    # Regresja Logarytmiczna
    lc = np.log(d['Close'].values)
    x = np.arange(len(lc))
    m, c = np.polyfit(x, lc, 1)
    sd = np.std(lc - (m*x + c))
    d['Reg_Lower'] = np.exp((m*x + c) - 2*sd)
    d['Reg_Upper'] = np.exp((m*x + c) + 2*sd)
    return d

data = get_data(ticker, years)

# --- AUTO OPTYMALIZACJA ---
if st.sidebar.button("🚀 URUCHOM AUTO-OPTYMALIZACJĘ"):
    st.toast("Szukam najlepszej kombinacji... to może chwilę potrwać.")
    best_res, best_params = 0, {}
    # Grid Search
    for r in [50, 60, 70]:
        for f in [9, 12, 20]:
            for s in [21, 50]:
                res = run_backtest(data, r, f, s, sl_pct, profile)
                if res > best_res:
                    best_res, best_params = res, {'rsi': r, 'f': f, 's': s}
    st.success(f"Znaleziono! Najlepsze RSI: {best_params['rsi']}, EMA: {best_params['f']}/{best_params['s']}")
    rsi_val, ema_f, ema_s = best_params['rsi'], best_params['f'], best_params['s']

# --- WYNIK KOŃCOWY ---
final_money = run_backtest(data, rsi_val, ema_f, ema_s, sl_pct, profile)
profit = (final_money - capital) / capital

st.title(f"📊 Raport: {ticker}")
c1, c2, c3 = st.columns(3)
c1.metric("Wynik Strategii", f"{profit:.2%}")
c2.metric("Kapitał Końcowy", f"${final_money:,.2f}")
c3.metric("Rynek (B&H)", f"{((data['Close'].iloc[-1]-data['Close'].iloc[0])/data['Close'].iloc[0]):.2%}")

# Wykres
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Cena", line=dict(color='white')))
fig.add_trace(go.Scatter(x=data.index, y=data['Reg_Lower'], name="-2σ Dno", line=dict(color='cyan', dash='dot')))
fig.add_trace(go.Scatter(x=data.index, y=data['Reg_Upper'], name="+2σ Szczyt", line=dict(color='magenta', dash='dot')))
fig.update_layout(template="plotly_dark", height=500, yaxis_type="log")
st.plotly_chart(fig, use_container_width=True)
