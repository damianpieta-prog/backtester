import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- KONFIGURACJA ---
st.set_page_config(page_title="Backtester Ultra | Max Deviation", layout="wide")

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; }
    div[data-testid="stMetric"] { background-color: #1e2227; border: 1px solid #3e444e; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: WYBÓR INSTRUMENTU ---
st.sidebar.header("🔍 Instrument")
popular_tickers = {"NVIDIA": "NVDA", "Bitcoin": "BTC-USD", "Tesla": "TSLA", "Berkshire": "BRK-B", "CD Projekt": "CDR.WA", "Własny...": "CUSTOM"}
selection = st.sidebar.selectbox("Instrument", list(popular_tickers.keys()))
ticker = st.sidebar.text_input("Symbol", value="NVDA").upper() if popular_tickers[selection] == "CUSTOM" else popular_tickers[selection]

# --- PARAMETRY ---
st.sidebar.header("🎯 Parametry")
capital = st.sidebar.number_input("Kapitał ($)", value=10000)
years = st.sidebar.slider("Lata", 1, 10, 2)

# Globalne suwaki (zmieniają się po AUTO)
rsi_val = st.sidebar.slider("RSI Próg", 20, 80, 65)
ema_f_val = st.sidebar.slider("Szybka EMA", 5, 30, 9)
ema_s_val = st.sidebar.slider("Wolna EMA", 10, 100, 17)
sl_val = st.sidebar.slider("Stop Loss %", 0.5, 15.0, 3.0) / 100

# --- SILNIK OBLICZENIOWY ---
def run_backtest(df_in, rsi_t, ef, es, sl):
    df = df_in.copy()
    df['EMA_F'] = ta.ema(df['Close'], length=ef)
    df['EMA_S'] = ta.ema(df['Close'], length=es)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df = df.dropna()
    
    cash, shares, entry_p = float(capital), 0.0, 0.0
    for i in range(len(df)):
        p = float(df['Close'].iloc[i])
        if shares == 0:
            # Wejście: Trend + RSI + Filtr "Ekstremalnie Tanio"
            if (df['EMA_F'].iloc[i] > df['EMA_S'].iloc[i] and df['RSI'].iloc[i] >= rsi_t):
                shares = cash / p
                entry_p, cash = p, 0.0
        elif shares > 0:
            # Wyjście: SL (zoptymalizowany) lub przecięcie EMA
            if p <= entry_p * (1 - sl) or df['EMA_F'].iloc[i] < df['EMA_S'].iloc[i]:
                cash = shares * p
                shares = 0.0
    return cash if shares == 0 else shares * float(df['Close'].iloc[-1])

# --- DANE I REGRESJA ---
@st.cache_data
def get_processed_data(symbol, yrs):
    d = yf.download(symbol, start=datetime.now()-timedelta(days=yrs*365), end=datetime.now())
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    
    # Logarytmiczna regresja liniowa
    lc = np.log(d['Close'].values)
    x = np.arange(len(lc))
    m, c = np.polyfit(x, lc, 1)
    line = m*x + c
    
    # Znajdowanie NAJWIĘKSZEGO historycznego odchylenia (Max Downside)
    diff = lc - line
    max_neg_dev = np.min(diff) # Największe wychylenie w dół
    
    d['Reg_Mid'] = np.exp(line)
    d['Reg_Max_Lower'] = np.exp(line + max_neg_dev) # Statystyczne "Dno Dna"
    return d

data = get_processed_data(ticker, years)

# --- AUTO OPTYMALIZACJA (GRID SEARCH PRO) ---
if st.sidebar.button("🚀 URUCHOM PEŁNE AUTO (EMA + RSI + SL)"):
    with st.spinner("Przeszukuję setki kombinacji..."):
        best_res, best_params = 0, {}
        for r in [50, 60, 70]:
            for f in [9, 13]:
                for s in [21, 50]:
                    for sl_opt in [0.02, 0.05, 0.08]: # Testowanie SL 2%, 5% i 8%
                        res = run_backtest(data, r, f, s, sl_opt)
                        if res > best_res:
                            best_res, best_params = res, {'rsi': r, 'f': f, 's': s, 'sl': sl_opt}
        
        st.success(f"ZNALEZIONO! Najlepszy SL: {best_params['sl']*100}% | RSI: {best_params['rsi']} | EMA: {best_params['f']}/{best_params['s']}")
        rsi_val, ema_f_val, ema_s_val, sl_val = best_params['rsi'], best_params['f'], best_params['s'], best_params['sl']

# --- WIZUALIZACJA ---
final_money = run_backtest(data, rsi_val, ema_f_val, ema_s_val, sl_val)
st.title(f"🚀 Wyniki dla {ticker}")

c1, c2, c3 = st.columns(3)
c1.metric("Wynik Strategii", f"{((final_money-capital)/capital):.2%}")
c2.metric("Kapitał Końcowy", f"${final_money:,.2f}")
c3.metric("Najlepszy SL (Auto)", f"{sl_val*100}%")

# Wykres z Max Deviation
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Cena", line=dict(color='white')))
fig.add_trace(go.Scatter(x=data.index, y=data['Reg_Max_Lower'], name="HISTORYCZNE DNO (Max Dev)", line=dict(color='yellow', dash='longdash')))
fig.update_layout(template="plotly_dark", height=600, yaxis_type="log")
st.plotly_chart(fig, use_container_width=True)

st.info("Żółta linia to 'największe historyczne odchylenie'. Jeśli cena ją dotyka, statystycznie taniej w tym okresie nie było.")
