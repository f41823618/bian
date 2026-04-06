import requests
import pandas as pd
import ta
import datetime
import os
import numpy as np
import google.generativeai as genai

# --- 配置区 ---
SYMBOLS = [
    ("BTC/USDT", "BTC-USDT"), ("ETH/USDT", "ETH-USDT"), ("XRP/USDT", "XRP-USDT"),
    ("BNB/USDT", "BNB-USDT"), ("SOL/USDT", "SOL-USDT"), ("DOGE/USDT", "DOGE-USDT"),
    ("TRX/USDT", "TRX-USDT"), ("ADA/USDT", "ADA-USDT"), ("AVAX/USDT", "AVAX-USDT"),
    ("LINK/USDT", "LINK-USDT")
]
KUCOIN_BASE = "https://api.kucoin.com"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# 初始化 Gemini
genai.configure(api_key=GEMINI_API_KEY)
ai_model = genai.GenerativeModel('gemini-1.5-flash')

def fetch_ohlcv(symbol_kucoin, limit=100):
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    resp = requests.get(url, params={"type": "1hour", "symbol": symbol_kucoin}, timeout=15)
    candles = resp.json()["data"]
    candles.reverse()
    df = pd.DataFrame(candles[-limit:], columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
    df[["close", "high", "low"]] = df[["close", "high", "low"]].astype(float)
    return df

def calculate_beta(coin_close, btc_close):
    coin_pct = coin_close.pct_change().dropna()
    btc_pct = btc_close.pct_change().dropna()
    min_len = min(len(coin_pct), len(btc_pct))
    if min_len < 2: return 1.0
    return round(np.cov(coin_pct[-min_len:], btc_pct[-min_len:])[0][1] / np.var(btc_pct[-min_len:]), 2)

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TG_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)

def run_cycle():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    btc_df = fetch_ohlcv("BTC-USDT")
    results = []
    
    for display_sym, kucoin_sym in SYMBOLS:
        try:
            df = fetch_ohlcv(kucoin_sym)
            close = df["close"]
            # 计算指标
            rsi = ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1]
            bb = ta.volatility.BollingerBands(close)
            bb_pos = ((close.iloc[-1] - bb.bollinger_lband().iloc[-1]) / (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])) * 100
            atr = ta.volatility.AverageTrueRange(df["high"], df["low"], close).average_true_range().iloc[-1]
            beta = calculate_beta(close, btc_df["close"])
            
            results.append({
                "币种": display_sym, "价格": close.iloc[-1], "RSI": round(rsi, 1),
                "布林位置%": round(bb_pos, 1), "Beta": beta, "ATR": round(atr, 4)
            })
        except: continue

    if not results: return
    df_res = pd.DataFrame(results)
    
    # 构造 AI 提示词
    prompt = f"""
    作为量化专家，基于 180 USDT 总保证金和 10x 杠杆，分析以下数据：
    {df_res.to_markdown(index=False)}
    
    要求：
    1. 寻找严格对冲组合：多头(BB位置<0, RSI<30) vs 空头(BB位置>100, RSI>70)。
    2. 给出 5 个最佳组合表格，包含：多/空币种、基于Beta的配比金额、TP/SL建议。
    3. 提醒 ATR 超过价格 2% 的爆仓风险。
    """
    
    response = ai_model.generate_content(prompt)
    send_telegram(f"🚀 **AI 对冲策略报告 ({now})**\n\n{response.text}")

if __name__ == "__main__":
    run_cycle() # 运行一次即退出，适配 GitHub Actions
