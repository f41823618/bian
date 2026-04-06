import requests
import pandas as pd
import ta
import datetime
import os
import numpy as np
import google.generativeai as genai

# --- 配置 ---
SYMBOLS = [
    ("BTC/USDT", "BTC-USDT"), ("ETH/USDT", "ETH-USDT"), ("SOL/USDT", "SOL-USDT"),
    ("XRP/USDT", "XRP-USDT"), ("DOGE/USDT", "DOGE-USDT"), ("AVAX/USDT", "AVAX-USDT")
]
KUCOIN_BASE = "https://api.kucoin.com"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

def fetch_ohlcv(symbol_kucoin):
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    resp = requests.get(url, params={"type": "1hour", "symbol": symbol_kucoin}, timeout=15)
    data = resp.json()["data"]
    data.reverse()
    df = pd.DataFrame(data, columns=["ts", "o", "c", "h", "l", "v", "t"])
    df[["c", "h", "l"]] = df[["c", "h", "l"]].astype(float)
    return df

def run_cycle():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []
    for disp, sym in SYMBOLS:
        try:
            df = fetch_ohlcv(sym)
            close = df["c"]
            rsi = ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1]
            bb = ta.volatility.BollingerBands(close)
            bb_pos = ((close.iloc[-1] - bb.bollinger_lband().iloc[-1]) / (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])) * 100
            results.append({"币种": disp, "价格": close.iloc[-1], "RSI": round(rsi, 1), "布林位置%": round(bb_pos, 1)})
        except: continue

    if not results: return
    
    # AI 分析
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"账户180USDT, 10x杠杆。分析数据并给5对对冲组合(多单BB<0/RSI<35, 空单BB>100/RSI>65):\n{pd.DataFrame(results).to_markdown()}"
    
    try:
        response = model.generate_content(prompt)
        # 推送 Telegram
        msg = f"🤖 **AI 对冲日报 ({now})**\n\n{response.text}"
        requests.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage", json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        print("✅ 发送成功")
    except Exception as e:
        print(f"❌ 失败: {e}")

if __name__ == "__main__":
    run_cycle() # 运行一次即结束
