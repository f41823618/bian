import requests
import pandas as pd
import ta
import datetime
import os
import numpy as np
import google.generativeai as genai

# --- 1. 配置区 ---
SYMBOLS = [
    ("BTC/USDT", "BTC-USDT"), ("ETH/USDT", "ETH-USDT"), ("SOL/USDT", "SOL-USDT"),
    ("XRP/USDT", "XRP-USDT"), ("DOGE/USDT", "DOGE-USDT"), ("AVAX/USDT", "AVAX-USDT")
]
KUCOIN_BASE = "https://api.kucoin.com"

# 环境变量 (由 GitHub Secrets 提供)
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

def run_analysis():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = []
    
    # 抓取数据并计算基础指标
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
    
    # --- 2. AI 分析 (修正模型名称) ---
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # 注意：这里使用 gemini-1.5-flash，确保 API Key 有效
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        data_table = pd.DataFrame(results).to_markdown()
        prompt = f"账户180USDT, 杠杆10x。请根据以下数据，找出符合对冲逻辑(多单BB<0/RSI<35, 空单BB>100/RSI>65)的组合并给出下单金额建议：\n\n{data_table}"
        
        response = model.generate_content(prompt)
        ai_text = response.text
        
        # --- 3. Telegram 推送 ---
        msg = f"🚀 **AI 策略推送 ({now})**\n\n{ai_text}"
        tg_url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        resp = requests.post(tg_url, json={"chat_id": TG_CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        
        if resp.status_code == 200:
            print("✅ 推送成功！")
        else:
            print(f"❌ 推送失败: {resp.text}")
            
    except Exception as e:
        print(f"❌ AI 分析或推送环节出错: {e}")

if __name__ == "__main__":
    run_analysis() # 运行一次即结束，由 GitHub Actions 调度
