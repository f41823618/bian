import requests
import pandas as pd
import ta
import datetime
import os
import numpy as np
import google.generativeai as genai

# --- 1. 配置区 ---
SYMBOLS = [
    ("BTC/USDT", "BTC-USDT"), ("ETH/USDT", "ETH-USDT"), ("XRP/USDT", "XRP-USDT"),
    ("BNB/USDT", "BNB-USDT"), ("SOL/USDT", "SOL-USDT"), ("DOGE/USDT", "DOGE-USDT"),
    ("ADA/USDT", "ADA-USDT"), ("AVAX/USDT", "AVAX-USDT"), ("LINK/USDT", "LINK-USDT")
]
KUCOIN_BASE = "https://api.kucoin.com"

# 从 GitHub Secrets 获取密钥
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID")

# --- 2. 数据获取函数 ---
def fetch_ohlcv(symbol_kucoin, limit=100):
    url = f"{KUCOIN_BASE}/api/v1/market/candles"
    resp = requests.get(url, params={"type": "1hour", "symbol": symbol_kucoin}, timeout=15)
    data = resp.json()["data"]
    data.reverse()
    df = pd.DataFrame(data[-limit:], columns=["ts", "o", "c", "h", "l", "v", "t"])
    df[["c", "h", "l"]] = df[["c", "h", "l"]].astype(float)
    return df

def calculate_beta(coin_close, btc_close):
    coin_pct = coin_close.pct_change().dropna()
    btc_pct = btc_close.pct_change().dropna()
    min_len = min(len(coin_pct), len(btc_pct))
    if min_len < 2: return 1.0
    return round(np.cov(coin_pct[-min_len:], btc_pct[-min_len:])[0][1] / np.var(btc_pct[-min_len:]), 2)

# --- 3. 核心运行逻辑 ---
def run_analysis():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🚀 开始分析时间: {now}")
    
    btc_df = fetch_ohlcv("BTC-USDT")
    results = []
    
    for display_sym, kucoin_sym in SYMBOLS:
        try:
            df = fetch_ohlcv(kucoin_sym)
            close = df["c"]
            # 计算指标
            rsi = ta.momentum.RSIIndicator(close, 14).rsi().iloc[-1]
            bb = ta.volatility.BollingerBands(close)
            bb_pos = ((close.iloc[-1] - bb.bollinger_lband().iloc[-1]) / (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])) * 100
            atr = ta.volatility.AverageTrueRange(df["h"], df["l"], close).average_true_range().iloc[-1]
            beta = calculate_beta(close, btc_df["c"])
            
            results.append({
                "币种": display_sym, "价格": close.iloc[-1], "RSI": round(rsi, 1),
                "布林位置%": round(bb_pos, 1), "Beta": beta, "ATR%": round(atr/close.iloc[-1]*100, 3)
            })
        except Exception as e:
            print(f"获取 {display_sym} 失败: {e}")

    if not results: return
    
    # 转换为数据表格
    df_res = pd.DataFrame(results)
    
    # 4. 调用 AI (Gemini)
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = f"""
    作为高级量化策略大脑。账户资金：180 USDT，杠杆：10x。
    实时数据如下：
    {df_res.to_markdown(index=False)}
    
    任务：
    1. 严格筛选：多头(BB位置<0, RSI<35) 对冲 空头(BB位置>100, RSI>65)。
    2. 给出 5 个最佳组合表格：包含多/空币种、基于Beta的配比下单金额、TP/SL建议。
    3. 如果 ATR% > 1.5%，提示用户该币种对 10x 杠杆极度危险。
    """
    
    try:
        response = model.generate_content(prompt)
        # 5. 发送 Telegram
        url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
        report = f"📊 **AI 24H 对冲策略 ({now})**\n\n{response.text}"
        requests.post(url, json={"chat_id": TG_CHAT_ID, "text": report, "parse_mode": "Markdown"}, timeout=15)
        print("✅ 分析已发送到 Telegram")
    except Exception as e:
        print(f"AI 或 TG 发送失败: {e}")

if __name__ == "__main__":
    run_analysis() # 运行一次即结束，适配 GitHub Actions
