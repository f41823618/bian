import requests
import pandas as pd
import ta
import time
import datetime
import os
import numpy as np
import google.generativeai as genai

# --- 配置区 ---
SYMBOLS = ["SUI", "BNB", "DOGE", "SOL", "ETH", "BTC", "XRP", "AVAX", "LTC", "ADA", "LINK", "AAVE"]
BINANCE_FAPI_BASE = "https://fapi.binance.com"
# 变量建议通过环境变量获取
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "你的_API_KEY")
TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "你的_TG_TOKEN")
TG_CHAT_ID = os.getenv("TG_CHAT_ID", "你的_CHAT_ID")

genai.configure(api_key=GEMINI_API_KEY)
ai_model = genai.GenerativeModel('gemini-1.5-flash')

def fetch_klines(symbol, interval, limit=100):
    url = f"{BINANCE_FAPI_BASE}/fapi/v1/klines"
    params = {"symbol": f"{symbol}USDT", "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    df = pd.DataFrame(resp.json(), columns=["ts", "o", "h", "l", "c", "v", "ct", "qv", "nt", "tb", "tq", "i"])
    df["c"] = df["c"].astype(float)
    df["h"] = df["h"].astype(float)
    df["l"] = df["l"].astype(float)
    return df

def calculate_beta(coin_close, btc_close):
    """计算该币种相对于BTC的Beta系数 (最近100小时)"""
    coin_pct = coin_close.pct_change().dropna()
    btc_pct = btc_close.pct_change().dropna()
    # 确保长度一致
    min_len = min(len(coin_pct), len(btc_pct))
    covariance = np.cov(coin_pct[-min_len:], btc_pct[-min_len:])[0][1]
    variance = np.var(btc_pct[-min_len:])
    return round(covariance / variance, 2) if variance != 0 else 1.0

def get_derivatives(symbol):
    pair = f"{symbol}USDT"
    f_rate = float(requests.get(f"{BINANCE_FAPI_BASE}/fapi/v1/premiumIndex?symbol={pair}").json().get("lastFundingRate", 0)) * 100
    oi_data = requests.get(f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist", params={"symbol": pair, "period": "15m", "limit": 2}).json()
    oi_trend = ((float(oi_data[1]["sumOpenInterest"]) / float(oi_data[0]["sumOpenInterest"])) - 1) * 100 if len(oi_data) == 2 else 0
    return f_rate, oi_trend

def run_cycle():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] 深度分析引擎运行中...")
    
    # 1. 先获取 BTC 基础数据用于计算 Beta
    btc_df_1h = fetch_klines("BTC", "1h", 100)
    
    all_data = []
    for symbol in SYMBOLS:
        try:
            df_15m = fetch_klines(symbol, "15m", 100)
            df_1h = fetch_klines(symbol, "1h", 100)
            curr_p = df_15m["c"].iloc[-1]
            
            # 指标计算
            rsi6 = ta.momentum.RSIIndicator(df_1h["c"], 6).rsi().iloc[-1]
            macd = ta.trend.MACD(df_1h["c"])
            bb = ta.volatility.BollingerBands(df_1h["c"])
            bb_pos = ((curr_p - bb.bollinger_lband().iloc[-1]) / (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1])) * 100
            atr = ta.volatility.AverageTrueRange(df_1h["h"], df_1h["l"], df_1h["c"]).average_true_range().iloc[-1]
            
            funding, oi_trend = get_derivatives(symbol)
            beta = calculate_beta(df_1h["c"], btc_df_1h["c"])

            all_data.append({
                "币种": symbol, "价格": curr_p, 
                "15m%": round((curr_p/df_15m["c"].iloc[-2]-1)*100, 2),
                "24h%": round((curr_p/df_15m["c"].iloc[-97]-1)*100, 2),
                "RSI(6)": round(rsi6, 1), 
                "MACD_Hist": round(macd.macd_diff().iloc[-1], 4),
                "布林位置%": round(bb_pos, 1), "Beta": beta,
                "资金%": round(funding, 4), "OI%": round(oi_trend, 2), "ATR": round(atr, 4)
            })
        except Exception as e: print(f"Error {symbol}: {e}")

    if not all_data: return
    df_result = pd.DataFrame(all_data)
    
    # 2. 构造极其严谨的 AI Prompt
    prompt = f"""
    你现在是高级量化策略大脑。账户资金：180 USDT，杠杆：10x（总持仓额度 1800 USDT）。
    
    【核心数据表格】：
    {df_result.to_markdown(index=False)}
    
    【执行逻辑】：
    1. 严格筛选：多头币种布林位置 < 0% 且 RSI(6) < 30；空头币种布林位置 > 100% 且 RSI(6) > 80。
    2. 对冲配比：必须利用 Beta 系数。若多头 Beta 为 1.2，空头 Beta 为 0.8，则下单金额应调整以实现中性。
    3. 极端风险预警：若 ATR > 价格的 2%，严禁开满 10x 杠杆，需提示降低金额。
    4. 趋势确认：观察 OI 趋势。价格跌+OI涨（真跌）适合做空；价格涨+OI涨（真涨）适合做多。
    
    【输出格式要求】：
    请严格列出 5 个最佳多空对冲组合，并使用以下 Markdown 表格输出：
    | 多头 + 空头 | 下单金额(多/空) | 当前价格(多/空) | 胜率估算 | TP/SL价格(具体标注) | 理由(重点因子) |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    
    最后请给出整体风险提醒和基于 180 USDT 的仓位管理方案。
    """

    try:
        response = ai_model.generate_content(prompt)
        report = f"🎯 **24H对冲策略专家建议**\n\n{response.text}"
        
        # 3. 发送 Telegram
        requests.post(f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage", 
                      json={"chat_id": TG_CHAT_ID, "text": report, "parse_mode": "Markdown"})
        # 4. 覆盖写入 Excel
        df_result.to_excel("crypto_analysis.xlsx", index=False)
        print("✅ 策略已推送至 Telegram 并更新 Excel")
    except Exception as e:
        print(f"AI 响应失败: {e}")
def main():
    print("=" * 100)
    print("  KuCoin 多币种技术指标监控启动...")
    print("=" * 100)
    try:
        run_cycle() # 只运行一次
        print("\n✅ 本次分析完成。")
    except Exception as e:
        print(f"  ❌ 运行错误: {e}")

if __name__ == "__main__":
    main()
