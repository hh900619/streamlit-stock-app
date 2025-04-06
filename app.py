
#cd "C:\Users\0619h\OneDrive\Desktop\streamlit-stock-app"
#git add .
#git commit -m "改"
#git push origin main

import streamlit as st
st.set_page_config(page_title="比對", layout="wide")
import plotly.graph_objects as go
import time
from numpy.linalg import norm
from dtaidistance import dtw
import random
import datetime
import os
from dotenv import load_dotenv
from supabase import create_client
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta

################################✅ 1️⃣ 資料庫初始化區 ################################

# 載入本地 .env
load_dotenv('app.env')

# 本機優先讀 .env，Cloud 再讀 st.secrets
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    try:
        SUPABASE_URL = st.secrets["SUPABASE_URL"]
        SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    except Exception as e:
        st.error("❌ Supabase 設定錯誤，請檢查 .env 或 secrets.toml")
        st.stop()

# 建立 Supabase 連線
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 存資料函式
def save_stats_to_supabase(ticker, category, mode,
                           dtw_min, dtw_max, dtw_avg,
                           total_score_min, total_score_max, total_score_avg):
    data = {
        "ticker": ticker,
        "category": category,  # 將產品類型記錄下來
        "mode": mode,
        "dtw_min": dtw_min,
        "dtw_max": dtw_max,
        "dtw_avg": dtw_avg,
        "total_score_min": total_score_min,
        "total_score_max": total_score_max,
        "total_score_avg": total_score_avg,
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    response = supabase.table("stats").insert(data).execute()
    print("✅ 寫入 Supabase 完成", response)

################################✅ 2️⃣ 資料下載區 ################################

def download_data(ticker):
    df = yf.download(ticker, period='max', interval='1d', auto_adjust=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            df[col] = df["Close"]
    df = df[["Open", "High", "Low", "Close"]].dropna(how='any')
    df = df[~df.index.duplicated(keep='first')].sort_index()
    return df

def download_vix(start, end):
    vx = yf.download("^VIX", start=start, end=end, interval='1d', auto_adjust=False)
    if vx.empty:
        return pd.DataFrame()
    if isinstance(vx.columns, pd.MultiIndex):
        vx.columns = vx.columns.get_level_values(0)
    vx = vx[["Close"]].dropna(how='any')
    vx = vx[~vx.index.duplicated(keep='first')].sort_index()
    return vx

################################✅ 3️⃣ 基本工具函式區 ################################

def get_vix_value(dt, vix_df):
    if vix_df.empty:
        return 20.0
    if dt in vix_df.index:
        return float(vix_df.loc[dt, "Close"])
    valid_idx = vix_df.index[vix_df.index <= dt]
    if len(valid_idx) == 0:
        return float(vix_df["Close"].iloc[0])
    return float(vix_df.loc[valid_idx[-1], "Close"])

def classify_ticker(ticker):
    """
    自動判斷商品類型，僅用於資料庫記錄，不影響比對流程
    """
    ticker = ticker.upper()
    if any(keyword in ticker for keyword in ['GOLD', 'SILVER', 'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F', 'XAU', 'XAG']):
        return "metal"
    elif any(keyword in ticker for keyword in ['CL=F', 'BRENT', 'NG=F', 'OIL', 'RB=F', 'HO=F', 'XLE']):
        return "energy"
    elif any(keyword in ticker for keyword in ['CORN', 'SOY', 'WHEAT', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'COTTON', 'SUGAR']):
        return "agriculture"
    elif any(keyword in ticker for keyword in [
        'AAPL', 'MSFT', 'GOOG', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD', 'INTC', 'AMZN', 'NFLX', 'BABA', 'CRM', 'ADBE', 'ORCL'
    ]):
        return "tech"
    elif any(keyword in ticker for keyword in [
        'SPY', 'QQQ', 'IWM', 'VOO', 'VTI', 'XLK', 'XLF', 'XLE', 'XLV', 'XLU', 'SMH', 'ARKK'
    ]):
        return "etf"
    elif any(keyword in ticker for keyword in [
        'ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'SP500', 'NASDAQ', 'DOW', 'RUSSELL'
    ]):
        return "index_futures"
    elif any(keyword in ticker for keyword in [
        'JPY=X', 'EUR=X', 'AUD=X', 'GBP=X', 'USDCAD=X', 'USDCHF=X', 'USD', 'FX', 'EURUSD', 'GBPUSD'
    ]):
        return "forex"
    else:
        return "stock"

def generate_future_dates(history_dates, fut_len):
    start_date = history_dates[-1] + timedelta(days=1)
    future_dates = pd.bdate_range(start=start_date, periods=fut_len)
    return future_dates

def weighted_random_choice(candidates):
    scores = np.array([s for (_, s) in candidates])
    weights = scores / scores.sum()
    idx = np.random.choice(len(candidates), p=weights)
    return candidates[idx]

################################✅ 4️⃣ 統計查詢函式（連資料庫用）################################

def get_stat_ranges_from_supabase(ticker):
    response = supabase.table('stats').select(
        'dtw_min, dtw_max, dtw_avg, total_score_min, total_score_max, total_score_avg'
    ).eq('ticker', ticker).execute()

    if not response.data:
        return None

    dtw_min = float(min(float(item['dtw_min']) for item in response.data))
    dtw_max = float(max(float(item['dtw_max']) for item in response.data))
    dtw_avg = float(sum(float(item['dtw_avg']) for item in response.data) / len(response.data))
    score_min = float(min(float(item['total_score_min']) for item in response.data))
    score_max = float(max(float(item['total_score_max']) for item in response.data))
    score_avg = float(sum(float(item['total_score_avg']) for item in response.data) / len(response.data))

    return {
        'dtw_min': dtw_min,
        'dtw_max': dtw_max,
        'dtw_avg': dtw_avg,
        'score_min': score_min,
        'score_max': score_max,
        'score_avg': score_avg
    }

def get_stats_count_from_supabase(ticker):
    try:
        response = supabase.table('stats').select('id').eq('ticker', ticker).execute()
        if response.data:
            return len(response.data)
        else:
            return 0
    except Exception as e:
        st.warning(f"查詢 Supabase 失敗：{e}")
        return 0

################################✅ 5️⃣ 技術分析＆特徵計算區 ################################

def kbar_features(df):
    arr = []
    prev_close = None
    for i, (idx, row) in enumerate(df.iterrows()):
        o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
        if i == 0:
            prev_close = o
        gap_pct = (o - prev_close) / max(abs(prev_close), 1e-9) * 100
        body_pct = (c - o) / max(abs(o), 1e-9) * 100
        daily_chg = (c - prev_close) / max(abs(prev_close), 1e-9) * 100
        rng = (h - l) / max(abs(prev_close), 1e-9) * 100
        color = 1 if c >= o else 0
        arr.append([gap_pct, body_pct, daily_chg, rng, color])
        prev_close = c
    return np.array(arr)

def dtw_distance(series_a, series_b):
    a = series_a.values if isinstance(series_a, pd.Series) else np.array(series_a)
    b = series_b.values if isinstance(series_b, pd.Series) else np.array(series_b)

    # 1) 計算平均值、標準差
    mean_a, std_a = a.mean(), a.std()
    mean_b, std_b = b.mean(), b.std()

    # 2) 避免 std=0 的情況，用 1e-9 防呆
    a = (a - mean_a) / max(std_a, 1e-9)
    b = (b - mean_b) / max(std_b, 1e-9)

    # 3) 計算 DTW 距離
    dist = dtw.distance(a, b)

    return float(dist) if np.isscalar(dist) else float(np.array(dist).item())

def cosine_similarity(a, b):
    if len(a) != len(b):
        return -1
    a_flat = a.flatten()
    b_flat = b.flatten()
    cos_sim = np.dot(a_flat, b_flat) / (norm(a_flat) * norm(b_flat) + 1e-9)
    return cos_sim

def compute_volatility_score(curr_seg, sample_seg):
    curr_return = (curr_seg["Close"].iloc[-1] - curr_seg["Close"].iloc[0]) / max(abs(curr_seg["Close"].iloc[0]), 1e-9)
    sample_return = (sample_seg["Close"].iloc[-1] - sample_seg["Close"].iloc[0]) / max(abs(sample_seg["Close"].iloc[0]), 1e-9)
    dir_diff = abs(curr_return - sample_return)
    curr_vol = (curr_seg["Close"].max() - curr_seg["Close"].min()) / max(abs(curr_seg["Close"].iloc[0]), 1e-9)
    sample_vol = (sample_seg["Close"].max() - sample_seg["Close"].min()) / max(abs(sample_seg["Close"].iloc[0]), 1e-9)
    vol_diff = abs(curr_vol - sample_vol)
    score = max(0, (1 - (0.5 * dir_diff + 0.5 * vol_diff))) * 100
    return float(score)

def compute_slope_angle_similarity(curr_seg, sample_seg):
    x = np.arange(len(curr_seg))
    y_curr = curr_seg["Close"].values
    y_samp = sample_seg["Close"].values
    slope_curr, _ = np.polyfit(x, y_curr, 1)
    slope_samp, _ = np.polyfit(x, y_samp, 1)
    angle_curr = np.degrees(np.arctan(slope_curr))
    angle_samp = np.degrees(np.arctan(slope_samp))
    angle_diff = abs(angle_curr - angle_samp)
    score = max(0, 100 - angle_diff)
    return float(score)

################################✅ 6️⃣ 篩選邏輯區 ################################

def filter_by_shape(curr_seg, df, seg_len, fut_len, dtw_thr):
    """
    篩選候選樣本，僅根據 K 線特徵的 Cosine 相似度進行篩選
    注意：dtw_thr 參數不再用於篩選，但為了保持介面一致仍傳回
    """
    if len(df) < seg_len + fut_len:
        return [], dtw_thr

    curr_feat = kbar_features(curr_seg)
    candidates = []

    for i in range(seg_len, len(df) - fut_len):
        sample_seg = df.iloc[i - seg_len: i]
        samp_feat = kbar_features(sample_seg)
        cos_sim = float(cosine_similarity(curr_feat, samp_feat))
        # 僅依據 cosine similarity > 0.2 篩選
        if cos_sim > 0.2:
            candidates.append((i, cos_sim))
    
    # 可視需要印出統計資訊 (若不需要則可移除)
    st.write(f"🔎 透過 Shape 篩選後共獲得 {len(candidates)} 筆候選樣本")
    
    return candidates, dtw_thr

def compute_total_scores(curr_seg, df, shape_candidates, seg_len, total_score_thr, dtw_thr, ticker, category, mode):
    results, total_scores = [], []
    dtw_list = []
    for i, cos_sim in shape_candidates:
        sample_seg = df.iloc[i - seg_len: i]
        vol_score = float(compute_volatility_score(curr_seg, sample_seg))
        slope_score = float(compute_slope_angle_similarity(curr_seg, sample_seg))
        dtw_dist = float(dtw_distance(curr_seg["Close"], sample_seg["Close"]))
        dtw_list.append(dtw_dist)
        dtw_norm = min(dtw_dist / dtw_thr, 1.0)
        total_score = 0.3 * vol_score + 0.2 * slope_score + 0.25 * (1 - dtw_norm) * 100 + 0.25 * cos_sim * 100
        if isinstance(total_score, (pd.Series, np.ndarray)):
            total_score = float(total_score)
        total_scores.append(total_score)
        if total_score >= total_score_thr:
            results.append((i, total_score))
    if total_scores:
        min_score, max_score = min(total_scores), max(total_scores)
        avg_score = np.mean(total_scores)
        try:
            st.write(f"📏 綜合總分 min={min_score:.2f}, max={max_score:.2f}, avg={avg_score:.2f}")
        except:
            pass
    else:
        min_score = max_score = avg_score = 0
    if dtw_list:
        min_dtw, max_dtw = min(dtw_list), max(dtw_list)
        avg_dtw = np.mean(dtw_list)
        try:
            st.write(f"🌀 DTW 距離 min={min_dtw:.2f}, max={max_dtw:.2f}, avg={avg_dtw:.2f}")
        except:
            pass
    else:
        min_dtw = max_dtw = avg_dtw = 0
    save_stats_to_supabase(
        ticker=ticker,
        category=category,
        mode=mode,
        dtw_min=min_dtw,
        dtw_max=max_dtw,
        dtw_avg=avg_dtw,
        total_score_min=min_score,
        total_score_max=max_score,
        total_score_avg=avg_score
    )
    return results, total_score_thr

################################✅ 7️⃣ VIX 加權與最終篩選區 ################################

def compute_vix_score(curr_seg, sample_seg, vix_df):
    curr_end = curr_seg.index[-1]
    samp_end = sample_seg.index[-1]
    cv = get_vix_value(curr_end, vix_df)
    sv = get_vix_value(samp_end, vix_df)
    diff = abs(cv - sv) / max(cv, sv, 1e-9)
    vix_sc = max(0, (1 - diff)) * 100
    return vix_sc

def final_vix_filter(curr_seg, df, atr_list, seg_len, vix_df, category, topN):
    final = []
    # 動態決定 VIX 權重
    if category == "metal":
        vix_weight = 0.2
    elif category == "forex":
        vix_weight = 0.5
    else:
        vix_weight = 0.3
    for (i, vol_score) in atr_list:
        sample_seg = df.iloc[i - seg_len: i]
        vix_score = compute_vix_score(curr_seg, sample_seg, vix_df)
        total_score = (1 - vix_weight) * vol_score + vix_weight * vix_score
        final.append((i, total_score))
    if not final:
        return []
    final_sorted = sorted(final, key=lambda x: x[1], reverse=True)
    try:
        st.write(f"🟢 VIX 加權後樣本數：{len(final_sorted)}, 分數範圍 min={min([s for (_, s) in final_sorted]):.2f}, max={max([s for (_, s) in final_sorted]):.2f}")
    except:
        pass
    return final_sorted[:topN]

################################✅ 8️⃣ 複製未來K棒邏輯 ################################

def copy_future_bars_percent_mode(df, best_i, fut_len):
    samp = df.iloc[best_i: best_i + fut_len].copy()
    if len(samp) < fut_len:
        return None, None
    last_close = df["Close"].iloc[-1]
    hist_prev_close = df["Close"].iloc[best_i - 1]
    new_ohlc = []
    future_dates = pd.bdate_range(
        start=df.index[-1] + pd.Timedelta(days=1),
        periods=fut_len,
        freq='B'
    )
    for i in range(fut_len):
        row = samp.iloc[i]
        ratio_open = row["Open"] / max(abs(hist_prev_close), 1e-9)
        new_open = last_close * ratio_open
        ratio_h = row["High"] / max(abs(row["Open"]), 1e-9)
        ratio_l = row["Low"] / max(abs(row["Open"]), 1e-9)
        ratio_c = row["Close"] / max(abs(row["Open"]), 1e-9)
        new_high = new_open * ratio_h
        new_low = new_open * ratio_l
        new_close = new_open * ratio_c
        hi = max(new_high, new_low, new_open, new_close)
        lo = min(new_high, new_low, new_open, new_close)
        new_ohlc.append([new_open, hi, lo, new_close])
        last_close = new_close
        hist_prev_close = row["Close"]
    return future_dates, new_ohlc

################################✅ 9️⃣ 自動分類決策邏輯 ################################
# 此部分原先依據產品類型選擇比對策略，但目前參數生成已不再依產品類型調整，
# 如無其他用途，可保留作資料庫記錄或視覺化標示
def select_matching_method(product_type):
    if product_type in ["metal", "energy", "agriculture"]:
        return "dtw_first"
    elif product_type in ["forex", "index_futures"]:
        return "volatility_first"
    else:
        return "balanced"

################################✅ 🔟 核心比對主流程 ################################

def find_best_match_advanced(df, vix_df, seg_len, fut_len, total_score_thr, dtw_thr, topN, ticker, category, mode):
    if len(df) < seg_len + fut_len:
        return None
    curr_seg = df.iloc[-seg_len:]
    shape_candidates, _ = filter_by_shape(curr_seg, df, seg_len, fut_len, dtw_thr)
    print(f"🔎 Shape 篩選後樣本數: {len(shape_candidates)}")
    if not shape_candidates:
        print("❌ 沒有通過 Shape 篩選")
        return None
    total_score_list, total_score_thr = compute_total_scores(
        curr_seg, df, shape_candidates, seg_len, total_score_thr, dtw_thr,
        ticker=ticker, category=category, mode=mode
    )
    print(f"✅ 綜合總分篩選後樣本數: {len(total_score_list)}")
    if not total_score_list:
        print("❌ 沒有通過總分篩選")
        return None
    final_list = final_vix_filter(curr_seg, df, total_score_list, seg_len, vix_df, category, topN)
    if not final_list:
        print("❌ 沒有通過 VIX 篩選")
        return None
    if len(final_list) < topN:
        print(f"⚠️ 最終樣本數 {len(final_list)}，小於 TopN={topN}")
    print(f"🧠 自動比對邏輯：{mode}")
    final_list_sorted = sorted(final_list, key=lambda x: x[1], reverse=True)
    return final_list_sorted

################################✅ 1️⃣1️⃣ 主程式 main() ################################

def get_default_params_v2(mode, dtw_range, score_range):
    """
    根據歷史統計區間與模式設定預設參數，因為正規化後 DTW 數值較小，
    所以 dtw_thr 也設定為較小的值。
    回傳 (total_score_thr, topN, dtw_thr)
    """
    dtw_min, dtw_max = dtw_range
    score_min, score_max = score_range

    if mode == "保守":
        # 取較低門檻，保留更多候選樣本
        total_score_thr = int((score_min + score_max) / 3)
        topN = 30
        dtw_thr = 5  # 由於正規化後 DTW 數值通常在 3 左右，故此處設定為 5
    elif mode == "平衡":
        total_score_thr = int((score_min + score_max) / 2)
        topN = 50
        dtw_thr = 8
    elif mode == "寬鬆":
        total_score_thr = int(score_min * 0.8)
        topN = 80
        dtw_thr = 10
    else:  # 自訂模式
        total_score_thr = st.slider("總分門檻 (0~100)", min_value=20.0, max_value=100.0, value=80.0, step=0.1)
        topN = st.slider("TopN 隨機選擇", 1, 200, 50)
        dtw_thr = st.slider("DTW 閾值", min_value=1, max_value=50, value=5)
    
    return total_score_thr, topN, dtw_thr


def main():
    st.title("股價比對")
    ticker = st.text_input("股票代號 (e.g. AAPL):", value="AAPL")
    seg_len = st.number_input("Segment Length(看幾根K棒)", 5, 50, 10)
    fut_len = st.number_input("Future Copy(複製幾根K棒)", 1, 20, 5)
    total_predict = st.slider("總預測天數", 5, 200, 50)

    # 取得產品類型（僅用於資料庫記錄）
    category = classify_ticker(ticker)

    # 讀取歷史統計資料
    stats_count = get_stats_count_from_supabase(ticker)
    stats = get_stat_ranges_from_supabase(ticker)
    if stats and stats_count >= 30:
        st.success(f"✅ 已累積 {stats_count} 筆歷史統計資料，啟用【歷史參數模式】")
        dtw_min, dtw_max = stats['dtw_min'], stats['dtw_max']
        score_min, score_max = stats['score_min'], stats['score_max']
        st.write(f"🌀 資料庫 DTW 統計: min={dtw_min:.2f}, max={dtw_max:.2f}")
        st.write(f"📏 資料庫 Score 統計: min={score_min:.2f}, max={score_max:.2f}")
        # 因為正規化後 DTW 數值通常偏小，這裡設定一個較小的建議範圍
        dtw_range = (max(dtw_min * 0.9, 0.1), min(dtw_max * 1.1, 20))
        score_range = (max(score_min * 0.9, 10), min(score_max * 1.1, 100))
        st.write(f"📈 建議 DTW 區間：{dtw_range}")
        st.write(f"📈 建議 Score 區間：{score_range}")
    else:
        st.warning(f"⚠️ 歷史資料僅 {stats_count} 筆，使用【預設參數模式】")
        dtw_range = (1, 10)
        score_range = (30, 60)
        st.write(f"📈 預設 DTW 區間：{dtw_range}")
        st.write(f"📈 預設 Score 區間：{score_range}")

    mode = st.selectbox("⚙️ 預設模式選擇", ["保守", "平衡", "寬鬆", "自訂"])
    total_score_thr, topN, dtw_thr = get_default_params_v2(mode, dtw_range, score_range)
    st.success(f"🎲 產生參數：total_score_thr={total_score_thr}, topN={topN}, dtw_thr={dtw_thr}")

    # 存入 session_state
    st.session_state['ticker'] = ticker
    st.session_state['category'] = category
    st.session_state['mode'] = mode
    st.session_state['topN'] = topN
    st.session_state['dtw_thr'] = dtw_thr
    st.session_state['total_score_thr'] = total_score_thr

    # 下載資料按鈕
    if st.button("下載資料"):
        df = download_data(ticker)
        if df.empty:
            st.error(f"無法取得 {ticker} 資料")
            return
        vix_df = download_vix(df.index.min(), df.index.max())
        st.session_state['df'] = df
        st.session_state['vix_df'] = vix_df
        st.session_state['min_date'] = df.index.min().date()
        st.session_state['max_date'] = df.index.max().date()
        st.success(f"✅ 資料期間：{st.session_state['min_date']} ～ {st.session_state['max_date']}")

    if 'df' in st.session_state:
        st.write(f"資料期間：{st.session_state['min_date']} ～ {st.session_state['max_date']}")
        user_start_date = st.date_input("顯示起始日 (必選):", value=st.session_state['min_date'],
                                        min_value=st.session_state['min_date'],
                                        max_value=st.session_state['max_date'])

        use_custom_line = st.checkbox("是否加入自定垂直線？")
        user_line_date = None
        if use_custom_line:
            user_line_date = st.date_input("垂直線日期（可選）:",
                                           value=st.session_state['min_date'],
                                           min_value=st.session_state['min_date'],
                                           max_value=st.session_state['max_date'])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🛑 停止運行"):
                st.session_state['stop'] = True
                st.warning("⚠️ 預測已停止，請重新操作")
        with col2:
            if st.button("🔄 重新輸入 / 清除"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        if 'stop' not in st.session_state:
            st.session_state['stop'] = False

        if st.button("執行預測"):
            st.write(f"✅ 參數確認 - total_score_thr: {total_score_thr}, TopN: {topN}")
            df, vix_df = st.session_state['df'], st.session_state['vix_df']
            pred_days, loop_count, logs = 0, 0, []
            start_time = time.time()
            st.session_state['total_score_thr'] = total_score_thr
            st.session_state['topN'] = topN
            st.session_state['seg_len'] = seg_len
            st.session_state['fut_len'] = fut_len
            st.session_state['dtw_thr'] = dtw_thr
            st.session_state['total_predict'] = total_predict
            st.session_state['stop'] = False

            last_real_date = df.index[-1]
            final_list = []

            while pred_days < st.session_state['total_predict'] and not st.session_state['stop']:
                loop_count += 1
                st.write(f"🔄 迴圈 {loop_count}，預測進度：{pred_days}/{total_predict} 天")

                final_list = find_best_match_advanced(
                    df,
                    vix_df,
                    seg_len=st.session_state['seg_len'],
                    fut_len=st.session_state['fut_len'],
                    total_score_thr=st.session_state['total_score_thr'],
                    dtw_thr=st.session_state['dtw_thr'],
                    topN=st.session_state['topN'],
                    ticker=ticker,
                    category=st.session_state['category'],
                    mode=mode
                )

                if not final_list or len(final_list) == 0:
                    logs.append(f"第 {loop_count} 次 => 找不到符合條件樣本，停止")
                    break

                topN_dynamic = random.randint(
                    max(3, int(st.session_state['topN'] * 0.8)),
                    int(st.session_state['topN'] * 1.2)
                )
                st.info(f"🎲 動態 TopN = {topN_dynamic}")

                final_list = [(idx, score + random.uniform(-5, 5)) for idx, score in final_list]
                final_list = sorted(final_list, key=lambda x: x[1], reverse=True)
                if len(final_list) > topN_dynamic:
                    final_list = final_list[:topN_dynamic]

                st.success(f"✅ 最後篩選樣本數：{len(final_list)} (動態TopN: {topN_dynamic})")
                scores = [s for (_, s) in final_list]
                st.write(f"分數範圍：min={min(scores):.2f}, max={max(scores):.2f}, avg={np.mean(scores):.2f}")

                best_i, best_score = weighted_random_choice(final_list)
                logs.append(f"第 {loop_count} 次選中 index={best_i}, 加權分數={best_score:.2f}")

                dates, new_ohlc = copy_future_bars_percent_mode(df, best_i, fut_len)
                if dates is None:
                    logs.append("⚠ 後續K棒不足，停止")
                    break

                new_rows = pd.DataFrame(new_ohlc, columns=["Open", "High", "Low", "Close"], index=dates)
                df = pd.concat([df, new_rows]).sort_index()
                pred_days += fut_len
                elapsed = time.time() - start_time
                st.info(f"⏳ 目前總耗時：{elapsed:.2f} 秒")

            total_time = time.time() - start_time
            st.success(f"✅ 預測完成！總耗時 {total_time:.2f} 秒")

            fdf = df.copy()
            fdf = fdf.loc[fdf.index >= pd.to_datetime(user_start_date)]

            fig = go.Figure()
            fig.update_layout(
                hovermode='x unified',
                dragmode='zoom',
                xaxis=dict(
                    tickformat="%Y-%m-%d",
                    showgrid=True,
                    rangeslider_visible=False
                ),
                yaxis=dict(
                    fixedrange=False
                ),
                font=dict(family="Microsoft JhengHei", size=14),
            )

            fig.add_trace(go.Candlestick(
                x=fdf.index,
                open=fdf["Open"], high=fdf["High"],
                low=fdf["Low"], close=fdf["Close"],
                increasing_line_color="red",
                decreasing_line_color="green",
                name="歷史＋預測"
            ))

            fig.add_shape(
                type="line",
                x0=last_real_date,
                x1=last_real_date,
                y0=0,
                y1=1,
                line=dict(color="blue", width=2, dash="dash"),
                xref='x',
                yref='paper'
            )

            if user_line_date:
                fig.add_vline(
                    x=pd.to_datetime(user_line_date),
                    line_width=2, line_dash="dot", line_color="yellow",
                    annotation_text="自定日期", annotation_position="bottom right"
                )

            fig.update_xaxes(
                type='date',
                tickformat="%Y-%m-%d",
                showgrid=True,
                rangeslider_visible=False
            )

            if st.button("🔄 一鍵重置圖表"):
                st.rerun()

            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,
                'displayModeBar': True,
                'displaylogo': False,
                'doubleClick': 'reset',
                'editable': True,
                'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'autoScale2d'],
                'modeBarButtonsToAdd': ['resetScale2d', 'drawline', 'drawopenpath', 'drawcircle', 'eraseshape'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'prediction_chart',
                    'height': 600,
                    'width': 1200,
                    'scale': 2
                }
            })

            st.success("✅ 預測完成! 可滑鼠拖曳、縮放，點『🔄 一鍵重置圖表』恢復原視角")

def run_app():
    main()

if __name__ == "__main__":
    run_app()
    