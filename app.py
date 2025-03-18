import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import timedelta
from numpy.linalg import norm
from dtaidistance import dtw
import random

#cd "C:\Users\0619h\OneDrive\Desktop\streamlit-stock-app"
#git add .
#git commit -m "你這次改了什麼"
#git push origin main

######################## 幫助函式 ########################

# ======= 自動判斷類別 =======
def classify_ticker(ticker):
    """
    自動判斷商品類型，yfinance 正確代號版本
    """
    ticker = ticker.upper()
    if any(keyword in ticker for keyword in ['GOLD', 'SILVER', 'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F']):
        return "metal"           # 貴金屬
    elif any(keyword in ticker for keyword in ['CL=F', 'BRENT', 'NG=F', 'OIL', 'RB=F', 'HO=F']):
        return "energy"          # 能源
    elif any(keyword in ticker for keyword in ['CORN', 'SOY', 'WHEAT', 'ZC=F', 'ZS=F', 'ZW=F', 'KC=F']):
        return "agriculture"     # 農產品
    elif any(keyword in ticker for keyword in ['AAPL', 'MSFT', 'GOOG', 'META', 'TSLA', 'NVDA']):
        return "tech"            # 科技股
    elif any(keyword in ticker for keyword in ['SPY', 'QQQ', 'IWM', 'VOO', 'VTI']):
        return "etf"             # 指數ETF
    elif any(keyword in ticker for keyword in ['ES=F', 'NQ=F', 'YM=F', 'RTY=F']):
        return "index_futures"   # 股指期貨
    elif any(keyword in ticker for keyword in ['JPY=X', 'EUR=X', 'AUD=X', 'GBP=X', 'FX', 'USD']):
        return "forex"           # 外匯
    else:
        return "stock"           # 其他股票

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

def get_vix_value(dt, vix_df):
    if vix_df.empty:
        return 20.0
    if dt in vix_df.index:
        return float(vix_df.loc[dt, "Close"])
    valid_idx = vix_df.index[vix_df.index <= dt]
    if len(valid_idx) == 0:
        return float(vix_df["Close"].iloc[0])
    return float(vix_df.loc[valid_idx[-1], "Close"])

# 加權隨機選樣本函式 - 全程會用到
def weighted_random_choice(candidates):
    """
    加權隨機選擇樣本，分數越高權重越高
    :param candidates: List of (index, score) tuple
    :return: 選中的 (index, score)
    """
    scores = np.array([s for (_, s) in candidates])
    weights = scores / scores.sum()
    idx = np.random.choice(len(candidates), p=weights)
    return candidates[idx]

def generate_future_dates(history_dates, fut_len):
    # 直接從最後一天後面產生交易日
    start_date = history_dates[-1] + timedelta(days=1)
    future_dates = pd.bdate_range(start=start_date, periods=fut_len)
    return future_dates

###################### Step1: 圖形 ######################

def kbar_features(df):
    arr=[]
    prev_close=None
    for i,(idx,row) in enumerate(df.iterrows()):
        o,h,l,c= row["Open"], row["High"], row["Low"], row["Close"]
        if i==0:
            prev_close= o
        gap_pct= (o- prev_close)/max(abs(prev_close),1e-9)*100
        body_pct= (c- o)/max(abs(o),1e-9)*100
        daily_chg= (c- prev_close)/max(abs(prev_close),1e-9)*100
        rng= (h-l)/max(abs(prev_close),1e-9)*100
        color= 1 if c>= o else 0
        arr.append([gap_pct, body_pct, daily_chg, rng, color])
        prev_close= c
    return np.array(arr)

def shape_distance(A, B):
    if A.shape != B.shape:
        return 999999
    dist = np.sqrt(((A - B) ** 2).sum(axis=1)).mean()
    return dist * 100 

def dtw_distance(series_a, series_b):
    """
    計算兩個價格序列的 DTW 距離
    """
    a = series_a.values if isinstance(series_a, pd.Series) else np.array(series_a)
    b = series_b.values if isinstance(series_b, pd.Series) else np.array(series_b)
    dist = dtw.distance(a, b)
    return dist

def cosine_similarity(a, b):
    """
    計算兩段 K 線特徵的 Cosine 相似度
    """
    if len(a) != len(b):
        return -1  # 無法比對
    a_flat = a.flatten()
    b_flat = b.flatten()
    cos_sim = np.dot(a_flat, b_flat) / (norm(a_flat) * norm(b_flat) + 1e-9)
    return cos_sim  # 越接近1越相似

def filter_by_shape(curr_seg, df, seg_len, fut_len, shape_thr, dtw_thr):
    if len(df) < seg_len + fut_len:
        return [], shape_thr

    curr_feat = kbar_features(curr_seg)
    candidates = []
    all_dists, all_cosines = [], []

    for i in range(seg_len, len(df) - fut_len):
        sample_seg = df.iloc[i - seg_len: i]
        samp_feat = kbar_features(sample_seg)

        dist = shape_distance(curr_feat, samp_feat)
        dtw_dist = dtw_distance(curr_seg["Close"], sample_seg["Close"])
        cos_sim = cosine_similarity(curr_feat, samp_feat)

        if dist < shape_thr and dtw_dist < dtw_thr and cos_sim > 0.2:
            candidates.append((i, cos_sim))

        all_dists.append(dist)
        all_cosines.append(cos_sim)

    if all_dists:
        min_shape, max_shape = min(all_dists), max(all_dists)
        avg_shape = np.mean(all_dists)
        st.write(f"📏 Shape 距離 min={min_shape:.2f}, max={max_shape:.2f}, avg={avg_shape:.2f}")

        # ✅ 自動調整 shape_thr 並回傳
        if shape_thr < min_shape:
            shape_thr = min_shape + 50
            st.warning(f"⚠️ Shape Threshold 過低，自動調整為 {shape_thr:.2f}")
        elif shape_thr > max_shape:
            shape_thr = max_shape - 50
            st.warning(f"⚠️ Shape Threshold 過高，自動調整為 {shape_thr:.2f}")

    return candidates, shape_thr

###################### Step2: 平均絕對百分比 ######################

def compute_volatility_score(curr_seg, sample_seg):
    curr_return = (curr_seg["Close"].iloc[-1] - curr_seg["Close"].iloc[0]) / max(abs(curr_seg["Close"].iloc[0]), 1e-9)
    sample_return = (sample_seg["Close"].iloc[-1] - sample_seg["Close"].iloc[0]) / max(abs(sample_seg["Close"].iloc[0]), 1e-9)
    dir_diff = abs(curr_return - sample_return)

    curr_vol = (curr_seg["Close"].max() - curr_seg["Close"].min()) / max(abs(curr_seg["Close"].iloc[0]), 1e-9)
    sample_vol = (sample_seg["Close"].max() - sample_seg["Close"].min()) / max(abs(sample_seg["Close"].iloc[0]), 1e-9)
    vol_diff = abs(curr_vol - sample_vol)

    score = max(0, (1 - (0.5 * dir_diff + 0.5 * vol_diff))) * 100
    return score

def compute_slope_angle_similarity(curr_seg, sample_seg):
    """
    計算當前區段與樣本區段的斜率＆角度相似度
    """
    # 計算兩段的斜率 (收盤價)
    x = np.arange(len(curr_seg))
    y_curr = curr_seg["Close"].values
    y_samp = sample_seg["Close"].values

    slope_curr, _ = np.polyfit(x, y_curr, 1)
    slope_samp, _ = np.polyfit(x, y_samp, 1)

    # 轉換成角度（弧度轉角度）
    angle_curr = np.degrees(np.arctan(slope_curr))
    angle_samp = np.degrees(np.arctan(slope_samp))

    # 相差越小越相似
    angle_diff = abs(angle_curr - angle_samp)

    # 分數轉換：差越小分數越高
    score = max(0, 100 - angle_diff)  # 你可自行調整 100 這個滿分上限

    return score

def compute_total_scores(curr_seg, df, shape_candidates, seg_len, total_score_thr, dtw_thr):
    results, total_scores = [], []
    dtw_list = []  # ⬅️ 新增收集 DTW 統計

    for i, cos_sim in shape_candidates:
        sample_seg = df.iloc[i - seg_len: i]
        vol_score = compute_volatility_score(curr_seg, sample_seg)
        slope_score = compute_slope_angle_similarity(curr_seg, sample_seg)
        dtw_dist = dtw_distance(curr_seg["Close"], sample_seg["Close"])
        dtw_list.append(dtw_dist)
        dtw_norm = min(dtw_dist / dtw_thr, 1.0)

        # ✅ 綜合加權計算（斜率、波動、DTW、Cosine）
        total_score = 0.3 * vol_score + 0.2 * slope_score + 0.25 * (1 - dtw_norm) * 100 + 0.25 * cos_sim * 100

        total_scores.append(total_score)
        if total_score >= total_score_thr:
            results.append((i, total_score))

    if total_scores:
        min_score, max_score = min(total_scores), max(total_scores)
        avg_score = np.mean(total_scores)
        st.write(f"📏 綜合總分 min={min_score:.2f}, max={max_score:.2f}, avg={avg_score:.2f}")

        # 自動調整 total_score_thr
        if total_score_thr < min_score:
            adjust_thr = min_score + 5
            st.warning(f"⚠️ 目前 total_score_thr={total_score_thr}，過低，自動拉高為 {adjust_thr:.2f}")
            total_score_thr = adjust_thr
        elif total_score_thr > max_score:
            adjust_thr = max_score - 5
            st.warning(f"⚠️ 目前 total_score_thr={total_score_thr}，過高，自動降低為 {adjust_thr:.2f}")
            total_score_thr = adjust_thr

    if dtw_list:
        min_dtw, max_dtw = min(dtw_list), max(dtw_list)
        avg_dtw = np.mean(dtw_list)
        st.write(f"🌀 DTW 距離 min={min_dtw:.2f}, max={max_dtw:.2f}, avg={avg_dtw:.2f}")
        if dtw_thr < min_dtw:
            dtw_thr = min_dtw + 5
            st.warning(f"⚠️ DTW Threshold 過低，自動調整為 {dtw_thr:.2f}")
        elif dtw_thr > max_dtw:
            dtw_thr = max_dtw + 5
            st.warning(f"⚠️ DTW Threshold 過高，自動調整為 {dtw_thr:.2f}")

    return results, total_score_thr

###################### Step3: VIX ######################

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

    # ✅ 動態決定 VIX 權重
    if category == "metal":
        vix_weight = 0.2  # 金屬類 VIX 影響小
    elif category == "forex":
        vix_weight = 0.5  # 外匯類 VIX 影響大
    else:
        vix_weight = 0.3  # 其他類別預設 0.3

    for (i, vol_score) in atr_list:
        sample_seg = df.iloc[i - seg_len: i]
        vix_score = compute_vix_score(curr_seg, sample_seg, vix_df)

        # ✅ 動態加權
        total_score = (1 - vix_weight) * vol_score + vix_weight * vix_score
        final.append((i, total_score))

    if not final:
        return []

    # ✅ 重要！強制排序取 TopN
    final_sorted = sorted(final, key=lambda x: x[1], reverse=True)
    st.write(f"🟢 VIX 加權後樣本數：{len(final_sorted)}, 分數範圍 min={min([s for (_, s) in final_sorted]):.2f}, max={max([s for (_, s) in final_sorted]):.2f}")

    # ✅ 強制只取 topN（如果樣本不夠，就取全部）
    return final_sorted[:topN]

# ======= 根據商品類型決定比對方法 =======
def select_matching_method(product_type):
    """
    依據商品類型決定比對方法：
    - metal、energy、agriculture 偏向技術面形態主導
    - forex、index_futures 加強波動比對
    - tech、etf、stock 綜合考量
    """
    if product_type in ["metal", "energy", "agriculture"]:
        return "shape_first"  # 圖形優先
    elif product_type in ["forex", "index_futures"]:
        return "volatility_first"  # 波動優先
    else:
        return "balanced"  # 綜合考量

def find_best_match_advanced(df, vix_df, seg_len, fut_len, shape_thr, total_score_thr, topN, dtw_thr):
    if len(df) < seg_len + fut_len:
        return None

    curr_seg = df.iloc[-seg_len:]
    shape_candidates, shape_thr = filter_by_shape(curr_seg, df, seg_len, fut_len, shape_thr, dtw_thr)
    st.write(f"🔎 Shape 篩選後樣本數: {len(shape_candidates)}")
    if not shape_candidates:
        st.warning("❌ 沒有通過 Shape 篩選")
        return None

    total_score_list, total_score_thr = compute_total_scores(curr_seg, df, shape_candidates, seg_len, total_score_thr, dtw_thr)
    st.write(f"✅ 綜合總分篩選後樣本數: {len(total_score_list)}")
    if not total_score_list:
        st.warning("❌ 沒有通過總分篩選")
        return None

    final_list = final_vix_filter(curr_seg, df, total_score_list, seg_len, vix_df, st.session_state['category'], st.session_state['topN'])
    if not final_list:
        st.warning("❌ 沒有通過 VIX 篩選")
        return None

    if len(final_list) < topN:
        st.warning(f"⚠️ 最終樣本數 {len(final_list)}，小於 TopN={topN}")

    match_mode = st.session_state.get('match_mode', 'balanced')
    st.write(f"🧠 自動比對邏輯：{match_mode}")

    final_list_sorted = sorted(final_list, key=lambda x: x[1], reverse=True)
    return final_list_sorted

###################### 複製未來K棒 (百分比) ######################

def copy_future_bars_percent_mode(df, best_i, fut_len):
    samp = df.iloc[best_i: best_i + fut_len].copy()
    if len(samp) < fut_len:
        return None, None
    last_close = df["Close"].iloc[-1]
    hist_prev_close = df["Close"].iloc[best_i - 1]
    new_ohlc = []

    # ✅ 產生未來交易日，確保時間對齊不跳回去
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

################### Streamlit App #####################

def main():
    st.title("股價比對")
    ticker = st.text_input("股票代號 (e.g. AAPL):", value="AAPL")
    seg_len = st.number_input("Segment Length(看幾根K棒)", 5, 50, 10)
    fut_len = st.number_input("Future Copy(複製幾根K棒)", 1, 20, 5)
    total_predict = st.slider("總預測天數", 5, 200, 50)

    # 根據類別給不同預設參數
    def get_default_params(category, mode):
        """
        回傳 (shape_thr, total_score_thr, topN, dtw_thr)，皆有隨機範圍
        - shape_thr：形態距離過濾
        - total_score_thr：綜合總分過濾
        - topN：進入隨機抽樣的樣本數
        - dtw_thr：DTW 閾值，直接影響 DTW 正規化
        """
        if category == "metal":
            if mode == "保守": return random.randint(300, 500), random.randint(50, 60), random.randint(5, 8), random.randint(20, 30)
            if mode == "平衡": return random.randint(400, 700), random.randint(45, 55), random.randint(10, 15), random.randint(30, 40)
            if mode == "寬鬆": return random.randint(600, 900), random.randint(40, 50), random.randint(15, 20), random.randint(40, 50)

        elif category == "energy":
            if mode == "保守": return random.randint(400, 600), random.randint(50, 60), random.randint(5, 8), random.randint(25, 35)
            if mode == "平衡": return random.randint(600, 900), random.randint(45, 55), random.randint(10, 15), random.randint(35, 45)
            if mode == "寬鬆": return random.randint(800, 1100), random.randint(40, 50), random.randint(15, 20), random.randint(45, 55)

        elif category == "agriculture":
            if mode == "保守": return random.randint(300, 500), random.randint(50, 60), random.randint(5, 8), random.randint(20, 30)
            if mode == "平衡": return random.randint(400, 700), random.randint(45, 55), random.randint(10, 15), random.randint(30, 40)
            if mode == "寬鬆": return random.randint(600, 900), random.randint(40, 50), random.randint(15, 20), random.randint(40, 50)

        elif category == "tech":  # ✅ AAPL、NVDA
            if mode == "保守": return random.randint(500, 800), random.randint(45, 55), random.randint(5, 8), random.randint(10, 20)
            if mode == "平衡": return random.randint(700, 1000), random.randint(40, 50), random.randint(10, 15), random.randint(20, 30)
            if mode == "寬鬆": return random.randint(900, 1300), random.randint(35, 45), random.randint(15, 20), random.randint(30, 40)

        elif category == "etf":
            if mode == "保守": return random.randint(600, 900), random.randint(50, 60), random.randint(5, 8), random.randint(25, 35)
            if mode == "平衡": return random.randint(800, 1100), random.randint(45, 55), random.randint(10, 15), random.randint(35, 45)
            if mode == "寬鬆": return random.randint(1000, 1400), random.randint(40, 50), random.randint(15, 20), random.randint(45, 55)

        elif category == "index_futures":
            if mode == "保守": return random.randint(500, 800), random.randint(50, 60), random.randint(5, 8), random.randint(20, 30)
            if mode == "平衡": return random.randint(700, 1000), random.randint(45, 55), random.randint(10, 15), random.randint(30, 40)
            if mode == "寬鬆": return random.randint(900, 1300), random.randint(40, 50), random.randint(15, 20), random.randint(40, 50)

        elif category == "forex":
            if mode == "保守": return random.randint(300, 500), random.randint(50, 60), random.randint(5, 8), random.randint(15, 25)
            if mode == "平衡": return random.randint(400, 700), random.randint(45, 55), random.randint(10, 15), random.randint(25, 35)
            if mode == "寬鬆": return random.randint(600, 900), random.randint(40, 50), random.randint(15, 20), random.randint(35, 45)

        else:  # stock 其他一般股票
            if mode == "保守": return random.randint(600, 900), random.randint(50, 60), random.randint(5, 8), random.randint(20, 30)
            if mode == "平衡": return random.randint(800, 1200), random.randint(45, 55), random.randint(10, 15), random.randint(30, 40)
            if mode == "寬鬆": return random.randint(1100, 1500), random.randint(40, 50), random.randint(15, 20), random.randint(40, 50)

    # 模式切換 + 自訂
    category = classify_ticker(ticker)
    match_mode = select_matching_method(category)
    st.write(f"📊 系統判斷類型：{category}")
    st.write(f"🧠 比對策略：{match_mode}")
    st.session_state['match_mode'] = match_mode
    st.session_state['category'] = category  # ✅ 保存類別，讓後面 VIX 動態加權能用

    mode = st.selectbox("⚙️ 預設模式選擇", ["保守", "平衡", "寬鬆", "自訂"])

    if mode != "自訂":
        # ✅ 加入隨機參數邏輯
        shape_thr, total_score_thr, topN, dtw_thr = get_default_params(category, mode)
        st.success(f"🎲 隨機參數已生成 ➔ Shape_thr: {shape_thr}, Total_score_thr: {total_score_thr}, TopN: {topN}, DTW_thr: {dtw_thr}")
    else:
        shape_thr = st.slider("Shape Threshold (圖形閾值)", 0, 3000, 1500)
        total_score_thr = st.slider("總分門檻 (0~100)", min_value=20.0, max_value=100.0, value=80.0, step=0.1)
        topN = st.slider("TopN 隨機選擇", 1, 50, 20)
        dtw_thr = st.slider("DTW 閾值 (建議30~50)", min_value=10, max_value=100, value=40)

    st.write(f"✅ 當前模式：{mode}")
    st.write(f"✅ Shape Threshold：{shape_thr}")
    st.write(f"✅ Total Score Threshold：{total_score_thr}")
    st.write(f"✅ TopN：{topN}")
    st.write(f"✅ DTW Threshold：{dtw_thr}")

    # ✅ 顯示比對策略提示區塊
    if match_mode == "shape_first":
        st.info("🟢 **比對策略：技術面形態優先（Shape First）**\n適用：貴金屬、能源、農產品")
    elif match_mode == "volatility_first":
        st.info("🔵 **比對策略：波動特性優先（Volatility First）**\n適用：外匯、股指期貨")
    else:
        st.info("🟡 **比對策略：綜合考量（Balanced）**\n適用：科技股、ETF、一般股票")

    # ✅ 下載資料
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
        st.session_state['shape_thr'] = shape_thr
        st.success(f"✅ 資料期間：{st.session_state['min_date']} ～ {st.session_state['max_date']}")

    if 'df' in st.session_state:
        st.write(f"資料期間：{st.session_state['min_date']} ～ {st.session_state['max_date']}")
        user_start_date = st.date_input("顯示起始日 (必選):", value=st.session_state['min_date'],
                                        min_value=st.session_state['min_date'],
                                        max_value=st.session_state['max_date'])

        # ✅ 自訂垂直線
        use_custom_line = st.checkbox("是否加入自定垂直線？")
        user_line_date = None
        if use_custom_line:
            user_line_date = st.date_input("垂直線日期（可選）:",
                                           value=st.session_state['min_date'],
                                           min_value=st.session_state['min_date'],
                                           max_value=st.session_state['max_date'])

        # ✅ 停止與重置
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

        # ✅ 預測按鈕
        if st.button("執行預測"):
            st.write(f"✅ 參數確認 - Shape: {shape_thr}, total_score_thr: {total_score_thr}, TopN: {topN}")
            df, vix_df = st.session_state['df'], st.session_state['vix_df']
            pred_days, loop_count, logs = 0, 0, []
            start_time = time.time()

            # 強制同步參數到 session
            st.session_state['shape_thr'] = shape_thr
            st.session_state['total_score_thr'] = total_score_thr
            st.session_state['topN'] = topN
            st.session_state['seg_len'] = seg_len
            st.session_state['fut_len'] = fut_len
            st.session_state['dtw_thr'] = dtw_thr
            st.write(f"🧠 目前 shape_thr = {shape_thr}")

            st.session_state['total_predict'] = total_predict  # ✅ 直接存

            start_time = time.time()
            st.session_state['stop'] = False

            last_real_date = df.index[-1]  # ✅ 預測前記錄歷史最後日期

            # ✅ 初始化 final_list
            final_list = []

            while pred_days < st.session_state['total_predict'] and not st.session_state['stop']:
                loop_count += 1
                st.write(f"🔄 迴圈 {loop_count}，預測進度：{pred_days}/{total_predict} 天")

                final_list = find_best_match_advanced(
                    df,
                    vix_df,
                    seg_len=st.session_state['seg_len'],
                    fut_len=st.session_state['fut_len'],
                    shape_thr=st.session_state['shape_thr'],
                    total_score_thr=st.session_state['total_score_thr'],
                    dtw_thr=st.session_state['dtw_thr'],
                    topN=st.session_state['topN']
                )

                if not final_list or len(final_list) == 0:
                    logs.append(f"第 {loop_count} 次 => 找不到符合條件樣本，停止")
                    break

                # ✅ 動態隨機 topN（增加隨機感）
                topN_dynamic = random.randint(max(3, int(st.session_state['topN'] * 0.8)),
                                            int(st.session_state['topN'] * 1.2))
                st.info(f"🎲 動態 TopN = {topN_dynamic}")

                # ✅ 加入隨機微擾，讓分數更有波動感
                final_list = [(idx, score + random.uniform(-5, 5)) for idx, score in final_list]
                final_list = sorted(final_list, key=lambda x: x[1], reverse=True)

                # ✅ 強制截斷到 topN_dynamic
                if len(final_list) > topN_dynamic:
                    final_list = final_list[:topN_dynamic]

                st.success(f"✅ 最後篩選樣本數：{len(final_list)} (動態TopN: {topN_dynamic})")
                scores = [s for (_, s) in final_list]
                st.write(f"分數範圍：min={min(scores):.2f}, max={max(scores):.2f}, avg={np.mean(scores):.2f}")

                # ✅ 加權抽樣選出最佳樣本（範圍拉大）
                best_i, best_score = weighted_random_choice(final_list)
                logs.append(f"第 {loop_count} 次選中 index={best_i}, 加權分數={best_score:.2f}")

                dates, new_ohlc = copy_future_bars_percent_mode(df, best_i, fut_len)
                if dates is None:
                    logs.append("⚠ 後續K棒不足，停止")
                    break
                new_rows = pd.DataFrame(new_ohlc, columns=["Open", "High", "Low", "Close"], index=dates)

                # ✅ 更新 df（產生未來 K 棒後）
                df = pd.concat([df, new_rows]).sort_index()
                df = df.sort_index()  # 強制檢查時間軸有沒有斷開

                pred_days += fut_len
                elapsed = time.time() - start_time
                st.info(f"⏳ 目前總耗時：{elapsed:.2f} 秒")

            total_time = time.time() - start_time
            st.success(f"✅ 預測完成！總耗時 {total_time:.2f} 秒")

            # ✅ 畫圖
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

            # ✅ 解決你的報錯，Timestamp 轉字串
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

            st.plotly_chart(fig, use_container_width=True, config={
                'scrollZoom': True,
                'displayModeBar': True,  # 顯示右上工具列，方便手動放大
                'doubleClick': 'reset',  # 雙擊還原視角
                'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d', 'autoScale2d'],  # 移除內建縮放鍵，避免干擾
            })
            st.success("✅ 預測完成! 可滑鼠拖曳、縮放")

def run_app():
    st.set_page_config(page_title="三步驟比對", layout="wide")
    main()

if __name__ == "__main__":
    run_app()
