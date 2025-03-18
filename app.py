import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from datetime import timedelta

######################## 幫助函式 ########################

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

def filter_by_shape(curr_seg, df, seg_len, fut_len, shape_thr):
    """
    Shape 篩選：比對當前 segment 與歷史樣本的形狀距離
    """
    if len(df) < seg_len + fut_len:
        return []

    curr_feat = kbar_features(curr_seg)
    candidates = []
    all_dists = []
    for i in range(seg_len, len(df) - fut_len):
        sample_seg = df.iloc[i - seg_len: i]
        samp_feat = kbar_features(sample_seg)
        dist = shape_distance(curr_feat, samp_feat)
        all_dists.append(dist)
        if dist < shape_thr:
            candidates.append(i)

    if all_dists:
        st.write(f"📏 Shape 距離 min={min(all_dists):.2f}, max={max(all_dists):.2f}, avg={np.mean(all_dists):.2f}")

    return candidates

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

def filter_by_volatility(curr_seg, df, shape_candidates, seg_len, vol_thr):
    results, vol_list = [], []
    for i in shape_candidates:
        sample_seg = df.iloc[i - seg_len: i]
        vol_score = compute_volatility_score(curr_seg, sample_seg)
        vol_list.append(vol_score)
        if vol_score >= vol_thr:
            results.append((i, vol_score))
    if vol_list:
        st.write(f"📏 Volatility 範圍 min={min(vol_list):.2f}, max={max(vol_list):.2f}, avg={np.mean(vol_list):.2f}")
    return results

###################### Step3: VIX ######################

def compute_vix_score(curr_seg, sample_seg, vix_df):
    curr_end = curr_seg.index[-1]
    samp_end = sample_seg.index[-1]
    cv = get_vix_value(curr_end, vix_df)
    sv = get_vix_value(samp_end, vix_df)
    diff = abs(cv - sv) / max(cv, sv, 1e-9)
    vix_sc = max(0, (1 - diff)) * 100
    return vix_sc

def final_vix_filter(curr_seg, df, atr_list, seg_len, vix_df):
    final = []
    for (i, vol_score) in atr_list:
        sample_seg = df.iloc[i - seg_len: i]
        vs = compute_vix_score(curr_seg, sample_seg, vix_df)
        total = 0.5 * vol_score + 0.5 * vs  # 權重可調
        final.append((i, total))
    return final

def find_best_match_advanced(df, vix_df, seg_len, fut_len, shape_thr, vol_thr, topN):
    if len(df) < seg_len + fut_len:
        return None

    curr_seg = df.iloc[-seg_len:]
    shape_candidates = filter_by_shape(curr_seg, df, seg_len, fut_len, shape_thr)
    st.write(f"🔎 Shape 篩選後樣本數: {len(shape_candidates)}")
    if not shape_candidates:
        st.warning("❌ 沒有通過 Shape 篩選")
        return None

    atr_list = filter_by_volatility(curr_seg, df, shape_candidates, seg_len, vol_thr)
    st.write(f"✅ Volatility 篩選後樣本數: {len(atr_list)}")
    if not atr_list:
        st.warning("❌ 沒有通過 Volatility 篩選")
        return None

    final_list = final_vix_filter(curr_seg, df, atr_list, seg_len, vix_df)
    if not final_list:
        st.warning("❌ 沒有通過 VIX 篩選")
        return None

    if len(final_list) < topN:
        st.warning(f"⚠️ 最終樣本數 {len(final_list)}，小於 TopN={topN}")
    return final_list

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
    # 模式切換 + 自訂
    mode = st.selectbox("⚙️ 預設模式選擇", ["保守", "平衡", "寬鬆", "自訂"])

    if mode == "保守":
        shape_thr = 800
        vol_thr = 90.0
        topN = 5
    elif mode == "平衡":
        shape_thr = 1500
        vol_thr = 80.0
        topN = 20
    elif mode == "寬鬆":
        shape_thr = 2500
        vol_thr = 50.0
        topN = 40
    else:
        # 自訂模式才能動
        shape_thr = st.slider("Shape Threshold (圖形閾值)", 0, 3000, 1500)
        vol_thr = st.slider("波動門檻 (0~100)", min_value=20.0, max_value=100.0, value=80.0, step=0.1)
        topN = st.slider("TopN 隨機選擇", 1, 50, 20)

    st.write(f"✅ 當前模式：{mode}")
    st.write(f"✅ Shape Threshold：{shape_thr}")
    st.write(f"✅ Volatility Threshold：{vol_thr}")
    st.write(f"✅ TopN：{topN}")

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
            st.write(f"✅ 參數確認 - Shape: {shape_thr}, Volatility: {vol_thr}, TopN: {topN}")
            df, vix_df = st.session_state['df'], st.session_state['vix_df']
            pred_days, loop_count, logs = 0, 0, []
            start_time = time.time()

            # 強制同步參數到 session
            st.session_state['shape_thr'] = shape_thr
            st.session_state['vol_thr'] = vol_thr
            st.session_state['topN'] = topN
            st.session_state['seg_len'] = seg_len
            st.session_state['fut_len'] = fut_len
            st.write(f"🧠 目前 shape_thr = {shape_thr}")

            st.session_state['total_predict'] = total_predict  # ✅ 直接存

            pred_days, loop_count, logs = 0, 0, []
            start_time = time.time()
            st.session_state['stop'] = False

            last_real_date = df.index[-1]  # ✅ 預測前記錄歷史最後日期

            # ✅ 初始化 final_list
            final_list = []

            while pred_days < st.session_state['total_predict'] and not st.session_state['stop']:
                loop_count += 1
                st.write(f"🔄 迴圈 {loop_count}，預測進度：{pred_days}/{total_predict} 天")

                # ✅ 執行完整 Shape → Volatility → VIX 三層比對
                final_list = find_best_match_advanced(
                    df,
                    vix_df,
                    seg_len=st.session_state['seg_len'],
                    fut_len=st.session_state['fut_len'],
                    shape_thr=st.session_state['shape_thr'],
                    vol_thr=st.session_state['vol_thr'],
                    topN=st.session_state['topN']
                )

                if not final_list:
                    logs.append(f"第 {loop_count} 次 => 找不到符合條件樣本，停止")
                    break

                st.success(f"✅ 第 {loop_count} 次篩選樣本數：{len(final_list)}")
                scores = [s for (_, s) in final_list]
                st.write(f"分數範圍：min={min(scores):.2f}, max={max(scores):.2f}, avg={np.mean(scores):.2f}")

                # ✅ 加權抽樣選出最佳樣本
                best_i, best_score = weighted_random_choice(final_list)
                logs.append(f"第 {loop_count} 次選中 index={best_i}, 加權分數={best_score:.2f}")

                dates, new_ohlc = copy_future_bars_percent_mode(df, best_i, fut_len)
                if dates is None:
                    logs.append("⚠ 後續K棒不足，停止")
                    break
                new_rows = pd.DataFrame(new_ohlc, columns=["Open", "High", "Low", "Close"], index=dates)

                # ✅ 更新 df（產生未來 K 棒後）
                df = pd.concat([df, new_rows]).sort_index()

                # ✅ 強制檢查時間軸有沒有斷開
                df = df.sort_index()
                print(df.tail(10))  # 確認日期是否連續，未來 K 棒有接上

                pred_days += fut_len

                elapsed = time.time() - start_time
                st.info(f"⏳ 目前總耗時：{elapsed:.2f} 秒")

            total_time = time.time() - start_time
            st.success(f"✅ 預測完成！總耗時 {total_time:.2f} 秒")

            # ✅ 最後只印最後一次 final_list 前5筆，避免洗版
            if final_list:
                st.write(f"✅ 最後一次比對樣本共 {len(final_list)} 筆，前5筆如下：")
                for i, (idx, total_score) in enumerate(final_list[:5]):
                    st.write(f"樣本 index={idx}, 加權總分={total_score:.2f}")

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
                'modeBarButtonsToRemove': ['zoomIn2d', 'zoomOut2d'],  # 移除內建縮放鍵，避免干擾
            })
            st.success("✅ 預測完成! 可滑鼠拖曳、縮放")

def run_app():
    st.set_page_config(page_title="三步驟比對", layout="wide")
    main()

if __name__ == "__main__":
    run_app()
