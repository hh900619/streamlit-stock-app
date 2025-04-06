![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-red?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)

# 📈 Streamlit 股價型態預測系統

## 🔗 線上體驗（Live Demo）

👉 [點此開啟 Streamlit App](https://app-stock-appgit-8eqsptjnxprv6dfwjmtbkf.streamlit.app/)

> ※ 如遇閒置自動休眠，請稍等 10~15 秒自動喚醒

---

## 🧠 專案簡介

本系統是一套使用 Python 開發、基於 Streamlit 的股價預測應用，透過歷史 K 線形態與統計特徵的比對分析，預測未來可能的價格走勢。

主要應用於：

- 短中期趨勢預測
- 類型樣本篩選與統計
- 技術分析圖表呈現與互動式模擬

---

## 🔍 功能亮點

- 📈 **K 線相似比對**：使用 Cosine、DTW、Slope、Volatility 進行多維特徵比對  
- 🔎 **歷史樣本評分與排序**：自動篩選最具代表性的樣本資料  
- 🔁 **迭代預測模擬未來走勢**：每輪預測基於上一次延伸  
- 📊 **互動式 K 線圖**：支援放大、拖曳、標記、圖表下載等功能  
- 🧬 **結合 VIX 恐慌指數**：進一步過濾不一致的市場情境  
- 🗃️ **Supabase 數據儲存**：記錄每次預測的統計數據與參數區間

---
