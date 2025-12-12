"""
過去1年間バックテストページ
========================
過去データを使ってシグナル売買戦略をシミュレーション
（ウォークフォワードテスト: 各日は未来のデータを知らない状態で判定）
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from analysis.backtest_analyzer import analyze_backtest_results, print_analysis

st.set_page_config(
    page_title="📅 過去1年バックテスト",
    page_icon="📅",
    layout="wide"
)

st.title("📅 過去1年間バックテスト")
st.markdown("過去データでシグナル売買戦略をシミュレーション（ウォークフォワードテスト）")

db = DatabaseManager()


# ==================== シグナル計算関数（特定日時点） ====================

def calculate_signal_at_date(df: pd.DataFrame, target_idx: int) -> dict:
    """
    特定の日付時点でのシグナルを計算（改善版）
    target_idx: その日のインデックス（その日までのデータのみ使用）
    """
    min_required = 50  # 最低50日必要
    if target_idx < min_required:
        return None
    
    # その日までのデータのみ使用（未来のデータは見ない）
    df_slice = df.iloc[:target_idx + 1].copy()
    
    if len(df_slice) < min_required:
        return None
    
    # 最新価格（その日の終値）
    current_price = float(df_slice['Close'].iloc[-1])
    prev_price = float(df_slice['Close'].iloc[-2])
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    # ========== RSI (14日) ==========
    delta = df_slice['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = float(rsi.iloc[-1])
    rsi_prev = float(rsi.iloc[-2])
    
    # RSIシグナル（基本）
    if rsi_value < 30:
        rsi_signal = 1.0
    elif rsi_value > 70:
        rsi_signal = -1.0
    else:
        rsi_signal = (50 - rsi_value) / 50
    
    # ========== RSIダイバージェンス検出 ==========
    # 価格が上昇しているのにRSIが下降 → 弱気ダイバージェンス（売りサイン）
    # 価格が下降しているのにRSIが上昇 → 強気ダイバージェンス（買いサイン）
    price_5d_change = (current_price - float(df_slice['Close'].iloc[-6])) / float(df_slice['Close'].iloc[-6]) * 100
    rsi_5d_change = rsi_value - float(rsi.iloc[-6])
    
    divergence_signal = 0.0
    if price_5d_change > 2 and rsi_5d_change < -5:  # 弱気ダイバージェンス
        divergence_signal = -0.5
    elif price_5d_change < -2 and rsi_5d_change > 5:  # 強気ダイバージェンス
        divergence_signal = 0.5
    
    # ========== 移動平均 ==========
    sma5 = df_slice['Close'].rolling(window=5).mean()
    sma20 = df_slice['Close'].rolling(window=20).mean()
    sma50 = df_slice['Close'].rolling(window=50).mean()
    
    # 200日MAはデータが十分な場合のみ計算
    has_sma200 = len(df_slice) >= 200
    if has_sma200:
        sma200 = df_slice['Close'].rolling(window=200).mean()
        sma200_val = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else sma50.iloc[-1]
    else:
        sma200_val = float(sma50.iloc[-1])  # 代わりに50日MAを使用
    
    sma5_val = float(sma5.iloc[-1])
    sma20_val = float(sma20.iloc[-1])
    sma50_val = float(sma50.iloc[-1])
    
    # MAシグナル
    ma_signal = 0.0
    if current_price > sma5_val:
        ma_signal += 0.2
    if sma5_val > sma20_val:
        ma_signal += 0.3
    if sma20_val > sma50_val:
        ma_signal += 0.25
    if has_sma200 and sma50_val > sma200_val:  # ゴールデンクロス状態
        ma_signal += 0.25
    elif not has_sma200:
        ma_signal += 0.125  # 200日MAがない場合は中立
    ma_signal = (ma_signal - 0.5) * 2
    
    # ========== トレンドフィルター（50日MA > 200日MA） ==========
    is_uptrend = sma50_val > sma200_val if has_sma200 else True  # データ不足時はTrueとする
    
    # ========== MACD ==========
    ema12 = df_slice['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_slice['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    macd_val = float(macd_line.iloc[-1])
    macd_signal_val = float(signal_line.iloc[-1])
    macd_hist_val = float(macd_hist.iloc[-1])
    macd_hist_prev = float(macd_hist.iloc[-2])
    macd_hist_prev2 = float(macd_hist.iloc[-3])
    
    # MACDヒストグラムの傾き（モメンタム）
    macd_momentum = macd_hist_val - macd_hist_prev
    
    # v8.5: MACDクロスオーバー検出
    macd_crossover = 0
    macd_prev = float(macd_line.iloc[-2])
    macd_signal_prev = float(signal_line.iloc[-2])
    # ゴールデンクロス: MACDがシグナルを上抜け
    if macd_prev <= macd_signal_prev and macd_val > macd_signal_val:
        macd_crossover = 1
    # デッドクロス: MACDがシグナルを下抜け
    elif macd_prev >= macd_signal_prev and macd_val < macd_signal_val:
        macd_crossover = -1
    
    # v8.5: ヒストグラムの方向転換検出
    hist_reversal = 0
    if macd_hist_prev2 < macd_hist_prev and macd_hist_prev > macd_hist_val:
        hist_reversal = -1  # ピーク形成（売りシグナル）
    elif macd_hist_prev2 > macd_hist_prev and macd_hist_prev < macd_hist_val:
        hist_reversal = 1   # 底形成（買いシグナル）
    
    if macd_val > macd_signal_val and macd_hist_val > 0:
        macd_signal = 1.0
    elif macd_val < macd_signal_val and macd_hist_val < 0:
        macd_signal = -1.0
    else:
        macd_signal = macd_hist_val / (abs(macd_hist_val) + 0.01) * 0.5
    
    # MACDモメンタム補正
    if macd_momentum > 0:
        macd_signal = min(1.0, macd_signal + 0.2)
    elif macd_momentum < 0:
        macd_signal = max(-1.0, macd_signal - 0.2)
    
    # v8.5: クロスオーバーボーナス
    if macd_crossover == 1:
        macd_signal = min(1.0, macd_signal + 0.3)
    elif macd_crossover == -1:
        macd_signal = max(-1.0, macd_signal - 0.3)
    
    # ========== ボリンジャーバンド ==========
    bb_std = df_slice['Close'].rolling(window=20).std().iloc[-1]
    bb_upper = sma20_val + 2 * bb_std
    bb_lower = sma20_val - 2 * bb_std
    
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    bb_signal = (0.5 - bb_position) * 2
    
    # ========== v8.1: 出来高プロファイル強化 ==========
    vol_sma = df_slice['Volume'].rolling(window=20).mean()
    vol_ratio = float(df_slice['Volume'].iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1.0
    
    # 過去5日の出来高トレンドも確認
    vol_5d_avg = float(df_slice['Volume'].iloc[-5:].mean())
    vol_prev_5d_avg = float(df_slice['Volume'].iloc[-10:-5].mean())
    vol_trend = (vol_5d_avg - vol_prev_5d_avg) / vol_prev_5d_avg if vol_prev_5d_avg > 0 else 0
    
    # 出来高急増 + 価格上昇 = 強い買いシグナル
    if vol_ratio > 2.0 and price_change > 1:  # 出来高2倍以上&上昇
        vol_signal = 1.5
    elif vol_ratio > 1.5 and price_change > 0:
        vol_signal = 1.0
    elif vol_ratio > 2.0 and price_change < -1:  # 出来高急増&下落=パニック売り
        vol_signal = -1.5
    elif vol_ratio > 1.5 and price_change < 0:
        vol_signal = -1.0
    else:
        vol_signal = 0.0
    
    # 出来高トレンドボーナス（機関投資家の参入を検出）
    vol_trend_bonus = min(0.3, max(-0.3, vol_trend * 0.5)) if vol_trend > 0.2 else 0
    
    # ========== ROC（Rate of Change）モメンタム ==========
    roc_10 = (current_price - float(df_slice['Close'].iloc[-11])) / float(df_slice['Close'].iloc[-11]) * 100
    if roc_10 > 5:
        roc_signal = 0.5
    elif roc_10 < -5:
        roc_signal = -0.5
    else:
        roc_signal = roc_10 / 10
    
    # ========== ATR（Average True Range）==========
    high = df_slice['High']
    low = df_slice['Low']
    close = df_slice['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = float(tr.rolling(window=14).mean().iloc[-1])
    atr_pct = (atr / current_price) * 100  # ATRを価格比率で
    
    # ========== v8.2: ATRブレイクアウト検出 ==========
    # ATRの縮小後の拡大を検出（ボラティリティブレイクアウト）
    atr_5d = float(tr.rolling(window=5).mean().iloc[-1])
    atr_20d = float(tr.rolling(window=20).mean().iloc[-1])
    atr_ratio = atr_5d / atr_20d if atr_20d > 0 else 1.0
    
    # 直近の価格レンジを確認
    recent_high = float(df_slice['High'].iloc[-5:].max())
    recent_low = float(df_slice['Low'].iloc[-5:].min())
    range_breakout = 0
    
    # 上方ブレイクアウト: 直近高値を更新 + ATR拡大
    if current_price > recent_high * 0.998 and atr_ratio > 1.2:
        range_breakout = 1
    # 下方ブレイクアウト: 直近安値を更新 + ATR拡大
    elif current_price < recent_low * 1.002 and atr_ratio > 1.2:
        range_breakout = -1
    
    # ========== v8.4: マルチタイムフレームモメンタム ==========
    # 複数期間のモメンタムを組み合わせ
    momentum_5d = (current_price - float(df_slice['Close'].iloc[-6])) / float(df_slice['Close'].iloc[-6]) * 100
    momentum_10d = (current_price - float(df_slice['Close'].iloc[-11])) / float(df_slice['Close'].iloc[-11]) * 100
    momentum_20d = (current_price - float(df_slice['Close'].iloc[-21])) / float(df_slice['Close'].iloc[-21]) * 100
    
    # モメンタム一貫性: 全期間でプラスなら強いトレンド
    momentum_consistency = 0
    if momentum_5d > 0 and momentum_10d > 0 and momentum_20d > 0:
        momentum_consistency = 1  # 強い上昇トレンド
    elif momentum_5d < 0 and momentum_10d < 0 and momentum_20d < 0:
        momentum_consistency = -1  # 強い下降トレンド
    
    # ========== v6: ボラティリティ調整済みリターン ==========
    volatility_20d = float(df_slice['Close'].pct_change().rolling(20).std().iloc[-1]) * 100
    risk_adjusted_momentum = momentum_20d / (volatility_20d + 0.1) if volatility_20d > 0 else momentum_20d
    
    # ========== v7新規: ケルトナーチャネル（Keltner Channel） ==========
    # ATRベースの動的チャネル - ボラティリティ適応型
    keltner_mid = float(df_slice['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
    keltner_upper = keltner_mid + 2.0 * atr
    keltner_lower = keltner_mid - 2.0 * atr
    keltner_position = (current_price - keltner_lower) / (keltner_upper - keltner_lower) if keltner_upper != keltner_lower else 0.5
    
    # ========== v7新規: ボラティリティスクイーズ検出（TTM Squeeze） ==========
    # BBがケルトナー内に収まっている = スクイーズ状態 = ブレイクアウト待機
    squeeze_on = (bb_lower > keltner_lower) and (bb_upper < keltner_upper)
    # スクイーズ解除後のモメンタム方向を確認
    squeeze_momentum = macd_hist_val  # MACDヒストグラムで方向判定
    
    # ========== v7新規: レジーム検出（市場状態分類） ==========
    # 簡易版: ボラティリティ + トレンドで4状態に分類
    vol_percentile_20 = volatility_20d
    vol_median = float(df_slice['Close'].pct_change().rolling(60).std().iloc[-1]) * 100 if len(df_slice) >= 60 else vol_percentile_20
    is_high_vol = vol_percentile_20 > vol_median * 1.2
    
    # レジーム: 0=低ボラ下降, 1=低ボラ上昇, 2=高ボラ下降, 3=高ボラ上昇
    if is_uptrend:
        regime = 3 if is_high_vol else 1  # 上昇トレンド
    else:
        regime = 2 if is_high_vol else 0  # 下降トレンド
    
    # レジーム別の推奨アクション
    regime_names = ['低ボラ下降', '低ボラ上昇', '高ボラ下降', '高ボラ上昇']
    regime_buy_mult = [0.0, 1.2, 0.3, 0.8]  # 各レジームでの買い倍率
    regime_stop_mult = [1.0, 0.8, 1.5, 1.2]  # 各レジームでの損切り倍率

    # ========== v9.2: 早期警戒シグナル（損失前の退出判定） ==========
    early_warning_score = 0  # 0=問題なし, 1以上=警戒レベル
    early_warning_reasons = []
    
    # 1. モメンタム悪化検知: 短期が長期を下回り始めた
    if momentum_5d < 0 and momentum_10d > 0:  # 短期で反転開始
        early_warning_score += 1
        early_warning_reasons.append("短期モメンタム悪化")
    if momentum_5d < momentum_10d < momentum_20d and momentum_5d < 0:  # 加速度的悪化
        early_warning_score += 1
        early_warning_reasons.append("モメンタム加速悪化")
    
    # 2. RSI反転検知: RSIが高値圏から下降開始
    rsi_3d_ago = float(rsi.iloc[-4]) if len(rsi) >= 4 else rsi_prev
    if rsi_value < rsi_prev < rsi_3d_ago and rsi_3d_ago > 60:  # 高値圏から連続下降
        early_warning_score += 1
        early_warning_reasons.append("RSI高値反転")
    if rsi_value < 50 and rsi_prev > 50:  # RSI50割れ（中立ライン割れ）
        early_warning_score += 1
        early_warning_reasons.append("RSI50割れ")
    
    # 3. MACD悪化検知: ヒストグラムがピークから下降
    if hist_reversal == -1:  # MACDヒストグラムがピーク形成
        early_warning_score += 1
        early_warning_reasons.append("MACDピーク反転")
    if macd_crossover == -1:  # MACDデッドクロス
        early_warning_score += 2  # 重要シグナル
        early_warning_reasons.append("MACDデッドクロス")
    
    # 4. 出来高パニック売り検知: 出来高急増＋価格下落
    if vol_ratio > 1.8 and price_change < -1:  # 出来高1.8倍以上で1%超下落
        early_warning_score += 2
        early_warning_reasons.append("出来高急増下落")
    
    # 5. SMA5がSMA20を下抜け（短期トレンド崩壊）
    sma5_prev = float(sma5.iloc[-2])
    sma20_prev = float(sma20.iloc[-2])
    if sma5_prev >= sma20_prev and sma5_val < sma20_val:
        early_warning_score += 1
        early_warning_reasons.append("SMA5/20デッドクロス")
    
    # 6. ボリンジャーバンド上限からの反落
    if bb_position < 0.7 and float((current_price - float(df_slice['Close'].iloc[-2])) / float(df_slice['Close'].iloc[-2]) * 100) < -1:
        # 前日BB上部にいて、今日1%以上下落
        prev_bb_position = (float(df_slice['Close'].iloc[-2]) - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        if prev_bb_position > 0.8:
            early_warning_score += 1
            early_warning_reasons.append("BB上限反落")

    # ========== 総合スコア計算（改善版） ==========
    weights = {
        'rsi': 0.15,
        'divergence': 0.10,  # 新規追加
        'ma': 0.20,
        'macd': 0.25,
        'bb': 0.10,
        'volume': 0.10,
        'roc': 0.10  # 新規追加
    }
    
    total_score = (
        rsi_signal * weights['rsi'] +
        divergence_signal * weights['divergence'] +
        ma_signal * weights['ma'] +
        macd_signal * weights['macd'] +
        bb_signal * weights['bb'] +
        vol_signal * weights['volume'] +
        roc_signal * weights['roc']
    )
    
    # v8.1: 出来高トレンドボーナスを追加
    total_score += vol_trend_bonus
    
    # 下降トレンドでの買いシグナルをペナルティ
    if not is_uptrend and total_score > 0:
        total_score *= 0.5
    
    # ========== v7新規: スクイーズブレイクアウトボーナス ==========
    # スクイーズ状態から解除されたタイミングで上方向ならボーナス
    squeeze_bonus = 0.0
    if squeeze_on and squeeze_momentum > 0:
        squeeze_bonus = 0.15  # スクイーズ中で上向きモメンタム
    
    # ========== v7新規: レジーム調整後スコア ==========
    regime_adjusted_score = total_score * regime_buy_mult[regime]
    
    # ========== v10.1: ADX（Average Directional Index）==========
    # トレンドの強さを測定（25以上で強いトレンド）
    plus_dm = df_slice['High'].diff()
    minus_dm = -df_slice['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr_14 = tr.rolling(window=14).sum()
    plus_di = 100 * plus_dm.rolling(window=14).sum() / tr_14
    minus_di = 100 * minus_dm.rolling(window=14).sum() / tr_14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.001)
    adx = dx.rolling(window=14).mean()
    
    adx_value = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 20.0
    plus_di_value = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 25.0
    minus_di_value = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 25.0
    
    # ADXシグナル: 強いトレンド + トレンド方向
    adx_signal = 0.0
    if adx_value >= 25:  # 強いトレンド
        if plus_di_value > minus_di_value:
            adx_signal = min(1.0, (adx_value - 25) / 25)  # 上昇トレンド
        else:
            adx_signal = max(-1.0, -(adx_value - 25) / 25)  # 下降トレンド
    
    # ========== v10.1: OBV（On Balance Volume）==========
    # 出来高の累積で資金流入を追跡
    obv = pd.Series(0, index=df_slice.index, dtype=float)
    obv.iloc[0] = df_slice['Volume'].iloc[0]
    for i in range(1, len(df_slice)):
        if df_slice['Close'].iloc[i] > df_slice['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + df_slice['Volume'].iloc[i]
        elif df_slice['Close'].iloc[i] < df_slice['Close'].iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - df_slice['Volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    obv_sma = obv.rolling(window=20).mean()
    obv_value = float(obv.iloc[-1])
    obv_sma_value = float(obv_sma.iloc[-1]) if not pd.isna(obv_sma.iloc[-1]) else obv_value
    
    # OBVシグナル: OBVがSMAを上回っていれば買い圧力
    obv_signal = 0.0
    if obv_sma_value != 0:
        obv_ratio = (obv_value - obv_sma_value) / abs(obv_sma_value)
        obv_signal = max(-1.0, min(1.0, obv_ratio * 5))  # スケーリング
    
    # v10.1: ADX/OBVをスコアに追加
    total_score += adx_signal * 0.1 + obv_signal * 0.1

    return {
        'price': current_price,
        'change': price_change,
        'rsi': rsi_value,
        'total_score': total_score,
        'is_uptrend': is_uptrend,
        'atr_pct': atr_pct,
        'bb_position': bb_position,
        'high_price': float(df_slice['High'].iloc[-20:].max()),
        'momentum_20d': momentum_20d,
        'risk_adjusted_momentum': risk_adjusted_momentum,
        'volatility': volatility_20d,
        # v7新規
        'regime': regime,
        'regime_name': regime_names[regime],
        'regime_buy_mult': regime_buy_mult[regime],
        'regime_stop_mult': regime_stop_mult[regime],
        'keltner_position': keltner_position,
        'squeeze_on': squeeze_on,
        'squeeze_bonus': squeeze_bonus,
        'regime_adjusted_score': regime_adjusted_score,
        # v9.2: 早期警戒シグナル
        'early_warning_score': early_warning_score,
        'early_warning_reasons': early_warning_reasons,
        'rsi_prev': rsi_prev,
        'macd_crossover': macd_crossover,
        'hist_reversal': hist_reversal,
        # v10.1: ADX/OBV
        'adx': adx_value,
        'plus_di': plus_di_value,
        'minus_di': minus_di_value,
        'adx_signal': adx_signal,
        'obv_signal': obv_signal
    }


def get_historical_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    過去データを取得（データベースキャッシュ優先）
    """
    # 期間を日数に変換
    period_days = {
        "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "max": 3650
    }
    days = period_days.get(period, 730)
    
    # まずデータベースから取得を試みる
    df = db.get_cached_prices(ticker, days=days)
    
    if df is not None and len(df) >= 50:
        return df
    
    # データベースにない場合はyfinanceから取得
    import ssl
    import urllib3
    import requests
    import time
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
    except:
        pass
    
    # レート制限対策: リクエスト間に遅延
    time.sleep(0.5)
    
    try:
        import yfinance as yf
        
        # SSL検証を無効化したセッションを使用
        session = requests.Session()
        session.verify = False
        
        stock = yf.Ticker(ticker, session=session)
        df = stock.history(period=period)
        
        if df is not None and len(df) >= 50:
            # データベースにキャッシュ
            db.cache_prices(ticker, df)
            return df
            
        return None
    except Exception as e:
        if '429' in str(e) or 'Too Many Requests' in str(e):
            st.warning(f"{ticker}: レート制限。DBキャッシュを確認中...")
        return None


def run_backtest(tickers: list, initial_cash: float = 1000000, 
                 start_days_ago: int = 252, progress_callback=None,
                 market_ticker: str = "SPY") -> dict:
    """
    過去1年間のバックテストを実行（改善版v6 - 大胆な発想）
    
    v7.0 改善点（最新研究 + 独自手法）:
    
    【学術研究ベース】
    - レジームスイッチング: 市場を4状態(低ボラ上昇/下降、高ボラ上昇/下降)に分類
    - ケリー基準: 勝率と期待リターンから最適ポジションサイズを計算
    - ボラティリティスクイーズ: TTM Squeezeでブレイクアウト検出
    - アダプティブ損切り: レジームに応じた動的損切り
    
    【独自手法】
    - セクター分散: 同一セクター集中を防ぐ
    - 相関フィルター: 高相関銘柄の重複保有制限
    - スクイーズブレイクアウト: 収縮後の拡大でエントリー
    
    【v6継続機能】
    - 勝者追跡、敗者ブラックリスト、モメンタムランキング
    """
    
    # 各銘柄の過去データを取得
    all_data = {}
    failed_tickers = []
    required_days = start_days_ago + 200
    
    st.info(f"必要データ日数: {required_days}日")
    
    for ticker in tickers:
        df = get_historical_data(ticker, "3y")
        if df is not None:
            if len(df) > required_days:
                all_data[ticker] = df
            elif len(df) > start_days_ago:
                all_data[ticker] = df
                st.warning(f"{ticker}: データが{len(df)}日のみ（一部指標が計算不可）")
            else:
                failed_tickers.append(f"{ticker}({len(df)}日)")
        else:
            failed_tickers.append(f"{ticker}(取得失敗)")
    
    if failed_tickers:
        st.warning(f"データ不足銘柄: {', '.join(failed_tickers[:10])}")
    
    # 市場データ（SPY）を取得
    market_data = get_historical_data(market_ticker, "3y")
    if market_data is None:
        st.warning("市場データ(SPY)が取得できません。市場フィルターを無効化します。")
    
    # ========== v10.0: VIXデータ取得（レジーム検知用） ==========
    # VIXは特殊なので、yfinanceから直接取得
    import yfinance as yf
    import requests
    try:
        session = requests.Session()
        session.verify = False
        vix_ticker = yf.Ticker("^VIX", session=session)
        vix_data = vix_ticker.history(period="3y")
        if vix_data is not None and len(vix_data) >= 50:
            st.info(f"✅ VIXデータ取得成功（{len(vix_data)}日分） - 高度なレジーム検知を有効化")
        else:
            vix_data = None
            st.warning("VIXデータが取得できません。VIXフィルターを無効化します。")
    except Exception as e:
        vix_data = None
        st.warning(f"VIXデータ取得エラー: {e}。VIXフィルターを無効化します。")
    
    if not all_data:
        return {'error': f"データ不足の銘柄: {', '.join(failed_tickers)}", 'failed_tickers': failed_tickers}
    
    # ========== v7新規: セクター情報（簡易版） ==========
    # 銘柄コードから簡易セクター推定（日本株は4桁コード）
    def get_sector(ticker):
        if '.T' in ticker:  # 日本株
            code = ticker.replace('.T', '')
            if code.startswith(('65', '66', '67', '68', '69')):
                return 'tech'  # 電気機器、精密機器
            elif code.startswith(('72', '73', '74', '75')):
                return 'auto'  # 輸送用機器
            elif code.startswith(('80', '81', '82', '83')):
                return 'finance'  # 金融
            elif code.startswith(('35', '36', '37', '38')):
                return 'materials'  # 化学、鉄鋼
            else:
                return 'other'
        else:  # 米国株
            us_tech = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'TSLA', 'AMZN', 'CRM', 'ORCL', 'ADBE', 'INTC', 'QCOM', 'AVGO']
            us_finance = ['JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'V', 'MA', 'AXP', 'BRK-B']
            us_health = ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'BMY', 'TMO', 'ABT']
            us_energy = ['XOM', 'CVX', 'COP', 'SLB', 'OXY']
            
            base = ticker.split('.')[0].upper()
            if base in us_tech:
                return 'tech'
            elif base in us_finance:
                return 'finance'
            elif base in us_health:
                return 'health'
            elif base in us_energy:
                return 'energy'
            else:
                return 'other'
    
    ticker_sectors = {t: get_sector(t) for t in all_data.keys()}
    
    # 共通の日付範囲を決定
    first_ticker = list(all_data.keys())[0]
    date_index = all_data[first_ticker].index[-start_days_ago:]
    
    # バックテスト状態
    cash = initial_cash
    portfolio = {}  # {ticker: {'shares', 'avg_cost', 'high_since_buy', 'buy_date', 'days_held'}}
    history = []
    trades = []
    
    # シグナル履歴（2日連続確認用）
    prev_day_signals = {}
    
    # ========== v6新規: 拡張トラッキング ==========
    ticker_performance = {t: {
        'wins': 0, 
        'losses': 0, 
        'total_pnl': 0.0, 
        'trade_count': 0,
        'big_wins': 0,  # +20%以上の大勝ち回数
        'consecutive_losses': 0,  # 連敗数
        'last_loss_day': 0,  # 最後に負けた日（ブラックリスト用）
        'best_pnl': 0.0  # 最高利益率
    } for t in all_data.keys()}
    
    # ========== v7新規: 市場レジーム履歴 ==========
    market_regime_history = []  # レジーム変化の追跡
    
    # ========== v8.3: セクターパフォーマンス追跡 ==========
    sector_performance = {
        'tech': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
        'finance': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
        'auto': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
        'health': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
        'energy': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
        'materials': {'wins': 0, 'losses': 0, 'total_pnl': 0.0},
        'other': {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
    }
    
    # ========== v10.0: ドローダウン追跡（DD連動ポジション縮小用） ==========
    peak_equity = initial_cash  # 最高資産額
    current_dd_multiplier = 1.0  # DD連動のポジション倍率
    
    # ========== v10.0: VIXレジーム履歴 ==========
    vix_regime_history = []  # VIXレジーム変化の追跡
    
    # ========== v10.1: ボラティリティターゲティング ==========
    TARGET_ANNUAL_VOL = 0.15  # 目標年率ボラティリティ 15%
    portfolio_returns_history = []  # 日次リターン履歴（20日分保持）
    vol_scaling_factor = 1.0  # ボラティリティスケーリング係数
    
    # ========== v10.1: ADX/OBVインジケーター用キャッシュ ==========
    indicator_cache = {}  # {ticker: {'adx': [], 'obv': [], 'obv_sma': []}}
    
    total_days = len(date_index)
    
    for day_num, current_date in enumerate(date_index):
        if progress_callback:
            progress_callback(day_num / total_days)
        
        # ========== 市場トレンド判定 ==========
        market_is_bullish = True
        if market_data is not None:
            market_mask = market_data.index <= current_date
            if market_mask.sum() >= 200:
                market_slice = market_data[market_mask]
                market_sma200 = market_slice['Close'].rolling(200).mean().iloc[-1]
                market_price = market_slice['Close'].iloc[-1]
                market_is_bullish = market_price > market_sma200
        
        # ========== v10.0: VIXベースのレジーム検知 ==========
        vix_level = 20.0  # デフォルト（データなしの場合）
        vix_regime = "NORMAL"  # NORMAL, CAUTION, FEAR, PANIC
        vix_position_multiplier = 1.0
        vix_momentum = 0.0
        
        if vix_data is not None:
            vix_mask = vix_data.index <= current_date
            if vix_mask.sum() >= 10:
                vix_slice = vix_data[vix_mask]
                vix_level = float(vix_slice['Close'].iloc[-1])
                
                # VIXモメンタム（5日変化率）
                if len(vix_slice) >= 6:
                    vix_5d_ago = float(vix_slice['Close'].iloc[-6])
                    vix_momentum = (vix_level - vix_5d_ago) / vix_5d_ago * 100
                
                # VIXレジーム判定
                if vix_level >= 35:
                    vix_regime = "PANIC"
                    vix_position_multiplier = 0.0  # 完全停止
                elif vix_level >= 30:
                    vix_regime = "FEAR"
                    vix_position_multiplier = 0.25  # 75%縮小
                elif vix_level >= 25:
                    vix_regime = "CAUTION"
                    vix_position_multiplier = 0.5  # 50%縮小
                elif vix_momentum > 15:  # VIXが5日で15%以上急上昇
                    vix_regime = "CAUTION"
                    vix_position_multiplier = 0.7  # 30%縮小
                else:
                    vix_regime = "NORMAL"
                    vix_position_multiplier = 1.0
        
        # その日のシグナルを計算（各銘柄）
        daily_signals = {}
        daily_prices = {}
        
        for ticker, df in all_data.items():
            mask = df.index <= current_date
            valid_idx = mask.sum() - 1
            
            if valid_idx < 50:  # 最低50日必要
                continue
            
            signal = calculate_signal_at_date(df, valid_idx)
            if signal:
                daily_signals[ticker] = signal
                daily_prices[ticker] = signal['price']
        
        # 現在の総資産を計算
        stock_value = sum(
            portfolio[t]['shares'] * daily_prices.get(t, portfolio[t]['avg_cost'])
            for t in portfolio if t in daily_prices
        )
        total_value = cash + stock_value
        
        # ========== v10.0: ドローダウン連動ポジション縮小 ==========
        # 最高資産を更新
        if total_value > peak_equity:
            peak_equity = total_value
        
        # 現在のドローダウンを計算
        current_drawdown = (peak_equity - total_value) / peak_equity if peak_equity > 0 else 0
        
        # DD連動のポジション倍率を決定
        if current_drawdown >= 0.05:  # 5%以上のDD
            current_dd_multiplier = 0.25  # 75%縮小
        elif current_drawdown >= 0.03:  # 3%以上のDD
            current_dd_multiplier = 0.5   # 50%縮小
        else:
            current_dd_multiplier = 1.0   # 通常
        
        # ========== v10.0: 総合ポジション倍率（VIX × DD） ==========
        combined_position_multiplier = vix_position_multiplier * current_dd_multiplier
        
        # ========== v10.1: ボラティリティターゲティング ==========
        # ポートフォリオの日次リターンを計算して履歴に追加
        if len(history) >= 1:
            prev_total = history[-1]['total_value']
            daily_return = (total_value - prev_total) / prev_total if prev_total > 0 else 0
            portfolio_returns_history.append(daily_return)
            
            # 直近20日のボラティリティを計算
            if len(portfolio_returns_history) > 20:
                portfolio_returns_history = portfolio_returns_history[-20:]
            
            if len(portfolio_returns_history) >= 10:
                import numpy as np
                realized_vol = np.std(portfolio_returns_history) * np.sqrt(252)  # 年率換算
                
                # ボラティリティスケーリング
                if realized_vol > 0:
                    vol_scaling_factor = min(2.0, max(0.3, TARGET_ANNUAL_VOL / realized_vol))
                else:
                    vol_scaling_factor = 1.0
        
        # v10.1: ボラティリティスケーリングを総合倍率に適用
        combined_position_multiplier *= vol_scaling_factor
        
        # ========== 保有日数更新 ==========
        for ticker in portfolio:
            portfolio[ticker]['days_held'] = portfolio[ticker].get('days_held', 0) + 1
        
        # ========== v10.0: VIX PANIC/FEARレジームでの強制ポジション縮小 ==========
        if vix_regime in ["PANIC", "FEAR"] and len(portfolio) > 0:
            for ticker in list(portfolio.keys()):
                if ticker not in daily_prices:
                    continue
                pos = portfolio[ticker]
                price = daily_prices[ticker]
                pnl_rate = ((price - pos['avg_cost']) / pos['avg_cost']) * 100
                
                sell_shares = 0
                sell_reason = None
                
                if vix_regime == "PANIC":
                    # PANICモード: 全ポジション即時売却
                    sell_shares = pos['shares']
                    sell_reason = f"VIX PANIC売却 (VIX:{vix_level:.1f}, PnL:{pnl_rate:.1f}%)"
                elif vix_regime == "FEAR":
                    # FEARモード: 損失ポジションは即売却、利益ポジションは50%売却
                    if pnl_rate <= 0:
                        sell_shares = pos['shares']
                        sell_reason = f"VIX FEAR損切り (VIX:{vix_level:.1f}, PnL:{pnl_rate:.1f}%)"
                    elif pnl_rate >= 3:
                        sell_shares = pos['shares'] * 0.5
                        sell_reason = f"VIX FEAR利確50% (VIX:{vix_level:.1f}, PnL:{pnl_rate:.1f}%)"
                
                if sell_shares > 0:
                    amount = sell_shares * price
                    cash += amount
                    
                    trades.append({
                        'date': current_date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': sell_shares,
                        'price': price,
                        'amount': amount,
                        'reason': sell_reason,
                        'pnl_rate': pnl_rate,
                        'vix_level': vix_level,
                        'vix_regime': vix_regime
                    })
                    
                    # パフォーマンス記録
                    if pnl_rate > 0:
                        ticker_performance[ticker]['wins'] += 1
                    else:
                        ticker_performance[ticker]['losses'] += 1
                    ticker_performance[ticker]['total_pnl'] += pnl_rate
                    ticker_performance[ticker]['trade_count'] += 1
                    
                    portfolio[ticker]['shares'] -= sell_shares
                    if portfolio[ticker]['shares'] < 0.01:
                        del portfolio[ticker]
        
        # ========== v9.0c: 弱気相場でのポジション縮小強化 ==========
        # 市場がSMA200を下回ったら、ポジションを積極的に縮小
        if not market_is_bullish and len(portfolio) > 0:
            for ticker in list(portfolio.keys()):
                if ticker not in daily_prices:
                    continue
                pos = portfolio[ticker]
                price = daily_prices[ticker]
                pnl_rate = ((price - pos['avg_cost']) / pos['avg_cost']) * 100
                
                sell_shares = 0
                sell_reason = None
                
                # 利益が出ているポジションは全売却
                if pnl_rate >= 5:  # +5%以上の利益
                    sell_shares = pos['shares']
                    sell_reason = f"弱気相場利確 ({pnl_rate:.1f}%)"
                elif pnl_rate >= 1:  # +1%〜5%の利益
                    sell_shares = pos['shares'] * 0.7  # 70%売却
                    sell_reason = f"弱気相場縮小 ({pnl_rate:.1f}%)"
                elif pnl_rate <= -3:  # -3%以下の損失は早めに損切り
                    sell_shares = pos['shares']
                    sell_reason = f"弱気相場損切り ({pnl_rate:.1f}%)"
                
                if sell_shares > 0:
                    amount = sell_shares * price
                    cash += amount
                    
                    trades.append({
                        'date': current_date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'shares': sell_shares,
                        'price': price,
                        'amount': amount,
                        'reason': sell_reason,
                        'pnl_rate': pnl_rate
                    })
                    
                    # パフォーマンス記録
                    if pnl_rate > 0:
                        ticker_performance[ticker]['wins'] += 1
                    else:
                        ticker_performance[ticker]['losses'] += 1
                    ticker_performance[ticker]['total_pnl'] += pnl_rate
                    ticker_performance[ticker]['trade_count'] += 1
                    
                    portfolio[ticker]['shares'] -= sell_shares
                    if portfolio[ticker]['shares'] < 0.01:
                        del portfolio[ticker]
        
        # ========== 売り処理（先に実行） ==========
        for ticker in list(portfolio.keys()):
            if ticker not in daily_signals or ticker not in daily_prices:
                continue
            
            pos = portfolio[ticker]
            price = daily_prices[ticker]
            score = daily_signals[ticker]['total_score']
            atr_pct = daily_signals[ticker].get('atr_pct', 2.0)
            pnl_rate = ((price - pos['avg_cost']) / pos['avg_cost']) * 100
            days_held = pos.get('days_held', 0)
            
            # 高値更新
            if price > pos.get('high_since_buy', pos['avg_cost']):
                portfolio[ticker]['high_since_buy'] = price
            
            high_since_buy = pos.get('high_since_buy', pos['avg_cost'])
            drop_from_high = ((high_since_buy - price) / high_since_buy) * 100 if high_since_buy > 0 else 0
            
            sell_reason = None
            sell_ratio = 0
            
            # ========== 改善v4: ポジションサイズの最低閾値 ==========
            # 総資産の0.5%未満のポジションは全売却して整理
            current_total = cash + sum(
                portfolio[t]['shares'] * daily_prices.get(t, portfolio[t]['avg_cost'])
                for t in portfolio
            )
            position_value = pos['shares'] * price
            position_ratio = position_value / current_total if current_total > 0 else 0
            
            if position_ratio < 0.005:  # 0.5%未満は清算
                sell_reason = f"ポジション整理 ({position_ratio*100:.2f}%)"
                sell_ratio = 1.0
            
            # ========== v9.0: 損切り厳格化 ==========
            # 最大損失を-10%に制限し、損切りを早める
            if sell_ratio == 0:
                regime_stop_mult = daily_signals[ticker].get('regime_stop_mult', 1.0)
                base_stop = min(8, atr_pct * 2.5)  # 基本損切りライン厳格化: 12→8, 3.5→2.5
                
                # レジーム調整: 高ボラ時は損切り幅を少し広げる（ただし上限あり）
                adjusted_stop = min(10, base_stop * regime_stop_mult)  # 最大-10%
                
                # 保有期間ボーナス: 10日ごとに1%緩和（最大2%）- 緩和を抑制
                holding_bonus = min(2, days_held // 10)
                
                # 過去好成績銘柄でも損切り緩和は控えめに
                perf = ticker_performance[ticker]
                if perf['trade_count'] >= 3 and perf['wins'] / perf['trade_count'] >= 0.7:
                    holding_bonus += 1  # 勝率70%以上なら+1%のみ
                
                dynamic_stop = min(10, adjusted_stop + holding_bonus)  # 絶対に-10%を超えない
                
                if days_held >= 3:  # 3日経過後から損切り
                    if pnl_rate <= -dynamic_stop:
                        regime_name = daily_signals[ticker].get('regime_name', '不明')
                        sell_reason = f"損切り ({pnl_rate:.1f}%, 閾値-{dynamic_stop:.1f}%, {regime_name})"
                        sell_ratio = 1.0
            
            # ========== v9.0: 絶対ハードストップ -10% ==========
            # どんな状況でも-10%で強制損切り
            if sell_ratio == 0 and pnl_rate <= -10:
                sell_reason = f"ハードストップ ({pnl_rate:.1f}%)"
                sell_ratio = 1.0
            
            # ========== v7.0: 段階的トレーリングストップ ==========
            # 利益が大きいほどトレーリングを緩く（利益を伸ばす）
            if sell_ratio == 0 and pnl_rate > 0:
                if pnl_rate >= 50:  # 50%以上の利益は高値から-15%で売却
                    trailing_threshold = 15
                elif pnl_rate >= 30:  # 30%以上は-12%
                    trailing_threshold = 12
                elif pnl_rate >= 15:  # 15%以上は-10%
                    trailing_threshold = 10
                else:  # それ以外は-7%
                    trailing_threshold = 7
                
                if drop_from_high >= trailing_threshold:
                    sell_reason = f"トレーリングストップ (高値から-{drop_from_high:.1f}%, 閾値{trailing_threshold}%)"
                    sell_ratio = 1.0
            
            # ========== v7.0: 利確ロジック ==========
            # 利確は段階的に、かつ全売却（半分売りの繰り返し問題を解消）
            if sell_ratio == 0:
                if pnl_rate >= 50:  # +50%以上で3/4売却
                    sell_reason = f"大幅利確 ({pnl_rate:.1f}%)"
                    sell_ratio = 0.75
                elif pnl_rate >= 30:  # +30%以上で半分売却
                    # ただし前回の利確から5日以上経過している場合のみ
                    last_partial_sell = pos.get('last_partial_sell_day', 0)
                    if days_held - last_partial_sell >= 5:
                        sell_reason = f"利確 ({pnl_rate:.1f}%)"
                        sell_ratio = 0.5
                        portfolio[ticker]['last_partial_sell_day'] = days_held
            
            # 強い売りシグナル
            if sell_ratio == 0 and score <= -0.5:
                sell_reason = f"強い売り (スコア {score:.2f})"
                sell_ratio = 1.0
            
            # ========== v7.5b: シグナル売り ==========
            # 利益時は半分売り、損失-3%以上のみ全売り
            if sell_ratio == 0 and score <= -0.2:
                if pnl_rate > 0:
                    sell_reason = f"売り (スコア {score:.2f})"
                    sell_ratio = 0.5
                elif pnl_rate <= -3:
                    sell_reason = f"損切り売り (スコア {score:.2f})"
                    sell_ratio = 1.0
            
            if sell_ratio > 0:
                shares_to_sell = pos['shares'] * sell_ratio
                amount = shares_to_sell * price
                cash += amount
                
                # ========== v6新規: 拡張実績トラッキング ==========
                if pnl_rate > 0:
                    ticker_performance[ticker]['wins'] += 1
                    ticker_performance[ticker]['consecutive_losses'] = 0  # 連敗リセット
                    if pnl_rate >= 20:  # 大勝ち記録
                        ticker_performance[ticker]['big_wins'] += 1
                    if pnl_rate > ticker_performance[ticker]['best_pnl']:
                        ticker_performance[ticker]['best_pnl'] = pnl_rate
                else:
                    ticker_performance[ticker]['losses'] += 1
                    ticker_performance[ticker]['consecutive_losses'] += 1
                    ticker_performance[ticker]['last_loss_day'] = day_num
                    
                ticker_performance[ticker]['total_pnl'] += pnl_rate
                ticker_performance[ticker]['trade_count'] += 1
                
                # v8.3: セクターパフォーマンス更新
                ticker_sector = ticker_sectors.get(ticker, 'other')
                if pnl_rate > 0:
                    sector_performance[ticker_sector]['wins'] += 1
                else:
                    sector_performance[ticker_sector]['losses'] += 1
                sector_performance[ticker_sector]['total_pnl'] += pnl_rate
                
                trades.append({
                    'date': current_date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'amount': amount,
                    'reason': sell_reason,
                    'pnl_rate': pnl_rate
                })
                
                if sell_ratio >= 1.0:
                    del portfolio[ticker]
                else:
                    portfolio[ticker]['shares'] -= shares_to_sell
        
        # ========== 買い処理 ==========
        # ========== v10.0: VIXレジーム + 市場トレンドによる買い制限 ==========
        if vix_regime == "PANIC":
            buy_budget_ratio = 0.0  # PANIC時は完全停止
            min_buy_score = 1.0
        elif vix_regime == "FEAR":
            buy_budget_ratio = 0.0  # FEAR時も買い停止
            min_buy_score = 1.0
        elif vix_regime == "CAUTION":
            if market_is_bullish:
                buy_budget_ratio = 0.5 * combined_position_multiplier  # 警戒モード: 50%に制限
                min_buy_score = 0.3  # より厳しいスコア要求
            else:
                buy_budget_ratio = 0.0
                min_buy_score = 1.0
        else:  # NORMAL
            if market_is_bullish:
                buy_budget_ratio = 1.0 * combined_position_multiplier
                min_buy_score = 0.2
            else:
                buy_budget_ratio = 0.0  # 弱気時は買わない
                min_buy_score = 1.0
        
        # ========== 改善v5: 既存ポジションへの買い増し ==========
        # 好成績銘柄が押し目にきたら買い増し（VIXがCAUTION以下で市場強気時のみ）
        if market_is_bullish and buy_budget_ratio > 0 and vix_regime in ["NORMAL", "CAUTION"]:
            for ticker in list(portfolio.keys()):
                if ticker not in daily_signals:
                    continue
                
                pos = portfolio[ticker]
                signal = daily_signals[ticker]
                price = daily_prices.get(ticker, pos['avg_cost'])
                score = signal['total_score']
                pnl_rate = ((price - pos['avg_cost']) / pos['avg_cost']) * 100
                bb_position = signal.get('bb_position', 0.5)
                
                # ========== v7.5b: 買い増し条件（ピラミッディング） ==========
                # 買い増し条件:
                # 1. 現在利益が出ている（+5%以上）
                # 2. 過去実績が良い（勝率50%以上）
                # 3. 押し目（BB中央より下）
                # 4. スコアがプラス
                perf = ticker_performance[ticker]
                
                if (pnl_rate >= 5 and
                    perf['trade_count'] >= 2 and 
                    perf['wins'] / perf['trade_count'] >= 0.5 and
                    bb_position < 0.5 and
                    score >= 0.15):
                    
                    current_total = cash + sum(
                        portfolio[t]['shares'] * daily_prices.get(t, portfolio[t]['avg_cost'])
                        for t in portfolio
                    )
                    position_value = pos['shares'] * price
                    position_ratio = position_value / current_total
                    
                    # 現在のポジションが15%未満なら買い増し可能（大勝ち銘柄は18%まで）
                    max_add_ratio = 0.18 if ticker_performance[ticker]['big_wins'] >= 1 else 0.15
                    if position_ratio < max_add_ratio and cash > current_total * 0.10:
                        # 買い増し額: 総資産の7%
                        add_amount = min(current_total * 0.07, cash - current_total * 0.08)
                        if add_amount > 15000:  # 最低1.5万円以上
                            add_shares = add_amount / price
                            cash -= add_amount
                            
                            # 平均コストを更新
                            total_shares = pos['shares'] + add_shares
                            new_avg_cost = (pos['shares'] * pos['avg_cost'] + add_shares * price) / total_shares
                            portfolio[ticker]['shares'] = total_shares
                            portfolio[ticker]['avg_cost'] = new_avg_cost
                            
                            trades.append({
                                'date': current_date,
                                'ticker': ticker,
                                'action': 'BUY',
                                'shares': add_shares,
                                'price': price,
                                'amount': add_amount,
                                'reason': f"買い増し (利益{pnl_rate:.1f}%, 勝率{perf['wins']/perf['trade_count']*100:.0f}%)"
                            })
        
        # 買い候補を選定
        buy_candidates = []
        for ticker, signal in daily_signals.items():
            if ticker in portfolio:  # 既存ポジションは買い増しで対応済み
                continue
            
            score = signal['total_score']
            price = signal['price']
            change = signal['change']
            is_uptrend = signal.get('is_uptrend', True)
            bb_position = signal.get('bb_position', 0.5)
            momentum_20d = signal.get('momentum_20d', 0)
            risk_adj_momentum = signal.get('risk_adjusted_momentum', 0)
            
            # 条件1: スコアが閾値以上（市場状況で変化）
            if score < min_buy_score:
                continue
            
            # 条件2: 銘柄自体が上昇トレンド（50日MA > 200日MA）
            if not is_uptrend:
                continue
            
            # 条件3: 2日連続で買いシグナル
            prev_signal = prev_day_signals.get(ticker, {})
            prev_score = prev_signal.get('total_score', 0) if prev_signal else 0
            if prev_score < 0.15:  # 前日は緩めの条件
                continue
            
            # 条件4: 高値追い回避（前日比+3%以上は見送り）
            if change > 3:
                continue
            
            # 条件5: 押し目買い優先（BB中央より下）
            if bb_position > 0.7:
                continue
            
            # ========== v6新規: 拡張フィルター ==========
            perf = ticker_performance[ticker]
            
            # ブラックリストチェック: 2連敗以上で20日間は買い禁止
            if perf['consecutive_losses'] >= 2:
                days_since_loss = day_num - perf['last_loss_day']
                if days_since_loss < 20:
                    continue  # ブラックリスト期間中
            
            # 過去実績フィルター
            if perf['trade_count'] >= 2:
                win_rate = perf['wins'] / perf['trade_count']
                avg_pnl = perf['total_pnl'] / perf['trade_count']
                
                # 勝率30%未満、または平均損益-5%以下の銘柄は買わない
                if win_rate < 0.3 or avg_pnl < -5:
                    continue
                
                # ========== v6新規: 拡張実績スコア ==========
                # 基本スコア
                perf_score = win_rate * 0.4 + min(max(avg_pnl / 30, -1), 1) * 0.3
                
                # 大勝ちボーナス: +20%以上の取引があった銘柄を優遇
                if perf['big_wins'] > 0:
                    perf_score += 0.2
                
                # 最高利益ボーナス
                if perf['best_pnl'] >= 30:
                    perf_score += 0.1
            else:
                perf_score = 0.5  # 実績不足は中立
            
            # ========== v7新規: スクイーズブレイクアウトボーナス ==========
            squeeze_bonus = signal.get('squeeze_bonus', 0.0)
            
            # ========== v7新規: ケリー基準による最適配分計算 ==========
            # Kelly % = W - [(1-W) / R]
            # W = 勝率, R = 勝ち時の平均利益 / 負け時の平均損失
            if perf['trade_count'] >= 3 and perf['wins'] > 0 and perf['losses'] > 0:
                win_rate = perf['wins'] / perf['trade_count']
                # 簡易計算: 勝ち時は+10%, 負け時は-5%と仮定（実際の平均を使うとより精密）
                avg_win = max(5, perf['best_pnl'] / 2)  # 推定勝ち幅
                avg_loss = 5  # 推定負け幅
                kelly_ratio = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                kelly_ratio = max(0, min(0.25, kelly_ratio))  # 0〜25%に制限（ハーフケリー推奨）
            else:
                kelly_ratio = 0.10  # デフォルト10%
            
            # ========== v7新規: レジーム調整スコア ==========
            regime_adjusted_score = signal.get('regime_adjusted_score', score)
            
            # ========== v8.3: セクターパフォーマンスボーナス ==========
            ticker_sector = ticker_sectors.get(ticker, 'other')
            sector_perf = sector_performance[ticker_sector]
            sector_trades = sector_perf['wins'] + sector_perf['losses']
            if sector_trades >= 3:
                sector_win_rate = sector_perf['wins'] / sector_trades
                sector_avg_pnl = sector_perf['total_pnl'] / sector_trades
                # 勝率60%以上かつ平均利益5%以上のセクターにボーナス
                if sector_win_rate >= 0.6 and sector_avg_pnl >= 5:
                    sector_bonus = 0.15
                elif sector_win_rate >= 0.5 and sector_avg_pnl >= 0:
                    sector_bonus = 0.05
                elif sector_win_rate < 0.4 or sector_avg_pnl < -5:
                    sector_bonus = -0.1  # 不調セクターはペナルティ
                else:
                    sector_bonus = 0
            else:
                sector_bonus = 0
            
            # 最終スコア = シグナルスコア + 実績ボーナス + モメンタム + スクイーズ + レジーム + セクター
            momentum_bonus = min(0.3, max(-0.3, risk_adj_momentum * 0.1))
            final_score = regime_adjusted_score + perf_score * 0.4 + momentum_bonus + squeeze_bonus + sector_bonus
            
            buy_candidates.append((ticker, signal, final_score, perf_score, momentum_20d, kelly_ratio))
        
        # ========== v6新規: モメンタムランキングでソート ==========
        # 最終スコアでソート（モメンタム込み）
        buy_candidates.sort(key=lambda x: x[2], reverse=True)
        
        # ========== v7新規: セクター分散フィルター ==========
        # 同一セクターからの購入は最大2銘柄まで
        sector_count = {}
        for t in portfolio:
            sector = ticker_sectors.get(t, 'other')
            sector_count[sector] = sector_count.get(sector, 0) + 1
        
        # ========== v7.5b: 上位銘柄に集中投資 ==========
        daily_buy_count = 0
        max_daily_buys = 2  # 1日2銘柄に絞って集中
        
        for ticker, signal, final_score, perf_score, momentum, kelly_ratio in buy_candidates:
            if daily_buy_count >= max_daily_buys:
                break
            
            # ========== v7新規: セクター集中回避 ==========
            ticker_sector = ticker_sectors.get(ticker, 'other')
            if sector_count.get(ticker_sector, 0) >= 2:
                continue  # 同一セクター2銘柄以上は回避
            
            price = daily_prices[ticker]
            score = signal['total_score']
            
            # v8.6: 現金比率を下げて投資機会を増やす
            min_cash_ratio = 0.05  # 8% → 5%
            current_total = cash + sum(
                portfolio[t]['shares'] * daily_prices.get(t, portfolio[t]['avg_cost'])
                for t in portfolio
            )
            if cash < current_total * min_cash_ratio:
                break
            
            # v8.6: 保有銘柄数を増やす（最大8銘柄）
            if len(portfolio) >= 8:
                break
            
            # ========== v8.6: 勝者への集中投資強化 ==========
            # ケリー比率を基本に、スコアと実績で調整
            base_ratio = max(0.12, kelly_ratio * 1.5)  # ベース比率を上げる（10%→12%）
            
            # 過去に大勝ちした銘柄は倍率アップ（最大3.0倍に強化）
            perf = ticker_performance[ticker]
            if perf['big_wins'] >= 2:  # 2回以上大勝ち
                alloc_multiplier = 3.0  # 2.5 → 3.0
            elif perf['big_wins'] >= 1:  # 1回大勝ち
                alloc_multiplier = 2.5  # 2.0 → 2.5
            elif perf_score >= 0.7:
                alloc_multiplier = 1.8  # 1.5 → 1.8
            elif perf_score >= 0.5:
                alloc_multiplier = 1.3  # 1.2 → 1.3
            else:
                alloc_multiplier = 1.0
            
            # スクイーズブレイクアウト時はさらに積極的に
            squeeze_bonus = signal.get('squeeze_bonus', 0.0)
            if squeeze_bonus > 0:
                alloc_multiplier *= 1.3
            
            buy_amount = current_total * base_ratio * alloc_multiplier * buy_budget_ratio
            
            # v8.6: 上限を引き上げ（最大25%まで）
            max_position = current_total * 0.25
            buy_amount = min(buy_amount, max_position)
            
            available_cash = cash - (current_total * min_cash_ratio)
            buy_amount = min(buy_amount, available_cash)
            
            # ========== v7.0: 最低購入額 ==========
            if buy_amount > 50000:  # 5万円以上のみ購入
                shares = buy_amount / price
                cash -= buy_amount
                
                portfolio[ticker] = {
                    'shares': shares,
                    'avg_cost': price,
                    'high_since_buy': price,
                    'buy_date': current_date,
                    'days_held': 0,
                    'last_partial_sell_day': 0
                }
                
                # セクターカウント更新
                sector_count[ticker_sector] = sector_count.get(ticker_sector, 0) + 1
                
                # 詳細な購入理由
                perf = ticker_performance[ticker]
                big_win_info = f", 大勝{perf['big_wins']}回" if perf['big_wins'] > 0 else ""
                momentum_info = f", M{momentum:.1f}%" if momentum != 0 else ""
                regime_name = signal.get('regime_name', '')
                squeeze_info = ", SQ" if squeeze_bonus > 0 else ""
                kelly_info = f", K{kelly_ratio*100:.0f}%"
                
                trades.append({
                    'date': current_date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'amount': buy_amount,
                    'reason': f"買い (S{score:.2f}{big_win_info}{momentum_info}{kelly_info}{squeeze_info}, {regime_name})"
                })
                
                daily_buy_count += 1
        
        # シグナル履歴を更新
        prev_day_signals = daily_signals.copy()
        
        # 日次記録
        stock_value = sum(
            portfolio[t]['shares'] * daily_prices.get(t, portfolio[t]['avg_cost'])
            for t in portfolio
        )
        total_value = cash + stock_value
        
        history.append({
            'date': current_date,
            'cash': cash,
            'stock_value': stock_value,
            'total_value': total_value,
            'num_positions': len(portfolio),
            'market_bullish': market_is_bullish,
            # v10.0: VIXレジーム情報
            'vix_level': vix_level,
            'vix_regime': vix_regime,
            'vix_momentum': vix_momentum,
            'current_drawdown': current_drawdown,
            'position_multiplier': combined_position_multiplier
        })
    
    if progress_callback:
        progress_callback(1.0)
    
    return {
        'history': history,
        'trades': trades,
        'final_portfolio': portfolio,
        'final_cash': cash,
        'ticker_performance': ticker_performance,
        # v10.0: VIXレジーム統計
        'vix_data_available': vix_data is not None
    }


# ==================== メインUI ====================

# サイドバー設定
st.sidebar.header("⚙️ バックテスト設定")

# キャッシュクリアボタン
if st.sidebar.button("🔄 データキャッシュをクリア"):
    st.cache_data.clear()
    st.sidebar.success("キャッシュをクリアしました")
    st.rerun()

# 期間選択
period_options = {
    "3ヶ月": 63,
    "6ヶ月": 126,
    "1年": 252,
    "2年": 504
}
selected_period = st.sidebar.selectbox("テスト期間", list(period_options.keys()), index=2)
test_days = period_options[selected_period]

# 初期資金
initial_cash = st.sidebar.number_input("初期資金（円）", min_value=100000, max_value=100000000, 
                                        value=1000000, step=100000)

# ウォッチリスト取得
watchlist = db.get_watchlist()

if not watchlist:
    st.warning("📭 ウォッチリストが空です。先に銘柄を追加してください。")
    st.stop()

# 銘柄選択
all_tickers = [w['ticker'] for w in watchlist]
st.sidebar.subheader("📊 対象銘柄")
select_all = st.sidebar.checkbox("全銘柄を選択", value=True)

if select_all:
    selected_tickers = all_tickers
else:
    selected_tickers = st.sidebar.multiselect("銘柄を選択", all_tickers, default=all_tickers[:10])

st.sidebar.metric("選択銘柄数", len(selected_tickers))

# アルゴリズム説明
with st.expander("📋 売買アルゴリズム（改善版 v10.1 - VIX + VolTarget + ADX/OBV）", expanded=False):
    st.markdown("""
    ### ウォークフォワードテストとは
    各日の判断は**その日までのデータのみ**を使用し、未来のデータは一切見ません。
    
    ### 🆕 v10.1 新機能（Phase 2-3実装）
    
    #### 📊 ボラティリティターゲティング（v10.1 Phase 2）
    ポートフォリオのボラティリティを目標値（年率15%）に維持：
    - 実現ボラが高い → ポジション縮小
    - 実現ボラが低い → ポジション拡大（最大2倍）
    - 急変動期に自動でリスク調整
    
    #### 📈 ADX（Average Directional Index）（v10.1 Phase 3）
    トレンドの強さを測定：
    - ADX ≥ 25: 強いトレンド → トレンド方向に従う
    - ADX < 25: レンジ相場 → 慎重運用
    - +DI > -DI: 上昇トレンド → 買いボーナス
    
    #### 💰 OBV（On Balance Volume）（v10.1 Phase 3）
    出来高の累積で機関投資家の資金流入を追跡：
    - OBV > SMA(20): 資金流入 → 買いシグナル
    - OBV < SMA(20): 資金流出 → 売りシグナル
    - 価格とOBVの乖離はダイバージェンス
    
    ---
    
    ### 🆕 v10.0 新機能（Deep Research統合）
    
    #### 🚨 VIXベースのレジーム検知（最優先改善）
    VIX（恐怖指数）を監視し、市場のパニック状態を早期検知：
    
    | VIX水準 | レジーム | ポジション倍率 | アクション |
    |--------|---------|--------------|-----------|
    | VIX < 20 | NORMAL | 100% | 通常運用 |
    | 20 ≤ VIX < 25 | NORMAL | 100% | 通常運用 |
    | 25 ≤ VIX < 30 | CAUTION | 50% | 警戒モード |
    | 30 ≤ VIX < 35 | FEAR | 25% | 損失ポジション即売却 |
    | VIX ≥ 35 | PANIC | 0% | 全ポジション売却 |
    
    **VIXモメンタム検知**: 5日で+15%急上昇 → CAUTIONへ移行
    
    #### 📉 ドローダウン連動ポジション縮小
    ポートフォリオ全体のDDを監視し、自動でリスク削減：
    
    | ドローダウン | ポジション倍率 | 説明 |
    |------------|--------------|------|
    | DD < 3% | 100% | 通常運用 |
    | 3% ≤ DD < 5% | 50% | 半分に縮小 |
    | DD ≥ 5% | 25% | 75%縮小 |
    
    **総合倍率 = VIX倍率 × DD倍率 × ボラ倍率**（全て適用）
    
    ---
    
    ### 🧪 v7.0 最新研究ベースの改善（継続）
    
    #### 📚 学術研究から導入した手法
    
    **1. レジームスイッチング（Market Regime Detection）**
    市場を4つの状態に分類し、各状態で最適な戦略を適用：
    | レジーム | 買い倍率 | 損切り倍率 | 特徴 |
    |---------|---------|-----------|------|
    | 低ボラ上昇 | 1.2x | 0.8x | 最良環境、積極投資 |
    | 高ボラ上昇 | 0.8x | 1.2x | 慎重に、利益は広めに |
    | 低ボラ下降 | 0.0x | 1.0x | 買い停止 |
    | 高ボラ下降 | 0.3x | 1.5x | ほぼ停止、損切り広め |
    
    **2. ケリー基準（Kelly Criterion）**
    $$f^* = W - \\frac{1-W}{R}$$
    - $W$ = 勝率、$R$ = 平均勝ち幅 / 平均負け幅
    - 各銘柄の過去実績から最適ポジションサイズを計算
    - ハーフケリー×1.5で積極的に運用
    
    **3. ボラティリティスクイーズ（TTM Squeeze）**
    - ボリンジャーバンドがケルトナーチャネル内に収まった状態を検出
    - 収縮後の拡大 = ブレイクアウトのチャンス
    - スクイーズ中 & 上向きモメンタム → 配分1.3倍ボーナス
    
    **4. ケルトナーチャネル（Keltner Channel）**
    - ATRベースの動的チャネル
    - ボリンジャーより安定したトレンド判定
    
    #### 🎨 独自手法
    
    **5. セクター分散**
    - 同一セクターは最大2銘柄まで
    - 米国株: tech, finance, health, energy, other
    - 日本株: 証券コードから推定
    
    **6. アダプティブ損切り**
    - レジームに応じて損切り幅を自動調整
    - 高ボラ時は広め、低ボラ時は狭め
    
    #### 💰 勝者優遇配分（v6継続）
    | 条件 | 配分倍率 |
    |------|---------|
    | 大勝ち2回以上 | **2.5倍** |
    | 大勝ち1回 | **2.0倍** |
    | スクイーズ中 | **×1.3倍** |
    | 通常 | 1.0倍 |
    
    #### 🚫 敗者ブラックリスト（v6継続）
    - **2連敗した銘柄は20日間買い禁止**
    
    ### 売りルール
    | 条件 | アクション |
    |------|-----------|
    | VIX PANIC | 全ポジション即売却 |
    | VIX FEAR + 損失 | 即売却 |
    | ポジション < 0.5% | 整理売却 |
    | 損切: レジーム×ATR×3 | 全売却（3日猶予） |
    | トレーリング | 利益に応じて-7%〜-15% |
    | 利確+50% | 3/4売却 |
    | 利確+30% | 半分売却 |
    | シグナル売り & 損失-3%超 | 全売却 |
    """)

# バックテスト実行
st.divider()

if st.button("🚀 バックテスト実行", type="primary", use_container_width=True):
    if len(selected_tickers) == 0:
        st.error("銘柄を選択してください")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(p):
            progress_bar.progress(p)
            status_text.text(f"処理中... {int(p * 100)}%")
        
        status_text.text("過去データを取得中...")
        
        with st.spinner("バックテスト実行中..."):
            result = run_backtest(
                selected_tickers, 
                initial_cash=initial_cash,
                start_days_ago=test_days,
                progress_callback=update_progress
            )
        
        progress_bar.empty()
        status_text.empty()
        
        if result is None:
            st.error("バックテストに失敗しました。銘柄データが不足している可能性があります。")
        elif 'error' in result:
            st.error(f"⚠️ {result['error']}")
            st.info(f"必要なデータ日数: {test_days + 200}日以上（{test_days}日テスト + 200日MA計算用）")
            st.info("💡 キャッシュをクリアして再試行してみてください")
            if st.button("🔄 キャッシュをクリア"):
                st.cache_data.clear()
                st.rerun()
        elif 'history' not in result:
            st.error("バックテスト結果が不正です")
        else:
            st.session_state['backtest_result'] = result
            st.success("✅ バックテスト完了！")
            st.rerun()

# 結果表示
if 'backtest_result' in st.session_state:
    result = st.session_state['backtest_result']
    history = result['history']
    trades = result['trades']
    
    # サマリー
    st.subheader("📊 バックテスト結果")
    
    initial = history[0]['total_value']
    final = history[-1]['total_value']
    profit = final - initial
    profit_rate = (profit / initial) * 100
    
    # 最大ドローダウン計算
    peak = initial
    max_drawdown = 0
    for h in history:
        if h['total_value'] > peak:
            peak = h['total_value']
        drawdown = (peak - h['total_value']) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    # 取引統計
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    
    # 勝率計算（売り取引で利益が出たもの）
    profitable_sells = [t for t in sell_trades if t.get('pnl_rate', 0) > 0]
    win_rate = (len(profitable_sells) / len(sell_trades) * 100) if sell_trades else 0
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 最終資産", f"¥{final:,.0f}", delta=f"¥{profit:+,.0f}")
    col2.metric("📈 収益率", f"{profit_rate:+.2f}%")
    col3.metric("📉 最大DD", f"-{max_drawdown:.2f}%")
    col4.metric("🔄 総取引数", len(trades))
    col5.metric("📊 勝率", f"{win_rate:.1f}%")
    
    # VIXレジーム統計（v10.0新機能）
    if result.get('vix_data_available', False) and 'vix_regime' in history[0]:
        st.divider()
        st.subheader("🚨 VIXレジーム分析 (v10.0)")
        
        # レジーム別日数カウント
        regime_counts = {'NORMAL': 0, 'CAUTION': 0, 'FEAR': 0, 'PANIC': 0}
        vix_values = []
        for h in history:
            regime = h.get('vix_regime', 'N/A')
            if regime in regime_counts:
                regime_counts[regime] += 1
            vix_level = h.get('vix_level')
            if vix_level is not None:
                vix_values.append(vix_level)
        
        total_days = sum(regime_counts.values())
        
        # レジーム統計表示
        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
        with rcol1:
            pct = (regime_counts['NORMAL'] / total_days * 100) if total_days > 0 else 0
            st.metric("🟢 NORMAL", f"{regime_counts['NORMAL']}日", f"{pct:.1f}%")
        with rcol2:
            pct = (regime_counts['CAUTION'] / total_days * 100) if total_days > 0 else 0
            st.metric("🟡 CAUTION", f"{regime_counts['CAUTION']}日", f"{pct:.1f}%")
        with rcol3:
            pct = (regime_counts['FEAR'] / total_days * 100) if total_days > 0 else 0
            st.metric("🟠 FEAR", f"{regime_counts['FEAR']}日", f"{pct:.1f}%", delta_color="inverse")
        with rcol4:
            pct = (regime_counts['PANIC'] / total_days * 100) if total_days > 0 else 0
            st.metric("🔴 PANIC", f"{regime_counts['PANIC']}日", f"{pct:.1f}%", delta_color="inverse")
        
        # VIX統計
        if vix_values:
            vcol1, vcol2, vcol3, vcol4 = st.columns(4)
            vcol1.metric("📊 VIX平均", f"{sum(vix_values)/len(vix_values):.1f}")
            vcol2.metric("📈 VIX最大", f"{max(vix_values):.1f}")
            vcol3.metric("📉 VIX最小", f"{min(vix_values):.1f}")
            vcol4.metric("📋 データ日数", f"{len(vix_values)}日")
        
        st.caption("💡 VIXレジームにより自動でポジション調整が行われます。FEAR/PANICモードでは損失ポジションを強制売却します。")
    
    st.divider()
    
    # 資産推移グラフ
    st.subheader("📈 資産推移")
    
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    df_history = pd.DataFrame(history)
    df_history['date'] = pd.to_datetime(df_history['date'])
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=("総資産推移", "保有銘柄数"))
    
    # 市場弱気期間を背景色で表示
    if 'market_bullish' in df_history.columns:
        bearish_periods = []
        in_bearish = False
        start_date = None
        
        for i, row in df_history.iterrows():
            if not row.get('market_bullish', True) and not in_bearish:
                in_bearish = True
                start_date = row['date']
            elif row.get('market_bullish', True) and in_bearish:
                in_bearish = False
                bearish_periods.append((start_date, row['date']))
        
        if in_bearish:
            bearish_periods.append((start_date, df_history['date'].iloc[-1]))
        
        for start, end in bearish_periods:
            fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.1, 
                          layer="below", line_width=0, row=1, col=1)
            fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.1, 
                          layer="below", line_width=0, row=2, col=1)
    
    # VIXレジーム期間を背景色で表示（v10.0新機能）
    if 'vix_regime' in df_history.columns:
        regime_colors = {
            'CAUTION': ('yellow', 0.15),
            'FEAR': ('orange', 0.2),
            'PANIC': ('red', 0.3)
        }
        
        for regime, (color, opacity) in regime_colors.items():
            regime_periods = []
            in_regime = False
            start_date = None
            
            for i, row in df_history.iterrows():
                if row.get('vix_regime') == regime and not in_regime:
                    in_regime = True
                    start_date = row['date']
                elif row.get('vix_regime') != regime and in_regime:
                    in_regime = False
                    regime_periods.append((start_date, row['date']))
            
            if in_regime:
                regime_periods.append((start_date, df_history['date'].iloc[-1]))
            
            for start, end in regime_periods:
                fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=opacity, 
                              layer="below", line_width=0, row=1, col=1,
                              annotation_text=regime if (end - start).days > 3 else "",
                              annotation_position="top left")
    
    # 総資産
    fig.add_trace(go.Scatter(
        x=df_history['date'], y=df_history['total_value'],
        name='総資産', line=dict(color='blue', width=2),
        fill='tozeroy', fillcolor='rgba(0,100,255,0.1)'
    ), row=1, col=1)
    
    # 現金
    fig.add_trace(go.Scatter(
        x=df_history['date'], y=df_history['cash'],
        name='現金', line=dict(color='green', width=1, dash='dash')
    ), row=1, col=1)
    
    # 初期資金ライン
    fig.add_hline(y=initial, line_dash="dash", line_color="gray", 
                  annotation_text="初期資金", row=1, col=1)
    
    # 保有銘柄数
    fig.add_trace(go.Scatter(
        x=df_history['date'], y=df_history['num_positions'],
        name='保有数', line=dict(color='orange'), fill='tozeroy'
    ), row=2, col=1)
    
    fig.update_layout(height=500, hovermode='x unified', showlegend=True)
    fig.update_yaxes(title_text="金額 (円)", row=1, col=1)
    fig.update_yaxes(title_text="銘柄数", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 取引履歴
    st.subheader("📜 取引履歴")
    
    tab1, tab2 = st.tabs(["買い取引", "売り取引"])
    
    with tab1:
        if buy_trades:
            buy_df = pd.DataFrame(buy_trades)
            buy_df['date'] = pd.to_datetime(buy_df['date']).dt.strftime('%Y-%m-%d')
            buy_df['price'] = buy_df['price'].apply(lambda x: f"${x:.2f}")
            buy_df['amount'] = buy_df['amount'].apply(lambda x: f"¥{x:,.0f}")
            buy_df['shares'] = buy_df['shares'].apply(lambda x: f"{x:.2f}")
            st.dataframe(
                buy_df[['date', 'ticker', 'shares', 'price', 'amount', 'reason']],
                use_container_width=True, hide_index=True
            )
        else:
            st.info("買い取引なし")
    
    with tab2:
        if sell_trades:
            sell_df = pd.DataFrame(sell_trades)
            sell_df['date'] = pd.to_datetime(sell_df['date']).dt.strftime('%Y-%m-%d')
            sell_df['price'] = sell_df['price'].apply(lambda x: f"${x:.2f}")
            sell_df['amount'] = sell_df['amount'].apply(lambda x: f"¥{x:,.0f}")
            sell_df['shares'] = sell_df['shares'].apply(lambda x: f"{x:.2f}")
            if 'pnl_rate' in sell_df.columns:
                sell_df['pnl'] = sell_df['pnl_rate'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "-")
                st.dataframe(
                    sell_df[['date', 'ticker', 'shares', 'price', 'amount', 'pnl', 'reason']],
                    use_container_width=True, hide_index=True
                )
            else:
                st.dataframe(
                    sell_df[['date', 'ticker', 'shares', 'price', 'amount', 'reason']],
                    use_container_width=True, hide_index=True
                )
        else:
            st.info("売り取引なし")
    
    # 月次リターン
    st.subheader("📅 月次リターン")
    
    df_history['month'] = pd.to_datetime(df_history['date']).dt.to_period('M')
    monthly = df_history.groupby('month').agg({
        'total_value': ['first', 'last']
    })
    monthly.columns = ['start', 'end']
    monthly['return'] = ((monthly['end'] - monthly['start']) / monthly['start'] * 100).round(2)
    
    fig_monthly = go.Figure(data=[
        go.Bar(
            x=[str(m) for m in monthly.index],
            y=monthly['return'],
            marker_color=['green' if r >= 0 else 'red' for r in monthly['return']],
            text=[f"{r:+.1f}%" for r in monthly['return']],
            textposition='outside'
        )
    ])
    fig_monthly.update_layout(
        title="月次リターン (%)",
        height=300,
        xaxis_title="月",
        yaxis_title="リターン (%)"
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # 最終ポートフォリオ
    if result['final_portfolio']:
        st.subheader("📦 最終保有銘柄")
        final_portfolio_data = []
        for ticker, pos in result['final_portfolio'].items():
            final_portfolio_data.append({
                '銘柄': ticker,
                '株数': f"{pos['shares']:.2f}",
                '平均取得単価': f"${pos['avg_cost']:.2f}"
            })
        st.dataframe(pd.DataFrame(final_portfolio_data), use_container_width=True, hide_index=True)
    
    # ==================== 詳細分析セクション ====================
    st.divider()
    st.subheader("🔬 詳細分析")
    
    # 分析実行ボタン
    if st.button("📊 詳細分析を実行", type="secondary"):
        with st.spinner("分析中..."):
            analysis = analyze_backtest_results(history, trades)
            st.session_state['backtest_analysis'] = analysis
    
    if 'backtest_analysis' in st.session_state:
        analysis = st.session_state['backtest_analysis']
        
        # タブで表示
        tab_basic, tab_risk, tab_trades, tab_tickers, tab_problems = st.tabs([
            "📈 基本統計", "⚠️ リスク指標", "🔄 取引分析", "📊 銘柄別", "💡 改善提案"
        ])
        
        with tab_basic:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 基本統計")
                for key, value in analysis.get('基本統計', {}).items():
                    st.write(f"**{key}**: {value}")
            with col2:
                st.markdown("### 月次パフォーマンス")
                monthly_data = analysis.get('月次パフォーマンス', [])
                if monthly_data:
                    df_monthly = pd.DataFrame(monthly_data)
                    st.dataframe(df_monthly, use_container_width=True, hide_index=True)
        
        with tab_risk:
            st.markdown("### リスク指標")
            for key, value in analysis.get('リスク指標', {}).items():
                st.write(f"**{key}**: {value}")
        
        with tab_trades:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 取引統計")
                for key, value in analysis.get('取引統計', {}).items():
                    st.write(f"**{key}**: {value}")
                
                st.markdown("### 売り取引分析")
                for key, value in analysis.get('売り取引分析', {}).items():
                    st.write(f"**{key}**: {value}")
            
            with col2:
                st.markdown("### 売却理由別")
                reason_data = analysis.get('売却理由別', {})
                if reason_data:
                    reason_df = []
                    for reason, stats in reason_data.items():
                        reason_df.append({
                            '理由': reason,
                            '回数': stats['回数'],
                            '平均損益': stats['平均損益']
                        })
                    st.dataframe(pd.DataFrame(reason_df), use_container_width=True, hide_index=True)
        
        with tab_tickers:
            st.markdown("### 銘柄別パフォーマンス")
            ticker_data = analysis.get('銘柄別パフォーマンス', [])
            if ticker_data:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### 🏆 上位5銘柄")
                    top_df = pd.DataFrame(ticker_data[:5])
                    top_df['平均損益率'] = top_df['平均損益率'].apply(lambda x: f"{x:.1f}%")
                    top_df['投資額合計'] = top_df['投資額合計'].apply(lambda x: f"¥{x:,.0f}")
                    st.dataframe(top_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("#### 📉 下位5銘柄")
                    bottom_df = pd.DataFrame(ticker_data[-5:])
                    bottom_df['平均損益率'] = bottom_df['平均損益率'].apply(lambda x: f"{x:.1f}%")
                    bottom_df['投資額合計'] = bottom_df['投資額合計'].apply(lambda x: f"¥{x:,.0f}")
                    st.dataframe(bottom_df, use_container_width=True, hide_index=True)
        
        with tab_problems:
            st.markdown("### ⚠️ 問題点")
            problems = analysis.get('問題点', [])
            if problems:
                for p in problems:
                    st.warning(p)
            else:
                st.success("大きな問題点は見つかりませんでした")
            
            st.markdown("### 💡 改善提案")
            suggestions = analysis.get('改善提案', [])
            for s in suggestions:
                st.info(s)
        
        # 結果保存ボタン
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 結果をJSONで保存"):
                # 保存用データ作成
                save_data = {
                    'timestamp': datetime.now().isoformat(),
                    'analysis': analysis,
                    'trades': trades,
                    'history_summary': {
                        'initial': history[0]['total_value'],
                        'final': history[-1]['total_value'],
                        'days': len(history)
                    }
                }
                
                # ファイル保存
                save_path = Path(__file__).parent.parent / "analysis" / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                # datetime変換
                def convert_datetime(obj):
                    if hasattr(obj, 'isoformat'):
                        return obj.isoformat()
                    elif hasattr(obj, '__str__'):
                        return str(obj)
                    return obj
                
                # tradesのdatetime変換
                trades_serializable = []
                for t in trades:
                    t_copy = t.copy()
                    if 'date' in t_copy:
                        t_copy['date'] = convert_datetime(t_copy['date'])
                    trades_serializable.append(t_copy)
                save_data['trades'] = trades_serializable
                
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
                
                st.success(f"✅ 保存しました: {save_path.name}")
        
        with col2:
            # CSVエクスポート
            if st.button("📥 取引履歴をCSV出力"):
                trades_df = pd.DataFrame(trades)
                csv_path = Path(__file__).parent.parent / "analysis" / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                trades_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                st.success(f"✅ 保存しました: {csv_path.name}")

else:
    st.info("👆 「バックテスト実行」ボタンを押してシミュレーションを開始してください")

# 注意事項
st.divider()
st.caption("""
⚠️ **注意事項**
- このバックテストは過去のデータに基づくシミュレーションであり、将来の結果を保証するものではありません
- 実際の取引では手数料、スリッページ、流動性などの要因が影響します
- 為替レートは考慮していません（米国株は1ドル=100円として計算）
""")
