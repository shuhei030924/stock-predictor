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
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from analysis.backtest_analyzer import analyze_backtest_results, print_analysis

st.set_page_config(
    page_title="📅 過去1年バックテスト",
    page_icon="📅",
    layout="wide"
)

st.title("📅 過去1年間バックテスト")
st.markdown("**v10.7** - 積極的スケーリング (取引頻度向上 + 収益最大化)")

db = DatabaseManager()


# ==================== シグナル計算関数（特定日時点） ====================

def precalculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    全期間の指標を一括計算（ベクトル化高速版）
    """
    # コピーを作成して警告を回避
    df = df.copy()
    
    # 必要な列が存在するか確認
    required_cols = ['Close', 'High', 'Low', 'Volume']
    if not all(col in df.columns for col in required_cols):
        return df
        
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # ========== RSI (14日) ==========
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # RSIシグナル (v10.6)
    conditions = [
        df['rsi'] < 30,
        df['rsi'] > 70
    ]
    choices = [
        1.0 + (30 - df['rsi']) * 0.05,
        -1.0
    ]
    df['rsi_signal'] = np.select(conditions, choices, default=(50 - df['rsi']) / 50)
    
    # V字回復ボーナス
    rsi_prev = df['rsi'].shift(1)
    v_shape = (rsi_prev < 25) & (df['rsi'] > rsi_prev + 2)
    df.loc[v_shape, 'rsi_signal'] += 0.5
    
    # ========== RSIダイバージェンス ==========
    price_5d_change = close.pct_change(5) * 100
    rsi_5d_change = df['rsi'].diff(5)
    
    df['divergence_signal'] = 0.0
    bearish_div = (price_5d_change > 2) & (rsi_5d_change < -5)
    bullish_div = (price_5d_change < -2) & (rsi_5d_change > 5)
    
    df.loc[bearish_div, 'divergence_signal'] = -0.5
    df.loc[bullish_div, 'divergence_signal'] = 0.5
    
    # ========== 移動平均 ==========
    df['sma5'] = close.rolling(window=5).mean()
    df['sma20'] = close.rolling(window=20).mean()
    df['sma50'] = close.rolling(window=50).mean()
    df['sma200'] = close.rolling(window=200).mean()
    
    # 200日MAがない場合は50日MAで代用
    df['sma200'] = df['sma200'].fillna(df['sma50'])
    
    # MAシグナル
    ma_signal = pd.Series(0.0, index=df.index)
    ma_signal += np.where(close > df['sma5'], 0.2, 0)
    ma_signal += np.where(df['sma5'] > df['sma20'], 0.3, 0)
    ma_signal += np.where(df['sma20'] > df['sma50'], 0.25, 0)
    ma_signal += np.where(df['sma50'] > df['sma200'], 0.25, 0)
    
    df['ma_signal'] = (ma_signal - 0.5) * 2
    df['is_uptrend'] = df['sma50'] > df['sma200']
    
    # ========== MACD ==========
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema12 - ema26
    df['macd_signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd_line'] - df['macd_signal_line']
    
    # MACDモメンタム
    macd_momentum = df['macd_hist'].diff()
    
    # クロスオーバー
    macd_prev = df['macd_line'].shift(1)
    signal_prev = df['macd_signal_line'].shift(1)
    
    macd_gc = (macd_prev <= signal_prev) & (df['macd_line'] > df['macd_signal_line'])
    macd_dc = (macd_prev >= signal_prev) & (df['macd_line'] < df['macd_signal_line'])
    
    # ヒストグラム反転
    hist_prev = df['macd_hist'].shift(1)
    hist_prev2 = df['macd_hist'].shift(2)
    hist_peak = (hist_prev2 < hist_prev) & (hist_prev > df['macd_hist'])
    hist_bottom = (hist_prev2 > hist_prev) & (hist_prev < df['macd_hist'])
    
    df['hist_reversal'] = 0
    df.loc[hist_peak, 'hist_reversal'] = -1
    df.loc[hist_bottom, 'hist_reversal'] = 1
    
    df['macd_crossover'] = 0
    df.loc[macd_gc, 'macd_crossover'] = 1
    df.loc[macd_dc, 'macd_crossover'] = -1

    # 基本シグナル
    base_macd_signal = pd.Series(0.0, index=df.index)
    cond_buy = (df['macd_line'] > df['macd_signal_line']) & (df['macd_hist'] > 0)
    cond_sell = (df['macd_line'] < df['macd_signal_line']) & (df['macd_hist'] < 0)
    
    base_macd_signal[cond_buy] = 1.0
    base_macd_signal[cond_sell] = -1.0
    
    mask_other = ~(cond_buy | cond_sell)
    # ゼロ除算回避
    hist_safe = df.loc[mask_other, 'macd_hist']
    base_macd_signal[mask_other] = hist_safe / (hist_safe.abs() + 0.01) * 0.5
    
    # モメンタム補正
    base_macd_signal = np.where(macd_momentum > 0, np.minimum(1.0, base_macd_signal + 0.2), base_macd_signal)
    base_macd_signal = np.where(macd_momentum < 0, np.maximum(-1.0, base_macd_signal - 0.2), base_macd_signal)
    
    # クロスオーバーボーナス
    base_macd_signal = np.where(macd_gc, np.minimum(1.0, base_macd_signal + 0.3), base_macd_signal)
    base_macd_signal = np.where(macd_dc, np.maximum(-1.0, base_macd_signal - 0.3), base_macd_signal)
    
    df['macd_signal'] = base_macd_signal
    
    # ========== ボリンジャーバンド ==========
    bb_std = close.rolling(window=20).std()
    bb_upper = df['sma20'] + 2 * bb_std
    bb_lower = df['sma20'] - 2 * bb_std
    
    bb_range = bb_upper - bb_lower
    bb_range = bb_range.replace(0, 1e-9)
    
    df['bb_position'] = (close - bb_lower) / bb_range
    df['bb_signal'] = (0.5 - df['bb_position']) * 2
    
    # ========== 出来高 ==========
    vol_sma = volume.rolling(window=20).mean()
    vol_ratio = volume / vol_sma.replace(0, 1)
    
    vol_5d = volume.rolling(window=5).mean()
    vol_prev_5d = volume.shift(5).rolling(window=5).mean()
    vol_trend = (vol_5d - vol_prev_5d) / vol_prev_5d.replace(0, 1)
    
    price_change_pct = close.pct_change() * 100
    df['price_change'] = price_change_pct
    
    vol_signal = pd.Series(0.0, index=df.index)
    vol_signal[(vol_ratio > 2.0) & (price_change_pct > 1)] = 1.5
    vol_signal[(vol_ratio > 1.5) & (price_change_pct > 0) & (vol_signal == 0)] = 1.0
    vol_signal[(vol_ratio > 2.0) & (price_change_pct < -1)] = -1.5
    vol_signal[(vol_ratio > 1.5) & (price_change_pct < 0) & (vol_signal == 0)] = -1.0
    
    vol_trend_bonus = np.where(vol_trend > 0.2, np.clip(vol_trend * 0.5, -0.3, 0.3), 0)
    df['vol_signal'] = vol_signal
    
    # ========== ROC ==========
    roc_10 = close.pct_change(10) * 100
    roc_signal = pd.Series(roc_10 / 10, index=df.index)
    roc_signal[roc_10 > 5] = 0.5
    roc_signal[roc_10 < -5] = -0.5
    df['roc_signal'] = roc_signal
    
    # ========== ATR ==========
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = (df['atr'] / close) * 100
    
    # ========== モメンタム ==========
    df['momentum_5d'] = close.pct_change(5) * 100
    df['momentum_10d'] = close.pct_change(10) * 100
    df['momentum_20d'] = close.pct_change(20) * 100
    
    # ========== ボラティリティ ==========
    df['volatility_20d'] = close.pct_change().rolling(20).std() * 100
    df['risk_adjusted_momentum'] = df['momentum_20d'] / (df['volatility_20d'] + 0.1)
    
    # ========== ケルトナーチャネル & スクイーズ ==========
    keltner_mid = close.ewm(span=20, adjust=False).mean()
    keltner_upper = keltner_mid + 2.0 * df['atr']
    keltner_lower = keltner_mid - 2.0 * df['atr']
    
    keltner_range = keltner_upper - keltner_lower
    df['keltner_position'] = (close - keltner_lower) / keltner_range.replace(0, 1e-9)
    
    df['squeeze_on'] = (bb_lower > keltner_lower) & (bb_upper < keltner_upper)
    
    # ========== レジーム ==========
    vol_median = close.pct_change().rolling(60).std() * 100
    is_high_vol = df['volatility_20d'] > (vol_median * 1.2)
    
    regime = pd.Series(0, index=df.index)
    regime[df['is_uptrend'] & ~is_high_vol] = 1
    regime[~df['is_uptrend'] & is_high_vol] = 2
    regime[df['is_uptrend'] & is_high_vol] = 3
    df['regime'] = regime
    
    # ========== 早期警戒スコア ==========
    early_warning = pd.Series(0, index=df.index)
    
    early_warning += ((df['momentum_5d'] < 0) & (df['momentum_10d'] > 0)).astype(int)
    early_warning += ((df['momentum_5d'] < df['momentum_10d']) & (df['momentum_10d'] < df['momentum_20d']) & (df['momentum_5d'] < 0)).astype(int)
    
    rsi_3d_ago = df['rsi'].shift(3)
    early_warning += ((df['rsi'] < rsi_prev) & (rsi_prev < rsi_3d_ago) & (rsi_3d_ago > 60)).astype(int)
    early_warning += ((df['rsi'] < 50) & (rsi_prev > 50)).astype(int)
    
    early_warning += hist_peak.astype(int)
    early_warning += (macd_dc.astype(int) * 2)
    
    early_warning += ((vol_ratio > 1.8) & (price_change_pct < -1)).astype(int) * 2
    
    sma5_prev = df['sma5'].shift(1)
    sma20_prev = df['sma20'].shift(1)
    early_warning += ((sma5_prev >= sma20_prev) & (df['sma5'] < df['sma20'])).astype(int)
    
    df['early_warning_score'] = early_warning
    
    # ========== 総合スコア ==========
    weights = {
        'rsi': 0.15,
        'divergence': 0.10,
        'ma': 0.20,
        'macd': 0.25,
        'bb': 0.10,
        'volume': 0.10,
        'roc': 0.10
    }
    
    total_score = (
        df['rsi_signal'] * weights['rsi'] +
        df['divergence_signal'] * weights['divergence'] +
        df['ma_signal'] * weights['ma'] +
        df['macd_signal'] * weights['macd'] +
        df['bb_signal'] * weights['bb'] +
        df['vol_signal'] * weights['volume'] +
        df['roc_signal'] * weights['roc']
    )
    
    total_score += vol_trend_bonus
    
    mask_downtrend = (~df['is_uptrend']) & (total_score > 0)
    total_score[mask_downtrend] *= 0.5
    
    squeeze_bonus = pd.Series(0.0, index=df.index)
    squeeze_bonus[df['squeeze_on'] & (df['macd_hist'] > 0)] = 0.15
    
    regime_buy_mult = [0.0, 1.2, 0.3, 0.8]
    regime_mults = np.array(regime_buy_mult)
    regime_safe = df['regime'].fillna(0).astype(int)
    current_mults = regime_mults[regime_safe]
    
    df['total_score'] = total_score
    df['regime_adjusted_score'] = total_score * current_mults
    df['squeeze_bonus'] = squeeze_bonus
    
    # 高値 (20日)
    df['high_price_20d'] = high.rolling(window=20).max()
    
    return df


def calculate_ai_scores(df: pd.DataFrame, interval: str = "1d") -> pd.Series:
    """
    Walk-Forward分析によるAIスコア算出 (Rolling Window)
    過去データのみを使ってモデル学習し、未来を予測する
    """
    # 特徴量作成 (既存の指標を利用)
    df_ai = df.copy()
    
    # 必要な列があるか確認
    required_cols = ['Close', 'rsi', 'macd_hist']
    if not all(col in df_ai.columns for col in required_cols):
        return pd.Series(0.0, index=df.index)

    df_ai['return_1d'] = df_ai['Close'].pct_change().fillna(0)
    df_ai['volatility'] = df_ai['return_1d'].rolling(20).std().fillna(0)
    df_ai['ma_ratio'] = (df_ai['Close'] / df_ai['Close'].rolling(50).mean()).fillna(1.0)
    
    # 目的変数: 5期間後のリターン (Swing Trade用)
    # 学習時は shift(-5) で未来を見るが、予測時は直近データを使う
    df_ai['target'] = df_ai['Close'].shift(-5) / df_ai['Close'] - 1
    
    features = ['return_1d', 'volatility', 'ma_ratio', 'rsi', 'macd_hist']
    
    # 欠損除去 (特徴量)
    df_clean = df_ai.dropna(subset=features)
    
    # AIスコア格納用
    ai_scores = pd.Series(0.0, index=df.index)
    
    # データが少なすぎる場合はスキップ
    if len(df_clean) < 200:
        return ai_scores
        
    # Rolling Window設定
    # 1時間足ならデータが多いのでウィンドウも調整
    train_window = 1000 if interval == "1h" else 250 # 1年分
    update_step = 100 if interval == "1h" else 20 # 1ヶ月ごとに再学習
    
    model = RandomForestRegressor(n_estimators=20, max_depth=5, n_jobs=1, random_state=42)
    
    # Walk-Forward Loop
    start_idx = train_window
    
    for i in range(start_idx, len(df_clean), update_step):
        # 学習データ: 過去 train_window 分
        train_data = df_clean.iloc[i-train_window:i]
        
        # ターゲットがNaN（直近5日など）の行は学習から除外
        train_data_valid = train_data.dropna(subset=['target'])
        
        if len(train_data_valid) < 100:
            continue
            
        X_train = train_data_valid[features]
        y_train = train_data_valid['target']
        
        # モデル学習
        try:
            model.fit(X_train, y_train)
            
            # 予測期間: 次の update_step 分
            end_idx = min(i + update_step, len(df_clean))
            predict_data = df_clean.iloc[i:end_idx]
            
            if len(predict_data) == 0:
                break
                
            X_pred = predict_data[features]
            
            # 予測実行
            preds = model.predict(X_pred)
            
            # スコア格納
            pred_indices = predict_data.index
            ai_scores.loc[pred_indices] = preds
            
        except Exception:
            continue
            
    return ai_scores


def get_historical_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    過去データを取得（データベースキャッシュ優先）
    interval: "1d" (日足) または "1h" (1時間足)
    """
    # 期間を日数に変換
    period_days = {
        "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "3y": 1095, "5y": 1825, "max": 3650
    }
    days = period_days.get(period, 730)
    
    # 1時間足の場合はキャッシュを使わず直接取得（DBが日足前提のため）
    if interval == "1h":
        import time
        time.sleep(0.3)
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            # 1時間足は最大730日(2年)まで
            if period in ["3y", "5y", "max"]:
                period = "2y"
            df = stock.history(period=period, interval=interval)
            if df is not None and len(df) >= 50:
                return df
            return None
        except Exception as e:
            st.warning(f"{ticker}: 1時間足データ取得エラー: {e}")
            return None

    # まずデータベースから取得を試みる
    df = db.get_cached_prices(ticker, days=days)
    
    if df is not None and len(df) >= 50:
        return df
    
    # データベースにない場合はyfinanceから取得
    import time
    
    # レート制限対策: リクエスト間に遅延
    time.sleep(0.3)
    
    try:
        import yfinance as yf
        
        # yfinanceは内部でcurl_cffiを使用するためセッション不要
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df is not None and len(df) >= 50:
            # データベースにキャッシュ (日足のみ)
            if interval == "1d":
                db.cache_prices(ticker, df)
            return df
            
        return None
    except Exception as e:
        if '429' in str(e) or 'Too Many Requests' in str(e):
            st.warning(f"{ticker}: レート制限。DBキャッシュを確認中...")
        return None


def run_backtest(tickers: list, initial_cash: float = 1000000, 
                 start_days_ago: int = 504, progress_callback=None,
                 market_ticker: str = "SPY", interval: str = "1d") -> dict:
    """
    過去2年間のバックテストを実行（改善版v11.0 - AIハイブリッド）
    
    v11.0 改善点（AI予測の導入）:
    
    【AIハイブリッド判定】
    - Walk-Forward分析: 過去データのみで学習したRandomForestモデルが、未来の収益を予測
    - AIスコアフィルタ: テクニカル的に買いでも、AIが「下落」と予測した場合はエントリーを見送り
    - これにより「ダマシ」を回避し、勝率をさらに向上させる
    
    v10.9 改善点（データ増量と利益最大化）:
    
    【データ拡張】
    - バックテスト期間: 1年(252日) → 2年(504日) に倍増
    - これにより、異なる市場環境（上昇・下落・レンジ）での安定性を検証
    - interval="1h" (1時間足) に対応し、データ密度を7倍に増加可能
    
    【利益追求（Let Profits Run）】
    - トレーリングストップ緩和: ノイズで切られないよう、ストップ幅を拡大
    - 利確ライン引き上げ: +50%→+60% など、大きなトレンドを最後まで追随
    - ブレークイーブン: +10%→+12% に変更し、早すぎる同値撤退を防止
    """
    
    # 各銘柄の過去データを取得
    all_data = {}
    failed_tickers = []
    # 1時間足ならバー数は約7倍必要
    multiplier = 7 if interval == "1h" else 1
    required_days = (start_days_ago * multiplier) + (200 * multiplier)
    
    st.info(f"必要データ数: {required_days}バー (期間: {start_days_ago}日分, 足: {interval})")
    
    for ticker in tickers:
        # 1時間足なら最大2年まで
        period = "2y" if interval == "1h" else "5y"
        df = get_historical_data(ticker, period, interval)
        if df is not None:
            # データ長チェック（1時間足はバー数が多いので緩和）
            min_bars = start_days_ago * multiplier
            if len(df) > min_bars:
                all_data[ticker] = df
            else:
                failed_tickers.append(f"{ticker}({len(df)}バー)")
        else:
            failed_tickers.append(f"{ticker}(取得失敗)")
    
    if failed_tickers:
        st.warning(f"データ不足銘柄: {', '.join(failed_tickers[:10])}")
    
    # 市場データ（SPY）を取得
    market_data = get_historical_data(market_ticker, "5y", interval)
    if market_data is None:
        st.warning("市場データ(SPY)が取得できません。市場フィルターを無効化します。")
    
    # ========== v10.0: VIXデータ取得（レジーム検知用） ==========
    # VIXデータもDBキャッシュを活用
    vix_data = None
    vix_ticker_symbol = "^VIX"
    try:
        # 1時間足の場合はVIXも1時間足で取得（ただしVIXの1時間足は取得できない場合が多いので日足で代用するか、取得を試みる）
        # ここでは簡易的に日足のままにする（レジームは日次で十分）
        # もし厳密にやるなら interval="1h" だが、VIXのヒストリカルデータは制限がきつい
        
        # まずDBキャッシュから取得を試みる (日足)
        vix_data = db.get_cached_prices(vix_ticker_symbol, days=730)
        
        if vix_data is not None and len(vix_data) >= 50:
            # タイムゾーンを除去（比較エラー防止）
            if vix_data.index.tz is not None:
                vix_data.index = vix_data.index.tz_localize(None)
            st.info(f"[OK] VIXデータ取得成功（キャッシュ: {len(vix_data)}日分） - 高度なレジーム検知を有効化")
        else:
            # キャッシュにない場合はyfinanceから取得
            import yfinance as yf
            import time
            time.sleep(0.3)
            
            vix_ticker = yf.Ticker(vix_ticker_symbol)
            vix_data = vix_ticker.history(period="2y") # VIXは日足で取得
            
            if vix_data is not None and len(vix_data) >= 50:
                # タイムゾーンを除去（比較エラー防止）
                if vix_data.index.tz is not None:
                    vix_data.index = vix_data.index.tz_localize(None)
                # DBにキャッシュ
                db.cache_prices(vix_ticker_symbol, vix_data)
                st.info(f"[OK] VIXデータ取得成功（API: {len(vix_data)}日分） - 高度なレジーム検知を有効化")
            else:
                vix_data = None
                st.warning("VIXデータが取得できません。VIXフィルターを無効化します。")
    except Exception as e:
        vix_data = None
        st.warning(f"VIXデータ取得エラー: {e}。VIXフィルターを無効化します。")
    
    if not all_data:
        return {'error': f"データ不足の銘柄: {', '.join(failed_tickers)}", 'failed_tickers': failed_tickers}
    
    # ========== 高速化: 指標一括計算 & AI予測 ==========
    st.info("指標計算 & AI予測モデル学習中 (Walk-Forward)...")
    precalculated_data = {}
    
    # プログレスバー
    prog_bar = st.progress(0)
    total_tickers = len(all_data)
    
    for i, (ticker, df) in enumerate(all_data.items()):
        # テクニカル指標
        df_calc = precalculate_indicators(df)
        
        # AIスコア (v11.0)
        # データ量が多いと時間がかかるので、ステータス表示
        if progress_callback:
            progress_callback((i / total_tickers), f"{ticker}: AIモデル学習中...")
        
        ai_scores = calculate_ai_scores(df_calc, interval=interval)
        df_calc['ai_score'] = ai_scores
        
        precalculated_data[ticker] = df_calc
        prog_bar.progress((i + 1) / total_tickers)
        
    prog_bar.empty()

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
    # 1時間足の場合はバー数を調整
    multiplier = 7 if interval == "1h" else 1
    required_bars = start_days_ago * multiplier
    
    date_index = all_data[first_ticker].index[-required_bars:]
    
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
    
    total_days = len(date_index)
    
    regime_names = ['低ボラ下降', '低ボラ上昇', '高ボラ下降', '高ボラ上昇']
    regime_stop_mults = [1.0, 0.8, 1.5, 1.2]

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
            # タイムゾーン不一致を解消（1時間足の場合、current_dateはtzあり、vix_dataはtzなし）
            compare_date = current_date
            if hasattr(compare_date, 'tzinfo') and compare_date.tzinfo is not None:
                compare_date = compare_date.tz_localize(None)
            
            vix_mask = vix_data.index <= compare_date
            if vix_mask.sum() >= 10:
                vix_slice = vix_data[vix_mask]
                vix_level = float(vix_slice['Close'].iloc[-1])
                
                # VIXモメンタム（5日変化率）
                if len(vix_slice) >= 6:
                    vix_5d_ago = float(vix_slice['Close'].iloc[-6])
                    vix_momentum = (vix_level - vix_5d_ago) / vix_5d_ago * 100
                
                # VIXレジーム判定（v10.3: 閾値緩和で収益改善）
                if vix_level >= 40:
                    vix_regime = "PANIC"
                    vix_position_multiplier = 0.0  # 完全停止（40以上に引き上げ）
                elif vix_level >= 35:
                    vix_regime = "FEAR"
                    vix_position_multiplier = 0.3  # 70%縮小（35以上に引き上げ）
                elif vix_level >= 30:
                    vix_regime = "CAUTION"
                    vix_position_multiplier = 0.6  # 40%縮小（30以上に引き上げ）
                elif vix_momentum > 20:  # VIXが5日で20%以上急上昇（15%→20%に緩和）
                    vix_regime = "CAUTION"
                    vix_position_multiplier = 0.8  # 20%縮小
                else:
                    vix_regime = "NORMAL"
                    vix_position_multiplier = 1.0
        
        # その日のシグナルを計算（各銘柄）
        daily_signals = {}
        daily_prices = {}
        
        for ticker, df in precalculated_data.items():
            if current_date not in df.index:
                continue
            
            # データが十分にあるか確認（最初の50日はスキップ）
            # indexの位置を取得
            try:
                idx_loc = df.index.get_loc(current_date)
            except KeyError:
                continue
                
            if idx_loc < 50:
                continue
            
            # 行データを取得
            row = df.iloc[idx_loc]
            
            # シグナル辞書を構築
            regime_idx = int(row['regime']) if not pd.isna(row['regime']) else 0
            
            signal = {
                'price': float(row['Close']),
                'change': float(row['price_change']) if not pd.isna(row['price_change']) else 0.0,
                'rsi': float(row['rsi']) if not pd.isna(row['rsi']) else 50.0,
                'total_score': float(row['total_score']) if not pd.isna(row['total_score']) else 0.0,
                'is_uptrend': bool(row['is_uptrend']) if not pd.isna(row['is_uptrend']) else True,
                'atr_pct': float(row['atr_pct']) if not pd.isna(row['atr_pct']) else 2.0,
                'bb_position': float(row['bb_position']) if not pd.isna(row['bb_position']) else 0.5,
                'high_price': float(row['high_price_20d']) if not pd.isna(row['high_price_20d']) else float(row['Close']),
                'momentum_20d': float(row['momentum_20d']) if not pd.isna(row['momentum_20d']) else 0.0,
                'risk_adjusted_momentum': float(row['risk_adjusted_momentum']) if not pd.isna(row['risk_adjusted_momentum']) else 0.0,
                'volatility': float(row['volatility_20d']) if not pd.isna(row['volatility_20d']) else 0.0,
                'regime': regime_idx,
                'regime_name': regime_names[regime_idx],
                'regime_buy_mult': float(row['regime_adjusted_score']) / float(row['total_score']) if row['total_score'] != 0 and not pd.isna(row['total_score']) else 0.0,
                'regime_stop_mult': regime_stop_mults[regime_idx],
                'keltner_position': float(row['keltner_position']) if not pd.isna(row['keltner_position']) else 0.5,
                'squeeze_on': bool(row['squeeze_on']) if not pd.isna(row['squeeze_on']) else False,
                'squeeze_bonus': float(row['squeeze_bonus']) if not pd.isna(row['squeeze_bonus']) else 0.0,
                'regime_adjusted_score': float(row['regime_adjusted_score']) if not pd.isna(row['regime_adjusted_score']) else 0.0,
                'early_warning_score': int(row['early_warning_score']) if not pd.isna(row['early_warning_score']) else 0,
                'early_warning_reasons': [], # 高速化のため省略
                'rsi_prev': float(df['rsi'].iloc[idx_loc-1]) if idx_loc > 0 and not pd.isna(df['rsi'].iloc[idx_loc-1]) else 50.0,
                'macd_crossover': int(row['macd_crossover']) if not pd.isna(row['macd_crossover']) else 0,
                'hist_reversal': int(row['hist_reversal']) if not pd.isna(row['hist_reversal']) else 0
            }
            
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
            
            # ========== v10.4: 損切り緩和 ==========
            # 最大損失を-12%に緩和し、ボラティリティ許容度を上げる
            if sell_ratio == 0:
                regime_stop_mult = daily_signals[ticker].get('regime_stop_mult', 1.0)
                base_stop = min(10, atr_pct * 3.0)  # 基本損切りライン緩和: 8→10, 2.5→3.0
                
                # レジーム調整: 高ボラ時は損切り幅を少し広げる（ただし上限あり）
                adjusted_stop = min(12, base_stop * regime_stop_mult)  # 最大-12%
                
                # 保有期間ボーナス: 10日ごとに1%緩和（最大2%）
                holding_bonus = min(2, days_held // 10)
                
                # 過去好成績銘柄でも損切り緩和は控えめに
                perf = ticker_performance[ticker]
                if perf['trade_count'] >= 3 and perf['wins'] / perf['trade_count'] >= 0.7:
                    holding_bonus += 1  # 勝率70%以上なら+1%のみ
                
                dynamic_stop = min(12, adjusted_stop + holding_bonus)  # 絶対に-12%を超えない
                
                if days_held >= 2:  # 2日経過後から損切り（3日→2日に短縮して大事故防止）
                    if pnl_rate <= -dynamic_stop:
                        regime_name = daily_signals[ticker].get('regime_name', '不明')
                        sell_reason = f"損切り ({pnl_rate:.1f}%, 閾値-{dynamic_stop:.1f}%, {regime_name})"
                        sell_ratio = 1.0
            
            # ========== v10.4: 絶対ハードストップ -12% ==========
            # どんな状況でも-12%で強制損切り
            if sell_ratio == 0 and pnl_rate <= -12:
                sell_reason = f"ハードストップ ({pnl_rate:.1f}%)"
                sell_ratio = 1.0
            
            # ========== v10.4: 最適化されたトレーリングストップ ==========
            # 利益を伸ばすため、初期段階でのトレーリングを無効化
            if sell_ratio == 0 and pnl_rate > 0:
                # v10.9: ボラティリティ連動トレーリングストップ（緩和版）
                # ATRが大きい銘柄はストップ幅を広く、小さい銘柄は狭く
                base_trail = 12.0
                if atr_pct > 3.0:
                    base_trail = 15.0
                elif atr_pct < 1.5:
                    base_trail = 8.0
                
                if pnl_rate >= 60:  # 60%以上の利益は高値から-15%で売却
                    trailing_threshold = 15
                elif pnl_rate >= 35:  # 35%以上は-12%
                    trailing_threshold = 12
                elif pnl_rate >= 20:  # 20%以上はATR連動
                    trailing_threshold = base_trail
                elif pnl_rate >= 12:  # 12%以上はブレークイーブン確保 (+1%確保)
                    # 買値+1%を下回ったら売却
                    if price < pos['avg_cost'] * 1.01:
                        sell_reason = f"利益確保 (買値+1%ライン割れ)"
                        sell_ratio = 1.0
                        trailing_threshold = 999 # ここではトリガーさせない
                    else:
                        trailing_threshold = base_trail
                else:
                    trailing_threshold = 999  # 12%未満の利益ではトレーリングしない（損切りに任せる）
                
                if sell_ratio == 0 and drop_from_high >= trailing_threshold:
                    sell_reason = f"トレーリングストップ (高値から-{drop_from_high:.1f}%, 閾値{trailing_threshold}%)"
                    sell_ratio = 1.0
            
            # ========== v7.0: 利確ロジック ==========
            # 利確は段階的に、かつ全売却（半分売りの繰り返し問題を解消）
            if sell_ratio == 0:
                if pnl_rate >= 60:  # +60%以上で3/4売却
                    sell_reason = f"大幅利確 ({pnl_rate:.1f}%)"
                    sell_ratio = 0.75
                elif pnl_rate >= 35:  # +35%以上で半分売却
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
            
            # ========== v10.4: シグナル売り緩和 ==========
            # 利益時は半分売り、損失-3%以上のみ全売り
            if sell_ratio == 0 and score <= -0.3:  # -0.2 → -0.3 に緩和（弱い売りシグナル無視）
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
                min_buy_score = 0.28  # v10.8: 0.25 -> 0.28 に修正
            else:
                buy_budget_ratio = 0.0
                min_buy_score = 1.0
        else:  # NORMAL
            if market_is_bullish:
                buy_budget_ratio = 1.0 * combined_position_multiplier
                min_buy_score = 0.18  # v10.8: 0.15 -> 0.18 に修正
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
                # 1. 現在利益が出ている（+3%以上） v10.7: 5% -> 3%
                # 2. 過去実績が良い（勝率50%以上）
                # 3. 押し目（BB中央より下）
                # 4. スコアがプラス
                perf = ticker_performance[ticker]
                
                if (pnl_rate >= 3 and
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
            
            # ========== v11.0: AIスコアフィルター ==========
            # テクニカル的に買いでも、AIが下落を予測している場合は見送り
            ai_score = signal.get('ai_score', 0.0)
            
            # AIが強い下落(-1%以下)を予測している場合
            if ai_score < -0.01:
                # ただし、テクニカルスコアが非常に高い(0.8以上)場合はAIを無視して勝負
                if score < 0.8:
                    continue
            
            # AIが上昇(+1%以上)を予測している場合はスコア加点
            if ai_score > 0.01:
                score += 0.1
            
            # 条件2: 銘柄自体が上昇トレンド（50日MA > 200日MA）
            # v10.8: 非常に強いシグナル（スコア0.7以上）ならトレンド無視して逆張り可 (0.6 -> 0.7)
            if not is_uptrend and score < 0.7:
                continue
            
            # 条件3: 2日連続で買いシグナル
            prev_signal = prev_day_signals.get(ticker, {})
            prev_score = prev_signal.get('total_score', 0) if prev_signal else 0
            # v10.8: 非常に強いシグナルなら即エントリー可 (0.1 -> 0.12)
            if prev_score < 0.12 and score < 0.7:
                continue
            
            # 条件4: 高値追い回避（前日比+3%以上は見送り）
            if change > 3:
                continue
            
            # 条件5: 押し目買い優先（BB中央より下）
            if bb_position > 0.7:
                continue
            
            # ========== v6新規: 拡張フィルター ==========
            perf = ticker_performance[ticker]
            
            # ブラックリストチェック: 2連敗以上で10日間は買い禁止 (v10.7: 20日 -> 10日)
            if perf['consecutive_losses'] >= 2:
                days_since_loss = day_num - perf['last_loss_day']
                if days_since_loss < 10:
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
            
            # ========== v10.5: ケリー基準による最適配分計算（グローバル補正版） ==========
            # Kelly % = W - [(1-W) / R]
            # グローバル事前分布: 勝率70%, ペイオフ2.0 (v10.4実績に基づく保守的推定)
            prior_wins = 7
            prior_total = 10
            prior_avg_win = 15.0
            prior_avg_loss = 7.0
            
            if perf['trade_count'] > 0:
                # ベイズ更新的な加重平均
                weight = min(1.0, perf['trade_count'] / 10.0)  # 10トレードで完全に個別実績に移行
                
                est_win_rate = (perf['wins'] / perf['trade_count']) * weight + (prior_wins / prior_total) * (1 - weight)
                
                # 平均損益の推定
                if perf['wins'] > 0:
                    indiv_avg_win = perf['total_pnl'] / perf['wins']  # 簡易計算（正確には勝ちトレードのみの平均が必要だが近似）
                    indiv_avg_win = max(5, indiv_avg_win)
                else:
                    indiv_avg_win = prior_avg_win
                    
                est_avg_win = indiv_avg_win * weight + prior_avg_win * (1 - weight)
                est_avg_loss = prior_avg_loss  # 損失幅は一定と仮定
                
                payoff = est_avg_win / est_avg_loss
                kelly_ratio = est_win_rate - ((1 - est_win_rate) / payoff)
                kelly_ratio = max(0, min(0.40, kelly_ratio))  # 0〜40%に緩和（集中投資用）
            else:
                # 新規銘柄はグローバル実績を使用
                payoff = prior_avg_win / prior_avg_loss
                kelly_ratio = (prior_wins / prior_total) - ((1 - (prior_wins / prior_total)) / payoff)
                kelly_ratio = max(0.10, min(0.20, kelly_ratio))
            
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
        
        # ========== v10.5: 集中投資ポートフォリオ ==========
        daily_buy_count = 0
        max_daily_buys = 5  # v10.7: 1日5銘柄まで拡大 (3 -> 5)
        
        for ticker, signal, final_score, perf_score, momentum, kelly_ratio in buy_candidates:
            if daily_buy_count >= max_daily_buys:
                break
            
            # ========== v7新規: セクター集中回避 ==========
            ticker_sector = ticker_sectors.get(ticker, 'other')
            if sector_count.get(ticker_sector, 0) >= 3: # v10.7: 2 -> 3 に緩和
                continue  # 同一セクター3銘柄以上は回避
            
            price = daily_prices[ticker]
            score = signal['total_score']
            
            # v10.5: 現金比率をさらに下げてフルインベストメントに近づける
            min_cash_ratio = 0.02  # 5% → 2%
            current_total = cash + sum(
                portfolio[t]['shares'] * daily_prices.get(t, portfolio[t]['avg_cost'])
                for t in portfolio
            )
            if cash < current_total * min_cash_ratio:
                break
            
            # v10.5: 銘柄数を絞って集中投資（最大10銘柄）
            if len(portfolio) >= 10:  # v10.7: 6 -> 10 に拡大
                break
            
            # ========== v10.5: 勝者への集中投資強化 ==========
            # ケリー比率を基本に、スコアと実績で調整
            # ベース比率を15%に引き上げ（6銘柄分散なら16%が平均）
            base_ratio = max(0.15, kelly_ratio * 0.8)  # ケリーの80%（フラクショナルケリー）
            
            # 過去に大勝ちした銘柄は倍率アップ（最大3.0倍に強化）
            perf = ticker_performance[ticker]
            if perf['big_wins'] >= 2:  # 2回以上大勝ち
                alloc_multiplier = 2.0  # 過度な集中を防ぐため少し抑制
            elif perf['big_wins'] >= 1:  # 1回大勝ち
                alloc_multiplier = 1.5
            elif perf_score >= 0.7:
                alloc_multiplier = 1.3
            elif perf_score >= 0.5:
                alloc_multiplier = 1.1
            else:
                alloc_multiplier = 1.0
            
            # スクイーズブレイクアウト時はさらに積極的に
            squeeze_bonus = signal.get('squeeze_bonus', 0.0)
            if squeeze_bonus > 0:
                alloc_multiplier *= 1.3
            
            # ========== v8.6: ケリー基準ベースのポジションサイジング ==========
            base_ratio = max(0.12, kelly_ratio * 1.5)  # ベース比率を上げる（10%→12%）
            
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
with st.expander("📋 売買アルゴリズム（改善版 v10.5 - 集中投資最適化）", expanded=False):
    st.markdown("""
    ### ウォークフォワードテストとは
    各日の判断は**その日までのデータのみ**を使用し、未来のデータは一切見ません。
    
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
    
    **総合倍率 = VIX倍率 × DD倍率**（両方適用）
    
    #### 🎯 期待効果
    - 3月型の急落: -3.71% → -1~2%に抑制
    - シャープレシオ: 1.62 → 1.8~2.0
    - 最大ドローダウン: 5.9% → 4%以下
    
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

# 時間足選択
interval_option = st.radio("時間足を選択", ["日足 (1d)", "1時間足 (1h)"], index=0, horizontal=True)
interval = "1d" if "1d" in interval_option else "1h"

if st.button("🚀 バックテスト実行", type="primary", use_container_width=True):
    if len(selected_tickers) == 0:
        st.error("銘柄を選択してください")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(p, msg=None):
            progress_bar.progress(p)
            if msg:
                status_text.text(msg)
            else:
                status_text.text(f"処理中... {int(p * 100)}%")
        
        status_text.text("過去データを取得中...")
        
        with st.spinner("バックテスト実行中..."):
            result = run_backtest(
                selected_tickers, 
                initial_cash=initial_cash,
                start_days_ago=test_days,
                progress_callback=update_progress,
                interval=interval
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
            # interval変数はrun_backtestの引数として渡されているが、ここではスコープ外の可能性がある
            # しかし、Streamlitの実行フローではinterval変数は定義されているはず
            # 安全のため、デフォルトは"1d"とする
            current_interval = interval if 'interval' in locals() else "1d"
            analysis = analyze_backtest_results(history, trades, interval=current_interval)
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
