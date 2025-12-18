"""
リアルタイムシグナル検出ページ
============================
現在の市場データからAIシグナルを検出し、通知を送信
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.xgb_model import StockPredictorXGB
from services.line_notify import LineNotifyService

st.set_page_config(
    page_title="🔔 リアルタイムシグナル",
    page_icon="🔔",
    layout="wide"
)

st.title("🔔 リアルタイムシグナル検出")
st.markdown("---")


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を計算"""
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ボリンジャーバンド
    df['bb_mid'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 移動平均
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    
    # リターン
    df['return_1d'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    
    # ボラティリティ
    df['volatility'] = df['return_1d'].rolling(20).std()
    
    # 出来高変化
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df


def detect_signals(df: pd.DataFrame, ticker: str) -> dict:
    """シグナルを検出"""
    if len(df) < 50:
        return None
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = {
        'ticker': ticker,
        'price': latest['Close'],
        'date': latest.name if hasattr(latest, 'name') else datetime.now(),
        'buy_signals': [],
        'sell_signals': [],
        'indicators': {
            'rsi': latest.get('rsi', 50),
            'macd': latest.get('macd', 0),
            'macd_hist': latest.get('macd_hist', 0),
            'bb_position': latest.get('bb_position', 0.5),
            'volume_ratio': latest.get('volume_ratio', 1),
        }
    }
    
    # 買いシグナル条件
    # 1. RSI < 30 (売られすぎ)
    if latest.get('rsi', 50) < 30:
        signals['buy_signals'].append(f"RSI売られすぎ ({latest['rsi']:.1f})")
    
    # 2. MACD ゴールデンクロス
    if prev.get('macd', 0) < prev.get('macd_signal', 0) and \
       latest.get('macd', 0) > latest.get('macd_signal', 0):
        signals['buy_signals'].append("MACDゴールデンクロス")
    
    # 3. ボリンジャーバンド下限タッチ
    if latest.get('bb_position', 0.5) < 0.1:
        signals['buy_signals'].append("BB下限接近")
    
    # 4. 出来高急増 + 上昇
    if latest.get('volume_ratio', 1) > 2 and latest.get('return_1d', 0) > 0.02:
        signals['buy_signals'].append("出来高急増+上昇")
    
    # 売りシグナル条件
    # 1. RSI > 70 (買われすぎ)
    if latest.get('rsi', 50) > 70:
        signals['sell_signals'].append(f"RSI買われすぎ ({latest['rsi']:.1f})")
    
    # 2. MACD デッドクロス
    if prev.get('macd', 0) > prev.get('macd_signal', 0) and \
       latest.get('macd', 0) < latest.get('macd_signal', 0):
        signals['sell_signals'].append("MACDデッドクロス")
    
    # 3. ボリンジャーバンド上限タッチ
    if latest.get('bb_position', 0.5) > 0.9:
        signals['sell_signals'].append("BB上限接近")
    
    return signals


def calculate_ai_score(df: pd.DataFrame, use_gpu: bool = True) -> float:
    """AIスコアを計算（簡易版）"""
    if len(df) < 100:
        return 50.0
    
    # 特徴量作成
    features = ['rsi', 'macd_hist', 'bb_position', 'return_1d', 'return_5d', 'volatility', 'volume_ratio']
    
    df_clean = df.dropna(subset=features)
    if len(df_clean) < 50:
        return 50.0
    
    # 最新のデータポイントに対してスコアリング
    latest = df_clean.iloc[-1]
    
    # シンプルなスコアリング（0-100）
    score = 50.0
    
    # RSI: 30以下で+15、70以上で-15
    rsi = latest.get('rsi', 50)
    if rsi < 30:
        score += 15
    elif rsi > 70:
        score -= 15
    
    # MACD: プラスで+10、マイナスで-10
    macd_hist = latest.get('macd_hist', 0)
    if macd_hist > 0:
        score += 10
    else:
        score -= 10
    
    # BB位置: 0.2以下で+10、0.8以上で-10
    bb_pos = latest.get('bb_position', 0.5)
    if bb_pos < 0.2:
        score += 10
    elif bb_pos > 0.8:
        score -= 10
    
    # モメンタム
    ret_5d = latest.get('return_5d', 0)
    score += min(max(ret_5d * 100, -15), 15)
    
    return max(0, min(100, score))


# サイドバー設定
st.sidebar.header("⚙️ 設定")

# 銘柄リスト
default_tickers = [
    "7203.T", "6758.T", "9984.T", "8306.T", "6902.T",
    "7267.T", "6501.T", "8035.T", "6861.T", "9433.T"
]

ticker_input = st.sidebar.text_area(
    "監視銘柄（1行1銘柄）",
    value="\n".join(default_tickers),
    height=200
)
tickers = [t.strip() for t in ticker_input.strip().split("\n") if t.strip()]

# 通知設定
st.sidebar.header("🔔 通知設定")
enable_line = st.sidebar.checkbox("LINE通知を有効化", value=False)
notify_buy = st.sidebar.checkbox("買いシグナルを通知", value=True)
notify_sell = st.sidebar.checkbox("売りシグナルを通知", value=True)

# GPU設定
use_gpu = st.sidebar.checkbox("GPU使用（XGBoost CUDA）", value=True)

# 実行ボタン
if st.button("🔍 シグナル検出実行", type="primary", use_container_width=True):
    
    line_service = LineNotifyService() if enable_line else None
    
    all_signals = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"分析中: {ticker} ({i+1}/{len(tickers)})")
        progress_bar.progress((i + 1) / len(tickers))
        
        try:
            # データ取得
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")
            
            if len(df) < 50:
                continue
            
            # 指標計算
            df = calculate_indicators(df)
            
            # シグナル検出
            signals = detect_signals(df, ticker)
            if signals:
                signals['ai_score'] = calculate_ai_score(df, use_gpu)
                all_signals.append(signals)
                
                # LINE通知
                if line_service:
                    if notify_buy and signals['buy_signals']:
                        line_service.send_buy_signal(
                            ticker=ticker,
                            price=signals['price'],
                            ai_score=signals['ai_score'],
                            reason=", ".join(signals['buy_signals']),
                            additional_info=signals['indicators']
                        )
                    
                    if notify_sell and signals['sell_signals']:
                        line_service.send_sell_signal(
                            ticker=ticker,
                            price=signals['price'],
                            profit_rate=0,  # 保有していないので0
                            reason=", ".join(signals['sell_signals'])
                        )
        
        except Exception as e:
            st.warning(f"{ticker}: データ取得エラー - {e}")
    
    progress_bar.empty()
    status_text.empty()
    
    # 結果表示
    st.markdown("---")
    st.subheader("📊 検出結果")
    
    # 買いシグナルがある銘柄
    buy_signals = [s for s in all_signals if s['buy_signals']]
    sell_signals = [s for s in all_signals if s['sell_signals']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 買いシグナル")
        if buy_signals:
            for sig in sorted(buy_signals, key=lambda x: -x['ai_score']):
                with st.expander(f"**{sig['ticker']}** - AIスコア: {sig['ai_score']:.0f}", expanded=True):
                    st.write(f"💰 現在値: ¥{sig['price']:,.0f}")
                    st.write(f"📝 シグナル: {', '.join(sig['buy_signals'])}")
                    st.write(f"📊 RSI: {sig['indicators']['rsi']:.1f}")
                    st.write(f"📈 MACD: {sig['indicators']['macd']:.2f}")
        else:
            st.info("買いシグナルはありません")
    
    with col2:
        st.markdown("### 🔴 売りシグナル")
        if sell_signals:
            for sig in sorted(sell_signals, key=lambda x: x['ai_score']):
                with st.expander(f"**{sig['ticker']}** - AIスコア: {sig['ai_score']:.0f}", expanded=True):
                    st.write(f"💰 現在値: ¥{sig['price']:,.0f}")
                    st.write(f"📝 シグナル: {', '.join(sig['sell_signals'])}")
                    st.write(f"📊 RSI: {sig['indicators']['rsi']:.1f}")
                    st.write(f"📈 MACD: {sig['indicators']['macd']:.2f}")
        else:
            st.info("売りシグナルはありません")
    
    # サマリーテーブル
    st.markdown("---")
    st.subheader("📋 全銘柄サマリー")
    
    if all_signals:
        summary_data = []
        for sig in all_signals:
            summary_data.append({
                '銘柄': sig['ticker'],
                '現在値': f"¥{sig['price']:,.0f}",
                'AIスコア': f"{sig['ai_score']:.0f}",
                'RSI': f"{sig['indicators']['rsi']:.1f}",
                '買いシグナル': len(sig['buy_signals']),
                '売りシグナル': len(sig['sell_signals']),
                'シグナル詳細': ', '.join(sig['buy_signals'] + sig['sell_signals']) or '-'
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
    
    # 通知送信数
    if enable_line and line_service:
        st.success(f"📨 LINE通知送信: 買い {len(buy_signals)}件, 売り {len(sell_signals)}件")

# 使い方説明
with st.expander("📖 使い方"):
    st.markdown("""
    ### シグナル検出の仕組み
    
    **買いシグナル条件:**
    - RSI < 30 (売られすぎ)
    - MACDゴールデンクロス
    - ボリンジャーバンド下限接近
    - 出来高急増 + 株価上昇
    
    **売りシグナル条件:**
    - RSI > 70 (買われすぎ)
    - MACDデッドクロス
    - ボリンジャーバンド上限接近
    
    ### LINE通知設定
    
    1. [LINE Notify](https://notify-bot.line.me/) にアクセス
    2. 「トークンを発行する」でアクセストークンを取得
    3. プロジェクトの `.env` ファイルに以下を追加:
       ```
       LINE_NOTIFY_TOKEN=your_token_here
       ```
    4. 「LINE通知を有効化」にチェック
    """)
