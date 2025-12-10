"""
シグナルヒートマップページ
========================
ウォッチリスト全銘柄の売買シグナルをヒートマップで可視化
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

st.set_page_config(
    page_title="🔥 シグナルヒートマップ",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 シグナルヒートマップ")
st.markdown("ウォッチリスト全銘柄の売買シグナルを一覧表示")

# データベース接続（新規インスタンス）
db = DatabaseManager()

# ==================== キャッシュ付きテクニカル計算 ====================

@st.cache_data(ttl=3600)  # 1時間キャッシュ
def calculate_signals_batch(tickers: tuple, use_cache: bool = True) -> dict:
    """複数銘柄のシグナルを並列計算"""
    results = {}
    
    def process_ticker(ticker):
        try:
            return ticker, calculate_single_signal(ticker, use_cache)
        except Exception as e:
            return ticker, None
    
    # 並列処理（最大10スレッド）
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_ticker, t): t for t in tickers}
        for future in as_completed(futures):
            ticker, result = future.result()
            if result:
                results[ticker] = result
    
    return results


@st.cache_data(ttl=1800)  # 30分キャッシュ
def get_ticker_detail(ticker: str) -> dict:
    """銘柄の詳細テクニカルデータを取得"""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(period="6mo")
        
        if df is None or len(df) < 50:
            return None
        
        # テクニカル指標計算
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # 移動平均
        sma5 = df['Close'].rolling(window=5).mean()
        sma20 = df['Close'].rolling(window=20).mean()
        sma50 = df['Close'].rolling(window=50).mean()
        
        # ボリンジャーバンド
        bb_std = df['Close'].rolling(window=20).std()
        bb_upper = sma20 + 2 * bb_std
        bb_lower = sma20 - 2 * bb_std
        
        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return {
            'df': df,
            'rsi': rsi,
            'sma5': sma5,
            'sma20': sma20,
            'sma50': sma50,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist
        }
    except Exception as e:
        print(f"Error getting detail for {ticker}: {e}")
        return None


def calculate_single_signal(ticker: str, use_cache: bool = True) -> dict:
    """単一銘柄のシグナル計算"""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # キャッシュからデータ取得を試みる
    df = None
    data_source = "API"
    
    if use_cache:
        cached = db.get_cached_prices(ticker, days=100)
        if cached is not None and len(cached) >= 50:
            df = cached
            data_source = "Cache"
    
    # キャッシュがなければAPI取得
    if df is None:
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")
            if df is None or len(df) < 50:
                return None
            data_source = "API"
        except:
            return None
    
    if len(df) < 50:
        return None
    
    # 最新価格
    current_price = float(df['Close'].iloc[-1])
    prev_price = float(df['Close'].iloc[-2])
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    # ==================== テクニカル指標計算 ====================
    
    # RSI (14日)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = float(rsi.iloc[-1])
    
    # RSIシグナル: -1 (売り) ~ +1 (買い)
    if rsi_value < 30:
        rsi_signal = 1.0  # 売られすぎ→買い
    elif rsi_value > 70:
        rsi_signal = -1.0  # 買われすぎ→売り
    else:
        rsi_signal = (50 - rsi_value) / 50  # 中間値
    
    # 移動平均 (5, 20, 50日)
    sma5 = df['Close'].rolling(window=5).mean()
    sma20 = df['Close'].rolling(window=20).mean()
    sma50 = df['Close'].rolling(window=50).mean()
    
    sma5_val = float(sma5.iloc[-1])
    sma20_val = float(sma20.iloc[-1])
    sma50_val = float(sma50.iloc[-1]) if len(df) >= 50 else sma20_val
    
    # MAシグナル: 短期が長期を上回っていれば買い
    ma_signal = 0.0
    if current_price > sma5_val:
        ma_signal += 0.3
    if sma5_val > sma20_val:
        ma_signal += 0.4
    if sma20_val > sma50_val:
        ma_signal += 0.3
    ma_signal = (ma_signal - 0.5) * 2  # -1 ~ +1 に正規化
    
    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    macd_val = float(macd_line.iloc[-1])
    macd_signal_val = float(signal_line.iloc[-1])
    macd_hist_val = float(macd_hist.iloc[-1])
    
    # MACDシグナル
    if macd_val > macd_signal_val and macd_hist_val > 0:
        macd_signal = 1.0
    elif macd_val < macd_signal_val and macd_hist_val < 0:
        macd_signal = -1.0
    else:
        macd_signal = macd_hist_val / (abs(macd_hist_val) + 0.01) * 0.5
    
    # ボリンジャーバンド
    bb_middle = sma20_val
    bb_std = df['Close'].rolling(window=20).std().iloc[-1]
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # BBシグナル: 下限に近ければ買い、上限に近ければ売り
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    bb_signal = (0.5 - bb_position) * 2  # -1 ~ +1
    
    # 出来高トレンド
    vol_sma = df['Volume'].rolling(window=20).mean()
    vol_ratio = float(df['Volume'].iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1.0
    
    # 出来高シグナル: 出来高増加 + 価格上昇 = 買い
    if vol_ratio > 1.5 and price_change > 0:
        vol_signal = 1.0
    elif vol_ratio > 1.5 and price_change < 0:
        vol_signal = -1.0
    else:
        vol_signal = 0.0
    
    # ==================== 総合スコア計算 ====================
    
    # 重み付け平均
    weights = {
        'rsi': 0.20,
        'ma': 0.25,
        'macd': 0.25,
        'bb': 0.15,
        'volume': 0.15
    }
    
    total_score = (
        rsi_signal * weights['rsi'] +
        ma_signal * weights['ma'] +
        macd_signal * weights['macd'] +
        bb_signal * weights['bb'] +
        vol_signal * weights['volume']
    )
    
    return {
        'ticker': ticker,
        'price': current_price,
        'change': price_change,
        'rsi': rsi_value,
        'rsi_signal': rsi_signal,
        'ma_signal': ma_signal,
        'macd_signal': macd_signal,
        'bb_signal': bb_signal,
        'vol_signal': vol_signal,
        'total_score': total_score,
        'data_source': data_source
    }


# ==================== メインUI ====================

# サイドバー設定
st.sidebar.header("⚙️ 設定")

cache_mode = st.sidebar.radio(
    "データ取得モード",
    ["⚡ 高速（DBキャッシュ）", "🔄 通常（価格キャッシュ）", "🌐 最新（API優先）"],
    index=0,
    help="高速モードは事前計算済みデータを使用"
)

auto_refresh = st.sidebar.checkbox("自動更新", value=False)

if auto_refresh:
    refresh_interval = st.sidebar.slider("更新間隔（秒）", 30, 300, 60)
    st.sidebar.info(f"⏱️ {refresh_interval}秒ごとに更新")

# ウォッチリスト取得
watchlist = db.get_watchlist()

if not watchlist:
    st.warning("📭 ウォッチリストが空です。先に銘柄を追加してください。")
    st.stop()

# 銘柄フィルタ
st.sidebar.subheader("📊 フィルタ")
markets = list(set(w.get('market', 'その他') or 'その他' for w in watchlist))
selected_markets = st.sidebar.multiselect("マーケット", markets, default=markets)

# フィルタ適用
filtered_tickers = [
    w['ticker'] for w in watchlist 
    if (w.get('market', 'その他') or 'その他') in selected_markets
]

st.sidebar.metric("対象銘柄数", len(filtered_tickers))

# DBキャッシュからの高速読み込み
def load_from_db_cache(tickers: list, max_age_minutes: int = 30) -> dict:
    """DBキャッシュから高速読み込み"""
    cached = db.get_signal_cache(max_age_minutes=max_age_minutes)
    return {c['ticker']: c for c in cached if c['ticker'] in tickers}

# 計算実行
col1, col2 = st.columns([1, 1])
with col1:
    refresh_btn = st.button("🔄 シグナル更新", type="primary")
with col2:
    force_refresh = st.button("🔃 強制再計算", help="キャッシュを無視して再計算")

if refresh_btn or force_refresh or 'signal_data' not in st.session_state:
    with st.spinner(f"📊 {len(filtered_tickers)}銘柄のシグナルを計算中..."):
        start_time = datetime.now()
        
        # 高速モード: DBキャッシュを最初にチェック
        if cache_mode == "⚡ 高速（DBキャッシュ）" and not force_refresh:
            signal_data = load_from_db_cache(filtered_tickers, max_age_minutes=30)
            missing_tickers = [t for t in filtered_tickers if t not in signal_data]
            
            if missing_tickers:
                # 不足分のみ計算
                use_cache = True
                new_signals = calculate_signals_batch(tuple(missing_tickers), use_cache)
                signal_data.update(new_signals)
                # DBに保存
                db.save_signals_batch(new_signals)
        else:
            # 通常/最新モード
            use_cache = (cache_mode != "🌐 最新（API優先）")
            signal_data = calculate_signals_batch(tuple(filtered_tickers), use_cache)
            # DBに保存
            db.save_signals_batch(signal_data)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        st.session_state['signal_data'] = signal_data
        st.session_state['last_update'] = datetime.now()
        
        st.success(f"✅ {len(signal_data)}銘柄を {elapsed:.1f}秒で計算完了")

# データ表示
if 'signal_data' in st.session_state and st.session_state['signal_data']:
    signal_data = st.session_state['signal_data']
    
    # 最終更新時刻
    if 'last_update' in st.session_state:
        st.caption(f"🕐 最終更新: {st.session_state['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # データフレーム作成
    df_signals = pd.DataFrame(signal_data.values())
    
    # 銘柄名を追加
    ticker_names = {w['ticker']: w.get('name', w['ticker']) for w in watchlist}
    df_signals['name'] = df_signals['ticker'].map(ticker_names)
    
    # ==================== サマリー ====================
    st.subheader("📊 シグナルサマリー")
    
    col1, col2, col3, col4 = st.columns(4)
    
    strong_buy = len(df_signals[df_signals['total_score'] > 0.5])
    buy = len(df_signals[(df_signals['total_score'] > 0) & (df_signals['total_score'] <= 0.5)])
    sell = len(df_signals[(df_signals['total_score'] < 0) & (df_signals['total_score'] >= -0.5)])
    strong_sell = len(df_signals[df_signals['total_score'] < -0.5])
    
    col1.metric("🟢 強い買い", strong_buy)
    col2.metric("🔵 買い", buy)
    col3.metric("🟠 売り", sell)
    col4.metric("🔴 強い売り", strong_sell)
    
    st.divider()
    
    # ==================== ヒートマップ ====================
    st.subheader("🔥 シグナルヒートマップ（行をクリックで銘柄選択）")
    
    # Plotly遅延ロード
    import plotly.graph_objects as go
    import plotly.express as px
    
    # ソート
    sort_by = st.selectbox("ソート", ["総合スコア", "RSI", "価格変動率"], index=0)
    if sort_by == "総合スコア":
        df_signals = df_signals.sort_values('total_score', ascending=False)
    elif sort_by == "RSI":
        df_signals = df_signals.sort_values('rsi')
    else:
        df_signals = df_signals.sort_values('change', ascending=False)
    
    # df_signalsのインデックスをリセット
    df_signals = df_signals.reset_index(drop=True)
    
    # ヒートマップ用データフレーム作成
    heatmap_df = df_signals[['ticker', 'rsi_signal', 'ma_signal', 'macd_signal', 'bb_signal', 'vol_signal', 'total_score']].copy()
    heatmap_df.columns = ['銘柄', 'RSI', 'MA', 'MACD', 'BB', '出来高', '総合']
    
    # スタイル関数（-1～+1を赤～緑にマッピング）
    def color_signal(val):
        if pd.isna(val) or not isinstance(val, (int, float)):
            return ''
        # -1: 赤, 0: 白, +1: 緑
        if val > 0:
            intensity = min(abs(val), 1) * 255
            return f'background-color: rgba(0, {int(intensity)}, 0, 0.7); color: white'
        elif val < 0:
            intensity = min(abs(val), 1) * 255
            return f'background-color: rgba({int(intensity)}, 0, 0, 0.7); color: white'
        else:
            return 'background-color: white'
    
    # スタイル適用
    styled_heatmap = heatmap_df.style.applymap(
        color_signal, 
        subset=['RSI', 'MA', 'MACD', 'BB', '出来高', '総合']
    ).format({
        'RSI': '{:.2f}',
        'MA': '{:.2f}',
        'MACD': '{:.2f}',
        'BB': '{:.2f}',
        '出来高': '{:.2f}',
        '総合': '{:.2f}'
    })
    
    # クリック可能なDataFrameとして表示
    clicked_heatmap = st.dataframe(
        styled_heatmap,
        use_container_width=True,
        hide_index=True,
        height=min(600, len(heatmap_df) * 35 + 40),
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # クリックされた行から銘柄を取得
    if clicked_heatmap.selection and clicked_heatmap.selection.rows:
        selected_row_idx = clicked_heatmap.selection.rows[0]
        clicked_ticker = heatmap_df.iloc[selected_row_idx]['銘柄']
        if clicked_ticker != st.session_state.get('selected_ticker'):
            st.session_state['selected_ticker'] = clicked_ticker
            st.rerun()
    
    # ==================== 銘柄詳細表示 ====================
    st.divider()
    st.subheader("🔍 銘柄詳細トレンド")
    
    # 銘柄選択用のデータ
    ticker_list = df_signals['ticker'].tolist()
    ticker_names_map = {t: ticker_names.get(t, '') for t in ticker_list}
    
    # 初期値設定
    if 'selected_ticker' not in st.session_state or st.session_state['selected_ticker'] not in ticker_list:
        st.session_state['selected_ticker'] = ticker_list[0] if ticker_list else None
    
    # 現在選択中の銘柄
    selected_ticker = st.session_state.get('selected_ticker')
    
    if selected_ticker and selected_ticker in ticker_list:
        st.success(f"📌 選択中: **{selected_ticker}** - {ticker_names_map.get(selected_ticker, '')}　（ヒートマップ横のボタン・テーブル・TOP5から変更可能）")
        
        # 詳細データ取得
        with st.spinner(f"{selected_ticker} のトレンドを取得中..."):
            detail_data = get_ticker_detail(selected_ticker)
        
        if detail_data:
            # 基本情報
            col1, col2, col3, col4 = st.columns(4)
            
            signal_rows = df_signals[df_signals['ticker'] == selected_ticker]
            if len(signal_rows) == 0:
                st.error("データが見つかりません")
                st.stop()
            signal_row = signal_rows.iloc[0]
            
            col1.metric(
                "現在値", 
                f"${signal_row['price']:.2f}" if pd.notna(signal_row['price']) else "-",
                f"{signal_row['change']:+.2f}%" if pd.notna(signal_row['change']) else None
            )
            col2.metric("RSI", f"{signal_row['rsi']:.1f}" if pd.notna(signal_row['rsi']) else "-")
            col3.metric("総合スコア", f"{signal_row['total_score']:+.2f}" if pd.notna(signal_row['total_score']) else "-")
            
            # 判定
            score = signal_row['total_score'] if pd.notna(signal_row['total_score']) else 0
            if score > 0.5:
                col4.metric("判定", "🟢 強い買い")
            elif score > 0:
                col4.metric("判定", "🔵 買い")
            elif score > -0.5:
                col4.metric("判定", "🟠 売り")
            else:
                col4.metric("判定", "🔴 強い売り")
            
            # チャートタブ
            tab1, tab2, tab3 = st.tabs(["📈 価格チャート", "📊 テクニカル指標", "📉 シグナル履歴"])
            
            with tab1:
                # 価格チャート + 移動平均
                fig_price = go.Figure()
                
                # ローソク足
                fig_price.add_trace(go.Candlestick(
                    x=detail_data['df'].index,
                    open=detail_data['df']['Open'],
                    high=detail_data['df']['High'],
                    low=detail_data['df']['Low'],
                    close=detail_data['df']['Close'],
                    name='価格'
                ))
                
                # 移動平均線
                fig_price.add_trace(go.Scatter(
                    x=detail_data['df'].index,
                    y=detail_data['sma5'],
                    name='SMA5',
                    line=dict(color='orange', width=1)
                ))
                fig_price.add_trace(go.Scatter(
                    x=detail_data['df'].index,
                    y=detail_data['sma20'],
                    name='SMA20',
                    line=dict(color='blue', width=1)
                ))
                fig_price.add_trace(go.Scatter(
                    x=detail_data['df'].index,
                    y=detail_data['sma50'],
                    name='SMA50',
                    line=dict(color='purple', width=1)
                ))
                
                # ボリンジャーバンド
                fig_price.add_trace(go.Scatter(
                    x=detail_data['df'].index,
                    y=detail_data['bb_upper'],
                    name='BB上限',
                    line=dict(color='gray', width=1, dash='dash'),
                    opacity=0.5
                ))
                fig_price.add_trace(go.Scatter(
                    x=detail_data['df'].index,
                    y=detail_data['bb_lower'],
                    name='BB下限',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    opacity=0.5
                ))
                
                fig_price.update_layout(
                    title=f"{selected_ticker} 価格チャート",
                    height=500,
                    xaxis_rangeslider_visible=False,
                    hovermode='x unified'
                )
                st.plotly_chart(fig_price, use_container_width=True)
            
            with tab2:
                # RSI
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=detail_data['df'].index,
                    y=detail_data['rsi'],
                    name='RSI',
                    line=dict(color='purple')
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="買われすぎ")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="売られすぎ")
                fig_rsi.update_layout(title="RSI (14日)", height=250, yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig_rsi, use_container_width=True)
                
                # MACD
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=detail_data['df'].index,
                    y=detail_data['macd'],
                    name='MACD',
                    line=dict(color='blue')
                ))
                fig_macd.add_trace(go.Scatter(
                    x=detail_data['df'].index,
                    y=detail_data['macd_signal'],
                    name='シグナル',
                    line=dict(color='orange')
                ))
                fig_macd.add_trace(go.Bar(
                    x=detail_data['df'].index,
                    y=detail_data['macd_hist'],
                    name='ヒストグラム',
                    marker_color=['green' if v >= 0 else 'red' for v in detail_data['macd_hist']]
                ))
                fig_macd.update_layout(title="MACD", height=250)
                st.plotly_chart(fig_macd, use_container_width=True)
                
                # 出来高
                fig_vol = go.Figure()
                colors = ['green' if detail_data['df']['Close'].iloc[i] >= detail_data['df']['Open'].iloc[i] 
                          else 'red' for i in range(len(detail_data['df']))]
                fig_vol.add_trace(go.Bar(
                    x=detail_data['df'].index,
                    y=detail_data['df']['Volume'],
                    name='出来高',
                    marker_color=colors
                ))
                fig_vol.update_layout(title="出来高", height=200)
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with tab3:
                # 各指標のシグナル詳細
                st.markdown("### 📊 シグナル詳細")
                
                signal_details = [
                    ("RSI", signal_row['rsi_signal'], f"RSI = {signal_row['rsi']:.1f}"),
                    ("移動平均", signal_row['ma_signal'], "短期>長期なら買い"),
                    ("MACD", signal_row['macd_signal'], "MACD>シグナルなら買い"),
                    ("ボリンジャーバンド", signal_row['bb_signal'], "下限付近なら買い"),
                    ("出来高", signal_row['vol_signal'], "出来高増+価格上昇なら買い"),
                ]
                
                for name, value, desc in signal_details:
                    col1, col2, col3 = st.columns([2, 2, 4])
                    col1.write(f"**{name}**")
                    
                    # スコアバー
                    if value > 0:
                        bar = "🟩" * int(value * 5) + "⬜" * (5 - int(value * 5))
                    else:
                        bar = "⬜" * (5 + int(value * 5)) + "🟥" * (-int(value * 5))
                    col2.write(f"{bar} {value:+.2f}")
                    col3.caption(desc)
                
                st.markdown("---")
                st.markdown(f"**総合スコア: {signal_row['total_score']:+.2f}**")
        else:
            st.error("データの取得に失敗しました")
    
    st.divider()
    
    # ==================== 詳細テーブル ====================
    st.subheader("📋 詳細データ（銘柄をクリックで詳細表示）")
    
    # data_sourceがない場合の対応
    if 'data_source' not in df_signals.columns:
        df_signals['data_source'] = 'Cache'
    
    # 表示用データフレーム
    display_df = df_signals[['ticker', 'name', 'price', 'change', 'rsi', 'total_score']].copy()
    display_df.columns = ['銘柄', '銘柄名', '現在値', '変動率%', 'RSI', '総合スコア']
    
    # スコアに応じた判定
    def get_signal_label(score):
        if score > 0.5:
            return '🟢 強い買い'
        elif score > 0:
            return '🔵 買い'
        elif score > -0.5:
            return '🟠 売り'
        else:
            return '🔴 強い売り'
    
    display_df['判定'] = df_signals['total_score'].apply(get_signal_label)
    display_df['現在値'] = display_df['現在値'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
    display_df['変動率%'] = display_df['変動率%'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
    display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")
    display_df['総合スコア'] = display_df['総合スコア'].apply(lambda x: f"{x:+.2f}")
    
    # クリック可能なテーブル
    event = st.dataframe(
        display_df[['銘柄', '銘柄名', '現在値', '変動率%', 'RSI', '総合スコア', '判定']],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row"
    )
    
    # テーブルで選択された銘柄を詳細表示
    if event.selection and event.selection.rows:
        selected_row_idx = event.selection.rows[0]
        clicked_ticker = df_signals.iloc[selected_row_idx]['ticker']
        if st.session_state.get('selected_ticker') != clicked_ticker:
            st.session_state['selected_ticker'] = clicked_ticker
            st.rerun()
    
    # ==================== トップ銘柄 ====================
    st.subheader("🏆⚠️ シグナルTOP5（クリックで詳細表示）")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 買いシグナル**")
        top_buy = df_signals.nlargest(5, 'total_score')[['ticker', 'name', 'total_score', 'change']]
        for _, row in top_buy.iterrows():
            score_bar = "🟩" * max(1, int((row['total_score'] + 1) * 2.5))
            is_selected = (st.session_state.get('selected_ticker') == row['ticker'])
            btn_label = f"{'→ ' if is_selected else ''}📈 {row['ticker']} ({row['total_score']:+.2f})"
            if st.button(btn_label, key=f"buy_{row['ticker']}", use_container_width=True):
                st.session_state['selected_ticker'] = row['ticker']
                st.rerun()
    
    with col2:
        st.markdown("**🔴 売りシグナル**")
        top_sell = df_signals.nsmallest(5, 'total_score')[['ticker', 'name', 'total_score', 'change']]
        for _, row in top_sell.iterrows():
            score_bar = "🟥" * max(1, int((1 - row['total_score']) * 2.5))
            is_selected = (st.session_state.get('selected_ticker') == row['ticker'])
            btn_label = f"{'→ ' if is_selected else ''}📉 {row['ticker']} ({row['total_score']:+.2f})"
            if st.button(btn_label, key=f"sell_{row['ticker']}", use_container_width=True):
                st.session_state['selected_ticker'] = row['ticker']
                st.rerun()
            st.caption(f"{row['name'] or ''} | {score_bar} スコア: {row['total_score']:+.2f}")

else:
    st.info("👆 「シグナル更新」ボタンを押してデータを取得してください")

# 自動更新
if auto_refresh and 'last_update' in st.session_state:
    import time
    time.sleep(refresh_interval)
    st.rerun()

# 凡例
with st.expander("📖 指標の説明"):
    st.markdown("""
    ### シグナル値の意味
    - **+1.0**: 強い買いシグナル
    - **0.0**: 中立
    - **-1.0**: 強い売りシグナル
    
    ### 各指標
    | 指標 | 説明 | 買いシグナル | 売りシグナル |
    |-----|------|-------------|-------------|
    | RSI | 相対力指数 | 30以下（売られすぎ） | 70以上（買われすぎ） |
    | MA | 移動平均トレンド | 短期>長期（上昇トレンド） | 短期<長期（下降トレンド） |
    | MACD | トレンド転換 | MACDライン>シグナル | MACDライン<シグナル |
    | BB | ボリンジャーバンド | 下限付近 | 上限付近 |
    | 出来高 | 出来高トレンド | 出来高増+価格上昇 | 出来高増+価格下落 |
    
    ### 総合スコアの重み
    - RSI: 20%
    - MA: 25%
    - MACD: 25%
    - BB: 15%
    - 出来高: 15%
    """)
