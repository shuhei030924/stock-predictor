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
    
    # 重み付け平均（MACDを重視、出来高は補助的）
    weights = {
        'rsi': 0.20,
        'ma': 0.25,
        'macd': 0.30,  # トレンド転換に最重要
        'bb': 0.15,
        'volume': 0.10  # 補助的指標
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
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 新しい閾値: ±0.2未満は中立
    strong_buy = len(df_signals[df_signals['total_score'] >= 0.5])
    buy = len(df_signals[(df_signals['total_score'] >= 0.2) & (df_signals['total_score'] < 0.5)])
    neutral = len(df_signals[(df_signals['total_score'] > -0.2) & (df_signals['total_score'] < 0.2)])
    sell = len(df_signals[(df_signals['total_score'] <= -0.2) & (df_signals['total_score'] > -0.5)])
    strong_sell = len(df_signals[df_signals['total_score'] <= -0.5])
    
    col1.metric("🟢 強い買い", strong_buy, help="スコア ≥ 0.5")
    col2.metric("🔵 買い", buy, help="スコア 0.2～0.5")
    col3.metric("⚪ 中立", neutral, help="スコア -0.2～0.2")
    col4.metric("🟠 売り", sell, help="スコア -0.5～-0.2")
    col5.metric("🔴 強い売り", strong_sell, help="スコア ≤ -0.5")
    
    st.divider()
    
    # ==================== ヒートマップ ====================
    st.subheader("🔥 シグナルヒートマップ（行をクリックで詳細をモーダル表示）")
    
    # Plotly遅延ロード
    import plotly.graph_objects as go
    import plotly.express as px
    
    # デフォルトは総合スコア降順
    df_signals = df_signals.sort_values('total_score', ascending=False).reset_index(drop=True)
    
    # ヒートマップ用データフレーム作成（情報追加: 銘柄名、価格、変動率）
    heatmap_df = df_signals[['ticker', 'name', 'price', 'change', 'rsi_signal', 'ma_signal', 'macd_signal', 'bb_signal', 'vol_signal', 'total_score']].copy()
    heatmap_df['price'] = heatmap_df['price'].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "-")
    heatmap_df['change'] = heatmap_df['change'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
    heatmap_df.columns = ['銘柄', '銘柄名', '価格', '変動率', 'RSI', 'MA', 'MACD', 'BB', '出来高', '総合']
    
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
    
    # 変動率の色分け
    def color_change(val):
        if pd.isna(val) or not isinstance(val, str):
            return ''
        try:
            num = float(val.replace('%', '').replace('+', ''))
            if num > 0:
                return 'color: green; font-weight: bold'
            elif num < 0:
                return 'color: red; font-weight: bold'
        except:
            pass
        return ''
    
    # スタイル適用
    styled_heatmap = heatmap_df.style.applymap(
        color_signal, 
        subset=['RSI', 'MA', 'MACD', 'BB', '出来高', '総合']
    ).applymap(
        color_change,
        subset=['変動率']
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
    
    # ==================== モーダルダイアログ定義 ====================
    @st.dialog("📊 銘柄詳細", width="large")
    def show_ticker_detail(ticker: str, ticker_name: str, signal_row: pd.Series):
        """モーダルダイアログで銘柄詳細を表示"""
        st.markdown(f"## {ticker} - {ticker_name}")
        
        # 基本情報
        col1, col2, col3, col4 = st.columns(4)
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
        
        # 詳細データ取得
        with st.spinner(f"{ticker} のトレンドを取得中..."):
            detail_data = get_ticker_detail(ticker)
        
        if detail_data:
            # チャートタブ
            tab1, tab2, tab3 = st.tabs(["📈 価格チャート", "📊 テクニカル", "📉 シグナル詳細"])
            
            with tab1:
                # ローソク足チャート
                fig_price = go.Figure()
                fig_price.add_trace(go.Candlestick(
                    x=detail_data['df'].index,
                    open=detail_data['df']['Open'],
                    high=detail_data['df']['High'],
                    low=detail_data['df']['Low'],
                    close=detail_data['df']['Close'],
                    name='価格'
                ))
                fig_price.add_trace(go.Scatter(x=detail_data['df'].index, y=detail_data['sma20'], name='SMA20', line=dict(color='blue', width=1)))
                fig_price.add_trace(go.Scatter(x=detail_data['df'].index, y=detail_data['sma50'], name='SMA50', line=dict(color='purple', width=1)))
                fig_price.update_layout(height=400, xaxis_rangeslider_visible=False, hovermode='x unified')
                st.plotly_chart(fig_price, use_container_width=True)
            
            with tab2:
                col_l, col_r = st.columns(2)
                with col_l:
                    # RSI
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=detail_data['df'].index, y=detail_data['rsi'], name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    fig_rsi.update_layout(title="RSI", height=200, yaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig_rsi, use_container_width=True)
                with col_r:
                    # MACD
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=detail_data['df'].index, y=detail_data['macd'], name='MACD', line=dict(color='blue')))
                    fig_macd.add_trace(go.Scatter(x=detail_data['df'].index, y=detail_data['macd_signal'], name='Signal', line=dict(color='orange')))
                    fig_macd.update_layout(title="MACD", height=200)
                    st.plotly_chart(fig_macd, use_container_width=True)
            
            with tab3:
                # シグナル詳細
                signal_details = [
                    ("RSI", signal_row['rsi_signal'], f"RSI = {signal_row['rsi']:.1f}"),
                    ("移動平均", signal_row['ma_signal'], "短期>長期なら買い"),
                    ("MACD", signal_row['macd_signal'], "MACD>シグナルなら買い"),
                    ("ボリンジャーバンド", signal_row['bb_signal'], "下限付近なら買い"),
                    ("出来高", signal_row['vol_signal'], "出来高増+価格上昇なら買い"),
                ]
                for name, value, desc in signal_details:
                    c1, c2, c3 = st.columns([2, 2, 4])
                    c1.write(f"**{name}**")
                    bar = "🟩" * max(0, int((value + 1) * 2.5)) + "🟥" * max(0, int((1 - value) * 2.5))
                    c2.write(f"{value:+.2f}")
                    c3.caption(desc)
                st.markdown(f"### 総合スコア: {signal_row['total_score']:+.2f}")
        else:
            st.error("データの取得に失敗しました")
        
        if st.button("閉じる", type="primary", use_container_width=True):
            st.rerun()
    
    # クリックされた行から銘柄を取得（session_stateに保存）
    if clicked_heatmap.selection and clicked_heatmap.selection.rows:
        selected_row_idx = clicked_heatmap.selection.rows[0]
        clicked_ticker = df_signals.iloc[selected_row_idx]['ticker']
        st.session_state['modal_ticker'] = clicked_ticker
        st.session_state['modal_ticker_idx'] = selected_row_idx
    
    st.divider()
    
    # ==================== トップ銘柄 ====================
    st.subheader("🏆⚠️ シグナルTOP5（クリックで詳細モーダル表示）")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🟢 買いシグナルTOP5**")
        top_buy = df_signals.nlargest(5, 'total_score')
        for idx, (orig_idx, row) in enumerate(top_buy.iterrows()):
            btn_label = f"📈 {row['ticker']} - {row['name'] or ''} ({row['total_score']:+.2f})"
            if st.button(btn_label, key=f"buy_{row['ticker']}", use_container_width=True):
                st.session_state['modal_ticker'] = row['ticker']
                # df_signalsでのインデックスを探す
                st.session_state['modal_ticker_idx'] = df_signals[df_signals['ticker'] == row['ticker']].index[0]
                st.rerun()
    
    with col2:
        st.markdown("**🔴 売りシグナルTOP5**")
        top_sell = df_signals.nsmallest(5, 'total_score')
        for idx, (orig_idx, row) in enumerate(top_sell.iterrows()):
            btn_label = f"📉 {row['ticker']} - {row['name'] or ''} ({row['total_score']:+.2f})"
            if st.button(btn_label, key=f"sell_{row['ticker']}", use_container_width=True):
                st.session_state['modal_ticker'] = row['ticker']
                st.session_state['modal_ticker_idx'] = df_signals[df_signals['ticker'] == row['ticker']].index[0]
                st.rerun()
    
    # ==================== モーダル表示（1か所で制御） ====================
    if 'modal_ticker' in st.session_state and st.session_state['modal_ticker']:
        modal_ticker = st.session_state['modal_ticker']
        modal_idx = st.session_state.get('modal_ticker_idx', 0)
        
        # df_signalsから該当行を取得
        modal_rows = df_signals[df_signals['ticker'] == modal_ticker]
        if len(modal_rows) > 0:
            modal_row = modal_rows.iloc[0]
            modal_name = modal_row['name'] or ''
            show_ticker_detail(modal_ticker, modal_name, modal_row)
            # モーダルを閉じた後にリセット
            st.session_state['modal_ticker'] = None

else:
    st.info("👆 「シグナル更新」ボタンを押してデータを取得してください")

# ==================== バックテストセクション ====================
st.divider()
st.subheader("🧪 シグナルバックテスト（リスク管理型）")

# アルゴリズム説明
with st.expander("📋 売買アルゴリズム（案C: リスク管理型）"):
    st.markdown("""
    ### 資金管理ルール
    - **現金比率**: 常に20%以上キープ
    - **1銘柄上限**: 総資産の10%まで
    - **最大保有銘柄数**: 10銘柄
    
    ### 買いルール
    | スコア | 条件 | 購入額 |
    |--------|------|--------|
    | ≥ 0.5 (強い買い) | 未保有のみ | 総資産の8% |
    | ≥ 0.2 (買い) | 未保有のみ | 総資産の5% |
    | < 0.2 | - | 購入しない |
    
    ### 売りルール
    | 条件 | アクション |
    |------|-----------|
    | スコア ≤ -0.5 | 全株売却 |
    | スコア ≤ -0.2 | 半分売却 |
    | **利確**: +20%到達 | 半分売却 |
    | **損切**: -10%到達 | 全株売却 |
    """)

# バックテスト残高取得
bt_balance = db.backtest_get_balance()
bt_portfolio = db.backtest_get_portfolio()

# 保有株の時価を計算（現在のシグナルデータから価格取得）
stock_value = 0
portfolio_with_pnl = []  # 損益計算済みポートフォリオ

if bt_portfolio:
    signal_data = st.session_state.get('signal_data', {})
    for pos in bt_portfolio:
        if pos['ticker'] in signal_data:
            price = signal_data[pos['ticker']]['price']
        else:
            price = pos['current_price']
        
        value = pos['shares'] * price
        cost = pos['shares'] * pos['avg_cost']
        pnl_rate = ((price - pos['avg_cost']) / pos['avg_cost']) * 100 if pos['avg_cost'] > 0 else 0
        
        stock_value += value
        portfolio_with_pnl.append({
            'ticker': pos['ticker'],
            'shares': pos['shares'],
            'avg_cost': pos['avg_cost'],
            'current_price': price,
            'value': value,
            'pnl_rate': pnl_rate
        })

total_value = bt_balance['cash'] + stock_value
initial_value = 1000000
profit_rate = ((total_value - initial_value) / initial_value) * 100
cash_ratio = (bt_balance['cash'] / total_value) * 100 if total_value > 0 else 100
held_tickers = set(p['ticker'] for p in bt_portfolio)

# 残高表示
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💵 現金", f"¥{bt_balance['cash']:,.0f}")
col2.metric("📈 株式評価額", f"¥{stock_value:,.0f}")
col3.metric("💰 総資産", f"¥{total_value:,.0f}")
col4.metric("📊 損益率", f"{profit_rate:+.2f}%", delta=f"¥{total_value - initial_value:+,.0f}")
col5.metric("💵 現金比率", f"{cash_ratio:.1f}%", delta="OK" if cash_ratio >= 20 else "⚠️低い")

# バックテスト実行ボタン
st.markdown("### 🎯 テスト実行")
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    if st.button("▶️ シグナルに基づいて売買実行", type="primary", use_container_width=True):
        if 'signal_data' not in st.session_state:
            st.error("先にシグナルを更新してください")
        else:
            signal_data = st.session_state['signal_data']
            executed_trades = []
            
            # === 売り処理を先に実行（現金確保）===
            for pos in portfolio_with_pnl:
                ticker = pos['ticker']
                price = pos['current_price']
                pnl_rate = pos['pnl_rate']
                score = signal_data.get(ticker, {}).get('total_score', 0)
                
                # 損切り: -10%以下
                if pnl_rate <= -10:
                    if db.backtest_sell(ticker, 1.0, price, score, f"損切り ({pnl_rate:.1f}%)"):
                        executed_trades.append(f"🔴 {ticker}: 全株売却（損切り {pnl_rate:.1f}%）")
                    continue
                
                # 利確: +20%以上
                if pnl_rate >= 20:
                    if db.backtest_sell(ticker, 0.5, price, score, f"利確 ({pnl_rate:.1f}%)"):
                        executed_trades.append(f"🟡 {ticker}: 半分売却（利確 {pnl_rate:.1f}%）")
                    continue
                
                # 強い売りシグナル
                if score <= -0.5:
                    if db.backtest_sell(ticker, 1.0, price, score, "強い売りシグナル"):
                        executed_trades.append(f"🔴 {ticker}: 全株売却（スコア {score:.2f}）")
                # 売りシグナル
                elif score <= -0.2:
                    if db.backtest_sell(ticker, 0.5, price, score, "売りシグナル"):
                        executed_trades.append(f"🟠 {ticker}: 半分売却（スコア {score:.2f}）")
            
            # 残高再取得
            bt_balance = db.backtest_get_balance()
            bt_portfolio = db.backtest_get_portfolio()
            held_tickers = set(p['ticker'] for p in bt_portfolio)
            total_value = bt_balance['cash'] + stock_value
            
            # === 買い処理 ===
            # スコア順にソート
            buy_candidates = [
                (t, d) for t, d in signal_data.items() 
                if d['total_score'] >= 0.2 and t not in held_tickers
            ]
            buy_candidates.sort(key=lambda x: x[1]['total_score'], reverse=True)
            
            for ticker, data in buy_candidates:
                score = data['total_score']
                price = data['price']
                
                # 現金比率チェック（20%以上キープ）
                current_cash = db.backtest_get_balance()['cash']
                if current_cash < total_value * 0.20:
                    break
                
                # 保有銘柄数チェック（最大10銘柄）
                current_portfolio = db.backtest_get_portfolio()
                if len(current_portfolio) >= 10:
                    break
                
                # 購入額決定
                if score >= 0.5:
                    buy_amount = total_value * 0.08  # 総資産の8%
                    reason = "強い買いシグナル"
                else:
                    buy_amount = total_value * 0.05  # 総資産の5%
                    reason = "買いシグナル"
                
                # 1銘柄上限チェック（総資産の10%）
                max_position = total_value * 0.10
                buy_amount = min(buy_amount, max_position)
                
                # 現金残高チェック（20%キープ分を除く）
                available_cash = current_cash - (total_value * 0.20)
                buy_amount = min(buy_amount, available_cash)
                
                if buy_amount > 10000:  # 最低1万円以上
                    if db.backtest_buy(ticker, buy_amount, price, score, reason):
                        executed_trades.append(f"🟢 {ticker}: ¥{buy_amount:,.0f} 購入（スコア {score:.2f}）")
                        held_tickers.add(ticker)
            
            if executed_trades:
                st.success(f"✅ {len(executed_trades)}件の取引を実行しました")
                for trade in executed_trades[:15]:
                    st.write(trade)
                if len(executed_trades) > 15:
                    st.caption(f"他 {len(executed_trades) - 15}件...")
                st.rerun()
            else:
                st.info("売買対象の銘柄がありませんでした")

with col2:
    if st.button("🔄 価格更新のみ", use_container_width=True, help="売買せず保有株の価格だけ更新"):
        if 'signal_data' in st.session_state:
            signal_data = st.session_state['signal_data']
            price_map = {t: d['price'] for t, d in signal_data.items()}
            db.backtest_update_prices(price_map)
            st.success("価格を更新しました")
            st.rerun()

with col3:
    if st.button("🗑️ リセット", use_container_width=True):
        if db.backtest_reset(1000000):
            st.success("バックテストをリセットしました（初期資金: 100万円）")
            st.rerun()

# 保有ポートフォリオ表示
if portfolio_with_pnl:
    st.markdown("### 📦 保有ポートフォリオ")
    portfolio_data = []
    for pos in portfolio_with_pnl:
        # シグナルスコア取得
        score = st.session_state.get('signal_data', {}).get(pos['ticker'], {}).get('total_score', None)
        score_str = f"{score:+.2f}" if score is not None else "-"
        
        # 損益率による色分け表示
        pnl_rate = pos['pnl_rate']
        if pnl_rate >= 20:
            status = "🟡 利確検討"
        elif pnl_rate <= -10:
            status = "🔴 損切り検討"
        elif pnl_rate > 0:
            status = "🟢 含み益"
        else:
            status = "🔵 含み損"
        
        portfolio_data.append({
            '銘柄': pos['ticker'],
            '株数': f"{pos['shares']:.2f}",
            '平均取得単価': f"${pos['avg_cost']:.2f}",
            '現在価格': f"${pos['current_price']:.2f}",
            '評価額': f"¥{pos['value']:,.0f}",
            '損益率': f"{pnl_rate:+.1f}%",
            'スコア': score_str,
            '状態': status
        })
    
    st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True, hide_index=True)

# 取引履歴表示
with st.expander("📜 取引履歴"):
    transactions = db.backtest_get_transactions(20)
    if transactions:
        tx_data = []
        for tx in transactions:
            tx_data.append({
                '日時': tx['created_at'][:16],
                '銘柄': tx['ticker'],
                '売買': '🟢 買い' if tx['action'] == 'BUY' else '🔴 売り',
                '株数': f"{tx['shares']:.2f}",
                '価格': f"${tx['price']:.2f}",
                '金額': f"¥{tx['amount']:,.0f}",
                'スコア': f"{tx['signal_score']:+.2f}" if tx['signal_score'] else "-",
                '理由': tx['reason'] or ""
            })
        st.dataframe(pd.DataFrame(tx_data), use_container_width=True, hide_index=True)
    else:
        st.info("取引履歴がありません")

# 資産推移グラフ
with st.expander("📈 資産推移"):
    balance_history = db.backtest_get_balance_history()
    if len(balance_history) > 1:
        import plotly.graph_objects as go
        
        dates = [b['created_at'] for b in balance_history]
        totals = [b['total_value'] for b in balance_history]
        cashes = [b['cash'] for b in balance_history]
        stocks = [b['stock_value'] for b in balance_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=totals, name='総資産', line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=dates, y=cashes, name='現金', line=dict(color='green', width=1)))
        fig.add_trace(go.Scatter(x=dates, y=stocks, name='株式', line=dict(color='orange', width=1)))
        fig.add_hline(y=1000000, line_dash="dash", line_color="gray", annotation_text="初期資金")
        fig.update_layout(title="資産推移", height=300, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("まだ取引がありません")

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
    
    ### 判定閾値
    | スコア | 判定 | バックテスト動作 |
    |--------|------|-----------------|
    | ≥ 0.5 | 🟢 強い買い | 10万円購入 |
    | 0.2～0.5 | 🔵 買い | 5万円購入 |
    | -0.2～0.2 | ⚪ 中立 | 何もしない |
    | -0.5～-0.2 | 🟠 売り | 半分売却 |
    | ≤ -0.5 | 🔴 強い売り | 全株売却 |
    
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
    - **MACD: 30%** （最重要）
    - BB: 15%
    - 出来高: 10%
    """)
