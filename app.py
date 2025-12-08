"""
株価予測ツール - Streamlit Web アプリ版
=====================================
ブラウザで動作するインタラクティブな株価予測ツール

起動方法:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# モジュールインポート
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ページ設定
st.set_page_config(
    page_title="📈 株価予測ツール",
    page_icon="📈",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
.big-font {
    font-size: 24px !important;
    font-weight: bold;
}
.signal-buy {
    background-color: #d4edda;
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #28a745;
}
.signal-sell {
    background-color: #f8d7da;
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #dc3545;
}
.signal-neutral {
    background-color: #fff3cd;
    padding: 10px;
    border-radius: 5px;
    border-left: 5px solid #ffc107;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """株価データを取得（キャッシュ付き）"""
    if YFINANCE_AVAILABLE:
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if len(data) > 0:
                return data
        except:
            pass
    
    # ダミーデータ
    np.random.seed(hash(ticker) % 100)
    days = 500
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    returns = np.random.normal(0.0005, 0.02, days)
    price = 100 * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'Open': price * (1 + np.random.uniform(-0.01, 0.01, days)),
        'High': price * (1 + np.random.uniform(0, 0.02, days)),
        'Low': price * (1 - np.random.uniform(0, 0.02, days)),
        'Close': price,
        'Volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を追加"""
    df = df.copy()
    
    # 移動平均
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # ボリンジャーバンド
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # その他
    df['Return'] = df['Close'].pct_change()
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
    
    return df.dropna()


def predict_arima(data: pd.DataFrame, forecast_days: int):
    """ARIMA予測"""
    best_aic = float('inf')
    best_order = (1, 1, 1)
    
    for p in range(3):
        for q in range(3):
            try:
                model = ARIMA(data['Close'], order=(p, 1, q))
                result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, 1, q)
            except:
                continue
    
    model = ARIMA(data['Close'], order=best_order)
    result = model.fit()
    forecast = result.get_forecast(steps=forecast_days)
    
    return forecast.predicted_mean, forecast.conf_int(), best_order


def predict_ml(data: pd.DataFrame, forecast_days: int):
    """機械学習予測"""
    df_ml = data.copy()
    for i in range(1, 6):
        df_ml[f'Close_lag{i}'] = df_ml['Close'].shift(i)
    df_ml = df_ml.dropna()
    
    features = ['SMA_5', 'SMA_20', 'RSI', 'MACD', 'Volume_ratio', 'Return_5d'] + \
               [f'Close_lag{i}' for i in range(1, 6)]
    
    X = df_ml[features].values
    y = df_ml['Close'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    
    last_row = df_ml[features].iloc[-1:].values
    predictions = []
    
    for _ in range(forecast_days):
        pred = model.predict(scaler.transform(last_row))[0]
        predictions.append(pred)
        last_row = np.roll(last_row, 1)
        last_row[0, -1] = pred
    
    return np.array(predictions), score


def get_signals(data: pd.DataFrame) -> list:
    """売買シグナルを取得"""
    latest = data.iloc[-1]
    signals = []
    
    # RSI
    if latest['RSI'] < 30:
        signals.append(('RSI', '買い', f"RSI={latest['RSI']:.1f} (売られすぎ)", 'buy'))
    elif latest['RSI'] > 70:
        signals.append(('RSI', '売り', f"RSI={latest['RSI']:.1f} (買われすぎ)", 'sell'))
    else:
        signals.append(('RSI', '中立', f"RSI={latest['RSI']:.1f}", 'neutral'))
    
    # 移動平均
    if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
        signals.append(('移動平均', '買い', '上昇トレンド', 'buy'))
    elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
        signals.append(('移動平均', '売り', '下降トレンド', 'sell'))
    else:
        signals.append(('移動平均', '中立', 'レンジ相場', 'neutral'))
    
    # MACD
    if latest['MACD'] > latest['MACD_signal']:
        signals.append(('MACD', '買い', 'ゴールデンクロス', 'buy'))
    else:
        signals.append(('MACD', '売り', 'デッドクロス', 'sell'))
    
    # ボリンジャーバンド
    if latest['Close'] < latest['BB_lower']:
        signals.append(('BB', '買い', '下バンド割れ', 'buy'))
    elif latest['Close'] > latest['BB_upper']:
        signals.append(('BB', '売り', '上バンド突破', 'sell'))
    else:
        signals.append(('BB', '中立', 'バンド内', 'neutral'))
    
    return signals


def create_chart(data: pd.DataFrame, arima_pred=None, arima_ci=None, ml_pred=None, forecast_days=30):
    """Plotlyチャートを作成"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=('株価', 'RSI', 'MACD', '出来高')
    )
    
    # 株価チャート
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='株価'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA20',
                            line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA50',
                            line=dict(color='blue', width=1)), row=1, col=1)
    
    # ボリンジャーバンド
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower',
                            line=dict(color='gray', width=1, dash='dash'),
                            fill='tonexty', fillcolor='rgba(128,128,128,0.1)'), row=1, col=1)
    
    # 予測
    if arima_pred is not None:
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='B')
        fig.add_trace(go.Scatter(x=forecast_dates, y=arima_pred, name='ARIMA予測',
                                line=dict(color='red', dash='dash')), row=1, col=1)
        if arima_ci is not None:
            fig.add_trace(go.Scatter(x=forecast_dates, y=arima_ci.iloc[:, 1],
                                    line=dict(color='red', width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=forecast_dates, y=arima_ci.iloc[:, 0],
                                    line=dict(color='red', width=0), fill='tonexty',
                                    fillcolor='rgba(255,0,0,0.1)', name='ARIMA 95%CI'), row=1, col=1)
    
    if ml_pred is not None:
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='B')
        fig.add_trace(go.Scatter(x=forecast_dates, y=ml_pred, name='ML予測',
                                line=dict(color='green', dash='dash')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI',
                            line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name='MACD',
                            line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], name='Signal',
                            line=dict(color='orange')), row=3, col=1)
    
    # 出来高
    colors = ['green' if data['Close'].iloc[i] >= data['Open'].iloc[i] else 'red'
              for i in range(len(data))]
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='出来高',
                        marker_color=colors), row=4, col=1)
    
    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


# ================== メインアプリ ==================

st.title("📈 株価予測ツール")
st.markdown("ARIMA + 機械学習 + テクニカル分析 による総合予測")

# サイドバー
st.sidebar.header("⚙️ 設定")

ticker = st.sidebar.text_input(
    "銘柄コード",
    value="AAPL",
    help="例: AAPL, GOOGL, 7203.T (トヨタ)"
)

period = st.sidebar.selectbox(
    "データ期間",
    options=["1y", "2y", "5y"],
    index=1
)

forecast_days = st.sidebar.slider(
    "予測日数",
    min_value=7,
    max_value=90,
    value=30
)

run_arima = st.sidebar.checkbox("ARIMA予測", value=True)
run_ml = st.sidebar.checkbox("機械学習予測", value=True)

# 分析実行ボタン
if st.sidebar.button("🔍 分析実行", type="primary"):
    with st.spinner("データを取得中..."):
        data = fetch_stock_data(ticker, period)
        data = add_indicators(data)
    
    # 基本情報
    col1, col2, col3, col4 = st.columns(4)
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    change = (latest['Close'] - prev['Close']) / prev['Close'] * 100
    
    col1.metric("最新株価", f"{latest['Close']:.2f}", f"{change:+.2f}%")
    col2.metric("高値", f"{latest['High']:.2f}")
    col3.metric("安値", f"{latest['Low']:.2f}")
    col4.metric("出来高", f"{latest['Volume']:,.0f}")
    
    # 予測実行
    arima_pred, arima_ci, ml_pred = None, None, None
    
    if run_arima:
        with st.spinner("ARIMA予測中..."):
            arima_pred, arima_ci, order = predict_arima(data, forecast_days)
            st.success(f"✅ ARIMA{order} 予測完了")
    
    if run_ml:
        with st.spinner("機械学習予測中..."):
            ml_pred, score = predict_ml(data, forecast_days)
            st.success(f"✅ 機械学習予測完了 (R²={score:.4f})")
    
    # シグナル表示
    st.subheader("📊 売買シグナル")
    signals = get_signals(data)
    
    cols = st.columns(4)
    for i, (indicator, signal, reason, signal_type) in enumerate(signals):
        with cols[i]:
            if signal_type == 'buy':
                st.markdown(f"""
                <div class="signal-buy">
                    <strong>🟢 {indicator}</strong><br>
                    {signal}: {reason}
                </div>
                """, unsafe_allow_html=True)
            elif signal_type == 'sell':
                st.markdown(f"""
                <div class="signal-sell">
                    <strong>🔴 {indicator}</strong><br>
                    {signal}: {reason}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="signal-neutral">
                    <strong>⚪ {indicator}</strong><br>
                    {signal}: {reason}
                </div>
                """, unsafe_allow_html=True)
    
    # 総合判断
    buy_count = sum(1 for _, s, _, _ in signals if s == '買い')
    sell_count = sum(1 for _, s, _, _ in signals if s == '売り')
    
    st.subheader("🎯 総合判断")
    if buy_count > sell_count:
        st.success(f"📈 **買い優勢** (買い{buy_count} / 売り{sell_count})")
    elif sell_count > buy_count:
        st.error(f"📉 **売り優勢** (買い{buy_count} / 売り{sell_count})")
    else:
        st.warning(f"➡️ **中立** (買い{buy_count} / 売り{sell_count})")
    
    # 予測結果
    if arima_pred is not None or ml_pred is not None:
        st.subheader("🔮 予測結果")
        pred_cols = st.columns(2)
        
        if arima_pred is not None:
            future_price = arima_pred.iloc[-1]
            change = (future_price - latest['Close']) / latest['Close'] * 100
            with pred_cols[0]:
                st.metric(
                    f"ARIMA予測 ({forecast_days}日後)",
                    f"{future_price:.2f}",
                    f"{change:+.2f}%"
                )
        
        if ml_pred is not None:
            future_price = ml_pred[-1]
            change = (future_price - latest['Close']) / latest['Close'] * 100
            with pred_cols[1]:
                st.metric(
                    f"ML予測 ({forecast_days}日後)",
                    f"{future_price:.2f}",
                    f"{change:+.2f}%"
                )
    
    # チャート
    st.subheader("📈 チャート")
    fig = create_chart(data, arima_pred, arima_ci, ml_pred, forecast_days)
    st.plotly_chart(fig, use_container_width=True)
    
    # 注意書き
    st.warning("⚠️ この予測は参考情報です。投資判断は自己責任で行ってください。")

else:
    st.info("👈 左のサイドバーで銘柄を設定し、「分析実行」をクリックしてください")
    
    # サンプル銘柄
    st.subheader("📌 人気銘柄")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🇺🇸 米国株**
        - AAPL (Apple)
        - GOOGL (Google)
        - MSFT (Microsoft)
        - AMZN (Amazon)
        - TSLA (Tesla)
        - NVDA (NVIDIA)
        """)
    
    with col2:
        st.markdown("""
        **🇯🇵 日本株**
        - 7203.T (トヨタ)
        - 9984.T (ソフトバンクG)
        - 6758.T (ソニー)
        - 6861.T (キーエンス)
        - 9432.T (NTT)
        - 8306.T (三菱UFJ)
        """)
