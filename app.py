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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ページ設定（最初に実行）
st.set_page_config(
    page_title="📈 株価予測ツール",
    page_icon="📈",
    layout="wide"
)

# 重いモジュールは遅延インポート（使用時にのみ読み込み）
@st.cache_resource
def load_plotly():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots

@st.cache_resource
def load_yfinance():
    try:
        import yfinance as yf
        return yf, True
    except ImportError:
        return None, False

@st.cache_resource
def load_statsmodels():
    from statsmodels.tsa.arima.model import ARIMA
    return ARIMA

@st.cache_resource
def load_sklearn():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    return RandomForestRegressor, StandardScaler, train_test_split

@st.cache_resource
def load_pytorch():
    try:
        import torch
        from models.lstm_model import StockLSTMPredictor, get_device
        return torch, StockLSTMPredictor, get_device, True
    except ImportError:
        return None, None, None, False

@st.cache_resource
def load_lightgbm():
    try:
        from models.lightgbm_model import StockLightGBMPredictor, LIGHTGBM_AVAILABLE
        return StockLightGBMPredictor, LIGHTGBM_AVAILABLE
    except ImportError:
        return None, False

@st.cache_resource
def load_garch():
    try:
        from models.garch_model import StockGARCHPredictor, ARCH_AVAILABLE
        return StockGARCHPredictor, ARCH_AVAILABLE
    except ImportError:
        return None, False

# モジュールの利用可能性を軽量チェック（サイドバー用）
@st.cache_resource
def check_lightgbm_available():
    try:
        import importlib.util
        return importlib.util.find_spec("lightgbm") is not None
    except:
        return False

@st.cache_resource
def check_garch_available():
    try:
        import importlib.util
        return importlib.util.find_spec("arch") is not None
    except:
        return False

@st.cache_resource
def check_pytorch_available():
    try:
        import importlib.util
        return importlib.util.find_spec("torch") is not None
    except:
        return False

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


def fetch_stock_data_from_api(ticker: str, period: str) -> pd.DataFrame:
    """株価データをAPIから取得（内部用）"""
    yf, yf_available = load_yfinance()
    if yf_available:
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if len(data) > 0:
                return data
        except:
            pass
    
    # ダミーデータ（APIが使えない場合）
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


def fetch_stock_data(ticker: str, period: str, use_smart_cache: bool = True):
    """
    株価データを取得（スマートキャッシュ対応）
    
    Returns:
        tuple: (DataFrame, source) - sourceは "cache", "api", "stale_cache" のいずれか
    """
    # DBが使えない場合は直接API
    if not db_available or not use_smart_cache:
        return fetch_stock_data_from_api(ticker, period), "api"
    
    # スマートキャッシュを使用
    try:
        from services import smart_fetch_stock_data
        return smart_fetch_stock_data(
            ticker=ticker,
            period=period,
            db_manager=db,
            api_fetch_func=fetch_stock_data_from_api,
            cache_max_age_hours=6  # 6時間以内のキャッシュは再利用
        )
    except ImportError:
        return fetch_stock_data_from_api(ticker, period), "api"


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
    """ARIMA予測（リターンベース）"""
    ARIMA = load_statsmodels()
    
    # リターンを予測（価格直接より安定）
    returns = data['Close'].pct_change().dropna()
    
    best_aic = float('inf')
    best_order = (1, 1, 1)
    
    for p in range(3):
        for q in range(3):
            try:
                model = ARIMA(returns, order=(p, 1, q))
                result = model.fit()
                if result.aic < best_aic:
                    best_aic = result.aic
                    best_order = (p, 1, q)
            except:
                continue
    
    model = ARIMA(returns, order=best_order)
    result = model.fit()
    forecast = result.get_forecast(steps=forecast_days)
    
    # リターン予測を価格に変換
    predicted_returns = forecast.predicted_mean.values
    last_price = data['Close'].iloc[-1]
    
    # 累積リターンで価格を計算
    predicted_prices = [last_price]
    for ret in predicted_returns:
        # リターンを制限（極端な予測を防ぐ）
        ret = np.clip(ret, -0.05, 0.05)  # 日次±5%以内
        predicted_prices.append(predicted_prices[-1] * (1 + ret))
    
    predicted_prices = np.array(predicted_prices[1:])
    
    # 信頼区間も調整（より厳しく制限）
    ci = forecast.conf_int()
    ci_lower = [last_price]
    ci_upper = [last_price]
    for i in range(len(ci)):
        # 日次±3%に厳しく制限
        ret_lower = np.clip(ci.iloc[i, 0], -0.03, 0.03)
        ret_upper = np.clip(ci.iloc[i, 1], -0.03, 0.03)
        ci_lower.append(ci_lower[-1] * (1 + ret_lower))
        ci_upper.append(ci_upper[-1] * (1 + ret_upper))
    
    ci_df = pd.DataFrame({
        'lower': ci_lower[1:],
        'upper': ci_upper[1:]
    })
    
    # 信頼区間が現在価格の±30%を超えたらNoneを返す（チャートに表示しない）
    if ci_df['upper'].max() > last_price * 1.3 or ci_df['lower'].min() < last_price * 0.7:
        ci_df = None
    
    return pd.Series(predicted_prices), ci_df, best_order


def predict_ml(data: pd.DataFrame, forecast_days: int):
    """機械学習予測（リターンベース + アンサンブル）"""
    RandomForestRegressor, StandardScaler, train_test_split = load_sklearn()
    from sklearn.ensemble import GradientBoostingRegressor
    
    df_ml = data.copy()
    
    # リターンを予測ターゲットに
    df_ml['Target_Return'] = df_ml['Close'].pct_change().shift(-1)
    
    # より多くの特徴量
    for i in range(1, 11):
        df_ml[f'Return_lag{i}'] = df_ml['Close'].pct_change().shift(i)
    
    df_ml['Volatility_10'] = df_ml['Close'].pct_change().rolling(10).std()
    df_ml['Volatility_20'] = df_ml['Close'].pct_change().rolling(20).std()
    df_ml['Price_SMA5_ratio'] = df_ml['Close'] / df_ml['SMA_5']
    df_ml['Price_SMA20_ratio'] = df_ml['Close'] / df_ml['SMA_20']
    
    df_ml = df_ml.dropna()
    
    features = ['RSI', 'MACD', 'Volume_ratio', 'Volatility_10', 'Volatility_20',
                'Price_SMA5_ratio', 'Price_SMA20_ratio'] + \
               [f'Return_lag{i}' for i in range(1, 11)]
    
    X = df_ml[features].values[:-1]  # 最後の行はターゲットがNaN
    y = df_ml['Target_Return'].values[:-1]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # アンサンブル: RF + GradientBoosting
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    rf_model.fit(X_train_scaled, y_train)
    gb_model.fit(X_train_scaled, y_train)
    
    # テストスコア（アンサンブル）
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    ensemble_pred = (rf_pred + gb_pred) / 2
    
    from sklearn.metrics import r2_score
    score = r2_score(y_test, ensemble_pred)
    
    # 予測
    last_price = data['Close'].iloc[-1]
    predictions = [last_price]
    
    # 最新の特徴量を取得
    current_features = df_ml[features].iloc[-1:].values
    
    for _ in range(forecast_days):
        current_scaled = scaler.transform(current_features)
        rf_ret = rf_model.predict(current_scaled)[0]
        gb_ret = gb_model.predict(current_scaled)[0]
        predicted_return = (rf_ret + gb_ret) / 2
        
        # リターンを制限（極端な予測を防ぐ）
        predicted_return = np.clip(predicted_return, -0.03, 0.03)  # 日次±3%以内
        
        next_price = predictions[-1] * (1 + predicted_return)
        predictions.append(next_price)
        
        # 特徴量を更新（リターンラグをシフト）
        current_features = np.roll(current_features, 1, axis=1)
        current_features[0, 6] = predicted_return  # Return_lag1を更新
    
    return np.array(predictions[1:]), score


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


def create_chart(data: pd.DataFrame, arima_pred=None, arima_ci=None, ml_pred=None, lstm_pred=None,
                 lightgbm_pred=None, garch_pred=None, garch_upper=None, garch_lower=None, forecast_days=30):
    """Plotlyチャートを作成"""
    go, make_subplots = load_plotly()
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
            fig.add_trace(go.Scatter(x=forecast_dates, y=arima_ci['upper'],
                                    line=dict(color='red', width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=forecast_dates, y=arima_ci['lower'],
                                    line=dict(color='red', width=0), fill='tonexty',
                                    fillcolor='rgba(255,0,0,0.1)', name='ARIMA 95%CI'), row=1, col=1)
    
    if ml_pred is not None:
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='B')
        fig.add_trace(go.Scatter(x=forecast_dates, y=ml_pred, name='ML予測',
                                line=dict(color='green', dash='dash')), row=1, col=1)
    
    if lstm_pred is not None:
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='B')
        fig.add_trace(go.Scatter(x=forecast_dates, y=lstm_pred, name='LSTM予測',
                                line=dict(color='purple', dash='dash', width=2)), row=1, col=1)
    
    if lightgbm_pred is not None:
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='B')
        fig.add_trace(go.Scatter(x=forecast_dates, y=lightgbm_pred, name='LightGBM予測',
                                line=dict(color='orange', dash='dash', width=2)), row=1, col=1)
    
    if garch_pred is not None:
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1),
                                       periods=forecast_days, freq='B')
        fig.add_trace(go.Scatter(x=forecast_dates, y=garch_pred, name='GARCH予測',
                                line=dict(color='cyan', dash='dash', width=2)), row=1, col=1)
        # 信頼区間は削除（スケールが崩れるため）
    
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

st.title("📈 株価分析ツール")
st.markdown("""
テクニカル分析 + 統計モデルによる株価分析  
⚠️ **注意**: 予測は参考値です。株価の正確な予測は原理的に困難です。
""")

# データベース接続（遅延ロード）
@st.cache_resource
def load_database():
    try:
        from database.db_manager import get_db
        return get_db(), True
    except ImportError:
        return None, False

db, db_available = load_database()

# サイドバー
st.sidebar.header("⚙️ 設定")

# ウォッチリストから選択（DBが利用可能な場合）
if db_available:
    watchlist = db.get_watchlist()
    if watchlist:
        watchlist_options = ["直接入力"] + [f"{w['ticker']} - {w['name'] or ''}" for w in watchlist]
        selected_from_watchlist = st.sidebar.selectbox("📋 ウォッチリストから選択", watchlist_options)
        
        if selected_from_watchlist != "直接入力":
            default_ticker = selected_from_watchlist.split(" - ")[0]
        else:
            default_ticker = st.session_state.get('selected_ticker', 'AAPL')
    else:
        default_ticker = st.session_state.get('selected_ticker', 'AAPL')
        st.sidebar.caption("📋 [ウォッチリストに銘柄を追加](/02_watchlist)")
else:
    default_ticker = 'AAPL'

ticker = st.sidebar.text_input(
    "銘柄コード",
    value=default_ticker,
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

st.sidebar.subheader("📊 予測モデル")
run_arima = st.sidebar.checkbox("ARIMA (時系列)", value=True)
run_ml = st.sidebar.checkbox("Random Forest", value=False)

# LightGBM オプション（軽量チェック）
lightgbm_available = check_lightgbm_available()
if lightgbm_available:
    run_lightgbm = st.sidebar.checkbox("⚡ LightGBM (推奨)", value=True)
else:
    run_lightgbm = False
    st.sidebar.warning("⚠️ LightGBM未インストール")

# GARCH オプション（軽量チェック）
arch_available = check_garch_available()
if arch_available:
    run_garch = st.sidebar.checkbox("📉 GARCH (ボラティリティ)", value=True)
else:
    run_garch = False
    st.sidebar.warning("⚠️ arch未インストール")

# LSTM (GPU対応) オプション（軽量チェック）
pytorch_available = check_pytorch_available()
if pytorch_available:
    run_lstm = st.sidebar.checkbox("🧠 LSTM (深層学習)", value=False)
    if run_lstm:
        # LSTMを使う場合のみPyTorchをロード
        torch, StockLSTMPredictor, get_device, _ = load_pytorch()
        device = get_device()
        if torch.cuda.is_available():
            st.sidebar.success(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.sidebar.info("💻 CPU モードで実行")
        lstm_epochs = st.sidebar.slider("LSTMエポック数", 50, 200, 100)
else:
    run_lstm = False

# 分析実行ボタン
if st.sidebar.button("🔍 分析実行", type="primary"):
    with st.spinner("データを取得中..."):
        data, data_source = fetch_stock_data(ticker, period)
        data = add_indicators(data)
    
    # データソースの表示
    source_icons = {
        "cache": "⚡ キャッシュ",
        "api": "🌐 API",
        "stale_cache": "📦 古いキャッシュ"
    }
    st.caption(f"データソース: {source_icons.get(data_source, data_source)}")
    
    # ウォッチリストに追加ボタン
    if db_available:
        # ウォッチリストに存在するかチェック
        watchlist_tickers = [w['ticker'] for w in db.get_watchlist()]
        if ticker.upper() not in watchlist_tickers:
            if st.button(f"📋 {ticker.upper()} をウォッチリストに追加"):
                db.add_to_watchlist(ticker)
                st.success(f"✅ {ticker.upper()} をウォッチリストに追加しました")
                st.rerun()
        else:
            st.caption(f"✅ {ticker.upper()} はウォッチリストに登録済み")
    
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
    arima_pred, arima_ci, ml_pred, lstm_pred = None, None, None, None
    lightgbm_pred, garch_pred, garch_upper, garch_lower = None, None, None, None
    
    if run_arima:
        with st.spinner("ARIMA予測中..."):
            arima_pred, arima_ci, order = predict_arima(data, forecast_days)
            st.success(f"✅ ARIMA{order} 予測完了")
    
    if run_ml:
        with st.spinner("Random Forest予測中..."):
            ml_pred, score = predict_ml(data, forecast_days)
            st.success(f"✅ Random Forest予測完了 (R²={score:.4f})")
    
    if run_lightgbm and lightgbm_available:
        with st.spinner("⚡ LightGBM予測中..."):
            try:
                StockLightGBMPredictor, _ = load_lightgbm()
                lgb_predictor = StockLightGBMPredictor(n_estimators=500, learning_rate=0.05)
                result = lgb_predictor.train(data, target_days=1, verbose=False)
                lightgbm_pred = lgb_predictor.predict(data, forecast_days=forecast_days)
                st.success(f"✅ LightGBM予測完了 (方向精度={result['direction_accuracy']:.1f}%)")
            except Exception as e:
                st.error(f"❌ LightGBM予測エラー: {e}")
    
    if run_garch and arch_available:
        with st.spinner("📉 GARCH予測中..."):
            try:
                StockGARCHPredictor, _ = load_garch()
                garch_predictor = StockGARCHPredictor(p=1, q=1)
                garch_predictor.train(data, verbose=False)
                price_range = garch_predictor.predict_price_range(data, forecast_days=forecast_days)
                garch_pred = price_range['Price_Mean'].values
                garch_upper = price_range['Price_Upper'].values
                garch_lower = price_range['Price_Lower'].values
                st.success(f"✅ GARCH予測完了 (ボラティリティ予測)")
            except Exception as e:
                st.error(f"❌ GARCH予測エラー: {e}")
    
    if run_lstm and pytorch_available:
        with st.spinner("🧠 LSTM予測中..."):
            try:
                torch, StockLSTMPredictor, get_device, _ = load_pytorch()
                predictor = StockLSTMPredictor(sequence_length=30, hidden_size=64)
                predictor.train(data['Close'].values, epochs=lstm_epochs, verbose=False)
                lstm_pred = predictor.predict(data['Close'].values, forecast_days=forecast_days)
                device_name = "GPU" if torch.cuda.is_available() else "CPU"
                st.success(f"✅ LSTM予測完了 ({device_name}使用)")
            except Exception as e:
                st.error(f"❌ LSTM予測エラー: {e}")
    
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
    has_predictions = any([arima_pred is not None, ml_pred is not None, 
                          lstm_pred is not None, lightgbm_pred is not None,
                          garch_pred is not None])
    
    if has_predictions:
        st.subheader("🔮 予測結果")
        
        # 予測カード表示
        pred_items = []
        if arima_pred is not None:
            pred_items.append(("ARIMA", arima_pred.iloc[-1]))
        if lightgbm_pred is not None:
            pred_items.append(("⚡ LightGBM", lightgbm_pred[-1]))
        if garch_pred is not None:
            pred_items.append(("📉 GARCH", garch_pred[-1]))
        if ml_pred is not None:
            pred_items.append(("RF", ml_pred[-1]))
        if lstm_pred is not None:
            pred_items.append(("🧠 LSTM", lstm_pred[-1]))
        
        pred_cols = st.columns(len(pred_items))
        for i, (name, future_price) in enumerate(pred_items):
            change = (future_price - latest['Close']) / latest['Close'] * 100
            with pred_cols[i]:
                st.metric(
                    f"{name} ({forecast_days}日後)",
                    f"{future_price:.2f}",
                    f"{change:+.2f}%"
                )
        
        # GARCHの価格レンジ表示
        if garch_pred is not None and garch_upper is not None:
            st.info(f"📉 GARCH 95%信頼区間: {garch_lower[-1]:.2f} ～ {garch_upper[-1]:.2f}")
    
    # チャート
    st.subheader("📈 チャート")
    fig = create_chart(data, arima_pred, arima_ci, ml_pred, lstm_pred, 
                      lightgbm_pred, garch_pred, garch_upper, garch_lower, forecast_days)
    st.plotly_chart(fig, use_container_width=True)
    
    # 予測の限界についての説明
    with st.expander("⚠️ 予測の限界について（重要）"):
        st.markdown("""
        ### 株価予測の現実
        
        **なぜ予測は外れるのか？**
        
        1. **効率的市場仮説**: 株価は既に全ての情報を織り込んでいるため、
           過去のデータから将来を予測することは理論上不可能です。
        
        2. **ランダムウォーク**: 短期的な株価変動はほぼランダムであり、
           統計モデルで捉えることが非常に困難です。
        
        3. **外部要因**: 決算発表、経済指標、地政学リスク、突発的ニュースなど、
           過去データに含まれない要因が株価を大きく動かします。
        
        **このツールの正しい使い方**
        
        - ❌ 予測値を信じて売買する
        - ✅ テクニカル指標の確認
        - ✅ ボラティリティ（変動リスク）の把握
        - ✅ トレンドの方向性の参考
        - ✅ 複数モデルの一致度を見る
        
        **学術研究の結論**
        
        > 「短期的な株価予測は、コインを投げるのと同程度の精度しかない」
        > - 多くの金融経済学者の見解
        """)
    
    # 注意書き
    st.error("⚠️ **重要**: この予測は参考情報です。予測精度は保証されません。投資判断は自己責任で行ってください。")

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
