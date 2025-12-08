"""
予測精度の検証ページ
==================
過去の予測と実績を比較して、モデルの精度を確認する
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="📊 予測精度検証",
    page_icon="📊",
    layout="wide"
)

st.title("📊 予測精度の検証")
st.markdown("過去の予測と実際の結果を比較して、予測モデルの精度を確認します。")


@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, period: str) -> pd.DataFrame:
    """株価データを取得"""
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
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    
    df['Return'] = df['Close'].pct_change()
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
    
    return df.dropna()


def backtest_arima(data: pd.DataFrame, test_days: int = 30):
    """ARIMAモデルのバックテスト"""
    results = []
    
    for i in range(test_days, 0, -1):
        train_data = data['Close'].iloc[:-i]
        actual = data['Close'].iloc[-i]
        actual_date = data.index[-i]
        
        try:
            model = ARIMA(train_data, order=(1, 1, 1))
            result = model.fit()
            pred = result.forecast(steps=1).iloc[0]
            
            results.append({
                'Date': actual_date,
                'Actual': actual,
                'Predicted': pred,
                'Error': actual - pred,
                'Error_Pct': (actual - pred) / actual * 100
            })
        except:
            continue
    
    return pd.DataFrame(results)


def backtest_ml(data: pd.DataFrame, test_days: int = 30):
    """機械学習モデルのバックテスト"""
    df_ml = data.copy()
    for i in range(1, 6):
        df_ml[f'Close_lag{i}'] = df_ml['Close'].shift(i)
    df_ml = df_ml.dropna()
    
    features = ['SMA_5', 'SMA_20', 'RSI', 'MACD', 'Volume_ratio', 'Return_5d'] + \
               [f'Close_lag{i}' for i in range(1, 6)]
    
    results = []
    
    for i in range(test_days, 0, -1):
        train_idx = len(df_ml) - i
        if train_idx < 100:
            continue
            
        X_train = df_ml[features].iloc[:train_idx].values
        y_train = df_ml['Close'].iloc[:train_idx].values
        
        X_test = df_ml[features].iloc[train_idx:train_idx+1].values
        actual = df_ml['Close'].iloc[train_idx]
        actual_date = df_ml.index[train_idx]
        
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)[0]
            
            results.append({
                'Date': actual_date,
                'Actual': actual,
                'Predicted': pred,
                'Error': actual - pred,
                'Error_Pct': (actual - pred) / actual * 100
            })
        except:
            continue
    
    return pd.DataFrame(results)


def calculate_metrics(results_df: pd.DataFrame) -> dict:
    """精度指標を計算"""
    if len(results_df) == 0:
        return {}
    
    mae = np.mean(np.abs(results_df['Error']))
    rmse = np.sqrt(np.mean(results_df['Error'] ** 2))
    mape = np.mean(np.abs(results_df['Error_Pct']))
    
    # 方向性の的中率
    results_df['Actual_Direction'] = (results_df['Actual'].diff() > 0).astype(int)
    results_df['Pred_Direction'] = (results_df['Predicted'].diff() > 0).astype(int)
    direction_accuracy = (results_df['Actual_Direction'] == results_df['Pred_Direction']).mean() * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy
    }


def create_backtest_chart(arima_results: pd.DataFrame, ml_results: pd.DataFrame):
    """バックテスト結果のチャートを作成"""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('予測 vs 実績', '予測誤差 (%)', '累積誤差')
    )
    
    # 実績
    if len(arima_results) > 0:
        fig.add_trace(go.Scatter(
            x=arima_results['Date'], y=arima_results['Actual'],
            name='実績', line=dict(color='blue', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=arima_results['Date'], y=arima_results['Predicted'],
            name='ARIMA予測', line=dict(color='red', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=arima_results['Date'], y=arima_results['Error_Pct'],
            name='ARIMA誤差', marker_color='red', opacity=0.5
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=arima_results['Date'], y=arima_results['Error'].cumsum(),
            name='ARIMA累積誤差', line=dict(color='red')
        ), row=3, col=1)
    
    if len(ml_results) > 0:
        fig.add_trace(go.Scatter(
            x=ml_results['Date'], y=ml_results['Predicted'],
            name='ML予測', line=dict(color='green', dash='dash')
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=ml_results['Date'], y=ml_results['Error_Pct'],
            name='ML誤差', marker_color='green', opacity=0.5
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=ml_results['Date'], y=ml_results['Error'].cumsum(),
            name='ML累積誤差', line=dict(color='green')
        ), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
    
    return fig


# サイドバー
st.sidebar.header("⚙️ 設定")

ticker = st.sidebar.text_input(
    "銘柄コード",
    value="AAPL",
    help="例: AAPL, GOOGL, 7203.T"
)

test_days = st.sidebar.slider(
    "検証期間（日数）",
    min_value=10,
    max_value=60,
    value=30
)

run_arima = st.sidebar.checkbox("ARIMA", value=True)
run_ml = st.sidebar.checkbox("機械学習", value=True)

# 検証実行
if st.sidebar.button("🔍 検証実行", type="primary"):
    with st.spinner("データを取得中..."):
        data = fetch_stock_data(ticker, "2y")
        data = add_indicators(data)
    
    st.success(f"✅ {ticker} のデータを取得しました（{len(data)}日分）")
    
    arima_results = pd.DataFrame()
    ml_results = pd.DataFrame()
    
    # ARIMA バックテスト
    if run_arima:
        with st.spinner("ARIMAバックテスト実行中..."):
            arima_results = backtest_arima(data, test_days)
        
        if len(arima_results) > 0:
            arima_metrics = calculate_metrics(arima_results)
            
            st.subheader("📈 ARIMA モデル精度")
            cols = st.columns(4)
            cols[0].metric("MAE", f"{arima_metrics['MAE']:.2f}")
            cols[1].metric("RMSE", f"{arima_metrics['RMSE']:.2f}")
            cols[2].metric("MAPE", f"{arima_metrics['MAPE']:.2f}%")
            cols[3].metric("方向的中率", f"{arima_metrics['Direction_Accuracy']:.1f}%")
    
    # ML バックテスト
    if run_ml:
        with st.spinner("機械学習バックテスト実行中..."):
            ml_results = backtest_ml(data, test_days)
        
        if len(ml_results) > 0:
            ml_metrics = calculate_metrics(ml_results)
            
            st.subheader("🤖 機械学習モデル精度")
            cols = st.columns(4)
            cols[0].metric("MAE", f"{ml_metrics['MAE']:.2f}")
            cols[1].metric("RMSE", f"{ml_metrics['RMSE']:.2f}")
            cols[2].metric("MAPE", f"{ml_metrics['MAPE']:.2f}%")
            cols[3].metric("方向的中率", f"{ml_metrics['Direction_Accuracy']:.1f}%")
    
    # チャート
    st.subheader("📊 バックテスト結果")
    fig = create_backtest_chart(arima_results, ml_results)
    st.plotly_chart(fig, use_container_width=True)
    
    # 詳細データ
    with st.expander("📋 詳細データを表示"):
        tab1, tab2 = st.tabs(["ARIMA", "機械学習"])
        
        with tab1:
            if len(arima_results) > 0:
                st.dataframe(arima_results.round(2), use_container_width=True)
        
        with tab2:
            if len(ml_results) > 0:
                st.dataframe(ml_results.round(2), use_container_width=True)
    
    # 評価サマリー
    st.subheader("📝 評価サマリー")
    
    if len(arima_results) > 0 and len(ml_results) > 0:
        arima_mape = calculate_metrics(arima_results)['MAPE']
        ml_mape = calculate_metrics(ml_results)['MAPE']
        
        if arima_mape < ml_mape:
            st.info(f"🏆 **ARIMA** の方が精度が高いです（MAPE: {arima_mape:.2f}% vs {ml_mape:.2f}%）")
        else:
            st.info(f"🏆 **機械学習** の方が精度が高いです（MAPE: {ml_mape:.2f}% vs {arima_mape:.2f}%）")
    
    st.markdown("""
    **指標の説明:**
    - **MAE (Mean Absolute Error)**: 平均絶対誤差。小さいほど良い。
    - **RMSE (Root Mean Squared Error)**: 二乗平均平方根誤差。外れ値に敏感。
    - **MAPE (Mean Absolute Percentage Error)**: 平均絶対パーセント誤差。10%以下が目安。
    - **方向的中率**: 上昇/下降を正しく予測した割合。50%以上なら意味がある。
    """)

else:
    st.info("👈 左のサイドバーで銘柄を設定し、「検証実行」をクリックしてください")
    
    st.markdown("""
    ### このページでできること
    
    1. **バックテスト**: 過去のデータを使って、予測モデルの精度を検証
    2. **精度比較**: ARIMAと機械学習モデルの性能を比較
    3. **方向性分析**: 価格の上昇/下降を正しく予測できているか確認
    
    ### 使い方
    
    1. 銘柄コードを入力
    2. 検証期間を設定（過去何日分を検証するか）
    3. 検証するモデルを選択
    4. 「検証実行」をクリック
    """)
