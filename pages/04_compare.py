"""
銘柄比較ページ
=============
複数銘柄の比較・相関分析
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db

st.set_page_config(
    page_title="📊 銘柄比較",
    page_icon="📊",
    layout="wide"
)

st.title("📊 銘柄比較")
st.markdown("複数銘柄のパフォーマンス比較・相関分析")

# データベース接続
db = get_db()

# Plotly遅延ロード
@st.cache_resource
def load_plotly():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    return go, make_subplots, px

# yfinance遅延ロード
@st.cache_resource
def load_yfinance():
    try:
        import yfinance as yf
        return yf, True
    except ImportError:
        return None, False

def fetch_stock_data(ticker: str, period: str = "1y"):
    """株価データを取得"""
    # まずキャッシュをチェック
    period_days = {"1y": 365, "2y": 730, "5y": 1825, "6mo": 180}.get(period, 365)
    
    cached = db.get_cached_prices(ticker, days=period_days)
    if cached is not None and len(cached) > period_days * 0.7:
        return cached
    
    # APIから取得
    yf, available = load_yfinance()
    if available:
        try:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            if len(data) > 0:
                db.cache_prices(ticker, data)
                return data
        except:
            pass
    
    return cached  # APIが失敗したら古いキャッシュを返す

# サイドバー
st.sidebar.header("⚙️ 比較設定")

# ウォッチリストから選択
watchlist = db.get_watchlist()
watchlist_tickers = [f"{w['ticker']} - {w['name'] or ''}" for w in watchlist]

selected_items = st.sidebar.multiselect(
    "比較する銘柄（最大5つ）",
    options=watchlist_tickers,
    max_selections=5,
    default=watchlist_tickers[:2] if len(watchlist_tickers) >= 2 else watchlist_tickers
)

# 選択された銘柄コードを抽出
selected_tickers = [item.split(" - ")[0] for item in selected_items]

# カスタム銘柄追加
custom_ticker = st.sidebar.text_input("カスタム銘柄を追加", placeholder="例: AAPL")
if custom_ticker and custom_ticker.upper() not in selected_tickers:
    if len(selected_tickers) < 5:
        selected_tickers.append(custom_ticker.upper())

# 期間選択
period = st.sidebar.selectbox(
    "比較期間",
    options=["6mo", "1y", "2y"],
    index=1,
    format_func=lambda x: {"6mo": "6ヶ月", "1y": "1年", "2y": "2年"}[x]
)

# 比較基準
normalize = st.sidebar.checkbox("パフォーマンス正規化（開始日=100）", value=True)

st.sidebar.divider()

if st.sidebar.button("📊 比較実行", type="primary"):
    if len(selected_tickers) < 2:
        st.warning("⚠️ 比較するには2つ以上の銘柄を選択してください")
    else:
        go, make_subplots, px = load_plotly()
        
        # データ取得
        with st.spinner("データを取得中..."):
            stock_data = {}
            for ticker in selected_tickers:
                data = fetch_stock_data(ticker, period)
                if data is not None and len(data) > 0:
                    stock_data[ticker] = data
        
        if len(stock_data) < 2:
            st.error("❌ 十分なデータを取得できませんでした")
        else:
            # 共通の日付範囲を見つける
            common_dates = None
            for ticker, data in stock_data.items():
                if common_dates is None:
                    common_dates = set(data.index)
                else:
                    common_dates = common_dates.intersection(set(data.index))
            
            common_dates = sorted(common_dates)
            
            if len(common_dates) == 0:
                st.error("❌ 共通の取引日がありません。別の銘柄を選択してください。")
                st.stop()
            
            # データフレームを作成
            price_df = pd.DataFrame(index=common_dates)
            for ticker, data in stock_data.items():
                price_df[ticker] = data.loc[common_dates, 'Close']
            
            # 空のカラムをチェック
            price_df = price_df.dropna(axis=1, how='all')
            if len(price_df.columns) < 2:
                st.error("❌ 有効なデータが不足しています")
                st.stop()
            
            # ==================== パフォーマンス比較 ====================
            st.subheader("📈 パフォーマンス比較")
            
            if normalize:
                # 開始日を100として正規化
                first_row = price_df.iloc[0]
                # ゼロ除算を避ける
                first_row = first_row.replace(0, np.nan)
                normalized_df = price_df / first_row * 100
                fig1 = go.Figure()
                for ticker in normalized_df.columns:
                    fig1.add_trace(go.Scatter(
                        x=normalized_df.index,
                        y=normalized_df[ticker],
                        name=ticker,
                        mode='lines'
                    ))
                fig1.update_layout(
                    title="正規化パフォーマンス（開始日=100）",
                    yaxis_title="パフォーマンス",
                    height=500,
                    hovermode='x unified'
                )
            else:
                fig1 = go.Figure()
                for ticker in price_df.columns:
                    fig1.add_trace(go.Scatter(
                        x=price_df.index,
                        y=price_df[ticker],
                        name=ticker,
                        mode='lines'
                    ))
                fig1.update_layout(
                    title="株価推移",
                    yaxis_title="株価",
                    height=500,
                    hovermode='x unified'
                )
            
            st.plotly_chart(fig1, use_container_width=True)
            
            # ==================== リターン比較 ====================
            st.subheader("📊 リターン比較")
            
            returns_df = price_df.pct_change().dropna()
            
            # 累積リターン
            cumulative_returns = (1 + returns_df).cumprod() - 1
            
            # 期間リターンの計算
            total_return = (price_df.iloc[-1] / price_df.iloc[0] - 1) * 100
            annualized_return = ((price_df.iloc[-1] / price_df.iloc[0]) ** (252 / len(price_df)) - 1) * 100
            volatility = returns_df.std() * np.sqrt(252) * 100
            sharpe = annualized_return / volatility
            max_drawdown = ((price_df / price_df.cummax()) - 1).min() * 100
            
            # メトリクス表示
            metrics_df = pd.DataFrame({
                '総リターン(%)': total_return.round(2),
                '年率リターン(%)': annualized_return.round(2),
                'ボラティリティ(%)': volatility.round(2),
                'シャープ比率': sharpe.round(2),
                '最大ドローダウン(%)': max_drawdown.round(2)
            })
            
            st.dataframe(metrics_df.T.style.format("{:.2f}"), use_container_width=True)
            
            # 棒グラフで比較
            col1, col2 = st.columns(2)
            
            with col1:
                fig2 = go.Figure(data=[
                    go.Bar(name='総リターン', x=selected_tickers, y=total_return.values, marker_color='blue'),
                ])
                fig2.update_layout(title="総リターン比較", height=300)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                fig3 = go.Figure(data=[
                    go.Bar(name='ボラティリティ', x=selected_tickers, y=volatility.values, marker_color='orange'),
                ])
                fig3.update_layout(title="ボラティリティ比較", height=300)
                st.plotly_chart(fig3, use_container_width=True)
            
            # ==================== 相関分析 ====================
            st.subheader("🔗 相関分析")
            
            correlation = returns_df.corr()
            
            # ヒートマップ
            fig4 = go.Figure(data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                colorscale='RdBu',
                zmid=0,
                text=correlation.round(2).values,
                texttemplate='%{text}',
                textfont={"size": 14},
                hoverongaps=False
            ))
            fig4.update_layout(title="相関係数マトリックス", height=400)
            st.plotly_chart(fig4, use_container_width=True)
            
            # 相関の解釈
            st.markdown("""
            **相関係数の解釈:**
            - **1.0**: 完全な正の相関（同じ方向に動く）
            - **0.0**: 相関なし（独立した動き）
            - **-1.0**: 完全な負の相関（逆方向に動く）
            
            💡 分散投資には相関の低い銘柄を組み合わせるのが効果的です。
            """)
            
            # ==================== リスク・リターン散布図 ====================
            st.subheader("⚖️ リスク・リターン分析")
            
            fig5 = go.Figure()
            
            for ticker in selected_tickers:
                fig5.add_trace(go.Scatter(
                    x=[volatility[ticker]],
                    y=[annualized_return[ticker]],
                    mode='markers+text',
                    name=ticker,
                    text=[ticker],
                    textposition='top center',
                    marker=dict(size=15)
                ))
            
            fig5.update_layout(
                title="リスク・リターン散布図",
                xaxis_title="ボラティリティ（リスク）%",
                yaxis_title="年率リターン %",
                height=400
            )
            st.plotly_chart(fig5, use_container_width=True)
            
            st.info("💡 右上にある銘柄ほど「高リターン・高リスク」、左上ほど「高リターン・低リスク」（理想的）")

else:
    st.info("👈 サイドバーで銘柄を選択し、「比較実行」をクリックしてください")
    
    # サンプル比較の提案
    st.subheader("💡 おすすめ比較")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🇺🇸 FAANG比較**
        - META (Facebook)
        - AAPL (Apple)
        - AMZN (Amazon)
        - NFLX (Netflix)
        - GOOGL (Google)
        """)
    
    with col2:
        st.markdown("""
        **🇯🇵 日本半導体関連**
        - 8035.T (東京エレクトロン)
        - 6857.T (アドバンテスト)
        - 6146.T (ディスコ)
        - 6723.T (ルネサス)
        """)
