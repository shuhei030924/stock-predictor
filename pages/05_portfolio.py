"""
ポートフォリオ管理ページ
=====================
保有株の管理・損益計算
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db

st.set_page_config(
    page_title="💼 ポートフォリオ",
    page_icon="💼",
    layout="wide"
)

st.title("💼 ポートフォリオ管理")
st.markdown("保有株の損益計算・資産配分")

# データベース接続
db = get_db()

# Plotly遅延ロード
@st.cache_resource
def load_plotly():
    import plotly.graph_objects as go
    import plotly.express as px
    return go, px

# yfinance遅延ロード
def get_current_price(ticker: str) -> float:
    """現在の株価を取得"""
    # まずキャッシュをチェック
    cached = db.get_cached_prices(ticker, days=7)
    if cached is not None and len(cached) > 0:
        return float(cached['Close'].iloc[-1])
    
    # APIから取得
    try:
        import yfinance as yf
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        stock = yf.Ticker(ticker)
        data = stock.history(period="5d")
        if len(data) > 0:
            return float(data['Close'].iloc[-1])
    except:
        pass
    
    return None

# サイドバー: 銘柄追加
st.sidebar.header("➕ 保有株を追加")

# ウォッチリストから選択
watchlist = db.get_watchlist()
watchlist_options = ["選択してください..."] + [f"{w['ticker']} - {w['name'] or ''}" for w in watchlist]

with st.sidebar.form("add_holding_form"):
    selected_ticker = st.selectbox("銘柄", options=watchlist_options)
    custom_ticker = st.text_input("または銘柄コードを入力", placeholder="例: AAPL")
    
    shares = st.number_input("保有株数", min_value=0.0, step=1.0, value=0.0)
    avg_cost = st.number_input("平均取得単価", min_value=0.0, step=0.01, value=0.0)
    purchase_date = st.date_input("購入日（任意）")
    notes = st.text_input("メモ（任意）")
    
    if st.form_submit_button("➕ 追加", type="primary"):
        # 銘柄コードを決定
        if custom_ticker:
            ticker = custom_ticker.upper()
        elif selected_ticker != "選択してください...":
            ticker = selected_ticker.split(" - ")[0]
        else:
            ticker = None
        
        if ticker and shares > 0 and avg_cost > 0:
            if db.add_portfolio_item(
                ticker=ticker,
                shares=shares,
                avg_cost=avg_cost,
                purchase_date=str(purchase_date),
                notes=notes if notes else None
            ):
                st.success(f"✅ {ticker} を追加しました")
                st.rerun()
            else:
                st.error("追加に失敗しました")
        else:
            st.warning("銘柄・株数・取得単価を入力してください")

# メインエリア
st.divider()

# ポートフォリオ取得
portfolio = db.get_portfolio()

if not portfolio:
    st.info("📭 ポートフォリオが空です。サイドバーから保有株を追加してください。")
else:
    go, px = load_plotly()
    
    # 現在価格を取得して損益計算
    with st.spinner("現在価格を取得中..."):
        portfolio_data = []
        for item in portfolio:
            current_price = get_current_price(item['ticker'])
            
            if current_price:
                market_value = item['shares'] * current_price
                cost_basis = item['shares'] * item['avg_cost']
                profit_loss = market_value - cost_basis
                profit_loss_pct = (profit_loss / cost_basis) * 100
            else:
                market_value = None
                cost_basis = item['shares'] * item['avg_cost']
                profit_loss = None
                profit_loss_pct = None
            
            portfolio_data.append({
                'id': item['id'],
                'ticker': item['ticker'],
                'name': item['ticker_name'] or item['ticker'],
                'shares': item['shares'],
                'avg_cost': item['avg_cost'],
                'current_price': current_price,
                'cost_basis': cost_basis,
                'market_value': market_value,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct,
                'notes': item['notes']
            })
    
    # サマリー計算
    total_cost = sum(p['cost_basis'] for p in portfolio_data)
    total_value = sum(p['market_value'] for p in portfolio_data if p['market_value'])
    total_profit = total_value - total_cost if total_value else None
    total_profit_pct = (total_profit / total_cost * 100) if total_profit else None
    
    # サマリーカード
    st.subheader("📊 ポートフォリオサマリー")
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("💰 投資総額", f"¥{total_cost:,.0f}" if total_cost < 10000 else f"${total_cost:,.2f}")
    col2.metric("📈 評価額", f"¥{total_value:,.0f}" if total_value and total_value < 10000 else f"${total_value:,.2f}" if total_value else "-")
    
    if total_profit:
        profit_color = "normal" if total_profit >= 0 else "inverse"
        col3.metric("💵 損益", f"${total_profit:,.2f}", f"{total_profit_pct:+.2f}%", delta_color=profit_color)
    else:
        col3.metric("💵 損益", "-")
    
    col4.metric("📦 銘柄数", len(set(p['ticker'] for p in portfolio_data)))
    
    st.divider()
    
    # 資産配分グラフ
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🥧 資産配分")
        
        # 銘柄別に集計
        allocation = {}
        for p in portfolio_data:
            if p['market_value']:
                if p['ticker'] in allocation:
                    allocation[p['ticker']] += p['market_value']
                else:
                    allocation[p['ticker']] = p['market_value']
        
        if allocation:
            fig1 = go.Figure(data=[go.Pie(
                labels=list(allocation.keys()),
                values=list(allocation.values()),
                hole=0.4,
                textinfo='label+percent',
                textposition='outside'
            )])
            fig1.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("価格データがありません")
    
    with col2:
        st.subheader("📊 損益比較")
        
        profit_data = [(p['ticker'], p['profit_loss']) for p in portfolio_data if p['profit_loss'] is not None]
        
        if profit_data:
            tickers = [d[0] for d in profit_data]
            profits = [d[1] for d in profit_data]
            colors = ['green' if p >= 0 else 'red' for p in profits]
            
            fig2 = go.Figure(data=[go.Bar(
                x=tickers,
                y=profits,
                marker_color=colors,
                text=[f"${p:,.0f}" for p in profits],
                textposition='outside'
            )])
            fig2.update_layout(
                height=350,
                yaxis_title="損益 ($)",
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("価格データがありません")
    
    st.divider()
    
    # 保有銘柄一覧
    st.subheader("📋 保有銘柄一覧")
    
    # データフレームに変換
    display_data = []
    for p in portfolio_data:
        row = {
            '銘柄': p['ticker'],
            '銘柄名': p['name'],
            '株数': f"{p['shares']:,.0f}",
            '取得単価': f"${p['avg_cost']:.2f}",
            '現在値': f"${p['current_price']:.2f}" if p['current_price'] else '-',
            '評価額': f"${p['market_value']:,.0f}" if p['market_value'] else '-',
            '損益': f"${p['profit_loss']:+,.0f}" if p['profit_loss'] is not None else '-',
            '損益率': f"{p['profit_loss_pct']:+.1f}%" if p['profit_loss_pct'] is not None else '-',
        }
        display_data.append(row)
    
    display_df = pd.DataFrame(display_data)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # 編集・削除
    st.subheader("✏️ 編集・削除")
    
    item_options = [f"{p['id']}: {p['ticker']} ({p['shares']}株 @ ${p['avg_cost']})" for p in portfolio_data]
    selected_item = st.selectbox("編集する項目", options=["選択してください..."] + item_options)
    
    if selected_item != "選択してください...":
        item_id = int(selected_item.split(":")[0])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ 削除", type="secondary"):
                if db.delete_portfolio_item(item_id):
                    st.success("✅ 削除しました")
                    st.rerun()
        
        with col2:
            with st.expander("✏️ 編集"):
                new_shares = st.number_input("新しい株数", min_value=0.0, step=1.0, key="edit_shares")
                new_cost = st.number_input("新しい取得単価", min_value=0.0, step=0.01, key="edit_cost")
                
                if st.button("💾 保存"):
                    updates = {}
                    if new_shares > 0:
                        updates['shares'] = new_shares
                    if new_cost > 0:
                        updates['avg_cost'] = new_cost
                    
                    if updates:
                        db.update_portfolio_item(item_id, **updates)
                        st.success("✅ 更新しました")
                        st.rerun()
