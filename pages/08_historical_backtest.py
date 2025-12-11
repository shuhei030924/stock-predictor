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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

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
    特定の日付時点でのシグナルを計算
    target_idx: その日のインデックス（その日までのデータのみ使用）
    """
    if target_idx < 50:  # 最低50日分のデータが必要
        return None
    
    # その日までのデータのみ使用（未来のデータは見ない）
    df_slice = df.iloc[:target_idx + 1].copy()
    
    if len(df_slice) < 50:
        return None
    
    # 最新価格（その日の終値）
    current_price = float(df_slice['Close'].iloc[-1])
    prev_price = float(df_slice['Close'].iloc[-2])
    price_change = ((current_price - prev_price) / prev_price) * 100
    
    # RSI (14日)
    delta = df_slice['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_value = float(rsi.iloc[-1])
    
    # RSIシグナル
    if rsi_value < 30:
        rsi_signal = 1.0
    elif rsi_value > 70:
        rsi_signal = -1.0
    else:
        rsi_signal = (50 - rsi_value) / 50
    
    # 移動平均
    sma5 = df_slice['Close'].rolling(window=5).mean()
    sma20 = df_slice['Close'].rolling(window=20).mean()
    sma50 = df_slice['Close'].rolling(window=50).mean()
    
    sma5_val = float(sma5.iloc[-1])
    sma20_val = float(sma20.iloc[-1])
    sma50_val = float(sma50.iloc[-1])
    
    # MAシグナル
    ma_signal = 0.0
    if current_price > sma5_val:
        ma_signal += 0.3
    if sma5_val > sma20_val:
        ma_signal += 0.4
    if sma20_val > sma50_val:
        ma_signal += 0.3
    ma_signal = (ma_signal - 0.5) * 2
    
    # MACD
    ema12 = df_slice['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_slice['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    macd_val = float(macd_line.iloc[-1])
    macd_signal_val = float(signal_line.iloc[-1])
    macd_hist_val = float(macd_hist.iloc[-1])
    
    if macd_val > macd_signal_val and macd_hist_val > 0:
        macd_signal = 1.0
    elif macd_val < macd_signal_val and macd_hist_val < 0:
        macd_signal = -1.0
    else:
        macd_signal = macd_hist_val / (abs(macd_hist_val) + 0.01) * 0.5
    
    # ボリンジャーバンド
    bb_std = df_slice['Close'].rolling(window=20).std().iloc[-1]
    bb_upper = sma20_val + 2 * bb_std
    bb_lower = sma20_val - 2 * bb_std
    
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    bb_signal = (0.5 - bb_position) * 2
    
    # 出来高
    vol_sma = df_slice['Volume'].rolling(window=20).mean()
    vol_ratio = float(df_slice['Volume'].iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1.0
    
    if vol_ratio > 1.5 and price_change > 0:
        vol_signal = 1.0
    elif vol_ratio > 1.5 and price_change < 0:
        vol_signal = -1.0
    else:
        vol_signal = 0.0
    
    # 総合スコア（案C: MACDを重視）
    weights = {
        'rsi': 0.20,
        'ma': 0.25,
        'macd': 0.30,
        'bb': 0.15,
        'volume': 0.10
    }
    
    total_score = (
        rsi_signal * weights['rsi'] +
        ma_signal * weights['ma'] +
        macd_signal * weights['macd'] +
        bb_signal * weights['bb'] +
        vol_signal * weights['volume']
    )
    
    return {
        'price': current_price,
        'change': price_change,
        'rsi': rsi_value,
        'total_score': total_score
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """過去データを取得"""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df is None or len(df) < 100:
            return None
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None


def run_backtest(tickers: list, initial_cash: float = 1000000, 
                 start_days_ago: int = 252, progress_callback=None) -> dict:
    """
    過去1年間のバックテストを実行
    
    Parameters:
    - tickers: 対象銘柄リスト
    - initial_cash: 初期資金
    - start_days_ago: 何日前から開始するか（252=約1年）
    - progress_callback: 進捗コールバック関数
    
    Returns:
    - バックテスト結果
    """
    
    # 各銘柄の過去データを取得
    all_data = {}
    for ticker in tickers:
        df = get_historical_data(ticker, "2y")
        if df is not None and len(df) > start_days_ago + 50:
            all_data[ticker] = df
    
    if not all_data:
        return None
    
    # 共通の日付範囲を決定
    first_ticker = list(all_data.keys())[0]
    date_index = all_data[first_ticker].index[-start_days_ago:]
    
    # バックテスト状態
    cash = initial_cash
    portfolio = {}  # {ticker: {'shares': float, 'avg_cost': float}}
    history = []  # 日次の資産推移
    trades = []  # 取引履歴
    
    total_days = len(date_index)
    
    for day_num, current_date in enumerate(date_index):
        if progress_callback:
            progress_callback(day_num / total_days)
        
        # その日のシグナルを計算（各銘柄）
        daily_signals = {}
        daily_prices = {}
        
        for ticker, df in all_data.items():
            # current_date以前のデータのみ使用
            mask = df.index <= current_date
            valid_idx = mask.sum() - 1
            
            if valid_idx < 50:
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
        
        # ========== 売り処理（先に実行） ==========
        for ticker in list(portfolio.keys()):
            if ticker not in daily_signals or ticker not in daily_prices:
                continue
            
            pos = portfolio[ticker]
            price = daily_prices[ticker]
            score = daily_signals[ticker]['total_score']
            pnl_rate = ((price - pos['avg_cost']) / pos['avg_cost']) * 100
            
            sell_reason = None
            sell_ratio = 0
            
            # 損切り: -10%以下
            if pnl_rate <= -10:
                sell_reason = f"損切り ({pnl_rate:.1f}%)"
                sell_ratio = 1.0
            # 利確: +20%以上
            elif pnl_rate >= 20:
                sell_reason = f"利確 ({pnl_rate:.1f}%)"
                sell_ratio = 0.5
            # 強い売りシグナル
            elif score <= -0.5:
                sell_reason = f"強い売り (スコア {score:.2f})"
                sell_ratio = 1.0
            # 売りシグナル
            elif score <= -0.2:
                sell_reason = f"売り (スコア {score:.2f})"
                sell_ratio = 0.5
            
            if sell_ratio > 0:
                shares_to_sell = pos['shares'] * sell_ratio
                amount = shares_to_sell * price
                cash += amount
                
                trades.append({
                    'date': current_date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares_to_sell,
                    'price': price,
                    'amount': amount,
                    'reason': sell_reason
                })
                
                if sell_ratio >= 1.0:
                    del portfolio[ticker]
                else:
                    portfolio[ticker]['shares'] -= shares_to_sell
        
        # ========== 買い処理 ==========
        # スコア順にソート
        buy_candidates = [
            (t, s) for t, s in daily_signals.items()
            if s['total_score'] >= 0.2 and t not in portfolio
        ]
        buy_candidates.sort(key=lambda x: x[1]['total_score'], reverse=True)
        
        for ticker, signal in buy_candidates:
            score = signal['total_score']
            price = daily_prices[ticker]
            
            # 現金比率チェック（20%キープ）
            current_total = cash + sum(
                portfolio[t]['shares'] * daily_prices.get(t, portfolio[t]['avg_cost'])
                for t in portfolio
            )
            if cash < current_total * 0.20:
                break
            
            # 保有銘柄数チェック（最大10銘柄）
            if len(portfolio) >= 10:
                break
            
            # 購入額決定
            if score >= 0.5:
                buy_amount = current_total * 0.08
            else:
                buy_amount = current_total * 0.05
            
            # 上限チェック
            max_position = current_total * 0.10
            buy_amount = min(buy_amount, max_position)
            
            available_cash = cash - (current_total * 0.20)
            buy_amount = min(buy_amount, available_cash)
            
            if buy_amount > 10000:
                shares = buy_amount / price
                cash -= buy_amount
                
                portfolio[ticker] = {
                    'shares': shares,
                    'avg_cost': price
                }
                
                trades.append({
                    'date': current_date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'amount': buy_amount,
                    'reason': f"{'強い' if score >= 0.5 else ''}買い (スコア {score:.2f})"
                })
        
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
            'num_positions': len(portfolio)
        })
    
    if progress_callback:
        progress_callback(1.0)
    
    return {
        'history': history,
        'trades': trades,
        'final_portfolio': portfolio,
        'final_cash': cash
    }


# ==================== メインUI ====================

# サイドバー設定
st.sidebar.header("⚙️ バックテスト設定")

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
with st.expander("📋 売買アルゴリズム（案C: リスク管理型）", expanded=False):
    st.markdown("""
    ### ウォークフォワードテストとは
    各日の判断は**その日までのデータのみ**を使用し、未来のデータは一切見ません。
    これにより実際の運用に近いシミュレーションが可能です。
    
    ### 資金管理ルール
    - **現金比率**: 常に20%以上キープ
    - **1銘柄上限**: 総資産の10%まで
    - **最大保有銘柄数**: 10銘柄
    
    ### 買いルール
    | スコア | 条件 | 購入額 |
    |--------|------|--------|
    | ≥ 0.5 | 未保有のみ | 総資産の8% |
    | ≥ 0.2 | 未保有のみ | 総資産の5% |
    
    ### 売りルール
    | 条件 | アクション |
    |------|-----------|
    | 損切: -10%到達 | 全売却 |
    | 利確: +20%到達 | 半分売却 |
    | スコア ≤ -0.5 | 全売却 |
    | スコア ≤ -0.2 | 半分売却 |
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
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 最終資産", f"¥{final:,.0f}", delta=f"¥{profit:+,.0f}")
    col2.metric("📈 収益率", f"{profit_rate:+.2f}%")
    col3.metric("📉 最大DD", f"-{max_drawdown:.2f}%")
    col4.metric("🔄 総取引数", len(trades))
    col5.metric("📊 勝率", f"{len([t for t in sell_trades if '利確' in t.get('reason', '')])}/{len(sell_trades)}")
    
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
