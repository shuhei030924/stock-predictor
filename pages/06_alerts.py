"""
アラート管理ページ
=================
価格アラート・通知設定
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import get_db

st.set_page_config(
    page_title="🔔 アラート",
    page_icon="🔔",
    layout="wide"
)

st.title("🔔 価格アラート")
st.markdown("目標価格に達したら通知")

# データベース接続
db = get_db()

# yfinance遅延ロード
def get_current_price(ticker: str) -> float:
    """現在の株価を取得"""
    cached = db.get_cached_prices(ticker, days=7)
    if cached is not None and len(cached) > 0:
        return float(cached['Close'].iloc[-1])
    
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

# サイドバー: アラート追加
st.sidebar.header("➕ アラート追加")

# ウォッチリストから選択
watchlist = db.get_watchlist()
watchlist_options = ["選択してください..."] + [f"{w['ticker']} - {w['name'] or ''}" for w in watchlist]

with st.sidebar.form("add_alert_form"):
    selected_ticker = st.selectbox("銘柄", options=watchlist_options)
    custom_ticker = st.text_input("または銘柄コードを入力", placeholder="例: AAPL")
    
    alert_type = st.radio("条件", ["以上になったら", "以下になったら"])
    target_price = st.number_input("目標価格 ($)", min_value=0.0, step=0.01, value=0.0)
    note = st.text_input("メモ（任意）")
    
    if st.form_submit_button("➕ アラート追加", type="primary"):
        # 銘柄コードを決定
        if custom_ticker:
            ticker = custom_ticker.upper()
        elif selected_ticker != "選択してください...":
            ticker = selected_ticker.split(" - ")[0]
        else:
            ticker = None
        
        if ticker and target_price > 0:
            condition = "above" if "以上" in alert_type else "below"
            if db.add_alert(
                ticker=ticker,
                target_price=target_price,
                condition=condition,
                note=note if note else None
            ):
                st.success(f"✅ アラートを追加しました")
                st.rerun()
            else:
                st.error("追加に失敗しました")
        else:
            st.warning("銘柄と目標価格を入力してください")

# メインエリア
st.divider()

# アラート取得
alerts = db.get_alerts()

# タブ
tab1, tab2 = st.tabs(["📊 アクティブなアラート", "📜 発動済み"])

with tab1:
    active_alerts = [a for a in alerts if a.get('is_active', True) and not a.get('triggered', False)]
    
    if not active_alerts:
        st.info("📭 アクティブなアラートはありません")
    else:
        # 現在価格チェック
        st.subheader(f"🔔 アクティブなアラート ({len(active_alerts)}件)")
        
        with st.spinner("価格をチェック中..."):
            alert_data = []
            triggered = []
            
            for alert in active_alerts:
                current_price = get_current_price(alert['ticker'])
                
                # 条件チェック
                is_triggered = False
                if current_price:
                    if alert['condition'] == 'above' and current_price >= alert['target_price']:
                        is_triggered = True
                    elif alert['condition'] == 'below' and current_price <= alert['target_price']:
                        is_triggered = True
                
                if is_triggered:
                    triggered.append(alert)
                
                # 現在価格との距離
                if current_price:
                    distance = ((alert['target_price'] - current_price) / current_price) * 100
                else:
                    distance = None
                
                alert_data.append({
                    'id': alert['id'],
                    'ticker': alert['ticker'],
                    'condition': '📈 以上' if alert['condition'] == 'above' else '📉 以下',
                    'target_price': alert['target_price'],
                    'current_price': current_price,
                    'distance': distance,
                    'note': alert.get('note', ''),
                    'is_triggered': is_triggered
                })
            
            # 発動したアラートを通知
            if triggered:
                st.warning(f"⚠️ {len(triggered)}件のアラートが発動しました！")
                for t in triggered:
                    price = get_current_price(t['ticker'])
                    condition_text = "以上" if t['condition'] == 'above' else "以下"
                    st.success(f"""
                    🎯 **{t['ticker']}** が目標価格 **${t['target_price']:.2f}** {condition_text}に達しました！
                    現在価格: **${price:.2f}**
                    """)
        
        # テーブル表示
        display_data = []
        for a in alert_data:
            row = {
                '銘柄': a['ticker'],
                '条件': a['condition'],
                '目標価格': f"${a['target_price']:.2f}",
                '現在価格': f"${a['current_price']:.2f}" if a['current_price'] else '-',
                '距離': f"{a['distance']:+.1f}%" if a['distance'] else '-',
                '状態': '🎯 発動!' if a['is_triggered'] else '⏳ 待機中',
                'メモ': a['note'] or ''
            }
            display_data.append(row)
        
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # 削除
        st.subheader("🗑️ アラート削除")
        
        alert_options = [f"{a['id']}: {a['ticker']} {a['condition']} ${a['target_price']:.2f}" for a in alert_data]
        selected_alert = st.selectbox("削除するアラート", options=["選択してください..."] + alert_options)
        
        if selected_alert != "選択してください...":
            alert_id = int(selected_alert.split(":")[0])
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🗑️ 削除", type="secondary"):
                    if db.delete_alert(alert_id):
                        st.success("✅ 削除しました")
                        st.rerun()
            
            with col2:
                if st.button("✅ 発動済みにする"):
                    if db.mark_alert_triggered(alert_id):
                        st.success("✅ 発動済みにしました")
                        st.rerun()

with tab2:
    triggered_alerts = [a for a in alerts if a.get('triggered', False)]
    
    if not triggered_alerts:
        st.info("📭 発動済みのアラートはありません")
    else:
        st.subheader(f"✅ 発動済みアラート ({len(triggered_alerts)}件)")
        
        display_data = []
        for a in triggered_alerts:
            row = {
                '銘柄': a['ticker'],
                '条件': '📈 以上' if a['condition'] == 'above' else '📉 以下',
                '目標価格': f"${a['target_price']:.2f}",
                '発動日時': a.get('triggered_at', '-'),
                'メモ': a.get('note', '')
            }
            display_data.append(row)
        
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        if st.button("🗑️ 発動済みをすべて削除"):
            for a in triggered_alerts:
                db.delete_alert(a['id'])
            st.success("✅ 削除しました")
            st.rerun()

# 使い方
st.divider()
with st.expander("💡 使い方"):
    st.markdown("""
    ### アラート機能について
    
    1. **サイドバーからアラートを追加**
       - 銘柄と目標価格を設定
       - 「以上になったら」または「以下になったら」を選択
    
    2. **アラートのチェック**
       - このページを開くと自動的に現在価格をチェック
       - 条件を満たしたアラートは「発動!」と表示
    
    3. **通知について**
       - 現在は画面上での通知のみ
       - 将来的にはメール/LINE通知も検討中
    
    💡 **Tips**: 利確や損切りラインの管理に活用できます
    """)
