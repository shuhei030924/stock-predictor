"""
データ管理ページ
===============
キャッシュ管理・バックグラウンド更新
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_db

st.set_page_config(
    page_title="🗄️ データ管理",
    page_icon="🗄️",
    layout="wide"
)

st.title("🗄️ データ管理")
st.markdown("キャッシュ管理・バックグラウンド更新・データベース操作")

# データベース接続
db = get_db()

# タブ
tab1, tab2, tab3 = st.tabs(["📊 キャッシュ状況", "🔄 一括更新", "⚙️ データベース"])

# ==================== キャッシュ状況 ====================
with tab1:
    st.subheader("📊 銘柄別キャッシュ状況")
    
    watchlist = db.get_watchlist()
    
    if not watchlist:
        st.info("ウォッチリストに銘柄がありません")
    else:
        # キャッシュ状況を取得
        cache_data = []
        for item in watchlist:
            ticker = item['ticker']
            status = db.get_cache_status(ticker)
            cache_data.append({
                '銘柄': ticker,
                '銘柄名': item['name'] or '-',
                'キャッシュ': '✅' if status['has_cache'] else '❌',
                'レコード数': status['record_count'],
                '最古データ': status['oldest_date'] or '-',
                '最新データ': status['latest_date'] or '-',
                '最終更新': status['last_update'][:16] if status['last_update'] else '-',
                '状態': '🟢 新鮮' if status['is_fresh'] else ('🟡 古い' if status['has_cache'] else '🔴 なし')
            })
        
        df = pd.DataFrame(cache_data)
        
        # フィルタ
        col1, col2 = st.columns([1, 3])
        with col1:
            filter_option = st.selectbox(
                "フィルタ",
                ["すべて", "🟢 新鮮のみ", "🟡 古いのみ", "🔴 キャッシュなし"]
            )
        
        if filter_option == "🟢 新鮮のみ":
            df = df[df['状態'] == '🟢 新鮮']
        elif filter_option == "🟡 古いのみ":
            df = df[df['状態'] == '🟡 古い']
        elif filter_option == "🔴 キャッシュなし":
            df = df[df['状態'] == '🔴 なし']
        
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # サマリー
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        total = len(cache_data)
        fresh = sum(1 for d in cache_data if d['状態'] == '🟢 新鮮')
        stale = sum(1 for d in cache_data if d['状態'] == '🟡 古い')
        no_cache = sum(1 for d in cache_data if d['状態'] == '🔴 なし')
        
        col1.metric("総銘柄数", total)
        col2.metric("🟢 新鮮", fresh)
        col3.metric("🟡 古い", stale)
        col4.metric("🔴 なし", no_cache)

# ==================== 一括更新 ====================
with tab2:
    st.subheader("🔄 株価データの一括更新")
    
    # 更新対象の選択
    update_option = st.radio(
        "更新対象",
        ["古いキャッシュのみ更新", "全銘柄を更新"],
        horizontal=True
    )
    
    # 古い銘柄のリスト
    stale_tickers = db.get_stale_tickers(max_age_hours=24)
    
    if update_option == "古いキャッシュのみ更新":
        st.info(f"📋 更新対象: {len(stale_tickers)}件")
        if stale_tickers:
            st.caption(", ".join(stale_tickers))
    else:
        all_tickers = db.get_all_watchlist_tickers()
        st.info(f"📋 更新対象: {len(all_tickers)}件（全銘柄）")
    
    # 更新実行
    if st.button("🚀 更新開始", type="primary"):
        # yfinanceをインポート
        try:
            import yfinance as yf
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            
            tickers_to_update = stale_tickers if update_option == "古いキャッシュのみ更新" else db.get_all_watchlist_tickers()
            
            if not tickers_to_update:
                st.success("✅ 更新が必要な銘柄はありません")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                for i, ticker in enumerate(tickers_to_update):
                    status_text.text(f"更新中: {ticker} ({i+1}/{len(tickers_to_update)})")
                    
                    try:
                        stock = yf.Ticker(ticker)
                        data = stock.history(period="2y")
                        
                        if len(data) > 0:
                            count = db.cache_prices(ticker, data)
                            results.append({'ticker': ticker, 'status': '✅', 'records': count})
                        else:
                            results.append({'ticker': ticker, 'status': '⚠️', 'records': 0})
                    except Exception as e:
                        results.append({'ticker': ticker, 'status': '❌', 'records': 0})
                    
                    progress_bar.progress((i + 1) / len(tickers_to_update))
                
                status_text.text("完了!")
                
                # 結果表示
                success = sum(1 for r in results if r['status'] == '✅')
                st.success(f"✅ 更新完了: {success}/{len(results)}件成功")
                
                # 詳細結果
                with st.expander("詳細結果"):
                    result_df = pd.DataFrame(results)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    
        except ImportError:
            st.error("❌ yfinanceがインストールされていません")
    
    st.divider()
    
    # 自動更新設定（将来の機能）
    st.subheader("⏰ 自動更新設定")
    st.info("🚧 自動更新機能は開発中です。現在は手動更新をご利用ください。")

# ==================== データベース ====================
with tab3:
    st.subheader("⚙️ データベース管理")
    
    # 統計情報
    stats = db.get_stats()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("ウォッチリスト", f"{stats['watchlist_count']}件")
    col2.metric("キャッシュ銘柄", f"{stats['cached_tickers']}件")
    col3.metric("価格レコード", f"{stats['cached_prices']:,}件")
    
    st.divider()
    
    # キャッシュ操作
    st.subheader("🗑️ キャッシュ操作")
    
    col1, col2 = st.columns(2)
    
    with col1:
        days_to_keep = st.number_input("保持する日数", min_value=30, max_value=1825, value=365)
        if st.button("古いデータを削除"):
            deleted = db.clear_old_cache(days=days_to_keep)
            st.success(f"✅ {deleted}件のレコードを削除しました")
    
    with col2:
        if st.button("全キャッシュをクリア", type="secondary"):
            if st.session_state.get('confirm_clear_cache'):
                deleted = db.clear_old_cache(days=0)
                st.session_state['confirm_clear_cache'] = False
                st.success(f"✅ 全キャッシュ（{deleted}件）を削除しました")
                st.rerun()
            else:
                st.session_state['confirm_clear_cache'] = True
                st.warning("⚠️ もう一度クリックで全削除を確定")
    
    st.divider()
    
    # データベースファイル情報
    st.subheader("📁 データベースファイル")
    
    db_path = Path(db.db_path)
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        st.info(f"""
        - **パス**: `{db_path}`
        - **サイズ**: {size_mb:.2f} MB
        - **最終更新**: {datetime.fromtimestamp(db_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}
        """)
    else:
        st.warning("データベースファイルが見つかりません")
