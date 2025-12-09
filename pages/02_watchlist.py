"""
ウォッチリスト管理ページ
=======================
銘柄の追加・削除・管理機能
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
    page_title="📋 ウォッチリスト",
    page_icon="📋",
    layout="wide"
)

st.title("📋 ウォッチリスト管理")
st.markdown("銘柄の追加・削除・お気に入り管理")

# データベース接続
db = get_db()

# サイドバー: 銘柄追加
st.sidebar.header("➕ 銘柄を追加")

# 人気銘柄のプリセット
presets = {
    "🇺🇸 米国株": {
        "AAPL": "Apple Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corp.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corp.",
        "META": "Meta Platforms Inc.",
    },
    "🇯🇵 日本株": {
        "7203.T": "トヨタ自動車",
        "9984.T": "ソフトバンクグループ",
        "6758.T": "ソニーグループ",
        "6861.T": "キーエンス",
        "9432.T": "NTT",
        "8306.T": "三菱UFJ",
        "6501.T": "日立製作所",
    },
    "📈 ETF": {
        "SPY": "S&P 500 ETF",
        "QQQ": "NASDAQ-100 ETF",
        "VTI": "Total Stock Market ETF",
        "1306.T": "TOPIX連動ETF",
        "1321.T": "日経225連動ETF",
    }
}

# プリセットから追加
preset_category = st.sidebar.selectbox("カテゴリ", list(presets.keys()))
preset_tickers = presets[preset_category]
selected_preset = st.sidebar.selectbox(
    "銘柄を選択",
    options=["選択してください..."] + list(preset_tickers.keys()),
    format_func=lambda x: f"{x} - {preset_tickers.get(x, '')}" if x in preset_tickers else x
)

if selected_preset != "選択してください..." and st.sidebar.button("📥 プリセットから追加"):
    ticker = selected_preset
    name = preset_tickers[ticker]
    if db.add_to_watchlist(ticker, name=name, market=preset_category):
        st.sidebar.success(f"✅ {ticker} を追加しました")
        st.rerun()
    else:
        st.sidebar.error("追加に失敗しました")

st.sidebar.divider()

# カスタム銘柄追加
st.sidebar.subheader("✏️ カスタム追加")
with st.sidebar.form("add_ticker_form"):
    new_ticker = st.text_input("銘柄コード", placeholder="例: AAPL, 7203.T")
    new_name = st.text_input("銘柄名（任意）", placeholder="例: Apple Inc.")
    new_sector = st.text_input("セクター（任意）", placeholder="例: Technology")
    new_notes = st.text_area("メモ（任意）", placeholder="投資理由など...")
    
    if st.form_submit_button("➕ 追加", type="primary"):
        if new_ticker:
            if db.add_to_watchlist(
                ticker=new_ticker,
                name=new_name if new_name else None,
                sector=new_sector if new_sector else None,
                notes=new_notes if new_notes else None
            ):
                st.success(f"✅ {new_ticker.upper()} を追加しました")
                st.rerun()
            else:
                st.error("追加に失敗しました")
        else:
            st.warning("銘柄コードを入力してください")

# メインエリア: ウォッチリスト表示
st.divider()

# フィルタオプション
col1, col2, col3 = st.columns([2, 2, 6])
with col1:
    show_favorites_only = st.checkbox("⭐ お気に入りのみ")
with col2:
    if st.button("🔄 更新"):
        st.rerun()

# ウォッチリスト取得
watchlist = db.get_watchlist(favorites_only=show_favorites_only)

if not watchlist:
    st.info("📭 ウォッチリストが空です。サイドバーから銘柄を追加してください。")
else:
    st.subheader(f"📊 登録銘柄 ({len(watchlist)}件)")
    
    # 銘柄カード表示
    for i in range(0, len(watchlist), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(watchlist):
                item = watchlist[i + j]
                with col:
                    with st.container(border=True):
                        # ヘッダー
                        header_col1, header_col2 = st.columns([4, 1])
                        with header_col1:
                            favorite_icon = "⭐" if item['is_favorite'] else "☆"
                            st.markdown(f"### {favorite_icon} {item['ticker']}")
                        with header_col2:
                            if st.button("🗑️", key=f"del_{item['ticker']}", help="削除"):
                                db.remove_from_watchlist(item['ticker'])
                                st.rerun()
                        
                        # 銘柄情報
                        if item['name']:
                            st.caption(item['name'])
                        
                        info_parts = []
                        if item['sector']:
                            info_parts.append(f"🏷️ {item['sector']}")
                        if item['market']:
                            info_parts.append(f"🌍 {item['market']}")
                        if info_parts:
                            st.caption(" | ".join(info_parts))
                        
                        # メモ
                        if item['notes']:
                            with st.expander("📝 メモ"):
                                st.write(item['notes'])
                        
                        # 追加日
                        added_at = datetime.fromisoformat(item['added_at'])
                        st.caption(f"追加: {added_at.strftime('%Y/%m/%d')}")
                        
                        # アクションボタン
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            fav_label = "☆ お気に入り解除" if item['is_favorite'] else "⭐ お気に入り"
                            if st.button(fav_label, key=f"fav_{item['ticker']}", use_container_width=True):
                                db.toggle_favorite(item['ticker'])
                                st.rerun()
                        with btn_col2:
                            if st.button("📈 分析", key=f"analyze_{item['ticker']}", use_container_width=True):
                                st.session_state['selected_ticker'] = item['ticker']
                                st.switch_page("app.py")

# 一括操作
st.divider()
st.subheader("⚙️ 一括操作")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📤 CSVエクスポート", use_container_width=True):
        if watchlist:
            df = pd.DataFrame(watchlist)
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 ダウンロード",
                data=csv,
                file_name=f"watchlist_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

with col2:
    uploaded_file = st.file_uploader("📥 CSVインポート", type=['csv'], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            import_df = pd.read_csv(uploaded_file)
            if 'ticker' in import_df.columns:
                count = 0
                for _, row in import_df.iterrows():
                    if db.add_to_watchlist(
                        ticker=row['ticker'],
                        name=row.get('name'),
                        sector=row.get('sector'),
                        notes=row.get('notes')
                    ):
                        count += 1
                st.success(f"✅ {count}件の銘柄をインポートしました")
                st.rerun()
            else:
                st.error("CSVに'ticker'列が必要です")
        except Exception as e:
            st.error(f"インポートエラー: {e}")

with col3:
    if st.button("🗑️ 全削除", type="secondary", use_container_width=True):
        if st.session_state.get('confirm_delete_all'):
            for item in watchlist:
                db.remove_from_watchlist(item['ticker'])
            st.session_state['confirm_delete_all'] = False
            st.success("✅ 全ての銘柄を削除しました")
            st.rerun()
        else:
            st.session_state['confirm_delete_all'] = True
            st.warning("⚠️ もう一度クリックで全削除を確定")

# 統計情報
st.divider()
stats = db.get_stats()
st.subheader("📊 データベース統計")

stat_cols = st.columns(5)
stat_cols[0].metric("ウォッチリスト", stats['watchlist_count'])
stat_cols[1].metric("キャッシュ銘柄数", stats['cached_tickers'])
stat_cols[2].metric("キャッシュ価格数", f"{stats['cached_prices']:,}")
stat_cols[3].metric("予測履歴", stats['total_predictions'])
stat_cols[4].metric("アクティブアラート", stats['active_alerts'])
