"""
問題銘柄を削除・更新するスクリプト
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from database.db_manager import DatabaseManager

# 削除する銘柄（上場廃止・シンボル変更）
REMOVE_TICKERS = [
    '2651.T',   # ローソン - 2024年7月上場廃止
    '8355.T',   # 静岡銀行 - Yahoo Financeでデータなし
    '9613.T',   # NTTデータ - 旧シンボル  
    'SQ',       # Block - 旧シンボル（XYZに変更）
]

# 追加する銘柄（代替）
ADD_TICKERS = {
    'XYZ': {'name': 'Block (旧Square)', 'sector': 'Financials', 'market': 'NYSE'},
}

def main():
    db = DatabaseManager()
    conn = db._get_connection()
    
    print("=" * 60)
    print("問題銘柄の修正")
    print("=" * 60)
    
    # 削除
    print("\n❌ 削除する銘柄:")
    for ticker in REMOVE_TICKERS:
        try:
            cursor = conn.execute("DELETE FROM watchlist WHERE ticker = ?", (ticker,))
            if cursor.rowcount > 0:
                print(f"   {ticker} - 削除完了")
            else:
                print(f"   {ticker} - 存在しませんでした")
        except Exception as e:
            print(f"   {ticker} - エラー: {e}")
    
    # キャッシュからも削除
    print("\n🗑️ キャッシュ削除:")
    for ticker in REMOVE_TICKERS:
        try:
            conn.execute("DELETE FROM price_cache WHERE ticker = ?", (ticker,))
            print(f"   {ticker} - キャッシュ削除")
        except:
            pass
    
    conn.commit()
    
    # 追加
    print("\n✅ 追加する銘柄:")
    for ticker, info in ADD_TICKERS.items():
        try:
            db.add_to_watchlist(ticker, info['name'], info['sector'], info['market'])
            print(f"   {ticker} ({info['name']}) - 追加完了")
        except Exception as e:
            print(f"   {ticker} - エラー: {e}")
    
    # 確認
    count = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
    print(f"\n📊 最終登録数: {count}銘柄")
    
    conn.close()

if __name__ == "__main__":
    main()
