"""
並列更新のテスト
================
10並列でデータ更新を実行し、速度を比較
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import yfinance as yf
from database.db_manager import DatabaseManager
from services.background_updater import BackgroundUpdater

def fetch_stock_data(ticker: str, period: str = "2y"):
    """yfinanceで株価データを取得"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def main():
    print("=" * 60)
    print("並列更新テスト")
    print("=" * 60)
    
    # データベース接続
    db = DatabaseManager()
    
    # ウォッチリストから取得
    tickers = db.get_all_watchlist_tickers()
    print(f"\n📊 ウォッチリスト銘柄数: {len(tickers)}")
    
    # テスト用に最初の50銘柄を使用
    test_tickers = tickers[:50]
    print(f"📋 テスト対象: {len(test_tickers)}銘柄")
    
    # 並列更新（10ワーカー）
    updater = BackgroundUpdater(db, fetch_stock_data, max_workers=10)
    
    print(f"\n🚀 10並列で更新開始...")
    start_time = time.time()
    
    results = updater.update_batch(test_tickers)
    
    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r['success'])
    
    print(f"\n" + "=" * 60)
    print("📈 結果")
    print("=" * 60)
    print(f"✅ 成功: {success_count}/{len(test_tickers)}")
    print(f"⏱️ 所要時間: {elapsed:.1f}秒")
    print(f"⚡ 速度: {len(test_tickers)/elapsed:.1f}銘柄/秒")
    print(f"\n📊 全銘柄({len(tickers)})の場合の推定時間:")
    print(f"   並列更新: 約{len(tickers)/max(len(test_tickers)/elapsed, 0.1):.0f}秒")
    print(f"   順次更新: 約{len(tickers)*0.5 + len(tickers)*1.5:.0f}秒 (0.5秒wait + 1.5秒/銘柄)")
    
    # エラーがあれば表示
    errors = [r for r in results if not r['success']]
    if errors:
        print(f"\n⚠️ エラー ({len(errors)}件):")
        for e in errors[:5]:
            print(f"   {e['ticker']}: {e['error']}")
        if len(errors) > 5:
            print(f"   ... 他{len(errors)-5}件")

if __name__ == "__main__":
    main()
