import sqlite3
import os

conn = sqlite3.connect('data/stock_predictor.db')
cursor = conn.cursor()

# バックテスト関連テーブルの確認
print('=== バックテスト結果 ===\n')

# backtest_balance（資産推移）
print('【資産推移 (backtest_balance)】')
cursor.execute("SELECT COUNT(*) FROM backtest_balance")
print(f"レコード数: {cursor.fetchone()[0]}")
cursor.execute("SELECT * FROM backtest_balance ORDER BY created_at DESC LIMIT 5")
for row in cursor.fetchall():
    print(f"  ID:{row[0]} 現金:¥{row[1]:,.0f} 株式:¥{row[2]:,.0f} 合計:¥{row[3]:,.0f} ({row[4]})")

# backtest_transactions（取引履歴）
print('\n【取引履歴 (backtest_transactions)】')
cursor.execute("SELECT COUNT(*) FROM backtest_transactions")
print(f"レコード数: {cursor.fetchone()[0]}")
cursor.execute("SELECT * FROM backtest_transactions ORDER BY created_at DESC LIMIT 10")
for row in cursor.fetchall():
    print(f"  {row[1]} {row[2]} {row[3]:.2f}株 @${row[4]:.2f} ({row[7]})")

# backtest_portfolio（最終ポートフォリオ）
print('\n【最終ポートフォリオ (backtest_portfolio)】')
cursor.execute("SELECT * FROM backtest_portfolio")
for row in cursor.fetchall():
    print(f"  {row[1]}: {row[2]:.2f}株 平均単価:${row[3]:.2f}")

conn.close()

# analysis フォルダの確認
print('\n=== analysis フォルダ ===')
analysis_path = 'analysis'
if os.path.exists(analysis_path):
    files = os.listdir(analysis_path)
    print(f"ファイル数: {len(files)}")
    for f in files:
        fpath = os.path.join(analysis_path, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size:,} bytes)")
else:
    print("analysis フォルダが見つかりません")
