"""DBテーブル確認"""
import sqlite3
conn = sqlite3.connect('data/stock_data.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cursor.fetchall()]
print('Tables:', tables)

for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  {table}: {count} rows")
    
    # サンプルデータ
    cursor.execute(f"SELECT * FROM {table} LIMIT 2")
    rows = cursor.fetchall()
    cursor.execute(f"PRAGMA table_info({table})")
    cols = [c[1] for c in cursor.fetchall()]
    print(f"    Columns: {cols}")
    for row in rows:
        print(f"    {row}")
    print()

conn.close()
