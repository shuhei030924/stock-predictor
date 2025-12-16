"""v10.3 バックテストテスト"""
import sys
import os
sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("開始...")
print(f"現在のディレクトリ: {os.getcwd()}")

# ファイル存在確認
path = 'pages/08_historical_backtest.py'
print(f"ファイル存在: {os.path.exists(path)}")

# コード読み込み
with open(path, encoding='utf-8') as f:
    content = f.read()
    
# 区切り文字を探す
if '# ==================== メインUI ====================' in content:
    print("区切り文字: 見つかりました")
    code = content.split('# ==================== メインUI ====================')[0]
    print(f"コード長: {len(code)} 文字")
else:
    print("区切り文字: 見つかりません")
    print("最初の1000文字:")
    print(content[:1000])
