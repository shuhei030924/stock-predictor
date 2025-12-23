"""データベースの内容を解析"""
import sqlite3
import pandas as pd
import json
from pathlib import Path

conn = sqlite3.connect('data/stock_data.db')

print('='*70)
print('📊 データベース解析レポート')
print('='*70)

# 1. 日足キャッシュ
print('\n1. 日足データ (price_cache)')
print('-'*70)
try:
    df_daily = pd.read_sql('SELECT * FROM price_cache LIMIT 3', conn)
    print(df_daily.to_string())
    count_daily = pd.read_sql('SELECT COUNT(*) as cnt, COUNT(DISTINCT ticker) as tickers FROM price_cache', conn)
    print(f'\n   総レコード: {count_daily.iloc[0]["cnt"]:,} | 銘柄数: {count_daily.iloc[0]["tickers"]}')
except Exception as e:
    print(f'   エラー: {e}')

# 2. 1時間足キャッシュ
print('\n2. 1時間足データ (price_cache_1h)')
print('-'*70)
try:
    df_1h = pd.read_sql('SELECT * FROM price_cache_1h LIMIT 3', conn)
    print(df_1h.to_string())
    count_1h = pd.read_sql('SELECT COUNT(*) as cnt, COUNT(DISTINCT ticker) as tickers FROM price_cache_1h', conn)
    print(f'\n   総レコード: {count_1h.iloc[0]["cnt"]:,} | 銘柄数: {count_1h.iloc[0]["tickers"]}')
except Exception as e:
    print(f'   エラー: {e}')

# 3. 各銘柄の期間（日足）
print('\n3. 銘柄別データ期間 (日足)')
print('-'*70)
try:
    periods = pd.read_sql('''
        SELECT ticker, MIN(date) as start_date, MAX(date) as end_date, COUNT(*) as rows
        FROM price_cache
        GROUP BY ticker
    ''', conn)
    print(periods.to_string())
except Exception as e:
    print(f'   エラー: {e}')

# 4. 各銘柄の期間（1時間足）
print('\n4. 銘柄別データ期間 (1時間足)')
print('-'*70)
try:
    periods_1h = pd.read_sql('''
        SELECT ticker, MIN(datetime) as start_dt, MAX(datetime) as end_dt, COUNT(*) as rows
        FROM price_cache_1h
        GROUP BY ticker
    ''', conn)
    print(periods_1h.to_string())
except Exception as e:
    print(f'   エラー: {e}')

conn.close()

# 5. バックテスト結果の解析
print('\n' + '='*70)
print('📈 バックテスト結果解析')
print('='*70)

analysis_dir = Path('analysis')
if analysis_dir.exists():
    json_files = sorted(analysis_dir.glob('backtest_*.json'), reverse=True)
    for jf in json_files[:3]:  # 最新3件
        print(f'\n📁 {jf.name}')
        print('-'*70)
        with open(jf, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'summary' in data:
            s = data['summary']
            print(f"   期間: {s.get('period', 'N/A')}")
            print(f"   総リターン: {s.get('total_return', 'N/A')}")
            print(f"   勝率: {s.get('win_rate', 'N/A')}")
            print(f"   取引数: {s.get('total_trades', 'N/A')}")
            print(f"   シャープレシオ: {s.get('sharpe_ratio', 'N/A')}")
            print(f"   最大DD: {s.get('max_drawdown', 'N/A')}")

print('\n' + '='*70)
print('🚀 次のステップ推奨')
print('='*70)
print('''
1. 【パフォーマンス比較】
   - 日足 vs 1時間足のバックテスト結果を比較
   - GPU vs CPU の処理時間を比較

2. 【モデル改善】
   - 特徴量エンジニアリング強化
   - ハイパーパラメータチューニング
   - アンサンブル学習（XGBoost + LightGBM）

3. 【リスク管理強化】
   - ポジションサイジングの最適化
   - 動的ストップロス
   - ボラティリティ調整

4. 【運用機能】
   - リアルタイムシグナル通知
   - 自動売買API連携
   - ダッシュボード強化
''')
