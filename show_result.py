import json
import os

# 最新のバックテスト結果を読み込む
analysis_dir = 'analysis'
files = sorted([f for f in os.listdir(analysis_dir) if f.startswith('backtest_') and f.endswith('.json')])

if files:
    latest = files[-1]
    print(f'最新ファイル: {latest}')
    
    with open(os.path.join(analysis_dir, latest), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"キー: {list(data.keys())}")
    analysis = data.get('analysis', {})
    print(f"\n=== 分析結果 ===")
    for k, v in analysis.items():
        print(f"{k}: {v}")
else:
    print('バックテスト結果が見つかりません')
