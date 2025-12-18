"""フルロバスト性テスト結果の解析"""
import json
import pandas as pd

with open('analysis/robustness_20251219_001520.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']
print('=' * 80)
print('🔬 フルロバスト性テスト結果 (317銘柄 x 6期間)')
print('=' * 80)

# サマリーテーブル作成
df = pd.DataFrame([{
    '期間': r['period'],
    'セット': r['ticker_set'],
    'リターン': r['avg_return'],
    '勝率': r['avg_win_rate'],
    'シャープ': r['avg_sharpe'],
    '銘柄数': r['n_tickers']
} for r in results])

# 期間別集計
print('\n📊 期間別サマリー')
print('-' * 80)
period_summary = df.groupby('期間').agg({
    'リターン': ['mean', 'min', 'max'],
    '勝率': 'mean',
    'シャープ': 'mean'
}).round(2)
print(period_summary)

# セット別集計
print('\n📈 セクター別サマリー (全期間平均)')
print('-' * 80)
set_summary = df.groupby('セット').agg({
    'リターン': 'mean',
    '勝率': 'mean',
    'シャープ': 'mean'
}).sort_values('リターン', ascending=False).round(2)
print(set_summary)

# 全体統計
print('\n📊 全体統計')
print('-' * 80)
positive_count = sum(1 for r in results if r['avg_return'] > 0)
total_count = len(results)
print(f'テストパターン数: {total_count}')
print(f'プラスリターン数: {positive_count} / {total_count} ({positive_count/total_count*100:.1f}%)')
print(f'平均リターン: {df["リターン"].mean():+.2f}%')
print(f'最小リターン: {df["リターン"].min():+.2f}%')
print(f'最大リターン: {df["リターン"].max():+.2f}%')
print(f'平均勝率: {df["勝率"].mean():.1f}%')
print(f'平均シャープ: {df["シャープ"].mean():.2f}')

# 判定
if positive_count / total_count >= 0.7:
    print('\n🎉 戦略は高いロバスト性を示しています！')
elif positive_count / total_count >= 0.5:
    print('\n⚠️ 戦略は中程度のロバスト性です。')
else:
    print('\n❌ 戦略のロバスト性に問題があります。')

# 詳細テーブル表示
print('\n' + '=' * 80)
print('📋 全102パターンの詳細結果')
print('=' * 80)
for period in ['直近1年', '1-2年前', '2-3年前', '3-4年前', '直近2年', '直近3年']:
    period_df = df[df['期間'] == period].copy()
    if len(period_df) > 0:
        print(f'\n📅 {period}')
        print('-' * 60)
        for _, row in period_df.iterrows():
            emoji = '✅' if row['リターン'] > 0 else '❌'
            print(f"  {emoji} {row['セット']}: {row['リターン']:+.1f}% | 勝率{row['勝率']:.0f}% | シャープ{row['シャープ']:.2f}")
