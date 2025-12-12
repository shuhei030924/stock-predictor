"""
v7.0 バックテスト実行スクリプト
"""
import sys
sys.path.insert(0, '.')

from database.db_manager import DatabaseManager
import pandas as pd
import numpy as np

# バックテスト関数を読み込み
with open('pages/08_historical_backtest.py', encoding='utf-8') as f:
    code = f.read().split('# ==================== メインUI ====================')[0]
exec(code)

db = DatabaseManager()
watchlist = db.get_watchlist()
tickers = [w['ticker'] for w in watchlist]
print(f'銘柄数: {len(tickers)}')
print(f'銘柄: {tickers}')

# バックテスト実行
print('\nバックテスト実行中...')
result = run_backtest(tickers, initial_cash=1000000, start_days_ago=252)

if 'error' in result:
    print(f'エラー: {result["error"]}')
else:
    history = result['history']
    trades = result['trades']
    
    initial = history[0]['total_value']
    final = history[-1]['total_value']
    total_return = ((final - initial) / initial) * 100
    
    # 日次リターン計算
    daily_returns = []
    for i in range(1, len(history)):
        prev = history[i-1]['total_value']
        curr = history[i]['total_value']
        daily_returns.append((curr - prev) / prev)
    
    # シャープレシオ
    if daily_returns:
        avg_return = np.mean(daily_returns) * 252
        std_return = np.std(daily_returns) * np.sqrt(252)
        sharpe = avg_return / std_return if std_return > 0 else 0
    else:
        sharpe = 0
    
    # 最大ドローダウン
    peak = history[0]['total_value']
    max_dd = 0
    for h in history:
        if h['total_value'] > peak:
            peak = h['total_value']
        dd = (peak - h['total_value']) / peak
        if dd > max_dd:
            max_dd = dd
    
    # 勝率計算
    sell_trades = [t for t in trades if t['action'] == 'SELL' and 'pnl_rate' in t]
    wins = len([t for t in sell_trades if t['pnl_rate'] > 0])
    losses = len([t for t in sell_trades if t['pnl_rate'] <= 0])
    win_rate = wins / len(sell_trades) * 100 if sell_trades else 0
    
    # 損切り回数
    stop_losses = len([t for t in trades if '損切' in t.get('reason', '')])
    
    print(f'\n{"="*50}')
    print(f'     v7.0 バックテスト結果')
    print(f'{"="*50}')
    print(f'初期資金:     ¥{initial:>12,.0f}')
    print(f'最終資産:     ¥{final:>12,.0f}')
    print(f'総リターン:   {total_return:>+12.2f}%')
    print(f'シャープレシオ: {sharpe:>10.2f}')
    print(f'最大DD:       {max_dd*100:>12.1f}%')
    print(f'勝率:         {win_rate:>11.1f}% ({wins}勝/{losses}敗)')
    print(f'損切り回数:   {stop_losses:>12}回')
    print(f'総取引数:     {len(trades):>12}回')
    
    # ベスト/ワースト
    if sell_trades:
        best = max(sell_trades, key=lambda x: x['pnl_rate'])
        worst = min(sell_trades, key=lambda x: x['pnl_rate'])
        print(f'\nベスト取引:  {best["ticker"]} {best["pnl_rate"]:+.1f}%')
        print(f'ワースト取引: {worst["ticker"]} {worst["pnl_rate"]:+.1f}%')
    
    # 銘柄別集計
    ticker_stats = {}
    for t in sell_trades:
        ticker = t['ticker']
        if ticker not in ticker_stats:
            ticker_stats[ticker] = {'pnl': 0, 'count': 0, 'wins': 0}
        ticker_stats[ticker]['pnl'] += t['pnl_rate']
        ticker_stats[ticker]['count'] += 1
        if t['pnl_rate'] > 0:
            ticker_stats[ticker]['wins'] += 1
    
    print(f'\n銘柄別成績 (取引2回以上):')
    sorted_stats = sorted(ticker_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
    for ticker, stats in sorted_stats:
        if stats['count'] >= 2:
            wr = stats['wins'] / stats['count'] * 100
            print(f'  {ticker:8}: {stats["pnl"]:>+6.1f}% ({stats["count"]:>2}回, 勝率{wr:>3.0f}%)')
    
    # レジーム別分析
    print(f'\n売り理由別集計:')
    reason_stats = {}
    for t in trades:
        if t['action'] == 'SELL':
            reason = t.get('reason', '不明')
            # 理由を簡略化
            if '損切' in reason:
                key = '損切り'
            elif 'トレーリング' in reason:
                key = 'トレーリング'
            elif '利確' in reason:
                key = '利確'
            elif '売り' in reason:
                key = 'シグナル売り'
            elif '整理' in reason:
                key = 'ポジション整理'
            else:
                key = 'その他'
            
            if key not in reason_stats:
                reason_stats[key] = {'count': 0, 'pnl': 0}
            reason_stats[key]['count'] += 1
            reason_stats[key]['pnl'] += t.get('pnl_rate', 0)
    
    for reason, stats in sorted(reason_stats.items(), key=lambda x: x[1]['count'], reverse=True):
        avg_pnl = stats['pnl'] / stats['count'] if stats['count'] > 0 else 0
        print(f'  {reason:12}: {stats["count"]:>3}回, 平均{avg_pnl:>+5.1f}%')
    
    print(f'\n{"="*50}')
