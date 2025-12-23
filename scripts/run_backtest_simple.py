"""
v7.0 バックテスト実行スクリプト (シンプル版)
"""
import sys
import warnings
warnings.filterwarnings('ignore')

# Streamlitの警告を抑制
class DummySt:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    def cache_data(self, func): return func
    
sys.modules['streamlit'] = type(sys)('streamlit')
sys.modules['streamlit'].info = lambda *a, **k: None
sys.modules['streamlit'].warning = lambda *a, **k: None
sys.modules['streamlit'].error = lambda *a, **k: None
sys.modules['streamlit'].set_page_config = lambda *a, **k: None
sys.modules['streamlit'].title = lambda *a, **k: None
sys.modules['streamlit'].markdown = lambda *a, **k: None
sys.modules['streamlit'].sidebar = type('sidebar', (), {'header': lambda *a, **k: None, 'button': lambda *a, **k: False, 'selectbox': lambda *a, **k: '1年', 'number_input': lambda *a, **k: 1000000, 'subheader': lambda *a, **k: None, 'checkbox': lambda *a, **k: True, 'multiselect': lambda *a, **k: [], 'metric': lambda *a, **k: None})()
sys.modules['streamlit'].expander = lambda *a, **k: type('exp', (), {'__enter__': lambda s: s, '__exit__': lambda *a: None})()
sys.modules['streamlit'].divider = lambda *a, **k: None
sys.modules['streamlit'].button = lambda *a, **k: False
sys.modules['streamlit'].progress = lambda *a, **k: type('prog', (), {'progress': lambda *a: None, 'empty': lambda *a: None})()
sys.modules['streamlit'].empty = lambda *a, **k: type('emp', (), {'text': lambda *a: None})()
sys.modules['streamlit'].spinner = lambda *a, **k: type('sp', (), {'__enter__': lambda s: s, '__exit__': lambda *a: None})()
sys.modules['streamlit'].stop = lambda *a, **k: None
sys.modules['streamlit'].tabs = lambda *a, **k: [type('tab', (), {'__enter__': lambda s: s, '__exit__': lambda *a: None})() for _ in a[0]]
sys.modules['streamlit'].metric = lambda *a, **k: None
sys.modules['streamlit'].columns = lambda *a, **k: [type('col', (), {'metric': lambda *a, **k: None})() for _ in range(a[0] if a else 3)]
sys.modules['streamlit'].dataframe = lambda *a, **k: None
sys.modules['streamlit'].plotly_chart = lambda *a, **k: None
sys.modules['streamlit'].json = lambda *a, **k: None
sys.modules['streamlit'].download_button = lambda *a, **k: None
sys.modules['streamlit'].success = lambda *a, **k: None
sys.modules['streamlit'].cache_data = lambda f: f

sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from database.db_manager import DatabaseManager

# バックテスト関数を読み込み
with open('pages/08_historical_backtest.py', encoding='utf-8') as f:
    code = f.read().split('# ==================== メインUI ====================')[0]

# st参照をダミーに置換
code = code.replace('st.info', 'print')
code = code.replace('st.warning', 'print')
exec(code)

db = DatabaseManager()
watchlist = db.get_watchlist()
tickers = [w['ticker'] for w in watchlist]
print(f'銘柄数: {len(tickers)}')

# バックテスト実行
print('バックテスト実行中...')
try:
    result = run_backtest(tickers, initial_cash=1000000, start_days_ago=252)
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
    result = {'error': str(e)}

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
    print(f'初期資金:     {initial:>12,.0f}円')
    print(f'最終資産:     {final:>12,.0f}円')
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
    
    print(f'\n銘柄別成績TOP10:')
    sorted_stats = sorted(ticker_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)[:10]
    for ticker, stats in sorted_stats:
        wr = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
        print(f'  {ticker:8}: {stats["pnl"]:>+6.1f}% ({stats["count"]:>2}回, 勝率{wr:>3.0f}%)')
    
    # 売り理由別集計
    print(f'\n売り理由別集計:')
    reason_stats = {}
    for t in trades:
        if t['action'] == 'SELL':
            reason = t.get('reason', '不明')
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
