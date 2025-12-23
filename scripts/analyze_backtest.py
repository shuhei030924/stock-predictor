"""
バックテスト詳細分析スクリプト
"""
import sys
import warnings
warnings.filterwarnings('ignore')

# Streamlitモック
class DummySt:
    def info(self, *args, **kwargs): pass
    def warning(self, *args, **kwargs): pass
    def error(self, *args, **kwargs): pass
    
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
sys.modules['streamlit'].cache_data = lambda f: f

sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from database.db_manager import DatabaseManager

# バックテスト関数を読み込み
with open('pages/08_historical_backtest.py', encoding='utf-8') as f:
    code = f.read().split('# ==================== メインUI ====================')[0]
code = code.replace('st.info', 'print')
code = code.replace('st.warning', 'print')
exec(code)

db = DatabaseManager()
watchlist = db.get_watchlist()
tickers = [w['ticker'] for w in watchlist]

print('バックテスト実行中...')
result = run_backtest(tickers, initial_cash=1000000, start_days_ago=252)

history = result['history']
trades = result['trades']

# 詳細分析
print('\n' + '='*60)
print('詳細分析')
print('='*60)

# 1. 月別パフォーマンス
print('\n【月別パフォーマンス】')
monthly_returns = {}
for i, h in enumerate(history):
    month = str(h['date'])[:7]
    if month not in monthly_returns:
        monthly_returns[month] = {'start': h['total_value'], 'end': h['total_value']}
    monthly_returns[month]['end'] = h['total_value']

prev_end = 1000000
for month in sorted(monthly_returns.keys()):
    data = monthly_returns[month]
    ret = (data['end'] - prev_end) / prev_end * 100
    print(f"  {month}: {ret:+.2f}%")
    prev_end = data['end']

# 2. 負け取引の分析
print('\n【損失取引TOP10】')
sell_trades = [t for t in trades if t['action'] == 'SELL' and 'pnl_rate' in t]
worst_trades = sorted(sell_trades, key=lambda x: x['pnl_rate'])[:10]
for t in worst_trades:
    print(f"  {t['ticker']:8} {t['pnl_rate']:+6.1f}% - {t['reason']}")

# 3. 利益を逃した可能性の分析
print('\n【半分売り後の動き（利益逃し分析）】')
partial_sells = [t for t in sell_trades if t.get('pnl_rate', 0) > 0 and 'shares' in t]
# 同一銘柄の連続取引を追跡
ticker_sequences = {}
for t in trades:
    ticker = t['ticker']
    if ticker not in ticker_sequences:
        ticker_sequences[ticker] = []
    ticker_sequences[ticker].append(t)

missed_profit = 0
for ticker, seq in ticker_sequences.items():
    for i, t in enumerate(seq):
        if t['action'] == 'SELL' and t.get('pnl_rate', 0) > 5:
            # 売った後の動きを見る
            for j in range(i+1, min(i+3, len(seq))):
                next_t = seq[j]
                if next_t['action'] == 'BUY':
                    # 売った後に買い戻し → 利益逃した可能性
                    missed_profit += 1

print(f"  売った後に買い戻した回数: {missed_profit}回")

# 4. 市場環境別パフォーマンス
print('\n【市場環境別パフォーマンス】')
bullish_days = sum(1 for h in history if h.get('market_bullish', True))
bearish_days = len(history) - bullish_days
print(f"  強気市場日数: {bullish_days}日")
print(f"  弱気市場日数: {bearish_days}日")

# 5. 現金比率の推移
print('\n【現金比率分析】')
cash_ratios = [h['cash'] / h['total_value'] * 100 for h in history]
print(f"  平均現金比率: {np.mean(cash_ratios):.1f}%")
print(f"  最小現金比率: {min(cash_ratios):.1f}%")
print(f"  最大現金比率: {max(cash_ratios):.1f}%")

# 6. 保有期間分析
print('\n【保有期間分析（勝ち取引 vs 負け取引）】')
# 取引から保有期間を推定（簡易版）
win_trades = [t for t in sell_trades if t.get('pnl_rate', 0) > 0]
loss_trades = [t for t in sell_trades if t.get('pnl_rate', 0) <= 0]
print(f"  勝ち取引数: {len(win_trades)}")
print(f"  負け取引数: {len(loss_trades)}")
print(f"  勝ち取引の平均利益: +{np.mean([t['pnl_rate'] for t in win_trades]):.1f}%")
print(f"  負け取引の平均損失: {np.mean([t['pnl_rate'] for t in loss_trades]):.1f}%")

# 7. 取引していない銘柄
print('\n【取引していない銘柄】')
traded_tickers = set(t['ticker'] for t in trades)
not_traded = set(tickers) - traded_tickers
if not_traded:
    print(f"  {', '.join(not_traded)}")
else:
    print("  全銘柄取引済み")

print('\n' + '='*60)
