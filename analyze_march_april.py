"""
3-4月の取引詳細分析
"""
import sys
import warnings
warnings.filterwarnings('ignore')

# Streamlitモック
sys.modules['streamlit'] = type(sys)('streamlit')
sys.modules['streamlit'].info = lambda *a, **k: None
sys.modules['streamlit'].warning = lambda *a, **k: None
sys.modules['streamlit'].error = lambda *a, **k: None
sys.modules['streamlit'].set_page_config = lambda *a, **k: None
sys.modules['streamlit'].title = lambda *a, **k: None
sys.modules['streamlit'].markdown = lambda *a, **k: None
sys.modules['streamlit'].sidebar = type('sidebar', (), {
    'header': lambda *a, **k: None, 
    'button': lambda *a, **k: False, 
    'selectbox': lambda *a, **k: '1年', 
    'number_input': lambda *a, **k: 1000000, 
    'subheader': lambda *a, **k: None, 
    'checkbox': lambda *a, **k: True, 
    'multiselect': lambda *a, **k: [], 
    'metric': lambda *a, **k: None
})()
sys.modules['streamlit'].expander = lambda *a, **k: type('exp', (), {'__enter__': lambda s: s, '__exit__': lambda *a: None})()
sys.modules['streamlit'].divider = lambda *a, **k: None
sys.modules['streamlit'].button = lambda *a, **k: False
sys.modules['streamlit'].progress = lambda *a, **k: type('prog', (), {'progress': lambda *a: None, 'empty': lambda *a: None})()
sys.modules['streamlit'].empty = lambda *a, **k: type('emp', (), {'text': lambda *a: None})()
sys.modules['streamlit'].spinner = lambda *a, **k: type('sp', (), {'__enter__': lambda s: s, '__exit__': lambda *a: None})()
sys.modules['streamlit'].stop = lambda *a, **k: None
sys.modules['streamlit'].cache_data = lambda f: f

sys.path.insert(0, '.')
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
trades = result['trades']
history = result['history']

# 3-4月の取引を分析
print('\n' + '='*70)
print('3-4月の取引詳細分析')
print('='*70)

march_april_trades = []
for t in trades:
    date_str = str(t['date'])
    if '2025-03' in date_str or '2025-04' in date_str:
        march_april_trades.append(t)
        if t['action'] == 'SELL':
            pnl = t.get('pnl_rate', 0)
            print(f"{date_str[:10]} {t['action']:4} {t['ticker']:8} {pnl:+6.1f}% - {t.get('reason', '')}")
        else:
            print(f"{date_str[:10]} {t['action']:4} {t['ticker']:8}")

# 3-4月の損益サマリー
print('\n【3-4月の損益サマリー】')
sells_34 = [t for t in march_april_trades if t['action'] == 'SELL' and 'pnl_rate' in t]
wins = [t for t in sells_34 if t['pnl_rate'] > 0]
losses = [t for t in sells_34 if t['pnl_rate'] <= 0]
print(f"  総取引数: {len(sells_34)}")
print(f"  勝ち: {len(wins)}件 (平均 +{sum(t['pnl_rate'] for t in wins)/len(wins) if wins else 0:.1f}%)")
print(f"  負け: {len(losses)}件 (平均 {sum(t['pnl_rate'] for t in losses)/len(losses) if losses else 0:.1f}%)")

# 市場全体の動き
print('\n【3-4月の市場環境】')
for h in history:
    date_str = str(h['date'])
    if '2025-03' in date_str or '2025-04' in date_str:
        if h['date'].day == 1 or h['date'].day == 15:
            bullish = h.get('market_bullish', True)
            pos_count = len(h.get('portfolio', []))
            cash_ratio = h['cash'] / h['total_value'] * 100
            print(f"  {date_str[:10]}: {'強気' if bullish else '弱気'}, ポジション{pos_count}個, 現金{cash_ratio:.0f}%")

# 弱気市場での取引
print('\n【弱気市場日に行われた取引】')
bearish_dates = set()
for h in history:
    if not h.get('market_bullish', True):
        bearish_dates.add(str(h['date'])[:10])

for t in trades:
    date_str = str(t['date'])[:10]
    if date_str in bearish_dates:
        if t['action'] == 'SELL':
            pnl = t.get('pnl_rate', 0)
            print(f"  {date_str} {t['action']:4} {t['ticker']:8} {pnl:+6.1f}%")
        else:
            print(f"  {date_str} {t['action']:4} {t['ticker']:8}")

print('\n' + '='*70)
