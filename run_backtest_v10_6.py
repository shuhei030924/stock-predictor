
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
            
    # 勝率
    wins = sum(1 for t in trades if t['action'] == 'SELL' and t['pnl_rate'] > 0)
    losses = sum(1 for t in trades if t['action'] == 'SELL' and t['pnl_rate'] <= 0)
    total_sells = wins + losses
    win_rate = (wins / total_sells * 100) if total_sells > 0 else 0
    
    output = f"""
==================================================
     v10.6 バックテスト結果
==================================================
初期資金:        {initial:,.0f}円
最終資産:        {final:,.0f}円
総リターン:         {total_return:+.2f}%
シャープレシオ:       {sharpe:.2f}
最大DD:                {max_dd*100:.1f}%
勝率:                {win_rate:.1f}% ({wins}勝/{losses}敗)
総取引数:               {len(trades)}回
==================================================
"""
    print(output)
    with open('backtest_result_v10_6_final.txt', 'w', encoding='utf-8') as f:
        f.write(output)
