import sys
print("DEBUG: Script started")
import pandas as pd
import numpy as np
from database.db_manager import DatabaseManager
import warnings
warnings.filterwarnings('ignore')

# Mock Streamlit
class DummySt:
    def info(self, *args, **kwargs): print(f"[INFO] {args[0]}")
    def warning(self, *args, **kwargs): print(f"[WARN] {args[0]}")
    def error(self, *args, **kwargs): print(f"[ERROR] {args[0]}")
    def cache_data(self, func): return func
    def progress(self, *args, **kwargs): return type('prog', (), {'progress': lambda *a: None, 'empty': lambda *a: None})()
    def empty(self, *args, **kwargs): return type('emp', (), {'text': lambda *a: None})()

sys.modules['streamlit'] = type(sys)('streamlit')
sys.modules['streamlit'].info = DummySt().info
sys.modules['streamlit'].warning = DummySt().warning
sys.modules['streamlit'].error = DummySt().error
sys.modules['streamlit'].cache_data = DummySt().cache_data
sys.modules['streamlit'].progress = DummySt().progress
sys.modules['streamlit'].empty = DummySt().empty
sys.modules['streamlit'].set_page_config = lambda *a, **k: None
sys.modules['streamlit'].title = lambda *a, **k: None
sys.modules['streamlit'].markdown = lambda *a, **k: None

# Import run_backtest by executing the file
with open('pages/08_historical_backtest.py', encoding='utf-8') as f:
    code = f.read()
    # Extract the functions we need
    # We need to be careful not to execute the main UI code
    # The file has "if __name__ == '__main__':" or similar? No, it's a streamlit script.
    # It runs top to bottom.
    # We can split by the main UI section if it exists.
    # Looking at the file content read previously, it has "# ==================== メインUI ===================="
    code = code.split('# ==================== メインUI ====================')[0]
    
    # Replace st calls in the code if necessary, but we mocked the module so it should be fine.
    exec(code)

# Now we have run_backtest and other functions in the local scope
# We need to make sure imports in the executed code work. 
# The file does `sys.path.insert(0, str(Path(__file__).parent.parent))` which might fail if __file__ is not defined.
# But we are running this script, so we should set up sys.path manually.
sys.path.insert(0, '.')

from analysis.backtest_analyzer import analyze_backtest_results

def main():
    print("Starting Backtest v12.0 (Simulation Mode)...")
    db = DatabaseManager()
    watchlist = db.get_watchlist()
    tickers = [w['ticker'] for w in watchlist]
    
    # Limit tickers for speed if needed, but let's try full list (usually 100)
    # tickers = tickers[:5] # DEBUG: Limit to 5 tickers for speed
    print(f"Tickers: {len(tickers)}")
    
    # Run backtest
    # Using interval="1d" for now to verify logic. 
    # If user wants 1h, we can change it, but 1d is standard for this app's logic verification.
    try:
        # Try 1h interval if user requested
        interval = "1h" 
        use_gpu = True # Enable GPU
        print(f"Running backtest with interval: {interval}, GPU: {use_gpu}")
        
        result = run_backtest(tickers, initial_cash=1000000, start_days_ago=252, interval=interval, use_gpu=use_gpu)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            # Fallback to 1d if 1h fails
            print("Falling back to 1d interval...")
            interval = "1d"
            result = run_backtest(tickers, initial_cash=1000000, start_days_ago=252, interval=interval, use_gpu=use_gpu)

        history = result['history']
        trades = result['trades']
        
        # Use the analyzer
        analysis = analyze_backtest_results(history, trades, initial_cash=1000000, interval=interval)
        
        print("\n" + "="*50)
        print("BACKTEST ANALYSIS REPORT (v12.0)")
        print("="*50)
        
        for category, metrics in analysis.items():
            if isinstance(metrics, dict):
                print(f"\n[{category}]")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
            else:
                # Handle non-dict items if any (though analyzer returns dict of dicts usually)
                pass

        # Save detailed results
        initial = history[0]['total_value']
        final = history[-1]['total_value']
        total_return = ((final - initial) / initial) * 100
        
        with open('backtest_result_v12.txt', 'w', encoding='utf-8') as f:
            f.write(f"Total Return: {total_return:.2f}%\n")
            f.write(f"Final Value: {final:,.0f}\n")
            f.write(f"Trade Count: {len(trades)}\n")
            
            # Calculate Win Rate
            sell_trades = [t for t in trades if t['action'] == 'SELL']
            wins = len([t for t in sell_trades if t.get('pnl_rate', 0) > 0])
            win_rate = (wins / len(sell_trades) * 100) if sell_trades else 0
            f.write(f"Win Rate: {win_rate:.2f}%\n")
            
            # Recent trades
            f.write("\nRecent Trades:\n")
            for t in trades[-10:]:
                pnl_str = f"PnL: {t.get('pnl_rate', 0):.1f}%" if 'pnl_rate' in t else ""
                f.write(f"{t['date']} {t['ticker']} {t['action']} {pnl_str} Reason: {t.get('reason', '')}\n")

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
