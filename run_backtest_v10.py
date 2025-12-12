"""
v10.0 バックテスト実行スクリプト
VIXレジーム検知 + DD連動ポジション縮小
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 銘柄リスト（デフォルト）
DEFAULT_TICKERS = [
    # 日本株
    "7203.T", "6758.T", "8306.T", "9432.T", "9433.T",
    "6367.T", "4502.T", "6902.T", "7974.T", "6501.T",
    "8035.T", "1306.T", "1321.T", "6594.T",
    # 米国株
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "META", "JPM", "NFLX", "MU"
]

def run_test():
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import requests
    import time
    from collections import defaultdict
    
    print("=" * 60)
    print("v10.0 バックテスト実行（VIXレジーム検知 + DD連動）")
    print("=" * 60)
    
    # データ取得
    print("\n📥 データ取得中...")
    
    all_data = {}
    for ticker in DEFAULT_TICKERS:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="3y")
            if df is not None and len(df) >= 50:
                all_data[ticker] = df
                print(f"  ✅ {ticker}: {len(df)}日分")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ❌ {ticker}: エラー - {e}")
    
    # SPY（市場データ）
    print("\n📊 市場データ取得...")
    spy = yf.Ticker("SPY")
    market_data = spy.history(period="3y")
    print(f"  SPY: {len(market_data)}日分")
    
    # VIXデータ
    print("\n🚨 VIXデータ取得...")
    vix_ticker = yf.Ticker("^VIX")
    vix_data = vix_ticker.history(period="3y")
    print(f"  VIX: {len(vix_data)}日分")
    
    # バックテストパラメータ
    initial_cash = 1000000
    start_days_ago = 252  # 1年
    
    # 取引日リスト作成
    market_dates = market_data.index.strftime('%Y-%m-%d').tolist()
    if len(market_dates) > start_days_ago:
        trade_dates = market_dates[-start_days_ago:]
    else:
        trade_dates = market_dates
    
    # VIXインデックス
    vix_dates = vix_data.index.strftime('%Y-%m-%d').tolist() if vix_data is not None else []
    
    print(f"\n📅 テスト期間: {trade_dates[0]} 〜 {trade_dates[-1]} ({len(trade_dates)}日)")
    
    # シミュレーション
    print("\n🔄 シミュレーション実行中...")
    
    cash = initial_cash
    positions = {}  # {ticker: {'shares': int, 'avg_price': float, 'buy_date': str}}
    trades = []
    history = []
    
    # DD追跡
    peak_equity = initial_cash
    
    # レジーム追跡
    regime_counts = {'NORMAL': 0, 'CAUTION': 0, 'FEAR': 0, 'PANIC': 0}
    vix_values = []
    
    for i, date in enumerate(trade_dates):
        # 現在のVIX取得
        vix_level = None
        vix_regime = 'NORMAL'
        vix_momentum = 0
        
        if date in vix_dates:
            idx = vix_dates.index(date)
            vix_level = vix_data['Close'].iloc[idx]
            vix_values.append(vix_level)
            
            # VIXモメンタム（5日変化率）
            if idx >= 5:
                vix_5d_ago = vix_data['Close'].iloc[idx - 5]
                vix_momentum = (vix_level - vix_5d_ago) / vix_5d_ago * 100
            
            # レジーム判定
            if vix_level >= 35:
                vix_regime = 'PANIC'
            elif vix_level >= 30:
                vix_regime = 'FEAR'
            elif vix_level >= 25 or vix_momentum > 15:
                vix_regime = 'CAUTION'
            else:
                vix_regime = 'NORMAL'
        
        regime_counts[vix_regime] += 1
        
        # ポートフォリオ評価
        portfolio_value = cash
        for ticker, pos in positions.items():
            if ticker in all_data:
                df = all_data[ticker]
                dates_list = df.index.strftime('%Y-%m-%d').tolist()
                if date in dates_list:
                    idx = dates_list.index(date)
                    price = df['Close'].iloc[idx]
                    portfolio_value += pos['shares'] * price
        
        # DD計算
        if portfolio_value > peak_equity:
            peak_equity = portfolio_value
        current_dd = (peak_equity - portfolio_value) / peak_equity * 100
        
        # DD連動ポジション倍率
        if current_dd >= 5:
            dd_multiplier = 0.25
        elif current_dd >= 3:
            dd_multiplier = 0.5
        else:
            dd_multiplier = 1.0
        
        # VIXポジション倍率
        if vix_regime == 'PANIC':
            vix_multiplier = 0.0
        elif vix_regime == 'FEAR':
            vix_multiplier = 0.25
        elif vix_regime == 'CAUTION':
            vix_multiplier = 0.5
        else:
            vix_multiplier = 1.0
        
        combined_multiplier = vix_multiplier * dd_multiplier
        
        # PANIC時: 全ポジション売却
        if vix_regime == 'PANIC' and positions:
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                if ticker in all_data:
                    df = all_data[ticker]
                    dates_list = df.index.strftime('%Y-%m-%d').tolist()
                    if date in dates_list:
                        idx = dates_list.index(date)
                        price = df['Close'].iloc[idx]
                        sell_value = pos['shares'] * price
                        pnl = (price - pos['avg_price']) / pos['avg_price'] * 100
                        cash += sell_value
                        trades.append({
                            'date': date, 'ticker': ticker, 'action': 'SELL',
                            'shares': pos['shares'], 'price': price,
                            'pnl_rate': pnl, 'reason': 'VIX PANIC売却'
                        })
                        del positions[ticker]
        
        # FEAR時: 損失ポジション売却
        elif vix_regime == 'FEAR':
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                if ticker in all_data:
                    df = all_data[ticker]
                    dates_list = df.index.strftime('%Y-%m-%d').tolist()
                    if date in dates_list:
                        idx = dates_list.index(date)
                        price = df['Close'].iloc[idx]
                        pnl = (price - pos['avg_price']) / pos['avg_price'] * 100
                        
                        if pnl < 0:  # 損失中
                            sell_value = pos['shares'] * price
                            cash += sell_value
                            trades.append({
                                'date': date, 'ticker': ticker, 'action': 'SELL',
                                'shares': pos['shares'], 'price': price,
                                'pnl_rate': pnl, 'reason': 'VIX FEAR損失売却'
                            })
                            del positions[ticker]
        
        # 買いロジック（NORMAL/CAUTION時のみ）
        if vix_regime in ['NORMAL', 'CAUTION'] and len(positions) < 5:
            budget = cash * 0.15 * combined_multiplier
            
            for ticker in DEFAULT_TICKERS:
                if ticker in positions or budget < 10000:
                    continue
                if ticker not in all_data:
                    continue
                    
                df = all_data[ticker]
                dates_list = df.index.strftime('%Y-%m-%d').tolist()
                if date not in dates_list:
                    continue
                    
                idx = dates_list.index(date)
                if idx < 20:  # 20日分のデータが必要
                    continue
                
                price = df['Close'].iloc[idx]
                ma20 = df['Close'].iloc[idx-20:idx].mean()
                
                # シンプルな買い条件: 20日MAを上回っている
                if price > ma20:
                    shares = int(budget / price)
                    if shares > 0:
                        cost = shares * price
                        cash -= cost
                        positions[ticker] = {
                            'shares': shares,
                            'avg_price': price,
                            'buy_date': date
                        }
                        trades.append({
                            'date': date, 'ticker': ticker, 'action': 'BUY',
                            'shares': shares, 'price': price
                        })
                        budget -= cost
                        
                        if len(positions) >= 5:
                            break
        
        # 利確・損切り（NORMALポジション管理）
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            if ticker not in all_data:
                continue
            df = all_data[ticker]
            dates_list = df.index.strftime('%Y-%m-%d').tolist()
            if date not in dates_list:
                continue
                
            idx = dates_list.index(date)
            price = df['Close'].iloc[idx]
            pnl = (price - pos['avg_price']) / pos['avg_price'] * 100
            
            sell_reason = None
            
            # 利確 +30%
            if pnl >= 30:
                sell_reason = '利確+30%'
            # 損切り -10%
            elif pnl <= -10:
                sell_reason = '損切り-10%'
            
            if sell_reason:
                sell_value = pos['shares'] * price
                cash += sell_value
                trades.append({
                    'date': date, 'ticker': ticker, 'action': 'SELL',
                    'shares': pos['shares'], 'price': price,
                    'pnl_rate': pnl, 'reason': sell_reason
                })
                del positions[ticker]
        
        # 履歴記録
        history.append({
            'date': date,
            'cash': cash,
            'total_value': portfolio_value,
            'num_positions': len(positions),
            'vix_level': vix_level,
            'vix_regime': vix_regime,
            'current_drawdown': current_dd,
            'position_multiplier': combined_multiplier
        })
        
        if i % 50 == 0:
            print(f"  [{i}/{len(trade_dates)}] {date} - ¥{portfolio_value:,.0f} - VIX:{vix_regime}")
    
    # 最終評価
    final_value = history[-1]['total_value']
    profit = final_value - initial_cash
    profit_rate = profit / initial_cash * 100
    
    # 最大DD再計算
    peak = initial_cash
    max_dd = 0
    for h in history:
        if h['total_value'] > peak:
            peak = h['total_value']
        dd = (peak - h['total_value']) / peak * 100
        max_dd = max(max_dd, dd)
    
    # シャープレシオ
    daily_returns = []
    for i in range(1, len(history)):
        prev = history[i-1]['total_value']
        curr = history[i]['total_value']
        daily_returns.append((curr - prev) / prev)
    daily_returns = np.array(daily_returns)
    sharpe = (np.mean(daily_returns) * 252) / (np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0
    
    # 勝率
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    profitable = [t for t in sell_trades if t.get('pnl_rate', 0) > 0]
    win_rate = len(profitable) / len(sell_trades) * 100 if sell_trades else 0
    
    # 月次
    monthly = defaultdict(lambda: {'start': None, 'end': None})
    for h in history:
        month = h['date'][:7]
        if monthly[month]['start'] is None:
            monthly[month]['start'] = h['total_value']
        monthly[month]['end'] = h['total_value']
    
    # 結果表示
    print("\n" + "=" * 60)
    print("📊 v10.0 バックテスト結果")
    print("=" * 60)
    
    print(f"\n💰 最終資産: ¥{final_value:,.0f} (初期: ¥{initial_cash:,.0f})")
    print(f"📈 総収益率: {profit_rate:+.2f}%")
    print(f"📉 最大DD: -{max_dd:.2f}%")
    print(f"📊 シャープレシオ: {sharpe:.2f}")
    print(f"🔄 総取引数: {len(trades)} (買:{len([t for t in trades if t['action']=='BUY'])}, 売:{len(sell_trades)})")
    print(f"✅ 勝率: {win_rate:.1f}%")
    
    # VIX統計
    if vix_values:
        print(f"\n🚨 VIXレジーム統計:")
        total_days = sum(regime_counts.values())
        print(f"  🟢 NORMAL: {regime_counts['NORMAL']}日 ({regime_counts['NORMAL']/total_days*100:.1f}%)")
        print(f"  🟡 CAUTION: {regime_counts['CAUTION']}日 ({regime_counts['CAUTION']/total_days*100:.1f}%)")
        print(f"  🟠 FEAR: {regime_counts['FEAR']}日 ({regime_counts['FEAR']/total_days*100:.1f}%)")
        print(f"  🔴 PANIC: {regime_counts['PANIC']}日 ({regime_counts['PANIC']/total_days*100:.1f}%)")
        print(f"\n  VIX平均: {sum(vix_values)/len(vix_values):.1f}")
        print(f"  VIX最大: {max(vix_values):.1f}")
        print(f"  VIX最小: {min(vix_values):.1f}")
    
    # 月次
    print(f"\n📅 月次パフォーマンス:")
    for month in sorted(monthly.keys()):
        data = monthly[month]
        if data['start'] and data['end']:
            ret = (data['end'] - data['start']) / data['start'] * 100
            emoji = "🟢" if ret > 0 else "🔴"
            print(f"  {month}: {emoji} {ret:+.2f}%")
    
    # v9.0c比較
    print("\n" + "=" * 60)
    print("📊 v9.0c との比較")
    print("=" * 60)
    print(f"{'指標':<15} {'v9.0c':<12} {'v10.0':<12} {'変化':<10}")
    print("-" * 50)
    print(f"{'収益率':<15} {'+19.03%':<12} {f'{profit_rate:+.2f}%':<12} {f'{profit_rate-19.03:+.2f}%':<10}")
    print(f"{'シャープ':<15} {'1.62':<12} {f'{sharpe:.2f}':<12} {f'{sharpe-1.62:+.2f}':<10}")
    print(f"{'最大DD':<15} {'-5.9%':<12} {f'-{max_dd:.2f}%':<12} {f'{5.9-max_dd:+.2f}%':<10}")

if __name__ == "__main__":
    run_test()
