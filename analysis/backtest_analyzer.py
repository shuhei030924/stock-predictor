"""
バックテスト結果の詳細分析スクリプト
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict

def analyze_backtest_results(history: list, trades: list, initial_cash: float = 1000000, interval: str = "1d"):
    """バックテスト結果を詳細に分析"""
    
    results = {}
    
    # 年率換算係数
    annual_factor = 252
    if interval == "1h":
        annual_factor = 252 * 7  # 1日7時間と仮定
    
    # ==================== 基本統計 ====================
    df_history = pd.DataFrame(history)
    df_history['date'] = pd.to_datetime(df_history['date'])
    
    initial_value = df_history['total_value'].iloc[0]
    final_value = df_history['total_value'].iloc[-1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    # 日次リターン
    df_history['daily_return'] = df_history['total_value'].pct_change() * 100
    
    results['基本統計'] = {
        '初期資金': f"¥{initial_value:,.0f}",
        '最終資産': f"¥{final_value:,.0f}",
        '損益': f"¥{final_value - initial_value:,.0f}",
        '総収益率': f"{total_return:.2f}%",
        '年率換算': f"{total_return * annual_factor / len(df_history):.2f}%",
        '取引日数': len(df_history),
    }
    
    # ==================== リスク指標 ====================
    # 最大ドローダウン
    peak = df_history['total_value'].expanding().max()
    drawdown = (df_history['total_value'] - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    # ボラティリティ
    daily_vol = df_history['daily_return'].std()
    annual_vol = daily_vol * np.sqrt(annual_factor)
    
    # シャープレシオ（無リスク金利0%と仮定）
    avg_daily_return = df_history['daily_return'].mean()
    sharpe_ratio = (avg_daily_return * annual_factor) / annual_vol if annual_vol > 0 else 0
    
    # 勝ち日数 vs 負け日数
    winning_days = (df_history['daily_return'] > 0).sum()
    losing_days = (df_history['daily_return'] < 0).sum()
    
    results['リスク指標'] = {
        '最大ドローダウン': f"{max_drawdown:.2f}%",
        '日次ボラティリティ': f"{daily_vol:.2f}%",
        '年率ボラティリティ': f"{annual_vol:.2f}%",
        'シャープレシオ': f"{sharpe_ratio:.2f}",
        '勝ち日数': winning_days,
        '負け日数': losing_days,
        '勝率(日次)': f"{winning_days / (winning_days + losing_days) * 100:.1f}%",
    }
    
    # ==================== 取引分析 ====================
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    
    results['取引統計'] = {
        '総取引数': len(trades),
        '買い取引数': len(buy_trades),
        '売り取引数': len(sell_trades),
    }
    
    # 売り取引の詳細分析
    if sell_trades:
        pnl_rates = [t.get('pnl_rate', 0) for t in sell_trades if 'pnl_rate' in t]
        if pnl_rates:
            winning_trades = [p for p in pnl_rates if p > 0]
            losing_trades = [p for p in pnl_rates if p < 0]
            
            results['売り取引分析'] = {
                '勝ち取引数': len(winning_trades),
                '負け取引数': len(losing_trades),
                '勝率': f"{len(winning_trades) / len(pnl_rates) * 100:.1f}%",
                '平均利益率': f"{np.mean(winning_trades):.2f}%" if winning_trades else "N/A",
                '平均損失率': f"{np.mean(losing_trades):.2f}%" if losing_trades else "N/A",
                '最大利益': f"{max(pnl_rates):.2f}%",
                '最大損失': f"{min(pnl_rates):.2f}%",
                '期待値': f"{np.mean(pnl_rates):.2f}%",
            }
            
            # プロフィットファクター
            total_profit = sum(winning_trades) if winning_trades else 0
            total_loss = abs(sum(losing_trades)) if losing_trades else 1
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            results['売り取引分析']['プロフィットファクター'] = f"{profit_factor:.2f}"
    
    # ==================== 売却理由別分析 ====================
    reason_stats = defaultdict(lambda: {'count': 0, 'pnl_rates': []})
    for t in sell_trades:
        reason = t.get('reason', 'Unknown')
        # 理由をカテゴリ化
        if '損切り' in reason:
            category = '損切り'
        elif 'トレーリング' in reason:
            category = 'トレーリングストップ'
        elif '利確' in reason:
            category = '利確'
        elif '強い売り' in reason:
            category = '強い売りシグナル'
        elif '売り' in reason:
            category = '売りシグナル'
        else:
            category = 'その他'
        
        reason_stats[category]['count'] += 1
        if 'pnl_rate' in t:
            reason_stats[category]['pnl_rates'].append(t['pnl_rate'])
    
    results['売却理由別'] = {}
    for category, stats in reason_stats.items():
        avg_pnl = np.mean(stats['pnl_rates']) if stats['pnl_rates'] else 0
        results['売却理由別'][category] = {
            '回数': stats['count'],
            '平均損益': f"{avg_pnl:.2f}%"
        }
    
    # ==================== 銘柄別分析 ====================
    ticker_stats = defaultdict(lambda: {'buys': 0, 'sells': 0, 'pnl_rates': [], 'total_amount': 0})
    
    for t in trades:
        ticker = t['ticker']
        if t['action'] == 'BUY':
            ticker_stats[ticker]['buys'] += 1
            ticker_stats[ticker]['total_amount'] += t['amount']
        else:
            ticker_stats[ticker]['sells'] += 1
            if 'pnl_rate' in t:
                ticker_stats[ticker]['pnl_rates'].append(t['pnl_rate'])
    
    # 損益でソート
    ticker_performance = []
    for ticker, stats in ticker_stats.items():
        avg_pnl = np.mean(stats['pnl_rates']) if stats['pnl_rates'] else 0
        ticker_performance.append({
            '銘柄': ticker,
            '買い回数': stats['buys'],
            '売り回数': stats['sells'],
            '投資額合計': stats['total_amount'],
            '平均損益率': avg_pnl,
        })
    
    ticker_performance.sort(key=lambda x: x['平均損益率'], reverse=True)
    results['銘柄別パフォーマンス'] = ticker_performance
    
    # ==================== 月次分析 ====================
    df_history['month'] = df_history['date'].dt.to_period('M')
    monthly_stats = []
    
    for month, group in df_history.groupby('month'):
        start_val = group['total_value'].iloc[0]
        end_val = group['total_value'].iloc[-1]
        monthly_return = (end_val - start_val) / start_val * 100
        
        # その月の取引数
        month_start = group['date'].iloc[0]
        month_end = group['date'].iloc[-1]
        month_trades = [t for t in trades 
                       if month_start <= pd.to_datetime(t['date']) <= month_end]
        
        monthly_stats.append({
            '月': str(month),
            'リターン': f"{monthly_return:.2f}%",
            '取引数': len(month_trades),
            '市場強気日数': group.get('market_bullish', pd.Series([True]*len(group))).sum() if 'market_bullish' in group else len(group)
        })
    
    results['月次パフォーマンス'] = monthly_stats
    
    # ==================== 問題点の特定 ====================
    problems = []
    
    # 勝率が低い場合
    if sell_trades:
        pnl_rates = [t.get('pnl_rate', 0) for t in sell_trades if 'pnl_rate' in t]
        if pnl_rates:
            win_rate = len([p for p in pnl_rates if p > 0]) / len(pnl_rates) * 100
            if win_rate < 40:
                problems.append(f"⚠️ 勝率が低い ({win_rate:.1f}%): エントリー条件を厳しくする必要あり")
            
            avg_win = np.mean([p for p in pnl_rates if p > 0]) if [p for p in pnl_rates if p > 0] else 0
            avg_loss = abs(np.mean([p for p in pnl_rates if p < 0])) if [p for p in pnl_rates if p < 0] else 0
            
            if avg_loss > avg_win:
                problems.append(f"⚠️ 平均損失({avg_loss:.1f}%)が平均利益({avg_win:.1f}%)より大きい: 損切りを早めるか、利確を遅らせる")
    
    # 損切りが多い場合
    stop_loss_count = reason_stats.get('損切り', {}).get('count', 0)
    if stop_loss_count > len(sell_trades) * 0.3:
        problems.append(f"⚠️ 損切りが多い ({stop_loss_count}回): エントリータイミングが悪い可能性")
    
    # 最大ドローダウンが大きい場合
    if max_drawdown < -15:
        problems.append(f"⚠️ ドローダウンが大きい ({max_drawdown:.1f}%): ポジションサイズを減らすか、相場環境フィルターを強化")
    
    # シャープレシオが低い場合
    if sharpe_ratio < 0.5:
        problems.append(f"⚠️ シャープレシオが低い ({sharpe_ratio:.2f}): リスクに見合ったリターンが出ていない")
    
    results['問題点'] = problems
    
    # ==================== 改善提案 ====================
    suggestions = []
    
    # 勝率改善
    if sell_trades:
        pnl_rates = [t.get('pnl_rate', 0) for t in sell_trades if 'pnl_rate' in t]
        if pnl_rates:
            win_rate = len([p for p in pnl_rates if p > 0]) / len(pnl_rates) * 100
            if win_rate < 50:
                suggestions.append("💡 エントリー条件をさらに厳しく: スコア閾値を0.3以上に引き上げ")
                suggestions.append("💡 3日連続シグナル確認に変更")
                suggestions.append("💡 RSI30-70の範囲でのみ買い")
    
    # 損切り改善
    if stop_loss_count > 5:
        suggestions.append("💡 損切りラインをATRベースでより広く設定（現在の2倍→2.5倍）")
        suggestions.append("💡 購入直後の変動を考慮し、2日間は損切りしない")
    
    # 利確改善
    profit_taking = reason_stats.get('利確', {}).get('count', 0)
    trailing_stop = reason_stats.get('トレーリングストップ', {}).get('count', 0)
    if profit_taking + trailing_stop < len(sell_trades) * 0.3:
        suggestions.append("💡 トレンドに乗れていない: 利確ラインを+20%に引き上げ")
        suggestions.append("💡 トレーリングストップを高値から-7%に緩和")
    
    # 一般的な改善
    suggestions.append("💡 出来高確認: 平均出来高の1.5倍以上の日のみ買い")
    suggestions.append("💡 セクターフィルター: テック株のみ or ディフェンシブ株のみに集中")
    suggestions.append("💡 決算シーズン回避: 決算発表前後1週間は新規購入しない")
    
    results['改善提案'] = suggestions
    
    return results


def print_analysis(results: dict):
    """分析結果を見やすく表示"""
    print("=" * 60)
    print("📊 バックテスト詳細分析レポート")
    print("=" * 60)
    
    for section, data in results.items():
        print(f"\n### {section}")
        print("-" * 40)
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    - {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        elif isinstance(data, list):
            if section == '問題点' or section == '改善提案':
                for item in data:
                    print(f"  {item}")
            elif section == '銘柄別パフォーマンス':
                # 上位5銘柄と下位5銘柄
                print("  【上位5銘柄】")
                for item in data[:5]:
                    print(f"    {item['銘柄']}: 平均{item['平均損益率']:.1f}% (取引{item['買い回数']}回)")
                print("  【下位5銘柄】")
                for item in data[-5:]:
                    print(f"    {item['銘柄']}: 平均{item['平均損益率']:.1f}% (取引{item['買い回数']}回)")
            else:
                for item in data:
                    if isinstance(item, dict):
                        print(f"  {item}")
                    else:
                        print(f"  - {item}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # テスト用
    print("このスクリプトはインポートして使用します")
