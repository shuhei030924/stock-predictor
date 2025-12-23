"""
定期シグナル検出スクリプト
========================
タスクスケジューラやcronで定期実行して、
シグナルを自動検出しLINE通知を送信

使用方法:
  python run_signal_check.py

タスクスケジューラ設定（Windows）:
  1. タスクスケジューラを開く
  2. 基本タスクの作成
  3. トリガー: 毎日 9:00, 15:00 など
  4. 操作: プログラムの開始
     - プログラム: python
     - 引数: run_signal_check.py
     - 開始: このスクリプトのディレクトリ
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
import argparse
import json

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from services.line_notify import LineNotifyService


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """テクニカル指標を計算"""
    df = df.copy()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ボリンジャーバンド
    df['bb_mid'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # リターン
    df['return_1d'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    
    # 出来高比率
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    return df


def detect_signals(df: pd.DataFrame, ticker: str) -> dict:
    """シグナルを検出"""
    if len(df) < 50:
        return None
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signals = {
        'ticker': ticker,
        'price': latest['Close'],
        'buy_signals': [],
        'sell_signals': [],
        'indicators': {
            'rsi': latest.get('rsi', 50),
            'macd': latest.get('macd', 0),
            'macd_hist': latest.get('macd_hist', 0),
            'bb_position': latest.get('bb_position', 0.5),
        }
    }
    
    # 買いシグナル
    if latest.get('rsi', 50) < 30:
        signals['buy_signals'].append(f"RSI売られすぎ ({latest['rsi']:.1f})")
    
    if prev.get('macd', 0) < prev.get('macd_signal', 0) and \
       latest.get('macd', 0) > latest.get('macd_signal', 0):
        signals['buy_signals'].append("MACDゴールデンクロス")
    
    if latest.get('bb_position', 0.5) < 0.1:
        signals['buy_signals'].append("BB下限接近")
    
    if latest.get('volume_ratio', 1) > 2 and latest.get('return_1d', 0) > 0.02:
        signals['buy_signals'].append("出来高急増+上昇")
    
    # 売りシグナル
    if latest.get('rsi', 50) > 70:
        signals['sell_signals'].append(f"RSI買われすぎ ({latest['rsi']:.1f})")
    
    if prev.get('macd', 0) > prev.get('macd_signal', 0) and \
       latest.get('macd', 0) < latest.get('macd_signal', 0):
        signals['sell_signals'].append("MACDデッドクロス")
    
    if latest.get('bb_position', 0.5) > 0.9:
        signals['sell_signals'].append("BB上限接近")
    
    return signals


def calculate_ai_score(df: pd.DataFrame) -> float:
    """簡易AIスコア"""
    if len(df) < 50:
        return 50.0
    
    latest = df.iloc[-1]
    score = 50.0
    
    rsi = latest.get('rsi', 50)
    if rsi < 30:
        score += 15
    elif rsi > 70:
        score -= 15
    
    macd_hist = latest.get('macd_hist', 0)
    if macd_hist > 0:
        score += 10
    else:
        score -= 10
    
    bb_pos = latest.get('bb_position', 0.5)
    if bb_pos < 0.2:
        score += 10
    elif bb_pos > 0.8:
        score -= 10
    
    ret_5d = latest.get('return_5d', 0)
    if not pd.isna(ret_5d):
        score += min(max(ret_5d * 100, -15), 15)
    
    return max(0, min(100, score))


def main():
    parser = argparse.ArgumentParser(description='株価シグナル検出 & LINE通知')
    parser.add_argument('--dry-run', action='store_true', help='通知を送信せずに結果を表示')
    parser.add_argument('--tickers', nargs='+', help='監視銘柄（スペース区切り）')
    args = parser.parse_args()
    
    # デフォルト監視銘柄
    default_tickers = [
        "7203.T", "6758.T", "9984.T", "8306.T", "6902.T",
        "7267.T", "6501.T", "8035.T", "6861.T", "9433.T",
        "7011.T", "7012.T", "8316.T", "8053.T", "6702.T"
    ]
    
    tickers = args.tickers or default_tickers
    
    print("=" * 60)
    print(f"🔔 シグナル検出開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   監視銘柄: {len(tickers)}件")
    print("=" * 60)
    
    # LINE通知サービス
    line_service = None if args.dry_run else LineNotifyService()
    
    all_buy_signals = []
    all_sell_signals = []
    
    for ticker in tickers:
        try:
            # データ取得
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")
            
            if len(df) < 50:
                print(f"⚠️ {ticker}: データ不足")
                continue
            
            # 指標計算
            df = calculate_indicators(df)
            
            # シグナル検出
            signals = detect_signals(df, ticker)
            if not signals:
                continue
            
            signals['ai_score'] = calculate_ai_score(df)
            
            # 買いシグナル
            if signals['buy_signals']:
                all_buy_signals.append(signals)
                print(f"🟢 {ticker}: {', '.join(signals['buy_signals'])} (AI:{signals['ai_score']:.0f})")
                
                if line_service:
                    line_service.send_buy_signal(
                        ticker=ticker,
                        price=signals['price'],
                        ai_score=signals['ai_score'],
                        reason=", ".join(signals['buy_signals']),
                        additional_info=signals['indicators']
                    )
            
            # 売りシグナル
            if signals['sell_signals']:
                all_sell_signals.append(signals)
                print(f"🔴 {ticker}: {', '.join(signals['sell_signals'])} (AI:{signals['ai_score']:.0f})")
                
                if line_service:
                    line_service.send_sell_signal(
                        ticker=ticker,
                        price=signals['price'],
                        profit_rate=0,
                        reason=", ".join(signals['sell_signals'])
                    )
        
        except Exception as e:
            print(f"❌ {ticker}: エラー - {e}")
    
    # サマリー
    print("\n" + "=" * 60)
    print("📊 検出サマリー")
    print("=" * 60)
    print(f"   買いシグナル: {len(all_buy_signals)}件")
    print(f"   売りシグナル: {len(all_sell_signals)}件")
    
    if args.dry_run:
        print("\n   ℹ️ --dry-run モード: LINE通知は送信されませんでした")
    
    # 結果をJSONに保存
    result = {
        'timestamp': datetime.now().isoformat(),
        'buy_signals': all_buy_signals,
        'sell_signals': all_sell_signals
    }
    
    output_dir = PROJECT_ROOT / 'analysis'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n   📁 結果保存: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
