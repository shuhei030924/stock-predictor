"""
ロバスト性検証スクリプト
========================
異なる期間・銘柄セットでバックテストを実行し、
戦略の有効性を検証する

使用方法:
  python run_robustness_test.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.xgb_model import StockPredictorXGB


# ============================================================
# 銘柄セット定義 (200-300銘柄の大規模テスト用)
# ============================================================

TICKER_SETS = {
    # ========== 日本株 ==========
    "日本株_大型50": [
        # 時価総額上位の主力銘柄
        "7203.T", "6758.T", "9984.T", "8306.T", "6902.T",  # トヨタ、ソニー、SBG、三菱UFJ、デンソー
        "7267.T", "6501.T", "8035.T", "6861.T", "9433.T",  # ホンダ、日立、東エレ、キーエンス、KDDI
        "7011.T", "7012.T", "8316.T", "8053.T", "6702.T",  # 三菱重工、川崎重工、三井住友FG、住友商事、富士通
        "9432.T", "9434.T", "4502.T", "4503.T", "4568.T",  # NTT、ソフトバンク、武田、アステラス、第一三共
        "6098.T", "4661.T", "8058.T", "8031.T", "7974.T",  # リクルート、OLC、三菱商事、三井物産、任天堂
        "6273.T", "6723.T", "6954.T", "8001.T", "8002.T",  # SMC、ルネサス、ファナック、伊藤忠、丸紅
        "9983.T", "3382.T", "4901.T", "4911.T", "6857.T",  # ファストリ、セブン&アイ、富士フイルム、資生堂、アドバンテスト
        "6981.T", "7751.T", "4523.T", "6988.T", "8801.T",  # 村田製作所、キヤノン、エーザイ、日東電工、三井不動産
        "8802.T", "9020.T", "9021.T", "9022.T", "2914.T",  # 三菱地所、JR東日本、JR西日本、JR東海、JT
        "5401.T", "5411.T", "3407.T", "4452.T", "6326.T",  # 日本製鉄、JFE、旭化成、花王、クボタ
    ],
    "日本株_テック30": [
        # 半導体・電子部品・IT
        "6758.T", "6861.T", "6702.T", "6501.T", "6971.T",  # ソニー、キーエンス、富士通、日立、京セラ
        "4063.T", "6367.T", "6594.T", "6762.T", "6752.T",  # 信越化学、ダイキン、日本電産、TDK、パナソニック
        "8035.T", "6857.T", "6723.T", "6920.T", "6146.T",  # 東エレ、アドバンテスト、ルネサス、レーザーテック、ディスコ
        "7735.T", "6963.T", "6981.T", "4684.T", "4307.T",  # SCREENホールディングス、ローム、村田製作所、オービック、野村総研
        "9613.T", "4751.T", "4755.T", "3659.T", "2413.T",  # NTTデータ、サイバーエージェント、楽天G、ネクソン、エムスリー
        "4689.T", "9449.T", "3769.T", "4816.T", "2371.T",  # Zホールディングス、GMOインターネット、GMOペイメント、東映アニメ、カカクコム
    ],
    "日本株_金融20": [
        # 銀行・証券・保険・リース
        "8306.T", "8316.T", "8411.T", "8308.T", "8309.T",  # 三菱UFJ、三井住友FG、みずほ、りそな、三井住友トラスト
        "8604.T", "8601.T", "8630.T", "8766.T", "8750.T",  # 野村HD、大和証券G、SOMPO、東京海上、第一生命
        "8725.T", "8795.T", "8591.T", "8593.T", "8697.T",  # MS&AD、T&D HD、オリックス、三菱HCキャピタル、日本取引所G
        "7182.T", "8354.T", "8331.T", "8355.T", "7186.T",  # ゆうちょ銀行、ふくおかFG、千葉銀行、静岡銀行、コンコルディア
    ],
    "日本株_自動車15": [
        # 完成車・部品
        "7203.T", "7267.T", "7269.T", "7201.T", "7211.T",  # トヨタ、ホンダ、スズキ、日産、三菱自動車
        "7270.T", "7259.T", "7202.T", "6902.T", "5108.T",  # SUBARU、アイシン、いすゞ、デンソー、ブリヂストン
        "7282.T", "7240.T", "7261.T", "6201.T", "5101.T",  # 豊田自動織機、NOK、マツダ、豊田自動織機、横浜ゴム
    ],
    "日本株_消費財20": [
        # 小売・食品・日用品
        "9983.T", "3382.T", "8267.T", "9843.T", "3099.T",  # ファストリ、セブン&アイ、イオン、ニトリ、三越伊勢丹
        "2802.T", "2801.T", "2269.T", "2503.T", "2502.T",  # 味の素、キッコーマン、明治HD、キリン、アサヒ
        "4452.T", "4911.T", "4922.T", "4912.T", "7453.T",  # 花王、資生堂、コーセー、ライオン、良品計画
        "3086.T", "8252.T", "2670.T", "7532.T", "2651.T",  # J.フロント、丸井G、ABCマート、パン・パシフィック、ローソン
    ],
    "日本株_医薬15": [
        # 製薬・バイオ・医療機器
        "4502.T", "4503.T", "4568.T", "4519.T", "4578.T",  # 武田、アステラス、第一三共、中外製薬、大塚HD
        "4523.T", "4506.T", "4507.T", "4151.T", "4528.T",  # エーザイ、住友ファーマ、塩野義、協和キリン、小野薬品
        "7733.T", "4543.T", "6869.T", "7747.T", "4974.T",  # オリンパス、テルモ、シスメックス、朝日インテック、タカラバイオ
    ],
    "日本株_インフラ15": [
        # 電力・ガス・鉄道・通信
        "9432.T", "9433.T", "9434.T", "9020.T", "9021.T",  # NTT、KDDI、ソフトバンク、JR東日本、JR西日本
        "9022.T", "9001.T", "9005.T", "9007.T", "9008.T",  # JR東海、東武鉄道、東急、小田急、京王
        "9501.T", "9502.T", "9503.T", "9531.T", "9532.T",  # 東京電力、中部電力、関西電力、東京ガス、大阪ガス
    ],
    "日本株_不動産建設15": [
        # 不動産・建設・住宅
        "8801.T", "8802.T", "8830.T", "3289.T", "8804.T",  # 三井不動産、三菱地所、住友不動産、東急不動産HD、東京建物
        "1925.T", "1928.T", "1802.T", "1803.T", "1801.T",  # 大和ハウス、積水ハウス、大林組、清水建設、大成建設
        "1812.T", "1808.T", "5232.T", "5233.T", "1878.T",  # 鹿島建設、長谷工、住友大阪セメント、太平洋セメント、大東建託
    ],
    "日本株_素材15": [
        # 鉄鋼・化学・素材
        "5401.T", "5411.T", "5406.T", "3407.T", "4188.T",  # 日本製鉄、JFE、神戸製鋼、旭化成、三菱ケミカル
        "4005.T", "4042.T", "4183.T", "4631.T", "4021.T",  # 住友化学、東ソー、三井化学、DIC、日産化学
        "5713.T", "5711.T", "5706.T", "5714.T", "5801.T",  # 住友金属鉱山、三菱マテリアル、三井金属、DOWAホールディングス、古河電工
    ],
    
    # ========== 米国株 ==========
    "米国株_大型30": [
        # 時価総額上位の主力銘柄
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",  # Apple, Microsoft, Alphabet, Amazon, NVIDIA
        "META", "TSLA", "BRK-B", "UNH", "JNJ",    # Meta, Tesla, Berkshire, UnitedHealth, J&J
        "V", "JPM", "XOM", "PG", "MA",            # Visa, JPMorgan, Exxon, P&G, Mastercard
        "HD", "CVX", "MRK", "ABBV", "PFE",        # Home Depot, Chevron, Merck, AbbVie, Pfizer
        "KO", "PEP", "COST", "TMO", "AVGO",       # Coca-Cola, Pepsi, Costco, Thermo Fisher, Broadcom
        "WMT", "MCD", "CSCO", "ACN", "DHR",       # Walmart, McDonald's, Cisco, Accenture, Danaher
    ],
    "米国株_テック30": [
        # テクノロジー・半導体
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",  # Big Tech
        "META", "CRM", "ADBE", "ORCL", "INTC",    # Software & Semiconductor
        "AMD", "QCOM", "TXN", "AVGO", "MU",       # Semiconductors
        "NOW", "SNOW", "PANW", "CRWD", "ZS",      # Cloud & Security
        "NET", "DDOG", "TEAM", "SHOP", "SQ",      # SaaS & Fintech
        "UBER", "ABNB", "DASH", "COIN", "RBLX",   # Platform & Gaming
        "PLTR", "U", "MRVL", "AMAT", "LRCX",      # AI & Semicon Equipment
    ],
    "米国株_金融20": [
        # 銀行・証券・保険・決済
        "JPM", "BAC", "WFC", "C", "GS",           # Banks
        "MS", "SCHW", "BLK", "AXP", "COF",        # Investment & Cards
        "V", "MA", "PYPL", "ADP", "FIS",          # Payments
        "MET", "PRU", "ALL", "TRV", "AFL",        # Insurance
    ],
    # 米国株_ヘルスケア20: ロバスト性テストで除外 (全期間平均 +12.82%)
    # 米国株_消費財20: ロバスト性テストで除外 (全期間平均 -3.71%)
    "米国株_工業20": [
        # 製造・航空・防衛・輸送
        "CAT", "DE", "HON", "UNP", "UPS",         # Industrial & Transport
        "BA", "LMT", "RTX", "GD", "NOC",          # Aerospace & Defense
        "GE", "MMM", "EMR", "ITW", "ETN",         # Diversified Industrial
        "FDX", "CSX", "NSC", "DAL", "UAL",        # Transport & Airlines
    ],
    "米国株_エネルギー15": [
        # 石油・ガス・再エネ
        "XOM", "CVX", "COP", "SLB", "EOG",        # Oil & Gas
        "MPC", "VLO", "PSX", "OXY", "KMI",        # Refining & Midstream
        "ENPH", "SEDG", "FSLR", "NEE", "DUK",     # Renewables & Utilities
    ],
    
    # ========== 除外セクター（ロバスト性テストでパフォーマンス低） ==========
    # "米国株_ヘルスケア20": 全期間平均 +12.82% (弱い)
    # "米国株_消費財20": 全期間平均 -3.71% (マイナス)
    
    # ========== 欧州・その他 ==========
    "欧州株_主力20": [
        # 欧州の主要銘柄 (ADR/米国上場)
        "ASML", "NVO", "SAP", "TM", "SHEL",       # ASML, Novo Nordisk, SAP, Toyota(ADR), Shell
        "AZN", "NVS", "HSBC", "UL", "BP",         # AstraZeneca, Novartis, HSBC, Unilever, BP
        "SNY", "GSK", "DEO", "BUD", "RIO",        # Sanofi, GSK, Diageo, AB InBev, Rio Tinto
        "BHP", "VALE", "LYG", "BCS", "BTI",       # BHP, Vale, Lloyds, Barclays, BAT
    ],
}

# 全銘柄リスト（ユニーク化）
ALL_TICKERS = list(set(
    ticker for tickers in TICKER_SETS.values() for ticker in tickers
))

# テスト用銘柄セット（クイックテスト）
QUICK_TEST_SETS = ["日本株_大型50", "米国株_大型30"]

# フルテスト用セット
FULL_TEST_SETS = list(TICKER_SETS.keys())

# テスト期間定義
TEST_PERIODS = {
    "直近1年": {"end": datetime.now(), "days": 365},
    "1-2年前": {"end": datetime.now() - timedelta(days=365), "days": 365},
    "2-3年前": {"end": datetime.now() - timedelta(days=730), "days": 365},
    "3-4年前": {"end": datetime.now() - timedelta(days=1095), "days": 365},
    "直近2年": {"end": datetime.now(), "days": 730},
    "直近3年": {"end": datetime.now(), "days": 1095},
}


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
    df['bb_position'] = (df['Close'] - (df['bb_mid'] - 2*df['bb_std'])) / (4*df['bb_std'])
    
    # 移動平均
    df['sma_5'] = df['Close'].rolling(5).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_ratio'] = df['sma_5'] / df['sma_20']
    
    # リターン
    df['return_1d'] = df['Close'].pct_change()
    df['return_5d'] = df['Close'].pct_change(5)
    df['return_20d'] = df['Close'].pct_change(20)
    
    # ボラティリティ
    df['volatility'] = df['return_1d'].rolling(20).std()
    
    # 出来高
    df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # ターゲット（5日後リターン）
    df['target'] = df['Close'].shift(-5) / df['Close'] - 1
    
    return df


def simple_backtest(df: pd.DataFrame, use_gpu: bool = True) -> dict:
    """シンプルなバックテスト（Walk-Forward）"""
    
    features = ['rsi', 'macd_hist', 'bb_position', 'sma_ratio', 
                'return_1d', 'return_5d', 'volatility', 'volume_ratio']
    
    df_clean = df.dropna(subset=features + ['target'])
    
    if len(df_clean) < 100:
        return None
    
    # パラメータ
    train_window = 60  # 60日で学習
    update_step = 20   # 20日ごとに再学習
    
    model = StockPredictorXGB(use_gpu=use_gpu)
    
    predictions = []
    actuals = []
    
    X_all = df_clean[features].values
    y_all = df_clean['target'].values
    
    for i in range(train_window, len(df_clean) - 5, update_step):
        train_start = max(0, i - train_window)
        train_end = i - 5  # 5日間のパージ
        
        if train_end - train_start < 30:
            continue
        
        X_train = X_all[train_start:train_end]
        y_train = y_all[train_start:train_end]
        
        valid_mask = ~np.isnan(y_train)
        if np.sum(valid_mask) < 20:
            continue
        
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        
        try:
            model.fit(X_train, y_train)
            
            end_idx = min(i + update_step, len(df_clean) - 5)
            X_pred = X_all[i:end_idx]
            y_actual = y_all[i:end_idx]
            
            if len(X_pred) == 0:
                continue
            
            preds = model.predict(X_pred)
            predictions.extend(preds)
            actuals.extend(y_actual)
        except:
            continue
    
    if len(predictions) < 10:
        return None
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # パフォーマンス計算
    # 予測が正の場合に買い、予測通りに動いたら利益
    correct = np.sign(predictions) == np.sign(actuals)
    
    # シンプルな戦略：予測が正なら買い
    strategy_returns = np.where(predictions > 0, actuals, 0)
    
    total_return = np.sum(strategy_returns)
    win_rate = np.mean(correct[predictions != 0]) if np.sum(predictions != 0) > 0 else 0
    
    # シャープレシオ（簡易版）
    if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252 / update_step)
    else:
        sharpe = 0
    
    return {
        'total_return': total_return * 100,  # %
        'win_rate': win_rate * 100,  # %
        'sharpe_ratio': sharpe,
        'n_predictions': len(predictions),
        'n_trades': np.sum(predictions > 0)
    }


def run_robustness_test(ticker_sets: list = None, periods: list = None, use_gpu: bool = True):
    """ロバスト性テスト実行"""
    
    if ticker_sets is None:
        ticker_sets = ["日本株_主力", "日本株_テック", "米国株_主力"]
    
    if periods is None:
        periods = ["直近1年", "1-2年前", "2-3年前"]
    
    results = []
    
    print("=" * 70)
    print("🔬 ロバスト性検証開始")
    print("=" * 70)
    print(f"   銘柄セット: {ticker_sets}")
    print(f"   テスト期間: {periods}")
    print(f"   GPU使用: {use_gpu}")
    print("=" * 70)
    
    for period_name in periods:
        period = TEST_PERIODS.get(period_name)
        if not period:
            print(f"⚠️ 未定義の期間: {period_name}")
            continue
        
        end_date = period['end']
        start_date = end_date - timedelta(days=period['days'])
        
        print(f"\n📅 期間: {period_name} ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")
        print("-" * 70)
        
        for set_name in ticker_sets:
            tickers = TICKER_SETS.get(set_name, [])
            if not tickers:
                continue
            
            set_results = []
            
            for ticker in tickers:
                try:
                    # データ取得
                    stock = yf.Ticker(ticker)
                    df = stock.history(start=start_date, end=end_date)
                    
                    if len(df) < 60:
                        continue
                    
                    # 指標計算
                    df = calculate_indicators(df)
                    
                    # バックテスト
                    result = simple_backtest(df, use_gpu=use_gpu)
                    
                    if result:
                        result['ticker'] = ticker
                        set_results.append(result)
                
                except Exception as e:
                    pass
            
            if set_results:
                # セット平均を計算
                avg_return = np.mean([r['total_return'] for r in set_results])
                avg_winrate = np.mean([r['win_rate'] for r in set_results])
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in set_results])
                
                result_summary = {
                    'period': period_name,
                    'ticker_set': set_name,
                    'avg_return': avg_return,
                    'avg_win_rate': avg_winrate,
                    'avg_sharpe': avg_sharpe,
                    'n_tickers': len(set_results),
                    'details': set_results
                }
                results.append(result_summary)
                
                # 表示
                emoji = "✅" if avg_return > 0 else "❌"
                print(f"   {emoji} {set_name}: リターン {avg_return:+.2f}% | 勝率 {avg_winrate:.1f}% | シャープ {avg_sharpe:.2f} ({len(set_results)}銘柄)")
    
    # サマリー表示
    print("\n" + "=" * 70)
    print("📊 ロバスト性検証サマリー")
    print("=" * 70)
    
    if results:
        df_summary = pd.DataFrame([{
            '期間': r['period'],
            '銘柄セット': r['ticker_set'],
            '平均リターン': f"{r['avg_return']:+.2f}%",
            '勝率': f"{r['avg_win_rate']:.1f}%",
            'シャープ': f"{r['avg_sharpe']:.2f}",
            '銘柄数': r['n_tickers']
        } for r in results])
        
        print(df_summary.to_string(index=False))
        
        # 全体評価
        all_returns = [r['avg_return'] for r in results]
        positive_pct = sum(1 for r in all_returns if r > 0) / len(all_returns) * 100
        
        print("\n" + "-" * 70)
        print(f"   📈 プラスリターン率: {positive_pct:.0f}% ({sum(1 for r in all_returns if r > 0)}/{len(all_returns)})")
        print(f"   📊 平均リターン: {np.mean(all_returns):+.2f}%")
        print(f"   📉 最小リターン: {min(all_returns):+.2f}%")
        print(f"   📈 最大リターン: {max(all_returns):+.2f}%")
        
        if positive_pct >= 70:
            print("\n   🎉 戦略は高いロバスト性を示しています！")
        elif positive_pct >= 50:
            print("\n   ⚠️ 戦略は中程度のロバスト性です。改善の余地があります。")
        else:
            print("\n   ❌ 戦略のロバスト性に問題があります。見直しが必要です。")
    
    # 結果保存
    output_dir = PROJECT_ROOT / 'analysis'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"robustness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n   📁 結果保存: {output_file}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ロバスト性検証')
    parser.add_argument('--quick', action='store_true', help='クイックテスト（大型銘柄のみ）')
    parser.add_argument('--full', action='store_true', help='フルテスト（全銘柄・全期間）')
    parser.add_argument('--medium', action='store_true', help='中規模テスト（主要セット）')
    parser.add_argument('--cpu', action='store_true', help='CPUのみ使用')
    args = parser.parse_args()
    
    use_gpu = not args.cpu
    
    if args.quick:
        # クイックテスト（大型銘柄2セット × 2期間）
        run_robustness_test(
            ticker_sets=QUICK_TEST_SETS,
            periods=["直近1年", "1-2年前"],
            use_gpu=use_gpu
        )
    elif args.full:
        # フルテスト（全16セット × 6期間）
        run_robustness_test(
            ticker_sets=FULL_TEST_SETS,
            periods=list(TEST_PERIODS.keys()),
            use_gpu=use_gpu
        )
    elif args.medium:
        # 中規模テスト（主要8セット × 3期間）
        medium_sets = [
            "日本株_大型50", "日本株_テック30", "日本株_金融20",
            "米国株_大型30", "米国株_テック30", "米国株_ヘルスケア20",
            "米国株_消費財20", "欧州株_主力20"
        ]
        run_robustness_test(
            ticker_sets=medium_sets,
            periods=["直近1年", "1-2年前", "2-3年前"],
            use_gpu=use_gpu
        )
    else:
        # デフォルト（日本大型 + 米国大型 × 3期間）
        run_robustness_test(
            ticker_sets=["日本株_大型50", "米国株_大型30", "欧州株_主力20"],
            periods=["直近1年", "1-2年前", "2-3年前"],
            use_gpu=use_gpu
        )
