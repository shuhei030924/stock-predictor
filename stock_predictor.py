"""
株価予測ツール (Stock Price Predictor)
=====================================
ARIMA、機械学習（Random Forest）、技術指標を組み合わせた株価予測ツール

使い方:
    python stock_predictor.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# データ取得
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Using dummy data.")

# 予測モデル
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class StockPredictor:
    """株価予測クラス"""
    
    def __init__(self, ticker: str, period: str = "2y"):
        """
        Args:
            ticker: 銘柄コード (例: "7203.T" トヨタ, "AAPL" Apple)
            period: データ取得期間 (例: "1y", "2y", "5y")
        """
        self.ticker = ticker
        self.period = period
        self.data = None
        self.predictions = {}
        
    def fetch_data(self) -> pd.DataFrame:
        """株価データを取得"""
        print(f"\n📊 {self.ticker} のデータを取得中...")
        
        if YFINANCE_AVAILABLE:
            try:
                import ssl
                ssl._create_default_https_context = ssl._create_unverified_context
                
                stock = yf.Ticker(self.ticker)
                self.data = stock.history(period=self.period)
                
                if len(self.data) == 0:
                    raise ValueError("データが取得できませんでした")
                    
                print(f"✓ {len(self.data)}日分のデータを取得しました")
                print(f"  期間: {self.data.index[0].strftime('%Y-%m-%d')} ~ {self.data.index[-1].strftime('%Y-%m-%d')}")
                
            except Exception as e:
                print(f"⚠ データ取得エラー: {e}")
                print("  ダミーデータを使用します")
                self.data = self._generate_dummy_data()
        else:
            self.data = self._generate_dummy_data()
            
        return self.data
    
    def _generate_dummy_data(self) -> pd.DataFrame:
        """ダミーデータを生成"""
        np.random.seed(42)
        days = 500
        dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
        
        # ランダムウォーク + トレンド
        returns = np.random.normal(0.0005, 0.02, days)
        price = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Open': price * (1 + np.random.uniform(-0.01, 0.01, days)),
            'High': price * (1 + np.random.uniform(0, 0.02, days)),
            'Low': price * (1 - np.random.uniform(0, 0.02, days)),
            'Close': price,
            'Volume': np.random.randint(1000000, 10000000, days)
        }, index=dates)
        
        print(f"✓ ダミーデータ {len(df)}日分を生成しました")
        return df
    
    def add_technical_indicators(self):
        """テクニカル指標を追加"""
        df = self.data.copy()
        
        # 移動平均
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # ボリンジャーバンド
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 変化率
        df['Return'] = df['Close'].pct_change()
        df['Return_5d'] = df['Close'].pct_change(5)
        
        # 出来高変化
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        self.data = df.dropna()
        print(f"✓ テクニカル指標を追加しました ({len(df.columns)}列)")
        
    def predict_arima(self, forecast_days: int = 30) -> pd.Series:
        """ARIMAモデルで予測"""
        print(f"\n🔮 ARIMA予測を実行中...")
        
        try:
            # 最適なパラメータを探索（簡易版）
            best_aic = float('inf')
            best_order = (1, 1, 1)
            
            for p in range(3):
                for q in range(3):
                    try:
                        model = ARIMA(self.data['Close'], order=(p, 1, q))
                        result = model.fit()
                        if result.aic < best_aic:
                            best_aic = result.aic
                            best_order = (p, 1, q)
                    except:
                        continue
            
            # 最適モデルで予測
            model = ARIMA(self.data['Close'], order=best_order)
            result = model.fit()
            
            forecast = result.get_forecast(steps=forecast_days)
            self.predictions['ARIMA'] = {
                'mean': forecast.predicted_mean,
                'ci': forecast.conf_int(),
                'order': best_order
            }
            
            print(f"✓ ARIMA{best_order} 予測完了")
            return forecast.predicted_mean
            
        except Exception as e:
            print(f"⚠ ARIMA予測エラー: {e}")
            return None
    
    def predict_ml(self, forecast_days: int = 30) -> np.ndarray:
        """機械学習（Random Forest）で予測"""
        print(f"\n🤖 機械学習予測を実行中...")
        
        try:
            # 特徴量の準備
            features = ['SMA_5', 'SMA_20', 'RSI', 'MACD', 'Volume_ratio', 'Return_5d']
            X = self.data[features].values
            y = self.data['Close'].values
            
            # 将来予測用にラグ特徴量を作成
            df_ml = self.data.copy()
            for i in range(1, 6):
                df_ml[f'Close_lag{i}'] = df_ml['Close'].shift(i)
            df_ml = df_ml.dropna()
            
            features_lag = features + [f'Close_lag{i}' for i in range(1, 6)]
            X = df_ml[features_lag].values
            y = df_ml['Close'].values
            
            # 学習/テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # モデル学習
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # テストスコア
            score = model.score(X_test_scaled, y_test)
            print(f"  R² Score: {score:.4f}")
            
            # 将来予測（逐次的に）
            last_row = df_ml[features_lag].iloc[-1:].values
            predictions = []
            
            for _ in range(forecast_days):
                pred = model.predict(scaler.transform(last_row))[0]
                predictions.append(pred)
                
                # 次の入力を準備（簡易的にシフト）
                last_row = np.roll(last_row, 1)
                last_row[0, -1] = pred
            
            self.predictions['ML'] = {
                'values': np.array(predictions),
                'score': score
            }
            
            print(f"✓ 機械学習予測完了")
            return np.array(predictions)
            
        except Exception as e:
            print(f"⚠ 機械学習予測エラー: {e}")
            return None
    
    def get_signal(self) -> dict:
        """売買シグナルを生成"""
        latest = self.data.iloc[-1]
        signals = []
        
        # RSI シグナル
        if latest['RSI'] < 30:
            signals.append(('RSI', '買い', f"RSI={latest['RSI']:.1f} (売られすぎ)"))
        elif latest['RSI'] > 70:
            signals.append(('RSI', '売り', f"RSI={latest['RSI']:.1f} (買われすぎ)"))
        else:
            signals.append(('RSI', '中立', f"RSI={latest['RSI']:.1f}"))
        
        # 移動平均シグナル
        if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
            signals.append(('MA', '買い', '上昇トレンド'))
        elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
            signals.append(('MA', '売り', '下降トレンド'))
        else:
            signals.append(('MA', '中立', 'レンジ相場'))
        
        # MACD シグナル
        if latest['MACD'] > latest['MACD_signal']:
            signals.append(('MACD', '買い', 'ゴールデンクロス'))
        else:
            signals.append(('MACD', '売り', 'デッドクロス'))
        
        # ボリンジャーバンド
        if latest['Close'] < latest['BB_lower']:
            signals.append(('BB', '買い', '下バンド割れ'))
        elif latest['Close'] > latest['BB_upper']:
            signals.append(('BB', '売り', '上バンド突破'))
        else:
            signals.append(('BB', '中立', 'バンド内'))
        
        return {
            'signals': signals,
            'latest_price': latest['Close'],
            'latest_date': self.data.index[-1]
        }
    
    def plot_analysis(self, forecast_days: int = 30):
        """分析結果を可視化"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        
        # 1. 株価チャート + 予測
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data['Close'], label='実績', color='blue')
        ax1.plot(self.data.index, self.data['SMA_20'], label='SMA20', alpha=0.7)
        ax1.plot(self.data.index, self.data['SMA_50'], label='SMA50', alpha=0.7)
        ax1.fill_between(self.data.index, self.data['BB_lower'], self.data['BB_upper'],
                         alpha=0.2, color='gray', label='BB')
        
        # ARIMA予測を追加
        if 'ARIMA' in self.predictions:
            forecast_dates = pd.date_range(
                start=self.data.index[-1] + timedelta(days=1),
                periods=forecast_days,
                freq='B'
            )
            pred = self.predictions['ARIMA']
            ax1.plot(forecast_dates, pred['mean'], 'r--', label='ARIMA予測')
            ax1.fill_between(forecast_dates, 
                           pred['ci'].iloc[:, 0], 
                           pred['ci'].iloc[:, 1],
                           alpha=0.2, color='red')
        
        # ML予測を追加
        if 'ML' in self.predictions:
            forecast_dates = pd.date_range(
                start=self.data.index[-1] + timedelta(days=1),
                periods=forecast_days,
                freq='B'
            )
            ax1.plot(forecast_dates, self.predictions['ML']['values'], 
                    'g--', label='ML予測')
        
        ax1.set_title(f'{self.ticker} 株価チャート & 予測', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = axes[1]
        ax2.plot(self.data.index, self.data['RSI'], color='purple')
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax2.fill_between(self.data.index, 30, 70, alpha=0.1, color='gray')
        ax2.set_title('RSI (Relative Strength Index)', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = axes[2]
        ax3.plot(self.data.index, self.data['MACD'], label='MACD', color='blue')
        ax3.plot(self.data.index, self.data['MACD_signal'], label='Signal', color='orange')
        ax3.bar(self.data.index, self.data['MACD'] - self.data['MACD_signal'], 
               alpha=0.3, color='gray')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('MACD', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. 出来高
        ax4 = axes[3]
        colors = ['green' if self.data['Close'].iloc[i] >= self.data['Open'].iloc[i] 
                  else 'red' for i in range(len(self.data))]
        ax4.bar(self.data.index, self.data['Volume'], color=colors, alpha=0.7)
        ax4.plot(self.data.index, self.data['Volume_SMA'], color='blue', label='SMA20')
        ax4.set_title('出来高', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'stock_analysis_{self.ticker.replace(".", "_")}.png', dpi=150)
        plt.show()
        print(f"\n📈 チャートを保存しました: stock_analysis_{self.ticker.replace('.', '_')}.png")
    
    def generate_report(self, forecast_days: int = 30) -> str:
        """分析レポートを生成"""
        signal_info = self.get_signal()
        
        report = f"""
{'='*60}
📊 株価分析レポート: {self.ticker}
{'='*60}

【基本情報】
  分析日: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  最新株価: {signal_info['latest_price']:.2f}
  データ期間: {self.data.index[0].strftime('%Y-%m-%d')} ~ {self.data.index[-1].strftime('%Y-%m-%d')}

【テクニカルシグナル】
"""
        for indicator, signal, reason in signal_info['signals']:
            emoji = '🟢' if signal == '買い' else '🔴' if signal == '売り' else '⚪'
            report += f"  {emoji} {indicator}: {signal} ({reason})\n"
        
        # 総合判断
        buy_count = sum(1 for _, s, _ in signal_info['signals'] if s == '買い')
        sell_count = sum(1 for _, s, _ in signal_info['signals'] if s == '売り')
        
        if buy_count > sell_count:
            overall = "買い優勢 📈"
        elif sell_count > buy_count:
            overall = "売り優勢 📉"
        else:
            overall = "中立 ➡️"
        
        report += f"\n【総合判断】 {overall} (買い{buy_count} / 売り{sell_count})\n"
        
        # 予測情報
        if 'ARIMA' in self.predictions:
            pred = self.predictions['ARIMA']
            future_price = pred['mean'].iloc[-1]
            change = (future_price - signal_info['latest_price']) / signal_info['latest_price'] * 100
            report += f"""
【ARIMA予測】
  モデル: ARIMA{pred['order']}
  {forecast_days}日後予測: {future_price:.2f} ({change:+.2f}%)
"""
        
        if 'ML' in self.predictions:
            ml_pred = self.predictions['ML']['values'][-1]
            change = (ml_pred - signal_info['latest_price']) / signal_info['latest_price'] * 100
            report += f"""
【機械学習予測】
  R² Score: {self.predictions['ML']['score']:.4f}
  {forecast_days}日後予測: {ml_pred:.2f} ({change:+.2f}%)
"""
        
        report += f"""
{'='*60}
⚠️ 注意: この予測は参考情報です。投資判断は自己責任で行ってください。
{'='*60}
"""
        return report
    
    def run_full_analysis(self, forecast_days: int = 30):
        """完全な分析を実行"""
        # データ取得
        self.fetch_data()
        
        # テクニカル指標追加
        self.add_technical_indicators()
        
        # 予測実行
        self.predict_arima(forecast_days)
        self.predict_ml(forecast_days)
        
        # レポート生成
        report = self.generate_report(forecast_days)
        print(report)
        
        # チャート表示
        self.plot_analysis(forecast_days)
        
        return report


def main():
    """メイン関数"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║          📈 株価予測ツール (Stock Predictor) 📉            ║
╠═══════════════════════════════════════════════════════════╣
║  ARIMA + 機械学習 + テクニカル分析 による総合予測          ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    # 銘柄コードの例
    print("【銘柄コードの例】")
    print("  日本株: 7203.T (トヨタ), 9984.T (ソフトバンクG), 6758.T (ソニー)")
    print("  米国株: AAPL (Apple), GOOGL (Google), MSFT (Microsoft)")
    print()
    
    # ユーザー入力
    ticker = input("銘柄コードを入力してください (デフォルト: AAPL): ").strip()
    if not ticker:
        ticker = "AAPL"
    
    forecast_days_input = input("予測日数を入力してください (デフォルト: 30): ").strip()
    forecast_days = int(forecast_days_input) if forecast_days_input else 30
    
    # 分析実行
    predictor = StockPredictor(ticker, period="2y")
    predictor.run_full_analysis(forecast_days)


if __name__ == "__main__":
    main()
