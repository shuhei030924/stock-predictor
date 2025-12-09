"""
GARCH モデル - ボラティリティ予測
================================
株価のボラティリティ（変動性）予測に特化したモデル
リスク管理や価格変動幅の予測に実務で広く使用される
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


class StockGARCHPredictor:
    """株価ボラティリティ予測用GARCHモデル"""
    
    def __init__(self, p: int = 1, q: int = 1, mean: str = 'AR', 
                 vol: str = 'GARCH', dist: str = 't'):
        """
        Parameters:
        -----------
        p : int - GARCH項の次数
        q : int - ARCH項の次数
        mean : str - 平均モデル ('Zero', 'Constant', 'AR', 'ARX')
        vol : str - ボラティリティモデル ('GARCH', 'EGARCH', 'TGARCH')
        dist : str - 分布 ('normal', 't', 'skewt')
        """
        if not ARCH_AVAILABLE:
            raise ImportError("archがインストールされていません: pip install arch")
        
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.dist = dist
        
        self.model = None
        self.result = None
        self.is_trained = False
        self.returns_std = None
    
    def prepare_returns(self, df: pd.DataFrame) -> pd.Series:
        """リターン系列を準備"""
        if isinstance(df, pd.DataFrame):
            prices = df['Close']
        else:
            prices = df
        
        # 対数リターン（%表示）
        returns = np.log(prices / prices.shift(1)).dropna() * 100
        return returns
    
    def train(self, df: pd.DataFrame, verbose: bool = True) -> dict:
        """モデルを訓練"""
        returns = self.prepare_returns(df)
        self.returns_std = returns.std()
        
        # GARCHモデルの構築
        self.model = arch_model(
            returns,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            q=self.q,
            dist=self.dist
        )
        
        # フィッティング
        self.result = self.model.fit(disp='off' if not verbose else 'final')
        self.is_trained = True
        
        if verbose:
            print("\n[GARCH] Model Summary:")
            print(f"  AIC: {self.result.aic:.2f}")
            print(f"  BIC: {self.result.bic:.2f}")
            print(f"  Log-Likelihood: {self.result.loglikelihood:.2f}")
        
        return {
            'aic': self.result.aic,
            'bic': self.result.bic,
            'loglikelihood': self.result.loglikelihood,
            'params': dict(self.result.params)
        }
    
    def predict_volatility(self, df: pd.DataFrame, forecast_days: int = 30) -> pd.DataFrame:
        """ボラティリティを予測"""
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")
        
        # フォーキャスト
        forecast = self.result.forecast(horizon=forecast_days)
        
        # 予測結果を取得
        variance_forecast = forecast.variance.iloc[-1].values
        mean_forecast = forecast.mean.iloc[-1].values if hasattr(forecast, 'mean') else np.zeros(forecast_days)
        
        # 年率ボラティリティに変換
        volatility = np.sqrt(variance_forecast) * np.sqrt(252)
        
        # 日付インデックスを生成
        last_date = df.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                       periods=forecast_days, freq='B')
        
        return pd.DataFrame({
            'Date': forecast_dates,
            'Volatility': volatility,
            'Variance': variance_forecast,
            'Mean_Return': mean_forecast
        })
    
    def predict_price_range(self, df: pd.DataFrame, forecast_days: int = 30,
                           confidence: float = 0.95) -> pd.DataFrame:
        """価格のレンジ（上限・下限）を予測"""
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")
        
        current_price = df['Close'].iloc[-1]
        vol_forecast = self.predict_volatility(df, forecast_days)
        
        # Z値（信頼区間）
        from scipy import stats
        z = stats.norm.ppf((1 + confidence) / 2)
        
        predictions = []
        cumulative_var = 0
        
        for i, row in vol_forecast.iterrows():
            days = i + 1
            daily_vol = row['Volatility'] / np.sqrt(252)
            cumulative_var += (daily_vol ** 2)
            cumulative_vol = np.sqrt(cumulative_var)
            
            # 対数正規分布を仮定した価格レンジ
            expected_return = row['Mean_Return'] / 100 * days
            
            upper = current_price * np.exp(expected_return + z * cumulative_vol)
            lower = current_price * np.exp(expected_return - z * cumulative_vol)
            mean = current_price * np.exp(expected_return)
            
            predictions.append({
                'Date': row['Date'],
                'Price_Mean': mean,
                'Price_Upper': upper,
                'Price_Lower': lower,
                'Daily_Volatility': daily_vol,
                'Cumulative_Volatility': cumulative_vol
            })
        
        return pd.DataFrame(predictions)
    
    def predict(self, df: pd.DataFrame, forecast_days: int = 30) -> np.ndarray:
        """価格の中央予測を返す（他のモデルとの互換性のため）"""
        price_range = self.predict_price_range(df, forecast_days)
        return price_range['Price_Mean'].values
    
    def backtest(self, df: pd.DataFrame, test_days: int = 30) -> pd.DataFrame:
        """バックテストを実行（ボラティリティ予測の精度）"""
        results = []
        returns = self.prepare_returns(df)
        
        for i in range(test_days, 0, -1):
            train_df = df.iloc[:-i]
            
            if len(train_df) < 200:
                continue
            
            try:
                # 訓練
                self.train(train_df, verbose=False)
                
                # 1日後のボラティリティを予測
                vol_forecast = self.predict_volatility(train_df, forecast_days=1)
                pred_vol = vol_forecast['Volatility'].iloc[0] / np.sqrt(252)  # 日次に変換
                
                # 実際のボラティリティ（実現ボラティリティの代わりに絶対リターンを使用）
                actual_return = abs(returns.iloc[-i])
                
                results.append({
                    'Date': df.index[-i],
                    'Actual_AbsReturn': actual_return,
                    'Predicted_Vol': pred_vol * 100,  # %表示
                    'Error': actual_return - pred_vol * 100
                })
            except Exception as e:
                continue
        
        return pd.DataFrame(results)
    
    def get_model_summary(self) -> str:
        """モデルのサマリーを取得"""
        if not self.is_trained:
            return "モデルが訓練されていません"
        return str(self.result.summary())


def check_garch_availability():
    """GARCHの利用可能性を確認"""
    print("=" * 50)
    print("GARCH (arch) Information")
    print("=" * 50)
    
    if ARCH_AVAILABLE:
        import arch
        print(f"[OK] arch Available: True")
        print(f"   Version: {arch.__version__}")
    else:
        print("[--] arch Available: False")
        print("   Install: pip install arch")
    
    print("=" * 50)
