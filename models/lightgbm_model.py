"""
LightGBM モデル - 高速・高精度な勾配ブースティング
================================================
株価予測で実務的によく使われるモデル
特徴量エンジニアリングを活かした予測が可能
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class StockLightGBMPredictor:
    """株価予測用LightGBMモデル"""
    
    def __init__(self, n_estimators: int = 500, learning_rate: float = 0.05,
                 max_depth: int = 6, num_leaves: int = 31):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBMがインストールされていません: pip install lightgbm")
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """テクニカル指標を特徴量として作成"""
        data = df.copy()
        close = data['Close']
        
        # トレンド指標
        for period in [5, 10, 20, 50]:
            data[f'SMA_{period}'] = close.rolling(window=period).mean()
            data[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
            data[f'SMA_ratio_{period}'] = close / data[f'SMA_{period}']
        
        # モメンタム指標
        for period in [5, 10, 20]:
            data[f'ROC_{period}'] = close.pct_change(period) * 100
            data[f'MOM_{period}'] = close - close.shift(period)
        
        # RSI
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        data['MACD'] = exp12 - exp26
        data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_hist'] = data['MACD'] - data['MACD_signal']
        
        # ボリンジャーバンド
        for period in [20]:
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            data[f'BB_upper_{period}'] = sma + 2 * std
            data[f'BB_lower_{period}'] = sma - 2 * std
            data[f'BB_width_{period}'] = (data[f'BB_upper_{period}'] - data[f'BB_lower_{period}']) / sma
            data[f'BB_position_{period}'] = (close - data[f'BB_lower_{period}']) / (data[f'BB_upper_{period}'] - data[f'BB_lower_{period}'])
        
        # ATR (Average True Range)
        if 'High' in data.columns and 'Low' in data.columns:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - close.shift())
            low_close = np.abs(data['Low'] - close.shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['ATR_14'] = tr.rolling(window=14).mean()
            data['ATR_ratio'] = data['ATR_14'] / close
        
        # 出来高関連
        if 'Volume' in data.columns:
            data['Volume_SMA_20'] = data['Volume'].rolling(window=20).mean()
            data['Volume_ratio'] = data['Volume'] / data['Volume_SMA_20']
            data['Volume_change'] = data['Volume'].pct_change()
        
        # ボラティリティ
        for period in [5, 10, 20]:
            data[f'Volatility_{period}'] = close.pct_change().rolling(window=period).std() * np.sqrt(252)
        
        # ラグ特徴量
        for lag in range(1, 6):
            data[f'Close_lag_{lag}'] = close.shift(lag)
            data[f'Return_lag_{lag}'] = close.pct_change().shift(lag)
        
        # 日付特徴量（曜日効果）
        if isinstance(data.index, pd.DatetimeIndex):
            data['DayOfWeek'] = data.index.dayofweek
            data['Month'] = data.index.month
        
        return data
    
    def prepare_data(self, df: pd.DataFrame, target_days: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """データを準備"""
        data = self.create_features(df)
        
        # ターゲット: N日後のリターン
        data['Target'] = data['Close'].shift(-target_days) / data['Close'] - 1
        
        # 欠損値を削除
        data = data.dropna()
        
        # 特徴量カラムを選択（Close, Open等の生データは除外）
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target', 
                       'Dividends', 'Stock Splits', 'Capital Gains']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X = data[feature_cols].values
        y = data['Target'].values
        
        return X, y, feature_cols
    
    def train(self, df: pd.DataFrame, target_days: int = 1, 
              validation_split: float = 0.2, verbose: bool = True) -> dict:
        """モデルを訓練"""
        X, y, feature_cols = self.prepare_data(df, target_days)
        self.feature_names = feature_cols
        
        # 時系列分割（シャッフルしない）
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # スケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # LightGBMデータセット
        train_data = lgb.Dataset(X_train_scaled, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
        
        # パラメータ
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        # 訓練
        callbacks = []
        if verbose:
            callbacks.append(lgb.log_evaluation(period=100))
        callbacks.append(lgb.early_stopping(stopping_rounds=50))
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        self.is_trained = True
        
        # 特徴量重要度
        importance = dict(zip(feature_cols, self.model.feature_importance()))
        
        # 検証スコア
        val_pred = self.model.predict(X_val_scaled)
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        
        # 方向性精度
        direction_accuracy = np.mean((val_pred > 0) == (y_val > 0)) * 100
        
        if verbose:
            print(f"\n[LightGBM] Validation RMSE: {val_rmse:.6f}")
            print(f"[LightGBM] Direction Accuracy: {direction_accuracy:.1f}%")
            print(f"\nTop 10 Features:")
            for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
                print(f"  {feat}: {imp}")
        
        return {
            'val_rmse': val_rmse,
            'direction_accuracy': direction_accuracy,
            'feature_importance': importance
        }
    
    def predict(self, df: pd.DataFrame, forecast_days: int = 30) -> np.ndarray:
        """将来の株価を予測"""
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません")
        
        data = self.create_features(df)
        data = data.dropna()
        
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Target',
                       'Dividends', 'Stock Splits', 'Capital Gains']
        feature_cols = [col for col in self.feature_names if col in data.columns]
        
        predictions = []
        current_price = df['Close'].iloc[-1]
        
        # 最新の特徴量を取得
        last_features = data[feature_cols].iloc[-1:].values
        last_features_scaled = self.scaler.transform(last_features)
        
        for _ in range(forecast_days):
            # リターンを予測
            pred_return = self.model.predict(last_features_scaled)[0]
            
            # 価格に変換
            next_price = current_price * (1 + pred_return)
            predictions.append(next_price)
            current_price = next_price
        
        return np.array(predictions)
    
    def backtest(self, df: pd.DataFrame, test_days: int = 30, 
                 target_days: int = 1) -> pd.DataFrame:
        """バックテストを実行"""
        results = []
        
        for i in range(test_days, 0, -1):
            train_df = df.iloc[:-i]
            
            if len(train_df) < 200:
                continue
            
            try:
                # 訓練
                self.train(train_df, target_days=target_days, verbose=False)
                
                # 1日後を予測
                pred_prices = self.predict(train_df, forecast_days=1)
                pred_price = pred_prices[0]
                
                actual_price = df['Close'].iloc[-i]
                actual_date = df.index[-i]
                
                results.append({
                    'Date': actual_date,
                    'Actual': actual_price,
                    'Predicted': pred_price,
                    'Error': actual_price - pred_price,
                    'Error_Pct': (actual_price - pred_price) / actual_price * 100
                })
            except Exception as e:
                continue
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """特徴量重要度を取得"""
        if not self.is_trained:
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importance()
        })
        return importance.sort_values('Importance', ascending=False)


def check_lightgbm_availability():
    """LightGBMの利用可能性を確認"""
    print("=" * 50)
    print("LightGBM Information")
    print("=" * 50)
    
    if LIGHTGBM_AVAILABLE:
        print(f"[OK] LightGBM Available: True")
        print(f"   Version: {lgb.__version__}")
    else:
        print("[--] LightGBM Available: False")
        print("   Install: pip install lightgbm")
    
    print("=" * 50)
