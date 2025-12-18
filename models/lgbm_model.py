import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import warnings

# Suppress LightGBM warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', module='lightgbm')

class StockPredictorLGBM(BaseEstimator, RegressorMixin):
    """
    LightGBM wrapper for Stock Prediction with Purged Cross-Validation support
    """
    def __init__(self, params=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.params = params if params else {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63, 
            'learning_rate': 0.03, 
            'feature_fraction': 0.8,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'lambda_l1': 0.1, 
            'lambda_l2': 0.1, 
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42
        }
        
        if self.use_gpu:
            # RTX 5070 (sm_120) は現在のLightGBM OpenCLで非対応
            # 代わりにCPU並列処理を最大化
            self.params['n_jobs'] = -1  # 全CPUコアを使用
            self.params['num_leaves'] = 127
            self.params['max_bin'] = 255
            # GPU設定は無効化 (動作しないため)
            # self.params['device'] = 'gpu'
            
        self.model = None
        self.feature_importance_ = None

    def fit(self, X, y, categorical_features=None):
        """
        Train the LightGBM model
        """
        # 学習ラウンド数をGPUモードで増やす
        num_rounds = 200 if self.use_gpu else 100
        
        # Suppress LightGBM output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Create dataset
            train_data = lgb.Dataset(X, label=y, categorical_feature=categorical_features or 'auto')
            
            # Train
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_rounds,
                callbacks=[lgb.log_evaluation(period=-1)] # Disable logging
            )
        
        self.feature_importance_ = self.model.feature_importance()
        return self

    def predict(self, X):
        """
        Predict using the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross Validation
    Ensures no data leakage by enforcing a gap between train and test sets
    based on the prediction horizon (embargo/purge).
    """
    def __init__(self, n_splits=5, purge_overlap=5):
        self.n_splits = n_splits
        self.purge_overlap = purge_overlap

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Simple TimeSeriesSplit logic but with purging
        # We want to split the data into (n_splits + 1) chunks
        # Train on [0...k], Test on [k+purge...k+purge+step]
        
        fold_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            train_end = (i + 1) * fold_size
            test_start = train_end + self.purge_overlap
            test_end = test_start + fold_size
            
            if test_start >= n_samples:
                break
                
            yield indices[:train_end], indices[test_start:min(test_end, n_samples)]

def train_lgbm_with_purging(df, features, target_col, train_window=252, purge_gap=5):
    """
    Train LightGBM with purging logic for Walk-Forward context.
    
    In a Walk-Forward loop (simulating real-time trading), we are at time T.
    We want to predict T+1 (or T+horizon).
    We can only train on data where the target is known.
    If target is 5-day return, we know the target for T-5 at time T.
    So we train on [0 ... T-5].
    
    This function is designed to be called inside the loop or handle the loop itself?
    The existing backtest loop iterates through time.
    
    If we replace the RandomForest logic in the loop:
    Current logic:
        train_data = df.iloc[:i]
        predict_data = df.iloc[i:i+step]
        
    New logic (Purged):
        # If target is future return (e.g. 5 days later), we must shift labels or cut training data.
        # Usually 'target' column in df is already shifted? 
        # Let's check how 'target' is defined in the main file.
        # It seems 'target' is usually "Close.shift(-5) / Close - 1".
        # If so, the row at index T contains the return realized at T+5.
        # At time T, we DON'T know the value in row T. We only know row T-5.
        
    So, we need to enforce:
    train_indices = indices[:current_index - purge_gap]
    """
    pass
