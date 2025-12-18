import xgboost as xgb
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

class StockPredictorXGB(BaseEstimator, RegressorMixin):
    """
    XGBoost wrapper for Stock Prediction with GPU support
    XGBoostはLightGBMより効率的にGPUを使用する (gpu_hist)
    """
    def __init__(self, params=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.params = params if params else {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': 0,
            'nthread': -1,
            'random_state': 42
        }
        
        if self.use_gpu:
            # XGBoost 3.x のGPU設定
            self.params['tree_method'] = 'hist'
            self.params['device'] = 'cuda:0'  # 明示的にGPU 0を指定
            self.params['nthread'] = 1
            self.params['max_depth'] = 8
            self.params['max_bin'] = 256
        else:
            self.params['tree_method'] = 'hist'
            self.params['device'] = 'cpu'
            
        self.model = None
        self.feature_importance_ = None

    def fit(self, X, y):
        """
        Train the XGBoost model with optimized settings for speed
        """
        # GPUはオーバーヘッドがあるので、小さいデータではラウンド数を減らす
        # 大きいデータではGPUの恩恵を受ける
        n_samples = len(X)
        if self.use_gpu and n_samples < 1000:
            # 小さいデータはCPUの方が速い場合がある
            num_rounds = 50
        elif self.use_gpu:
            num_rounds = 100  # GPU: 高速なので適度なラウンド数
        else:
            num_rounds = 100
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X, label=y)
            
            # Train with early stopping (validation split)
            # データが小さい場合はearly stoppingなし
            if n_samples > 500:
                # 10%をvalidationに使用
                val_size = int(n_samples * 0.1)
                dval = xgb.DMatrix(X[-val_size:], label=y[-val_size:])
                dtrain_sub = xgb.DMatrix(X[:-val_size], label=y[:-val_size])
                
                self.model = xgb.train(
                    self.params,
                    dtrain_sub,
                    num_boost_round=num_rounds,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=10,
                    verbose_eval=False
                )
            else:
                self.model = xgb.train(
                    self.params,
                    dtrain,
                    num_boost_round=num_rounds,
                    verbose_eval=False
                )
        
        self.feature_importance_ = self.model.get_score(importance_type='gain')
        return self

    def predict(self, X):
        """
        Predict using the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def get_feature_importance(self):
        """
        Get feature importance
        """
        return self.feature_importance_
