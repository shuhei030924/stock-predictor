import lightgbm as lgb
import numpy as np
import pandas as pd

def check_gpu():
    print("Checking LightGBM GPU support...")
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    train_data = lgb.Dataset(X, label=y)
    
    params = {
        'device': 'gpu',
        'verbose': -1
    }
    
    try:
        lgb.train(params, train_data, num_boost_round=1)
        print("GPU support is AVAILABLE.")
        return True
    except Exception as e:
        print(f"GPU support is NOT available: {e}")
        return False

if __name__ == "__main__":
    check_gpu()
