import xgboost as xgb
import numpy as np
import time

print(f"XGBoost Version: {xgb.__version__}")
print(f"Build Info: {xgb.build_info()}")

# テストデータ生成
n_samples = 500000
n_features = 50
np.random.seed(42)
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(n_samples) * 0.1 > 0).astype(np.int32)

dtrain = xgb.DMatrix(X, label=y)

print(f"\n=== XGBoost GPU Test ({n_samples:,} samples, {n_features} features) ===")

# GPU学習
params_gpu = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda:0',
    'max_depth': 6,
    'n_estimators': 100
}

start = time.time()
model_gpu = xgb.train(params_gpu, dtrain, num_boost_round=100)
gpu_time = time.time() - start
print(f"GPU Training Time: {gpu_time:.2f}秒")

# CPU学習
params_cpu = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cpu',
    'max_depth': 6,
    'n_estimators': 100
}

start = time.time()
model_cpu = xgb.train(params_cpu, dtrain, num_boost_round=100)
cpu_time = time.time() - start
print(f"CPU Training Time: {cpu_time:.2f}秒")

print(f"\n🚀 GPU Speedup: {cpu_time/gpu_time:.2f}x")

if gpu_time < cpu_time:
    print("✅ GPU is FASTER!")
else:
    print("⚠️ CPU is still faster (small data or GPU overhead)")
