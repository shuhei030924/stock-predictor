"""
XGBoost GPU学習と同時にnvidia-smiを実行してGPU使用状況を確認
"""
import xgboost as xgb
import numpy as np
import subprocess
import threading
import time

def monitor_gpu():
    """バックグラウンドでGPU監視"""
    for i in range(10):
        time.sleep(1)
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                 '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        if result.returncode == 0:
            gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
            print(f"  [監視 {i+1}s] GPU使用率: {gpu_util}% | メモリ: {mem_used}MB / {mem_total}MB")

print("=" * 70)
print("XGBoost GPU 使用状況モニタリングテスト")
print("=" * 70)

# データ生成
n_samples = 3000000  # 300万サンプル
n_features = 100
print(f"\n1. データ生成中... ({n_samples:,} samples)")
np.random.seed(42)
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(n_samples) * 0.1 > 0).astype(np.int32)
dtrain = xgb.DMatrix(X, label=y)
print("   完了")

# GPU監視スレッド開始
print("\n2. GPU監視スレッド開始...")
monitor_thread = threading.Thread(target=monitor_gpu, daemon=True)
monitor_thread.start()

# GPU学習
params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda:0',
    'max_depth': 10,
    'max_bin': 512,
    'verbosity': 0
}

print("\n3. XGBoost GPU学習開始 (300 rounds)...")
print("-" * 70)
start = time.time()
model = xgb.train(params, dtrain, num_boost_round=300)
elapsed = time.time() - start
print("-" * 70)

print(f"\n✅ 学習完了: {elapsed:.2f}秒")
print(f"   スループット: {n_samples / elapsed:,.0f} samples/sec")

# 最終GPU状態
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print("\n4. 最終GPU状態:")
print(result.stdout)
