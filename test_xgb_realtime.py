"""
XGBoost GPU学習中にnvidia-smiでプロセスを確認するテスト
このスクリプトを実行中に別ターミナルで nvidia-smi を実行してください
"""
import xgboost as xgb
import numpy as np
import time

print("=" * 60)
print("XGBoost GPU リアルタイムテスト")
print("=" * 60)
print("\n⚠️ このスクリプト実行中に別ターミナルで 'nvidia-smi' を実行してください\n")

# 大きめのデータを生成してGPU負荷をかける
n_samples = 2000000  # 200万サンプル
n_features = 100
print(f"データ生成中... ({n_samples:,} samples, {n_features} features)")

np.random.seed(42)
X = np.random.randn(n_samples, n_features).astype(np.float32)
y = (X[:, 0] + X[:, 1] * 2 + np.random.randn(n_samples) * 0.1 > 0).astype(np.int32)

dtrain = xgb.DMatrix(X, label=y)
print("データ生成完了\n")

# GPU学習パラメータ
params = {
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'device': 'cuda:0',
    'max_depth': 8,
    'max_bin': 256,
    'verbosity': 1
}

print("🚀 GPU学習開始 (200 rounds)...")
print("   別ターミナルで nvidia-smi を実行してGPU使用状況を確認！")
print("-" * 60)

start = time.time()
model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=20)
elapsed = time.time() - start

print("-" * 60)
print(f"\n✅ 学習完了: {elapsed:.2f}秒")
print("\nnvidia-smi で Python プロセスが表示されていれば、GPUが使われています！")
