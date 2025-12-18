"""
XGBoost GPU テスト
"""
import numpy as np
import time

def test_xgb_gpu():
    print("=" * 50)
    print("XGBoost GPU Test")
    print("=" * 50)
    
    try:
        from models.xgb_model import StockPredictorXGB
        print("[OK] XGBoost model imported successfully")
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    
    # テストデータ生成 (非常に大きなデータでGPU効果を確認)
    print("\nGenerating test data (500000 samples, 20 features)...")
    np.random.seed(42)
    X = np.random.rand(500000, 20)
    y = np.random.rand(500000)
    
    # CPU テスト
    print("\n--- CPU Mode ---")
    model_cpu = StockPredictorXGB(use_gpu=False)
    start = time.time()
    model_cpu.fit(X, y)
    cpu_time = time.time() - start
    print(f"Training time: {cpu_time:.3f} seconds")
    
    preds_cpu = model_cpu.predict(X[:100])
    print(f"Prediction sample: {preds_cpu[:3]}")
    
    # GPU テスト
    print("\n--- GPU Mode ---")
    model_gpu = StockPredictorXGB(use_gpu=True)
    start = time.time()
    try:
        model_gpu.fit(X, y)
        gpu_time = time.time() - start
        print(f"Training time: {gpu_time:.3f} seconds")
        
        preds_gpu = model_gpu.predict(X[:100])
        print(f"Prediction sample: {preds_gpu[:3]}")
        
        # 比較
        print("\n--- Comparison ---")
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        if gpu_time < cpu_time:
            print("[OK] GPU is faster!")
        else:
            print("[INFO] GPU may not be faster for small data")
            
        return True
        
    except Exception as e:
        print(f"[ERROR] GPU training failed: {e}")
        print("Falling back to CPU mode recommended")
        return False

if __name__ == "__main__":
    test_xgb_gpu()
