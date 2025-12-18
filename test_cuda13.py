import torch
import time

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # GPU計算テスト
    print("\n=== GPU Compute Test ===")
    x = torch.randn(5000, 5000, device='cuda')
    
    # ウォームアップ
    _ = torch.matmul(x, x)
    torch.cuda.synchronize()
    
    # 実際のベンチマーク
    start = time.time()
    for _ in range(10):
        y = torch.matmul(x, x)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"GPU: 5000x5000 行列積 x10回 = {gpu_time:.3f}秒")
    
    # CPU比較
    x_cpu = torch.randn(5000, 5000)
    start = time.time()
    for _ in range(10):
        y_cpu = torch.matmul(x_cpu, x_cpu)
    cpu_time = time.time() - start
    print(f"CPU: 5000x5000 行列積 x10回 = {cpu_time:.3f}秒")
    
    print(f"\n🚀 GPU Speedup: {cpu_time/gpu_time:.1f}x faster!")
    print("\n✅ RTX 5070 + CUDA 13 + PyTorch 2.9 = 動作確認OK!")
else:
    print("❌ CUDA is not available")
