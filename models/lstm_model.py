"""
LSTM モデル - GPU対応の時系列予測
================================
PyTorchを使用したLSTMモデルでGPU高速予測を実現
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional


# デバイス設定（GPU利用可能ならGPU、なければCPU）
def get_device():
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[CPU] Mode")
    return device


class LSTMModel(nn.Module):
    """LSTM時系列予測モデル"""
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 1, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        # LSTM層
        lstm_out, _ = self.lstm(x)
        # 最後の時点の出力を使用
        out = self.fc(lstm_out[:, -1, :])
        return out


class StockLSTMPredictor:
    """株価予測用LSTMラッパークラス"""
    
    def __init__(self, sequence_length: int = 30, hidden_size: int = 64,
                 num_layers: int = 2, learning_rate: float = 0.001):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        self.device = get_device()
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_trained = False
    
    def prepare_data(self, data: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """データをLSTM用に整形"""
        # スケーリング
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i - self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # PyTorchテンソルに変換
        X = torch.FloatTensor(X).unsqueeze(-1)  # (batch, seq_len, 1)
        y = torch.FloatTensor(y).unsqueeze(-1)  # (batch, 1)
        
        return X.to(self.device), y.to(self.device)
    
    def train(self, data: np.ndarray, epochs: int = 100, 
              batch_size: int = 32, verbose: bool = True) -> list:
        """モデルを訓練"""
        X, y = self.prepare_data(data)
        
        # モデル初期化
        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        # データローダー
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        losses = []
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        self.is_trained = True
        return losses
    
    def predict(self, data: np.ndarray, forecast_days: int = 30) -> np.ndarray:
        """将来の株価を予測"""
        if not self.is_trained:
            raise ValueError("モデルが訓練されていません。先にtrain()を実行してください。")
        
        self.model.eval()
        
        # データをスケーリング
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        predictions = []
        current_seq = scaled_data[-self.sequence_length:].flatten().tolist()
        
        with torch.no_grad():
            for _ in range(forecast_days):
                # 入力を準備
                seq_tensor = torch.FloatTensor(current_seq[-self.sequence_length:])
                seq_tensor = seq_tensor.unsqueeze(0).unsqueeze(-1).to(self.device)
                
                # 予測
                pred = self.model(seq_tensor)
                pred_value = pred.cpu().numpy()[0, 0]
                
                predictions.append(pred_value)
                current_seq.append(pred_value)
        
        # スケールを戻す
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def backtest(self, data: np.ndarray, test_days: int = 30, 
                 train_epochs: int = 50) -> pd.DataFrame:
        """バックテストを実行"""
        results = []
        
        for i in range(test_days, 0, -1):
            train_data = data[:-i]
            actual = data[-i]
            
            if len(train_data) < self.sequence_length + 50:
                continue
            
            try:
                # 訓練（少ないエポックで高速化）
                self.train(train_data, epochs=train_epochs, verbose=False)
                
                # 1日先を予測
                pred = self.predict(train_data, forecast_days=1)[0]
                
                results.append({
                    'Index': len(data) - i,
                    'Actual': actual,
                    'Predicted': pred,
                    'Error': actual - pred,
                    'Error_Pct': (actual - pred) / actual * 100
                })
            except Exception as e:
                continue
        
        return pd.DataFrame(results)
    
    def save_model(self, path: str):
        """モデルを保存"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers
            }, path)
            print(f"[OK] Model saved: {path}")
    
    def load_model(self, path: str):
        """モデルを読み込み"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.scaler = checkpoint['scaler']
        
        self.model = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        print(f"[OK] Model loaded: {path}")


def check_gpu_availability():
    """GPU情報を表示"""
    print("=" * 50)
    print("GPU Information")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"[OK] CUDA Available: True")
        print(f"   Device Name: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Device Count: {torch.cuda.device_count()}")
        
        # メモリ情報
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   Total Memory: {total_memory:.2f} GB")
    else:
        print("[--] CUDA Available: False")
        print("   Using CPU instead")
    
    print("=" * 50)
