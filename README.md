# 📈 株価予測ツール (Stock Predictor)

ARIMA + 機械学習 + LSTM深層学習 + テクニカル分析 を組み合わせた株価予測・分析ツールです。

## 🚀 機能

### 📊 株価予測
- **ARIMA**: 自己回帰和分移動平均モデル
- **Random Forest**: 機械学習アンサンブル
- **LightGBM**: 勾配ブースティング
- **GARCH**: ボラティリティ予測
- **LSTM**: 深層学習 (GPU対応)

### 📈 テクニカル分析
- 移動平均 (SMA)
- RSI (相対力指数)
- MACD
- ボリンジャーバンド
- 売買シグナル生成

### 💼 ポートフォリオ管理 (NEW!)
- 保有株の登録・管理
- リアルタイム損益計算
- 資産配分の可視化

### 🔔 価格アラート (NEW!)
- 目標価格でアラート設定
- 条件達成時の通知

### 📊 銘柄比較 (NEW!)
- 最大5銘柄の比較分析
- 相関係数ヒートマップ
- パフォーマンス比較

### ⚡ スマートキャッシュ
- 株価データの自動キャッシュ
- バックグラウンド更新
- API負荷軽減

## 🎮 GPU対応

PyTorchを使用したLSTMモデルでGPU高速予測に対応！

```bash
# CUDA 12.1 対応版
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 📦 インストール

```bash
# 仮想環境の作成（推奨）
python -m venv venv
venv\Scripts\activate  # Windows

# パッケージのインストール
pip install -r requirements.txt
```

または、`setup.bat` を実行してください。

## 🎮 使い方

### 方法1: バッチファイル（推奨）

`run.bat` をダブルクリック

### 方法2: コマンドライン

```bash
streamlit run app.py
```

## 📱 ページ構成

| ページ | 説明 |
|-------|------|
| 🏠 メイン | 株価予測・テクニカル分析 |
| 📊 精度評価 | 予測モデルの精度検証 |
| 📋 ウォッチリスト | 銘柄の管理 |
| 💾 データ管理 | キャッシュ管理 |
| 📈 銘柄比較 | 複数銘柄の比較分析 |
| 💼 ポートフォリオ | 保有株管理・損益計算 |
| 🔔 アラート | 価格通知設定 |

## 📊 対応銘柄

### 米国株
- `AAPL` - Apple
- `GOOGL` - Google
- `MSFT` - Microsoft
- `AMZN` - Amazon
- `TSLA` - Tesla
- `NVDA` - NVIDIA

### 日本株
- `7203.T` - トヨタ自動車
- `9984.T` - ソフトバンクグループ
- `6758.T` - ソニー
- `6861.T` - キーエンス
- `9432.T` - NTT
- `8306.T` - 三菱UFJ

## 📈 出力例

```
==========================================================
📊 株価分析レポート: AAPL
==========================================================

【基本情報】
  分析日: 2024-12-08 12:00
  最新株価: 195.50
  データ期間: 2022-12-08 ~ 2024-12-08

【テクニカルシグナル】
  🟢 RSI: 買い (RSI=28.5 売られすぎ)
  🟢 MA: 買い (上昇トレンド)
  🔴 MACD: 売り (デッドクロス)
  ⚪ BB: 中立 (バンド内)

【総合判断】 買い優勢 📈 (買い2 / 売り1)

【ARIMA予測】
  モデル: ARIMA(2, 1, 1)
  30日後予測: 201.23 (+2.93%)

【機械学習予測】
  R² Score: 0.9234
  30日後予測: 198.76 (+1.67%)
==========================================================
```

## ⚠️ 注意事項

- **投資判断は自己責任** で行ってください
- この予測は **参考情報** であり、将来の株価を保証するものではありません
- 実際の投資では、ファンダメンタル分析や市場環境も考慮してください

## 🛠️ ファイル構成

```
stock_predictor/
├── app.py                    # メインWebアプリ
├── stock_predictor.py        # コマンドライン版
├── run.bat                   # 実行スクリプト
├── setup.bat                 # セットアップスクリプト
├── requirements.txt          # 依存パッケージ
├── models/
│   ├── __init__.py
│   └── lstm_model.py         # LSTM深層学習モデル
├── database/
│   ├── __init__.py
│   └── db_manager.py         # SQLiteデータベース管理
├── services/
│   ├── __init__.py
│   └── background_updater.py # バックグラウンド更新
├── pages/
│   ├── 01_accuracy.py        # 予測精度検証
│   ├── 02_watchlist.py       # ウォッチリスト管理
│   ├── 03_data_management.py # データ管理
│   ├── 04_compare.py         # 銘柄比較
│   ├── 05_portfolio.py       # ポートフォリオ
│   └── 06_alerts.py          # 価格アラート
├── data/
│   └── stock_predictor.db    # SQLiteデータベース
└── README.md
```

## 📚 使用技術

- **Python 3.8+**
- **yfinance**: Yahoo Finance APIラッパー
- **statsmodels**: ARIMA時系列モデル
- **scikit-learn**: Random Forest機械学習
- **LightGBM**: 勾配ブースティング
- **arch**: GARCHボラティリティモデル
- **PyTorch**: LSTM深層学習 (GPU対応)
- **Streamlit**: Webアプリフレームワーク
- **Plotly**: インタラクティブチャート
- **SQLite**: ローカルデータベース
