# 日本株1時間足自動売買システム高度化の完全ガイド

勝率84%・PF5.0という成績は**過学習の強い兆候**であり、即座にOut-of-Sample検証とWalk-Forward Optimizationで検証すべきです。日本株特有の値幅制限・昼休みギャップを正確にシミュレートしないバックテストは信頼性が低く、実運用では**auカブコム証券API**が個人投資家向け唯一のREST APIとして最も実用的です。本レポートでは、過学習対策、AIモデル選択、実運用移行の3軸で、2024-2025年時点の最新ベストプラクティスを解説します。

---

## 過学習リスクの検出と対策

### 高成績が示す危険信号

勝率80%超、PF5.0超という成績は、実運用では達成困難な数値です。ある自己勘定取引会社が95%の勝率モデルを開発したが、実運用では**70%が損失トレード**となり会社が閉鎖された事例もあります。以下の検証を必ず実施してください。

**Deflated Sharpe Ratio（DSR）による補正**は、複数戦略テスト時の選択バイアスを考慮します。Bailey & López de Pradoの研究に基づき、試行回数が増えるほど「偶然良い結果」が出やすくなることを補正します。

```python
import numpy as np
from scipy.stats import norm

def compute_deflated_sharpe_ratio(estimated_sharpe, nb_trials, backtest_horizon, skew, kurtosis):
    """Deflated Sharpe Ratio計算（Bailey & López de Prado, 2014）"""
    gamma = 0.5772  # Euler-Mascheroni定数
    sharpe_variance = 0.5 / 252
    
    SR0 = np.sqrt(sharpe_variance) * (
        (1 - gamma) * norm.ppf(1 - 1/nb_trials) + 
        gamma * norm.ppf(1 - 1/(nb_trials * np.e))
    )
    
    numerator = (estimated_sharpe - SR0) * np.sqrt(backtest_horizon - 1)
    denominator = np.sqrt(1 - skew * estimated_sharpe + ((kurtosis - 1) / 4) * estimated_sharpe**2)
    
    return norm.cdf(numerator / denominator)

# 例：100戦略テスト、年率SR2.5、5年間
dsr = compute_deflated_sharpe_ratio(2.5/np.sqrt(252), 100, 1250, -0.5, 5)
print(f"DSR: {dsr:.2%}")  # 95%未満なら統計的に有意でない
```

### Walk-Forward Optimizationの実装

WFOは実運用環境を最もよく再現する検証手法です。**vectorbt**ライブラリを使用した実装例：

```python
import vectorbt as vbt

# ローリング分割（30分割、各2年ウィンドウ、180日テスト）
(in_price, in_indexes), (out_price, out_indexes) = price.vbt.rolling_split(
    n=30, window_len=365*2, set_lens=(180,), left_to_right=False
)

def simulate_all_params(price, windows):
    fast_ma, slow_ma = vbt.MA.run_combs(price, windows, r=2, short_names=["fast", "slow"])
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(price, entries, exits, direction="both", freq="h")
    return pf.sharpe_ratio()

# In-sample最適化 → Out-of-sample検証
in_sharpe = simulate_all_params(in_price, windows)
best_params = in_sharpe.idxmax()
out_sharpe = simulate_all_params(out_price, [best_params])
```

### Combinatorial Purged Cross-Validation

金融時系列特有の**情報漏洩を防ぐ**クロスバリデーション手法です。López de Prado（2018）が提唱し、**skfolio**や**mlfinlab**で実装されています。

```python
from skfolio.model_selection import CombinatorialPurgedCV

cv = CombinatorialPurgedCV(
    n_folds=6,           # 6グループに分割
    n_test_folds=2,      # 2グループをテストに使用
    purged_size=10,      # パージ期間（ラベル形成期間）
    embargo_size=5       # エンバーゴ期間（テスト後の隔離）
)
```

**推奨Embargo期間**：1時間足では**2-4時間**、一般的にはテストセットの1-5%が適切です。

---

## 日本株バックテストの特殊事項

### ストップ高・ストップ安の正確なシミュレーション

2024年時点の東証値幅制限は株価帯により異なり、**ストップ配分**では個人投資家は約定できない可能性が高いです。

| 基準価格 | 制限値幅 | 最大騰落率 |
|----------|----------|------------|
| 100円未満 | 30円 | ~30% |
| 500～700円 | 100円 | ~14-20% |
| 1,000～1,500円 | 300円 | ~20-30% |

```python
def apply_price_limit(base_price, target_price):
    """値幅制限を考慮した約定可能価格を計算"""
    limit_table = {
        (0, 100): 30, (100, 200): 50, (200, 500): 80,
        (500, 700): 100, (700, 1000): 150, (1000, 1500): 300
    }
    for (low, high), limit in limit_table.items():
        if low <= base_price < high:
            return max(base_price - limit, min(base_price + limit, target_price))
    return target_price
```

### 昼休みギャップと取引時間変更

**2024年11月5日からの東証取引時間**：
- 前場：9:00～11:30
- 昼休み：11:30～12:30
- 後場：12:30～15:30（30分延長）
- クロージング・オークション：15:25～15:30

1時間足では11:00-12:30が**2.5時間の「足」**になる点に注意が必要です。昼休み中のニュースで後場寄付きにギャップが発生するため、前場終値→後場始値のギャップを特徴量として活用できます。

---

## AIモデルアーキテクチャの選択

### 勾配ブースティング系モデルの比較

金融時系列データでは**LightGBM**が総合的に最も推奨されます。2023年の研究でXGBoost、AdaBoost、CatBoostを上回る性能を示しました。

| モデル | 予測精度 | 計算速度 | メモリ効率 | 過学習耐性 |
|--------|----------|----------|------------|------------|
| Random Forest | 高い | 中程度 | 高消費 | **強い** |
| **LightGBM** | **最高** | **最速** | **最効率** | やや弱い |
| XGBoost | 高い | 速い | 中程度 | 中程度 |
| CatBoost | 高い | 中程度 | 中程度 | 中程度 |

```python
import lightgbm as lgb

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.01,       # 金融データでは低めに設定
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'lambda_l1': 0.1,            # L1正則化
    'lambda_l2': 0.1,            # L2正則化
    'min_child_samples': 20,     # 過学習防止
}
```

### 最新の深層学習モデル（2023-2025）

**Temporal Fusion Transformer（TFT）** は金融時系列で最も注目されるモデルです。LSTM + Self-Attention + Variable Selection Networksを組み合わせ、**どの特徴量が重要か定量化可能**な解釈性を持ちます。

```python
from darts.models import TFTModel

model = TFTModel(
    input_chunk_length=72,    # 過去72時間（3日分）
    output_chunk_length=24,   # 24時間先を予測
    hidden_size=64,
    lstm_layers=2,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=100
)
model.fit(train_series, past_covariates=past_covs)
```

**N-BEATS**はM4コンペティションで統計モデルを**11%上回り**、純粋な深層学習モデルながらトレンドと季節性に分解可能な解釈性を持ちます。**PatchTST**は時系列をパッチに分割してTransformerに入力する手法で、2024年の研究でS&P500等での優位性が実証されています。

### データ量に応じたモデル選択ガイドライン

| データ量 | 推奨モデル |
|----------|------------|
| <1,000サンプル | LightGBM, Random Forest |
| 1,000-10,000 | XGBoost, LSTM, GRU |
| 10,000-100,000 | TFT, N-BEATS, DeepAR |
| >100,000 | PatchTST, TimesNet |

---

## 特徴量エンジニアリングの実践

### テクニカル指標を超える有効な特徴量

**出来高プロファイル（Volume Profile）** は価格帯別の出来高分布を可視化し、POC（Point of Control）からの乖離率が有効な特徴量となります。

```python
from marketprofile import MarketProfile

mp = MarketProfile(df)
mp_slice = mp[start:end]
poc_price = mp_slice.poc_price        # 最も出来高が多い価格帯
val, vah = mp_slice.value_area()      # Value Area (70%出来高集中帯)
```

**日本市場特有の時間特徴量**として、SQ日（毎月第2金曜日）、五十日（ごとおび：5,10,15,20,25,30日）、配当落ち日などをフラグ化します。

```python
import jpholiday
import numpy as np

# サイクリックエンコーディング
df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

# 日本市場特有
df['is_gotobi'] = df.index.day.isin([5, 10, 15, 20, 25, 30])
df['is_before_holiday'] = df.index.map(lambda x: jpholiday.is_holiday(x + pd.Timedelta(days=1)))
df['is_sq_week'] = df.index.map(is_sq_week)  # 第2金曜日の週
```

### SHAP値による特徴量重要度分析

TreeExplainerはLightGBM/XGBoostに最適で、高速かつ高精度です。

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary Plot（全体の重要度）
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Dependence Plot（特徴量間の相互作用）
shap.dependence_plot("volume", shap_values, X_test, interaction_index="volatility")
```

### ラグ特徴量のベストプラクティス

1時間足での推奨ラグ期間：**1, 2, 3, 6, 12, 24, 48, 120時間**（短期～1週間）

```python
from feature_engine.timeseries.forecasting import LagFeatures

lag_transformer = LagFeatures(
    variables=['return', 'volume', 'volatility'],
    periods=[1, 2, 3, 6, 12, 24, 48, 120],
    fill_value=None
)
df_lagged = lag_transformer.fit_transform(df)
```

**Look-ahead Bias防止**のため、ローリング統計量は必ず`shift(1)`を適用してから計算します。

---

## 日本市場向けオルタナティブデータ

### J-Quants API：個人投資家の最有力選択肢

JPX（日本取引所グループ）公式のAPIで、**信頼性最高**のデータソースです。

| プラン | 月額 | 株価遅延 | 取得期間 |
|--------|------|----------|----------|
| Free | 無料 | 12週間 | 過去2年 |
| Light | 1,650円 | 当日 | 過去5年 |
| Standard | 3,300円 | 当日 | 過去10年 |
| Premium | 16,500円 | 当日 | 全期間 |

```python
import jquantsapi

cli = jquantsapi.Client(mail_address="xxx", password="xxx")
df_price = cli.get_prices_daily_quotes(code="7203")  # トヨタ日足
df_fin = cli.get_fins_statements(code="7203")         # 財務情報
```

**重要な制限**：J-Quantsは**日足データのみ**で、1時間足は提供されていません。1時間足は楽天証券RSS（2年分取得可能）またはInteractive Brokers APIで取得する必要があります。

### 適時開示（TDnet）のNLP活用

TDnet APIサービスは有料（基本料7万円/月～）ですが、**有報キャッチャーAPI**で無料取得も可能です。日本語センチメント分析には**GiNZA**（spaCy統合）と**BERT-Japanese**（cl-tohoku/bert-japanese）を組み合わせます。

```python
import spacy
nlp = spacy.load('ja_ginza_electra')

from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="daigo/bert-base-japanese-sentiment")
result = classifier("決算は増収増益で好調に推移")
```

### 法的リスクのあるデータソース

| サイト | スクレイピング | 理由 |
|--------|---------------|------|
| **Kabutan（株探）** | **×禁止** | 利用規約で明確禁止、削除要請実例あり |
| Yahoo!ファイナンス | ×禁止 | ヘルプで明記 |
| TDnet | △注意 | 公式APIあり、robots.txt禁止なし |

---

## 証券会社API比較と実運用構成

### auカブコム証券 kabuステーションAPI

国内証券で**唯一のREST API直接提供**。個人投資家が本格的なシステムトレードを行う最も現実的な選択肢です。

```python
import requests

# トークン取得
token_url = "http://localhost:18080/kabusapi/token"
token_response = requests.post(token_url, json={"APIPassword": api_password})
token = token_response.json()['Token']

# 時価情報取得
headers = {'X-API-KEY': token}
board = requests.get("http://localhost:18080/kabusapi/board/7203@1", headers=headers).json()

# 注文発注
order_payload = {
    "Password": "取引パスワード", "Symbol": "7203", "Exchange": 1,
    "Side": "2", "CashMargin": 1, "Qty": 100, "Price": 0, "FrontOrderType": 10
}
requests.post("http://localhost:18080/kabusapi/sendorder", json=order_payload, headers=headers)
```

**利用条件**：kabuステーション Professionalプラン以上（前月1回以上の取引、または預かり資産100万円以上で無料）

### 楽天証券 MarketSpeed II RSS

Excel経由で**1時間足2年分**のヒストリカルデータを取得可能。Python連携はxlwings/win32com経由の間接連携となります。

```python
import win32com.client
xl = win32com.client.GetObject(Class="Excel.Application")
xl.Cells(1, 1).Formula = '=RssChart(,7203,"60M",100)'  # 1時間足100本
```

### Interactive Brokers（グローバル対応・高機能）

日本株含む150以上の市場にアクセス可能で、**1時間足ヒストリカルデータ取得可能**。ペーパートレード口座で無料練習できます。

```python
from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

contract = Stock('7203', 'TSEJ', 'JPY')  # 東証
bars = ib.reqHistoricalData(
    contract, endDateTime='', durationStr='30 D',
    barSizeSetting='1 hour', whatToShow='TRADES', useRTH=True
)
```

### yfinanceの問題点

yfinanceは**本番運用には非推奨**です。JSONDecodeError、タイムゾーンエラー、一部銘柄での長期間一定値問題などが2025年2月時点でも報告されています。バックテスト用途でも検証が必要です。

---

## 実運用インフラストラクチャ

### 推奨システム構成

```
┌─────────────────────────────────────────────────────────────────┐
│  データ取得 → シグナル生成 → リスク管理 → 注文執行             │
│      ↓                                         ↓              │
│  時系列DB (InfluxDB)          ログ・モニタリング              │
│                        (Prometheus + Grafana + Slack)         │
└─────────────────────────────────────────────────────────────────┘
```

### スケジューリング（APScheduler）

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Tokyo'))

# 前場寄り付き前（8:55）
scheduler.add_job(fetch_market_data, CronTrigger(hour=8, minute=55, day_of_week='mon-fri'))

# 後場寄り付き前（12:25）
scheduler.add_job(afternoon_session_start, CronTrigger(hour=12, minute=25, day_of_week='mon-fri'))

# 大引け後処理（15:35）- 2024年11月以降
scheduler.add_job(end_of_day_processing, CronTrigger(hour=15, minute=35, day_of_week='mon-fri'))

scheduler.start()
```

### 本番環境の選択

| 環境 | 月額コスト | 推奨度 |
|------|------------|--------|
| さくらVPS 2GB | 1,738円 | ⭐⭐⭐⭐⭐（低レイテンシ） |
| ConoHa VPS 2GB | 1,848円 | ⭐⭐⭐⭐⭐ |
| AWS EC2 t3.small | 約2,500円 | ⭐⭐⭐⭐ |

日本株取引では**国内VPS**が東証への低レイテンシでコスパ良好です。

### キルスイッチ（緊急停止機能）

```python
class TradingKillSwitch:
    def __init__(self, max_daily_loss_pct=0.03, max_drawdown_pct=0.10):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.is_active = False
    
    def check_and_activate(self, daily_pnl, total_capital):
        if daily_pnl / total_capital < -self.max_daily_loss_pct:
            self.is_active = True
            send_slack_alert("🚨 KILL SWITCH: Daily loss limit exceeded", "error")
            self.close_all_positions()
            return True
        return False
```

---

## 日本株特有の執行課題

### 東証の呼値単位（2025年5月更新）

TOPIX500構成銘柄とその他で呼値単位が異なります。

```python
def round_to_tick(price: float, is_topix500: bool = False) -> float:
    if is_topix500:
        tick_table = [(1000, 0.1), (3000, 0.5), (5000, 1), (10000, 1), (30000, 5)]
    else:
        tick_table = [(3000, 1), (5000, 5), (10000, 10), (30000, 10)]
    
    for threshold, tick in tick_table:
        if price <= threshold:
            return round(price / tick) * tick
    return price
```

### PTS（私設取引システム）の活用

- **ジャパンネクストPTS**：デイタイム8:20-16:00、ナイトタイム17:00-翌6:00
- SBI証券、楽天証券、松井証券で利用可能
- 決算発表後の夜間取引で優位性を確保可能

### 注文タイミング戦略

1時間足戦略では、シグナル発生時刻に応じて**寄成（9:00）、ザラバ（取引時間中）、引成（15:30）**を使い分けます。前場終了時のシグナルは後場寄付きに執行、14:30以降のシグナルは大引けに執行するのが一般的です。

---

## 結論：推奨アクションプラン

**短期（1-2週間）**：
1. 勝率84%・PF5.0の戦略をWalk-Forward OptimizationとDeflated Sharpe Ratioで再検証
2. J-Quants API（無料プラン）でデータ基盤を構築
3. 日本株特有の値幅制限・昼休みギャップをバックテストに組み込み

**中期（1-3ヶ月）**：
1. auカブコム証券でkabuステーションAPI環境を構築
2. LightGBMからTFTへのモデル高度化を検討
3. 楽天証券RSSで1時間足ヒストリカルデータを蓄積

**長期（3-6ヶ月）**：
1. 国内VPS（さくら/ConoHa）で本番環境を構築
2. Prometheus + Grafana + Slackでモニタリング体制を確立
3. ペーパートレード3ヶ月以上の実環境検証後に実運用開始

統計的に有意でない高成績は**実運用で必ず劣化**します。PBO（Probability of Backtest Overfitting）50%未満、DSR95%以上を確認してから実運用に移行することを強く推奨します。