"""問題銘柄の確認"""
import yfinance as yf

# 問題銘柄と代替シンボル候補
check_tickers = [
    # 問題の銘柄
    ('2651.T', 'ローソン - 2024年7月上場廃止'),
    ('8355.T', '静岡銀行 - シンボル確認'),
    ('9613.T', 'NTTデータ - 旧シンボル'),
    ('SQ', 'Block - 旧シンボル'),
    # 代替候補
    ('XYZ', 'Block - 新シンボル'),
    ('9747.T', 'NTTデータグループ - 新シンボル'),
    # 正常な銘柄（比較用）
    ('7203.T', 'トヨタ'),
    ('AAPL', 'Apple'),
]

print("=" * 60)
print("銘柄シンボル確認")
print("=" * 60)

for ticker, desc in check_tickers:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        status = "✅" if len(hist) > 0 else "❌"
        print(f"{status} {ticker}: {len(hist)} rows - {desc}")
    except Exception as e:
        print(f"❌ {ticker}: ERROR - {desc}")
