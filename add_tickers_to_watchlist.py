"""
ウォッチリストに銘柄を一括追加するスクリプト
================================================
ロバスト性テストで高パフォーマンスだったセクターの銘柄を追加

使用方法:
  python add_tickers_to_watchlist.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from database.db_manager import DatabaseManager

# ========================================
# 銘柄リスト（約300銘柄）
# ========================================

TICKERS_TO_ADD = {
    # ==================== 日本株_大型50 ====================
    "7203.T": {"name": "トヨタ自動車", "sector": "自動車", "market": "東証プライム"},
    "6758.T": {"name": "ソニーグループ", "sector": "電機", "market": "東証プライム"},
    "9984.T": {"name": "ソフトバンクグループ", "sector": "通信", "market": "東証プライム"},
    "8306.T": {"name": "三菱UFJフィナンシャル・グループ", "sector": "銀行", "market": "東証プライム"},
    "6902.T": {"name": "デンソー", "sector": "自動車部品", "market": "東証プライム"},
    "7267.T": {"name": "本田技研工業", "sector": "自動車", "market": "東証プライム"},
    "6501.T": {"name": "日立製作所", "sector": "電機", "market": "東証プライム"},
    "8035.T": {"name": "東京エレクトロン", "sector": "半導体", "market": "東証プライム"},
    "6861.T": {"name": "キーエンス", "sector": "電機", "market": "東証プライム"},
    "9433.T": {"name": "KDDI", "sector": "通信", "market": "東証プライム"},
    "7011.T": {"name": "三菱重工業", "sector": "機械", "market": "東証プライム"},
    "7012.T": {"name": "川崎重工業", "sector": "機械", "market": "東証プライム"},
    "8316.T": {"name": "三井住友フィナンシャルグループ", "sector": "銀行", "market": "東証プライム"},
    "8053.T": {"name": "住友商事", "sector": "商社", "market": "東証プライム"},
    "6702.T": {"name": "富士通", "sector": "電機", "market": "東証プライム"},
    "9432.T": {"name": "日本電信電話(NTT)", "sector": "通信", "market": "東証プライム"},
    "9434.T": {"name": "ソフトバンク", "sector": "通信", "market": "東証プライム"},
    "4502.T": {"name": "武田薬品工業", "sector": "医薬品", "market": "東証プライム"},
    "4503.T": {"name": "アステラス製薬", "sector": "医薬品", "market": "東証プライム"},
    "4568.T": {"name": "第一三共", "sector": "医薬品", "market": "東証プライム"},
    "6098.T": {"name": "リクルートホールディングス", "sector": "サービス", "market": "東証プライム"},
    "4661.T": {"name": "オリエンタルランド", "sector": "サービス", "market": "東証プライム"},
    "8058.T": {"name": "三菱商事", "sector": "商社", "market": "東証プライム"},
    "8031.T": {"name": "三井物産", "sector": "商社", "market": "東証プライム"},
    "7974.T": {"name": "任天堂", "sector": "ゲーム", "market": "東証プライム"},
    "6273.T": {"name": "SMC", "sector": "機械", "market": "東証プライム"},
    "6723.T": {"name": "ルネサスエレクトロニクス", "sector": "半導体", "market": "東証プライム"},
    "6954.T": {"name": "ファナック", "sector": "機械", "market": "東証プライム"},
    "8001.T": {"name": "伊藤忠商事", "sector": "商社", "market": "東証プライム"},
    "8002.T": {"name": "丸紅", "sector": "商社", "market": "東証プライム"},
    "9983.T": {"name": "ファーストリテイリング", "sector": "小売", "market": "東証プライム"},
    "3382.T": {"name": "セブン&アイ・ホールディングス", "sector": "小売", "market": "東証プライム"},
    "4901.T": {"name": "富士フイルムホールディングス", "sector": "化学", "market": "東証プライム"},
    "4911.T": {"name": "資生堂", "sector": "化学", "market": "東証プライム"},
    "6857.T": {"name": "アドバンテスト", "sector": "半導体", "market": "東証プライム"},
    "6981.T": {"name": "村田製作所", "sector": "電子部品", "market": "東証プライム"},
    "7751.T": {"name": "キヤノン", "sector": "電機", "market": "東証プライム"},
    "4523.T": {"name": "エーザイ", "sector": "医薬品", "market": "東証プライム"},
    "6988.T": {"name": "日東電工", "sector": "化学", "market": "東証プライム"},
    "8801.T": {"name": "三井不動産", "sector": "不動産", "market": "東証プライム"},
    "8802.T": {"name": "三菱地所", "sector": "不動産", "market": "東証プライム"},
    "9020.T": {"name": "東日本旅客鉄道(JR東日本)", "sector": "鉄道", "market": "東証プライム"},
    "9021.T": {"name": "西日本旅客鉄道(JR西日本)", "sector": "鉄道", "market": "東証プライム"},
    "9022.T": {"name": "東海旅客鉄道(JR東海)", "sector": "鉄道", "market": "東証プライム"},
    "2914.T": {"name": "日本たばこ産業(JT)", "sector": "食品", "market": "東証プライム"},
    "5401.T": {"name": "日本製鉄", "sector": "鉄鋼", "market": "東証プライム"},
    "5411.T": {"name": "JFEホールディングス", "sector": "鉄鋼", "market": "東証プライム"},
    "3407.T": {"name": "旭化成", "sector": "化学", "market": "東証プライム"},
    "4452.T": {"name": "花王", "sector": "化学", "market": "東証プライム"},
    "6326.T": {"name": "クボタ", "sector": "機械", "market": "東証プライム"},
    
    # ==================== 日本株_テック30 ====================
    "6971.T": {"name": "京セラ", "sector": "電子部品", "market": "東証プライム"},
    "4063.T": {"name": "信越化学工業", "sector": "化学", "market": "東証プライム"},
    "6367.T": {"name": "ダイキン工業", "sector": "機械", "market": "東証プライム"},
    "6594.T": {"name": "日本電産", "sector": "電機", "market": "東証プライム"},
    "6762.T": {"name": "TDK", "sector": "電子部品", "market": "東証プライム"},
    "6752.T": {"name": "パナソニック", "sector": "電機", "market": "東証プライム"},
    "6920.T": {"name": "レーザーテック", "sector": "半導体", "market": "東証プライム"},
    "6146.T": {"name": "ディスコ", "sector": "半導体", "market": "東証プライム"},
    "7735.T": {"name": "SCREENホールディングス", "sector": "半導体", "market": "東証プライム"},
    "6963.T": {"name": "ローム", "sector": "半導体", "market": "東証プライム"},
    "4684.T": {"name": "オービック", "sector": "IT", "market": "東証プライム"},
    "4307.T": {"name": "野村総合研究所", "sector": "IT", "market": "東証プライム"},
    # 9613.T (NTTデータ) - 上場廃止のため削除
    "4751.T": {"name": "サイバーエージェント", "sector": "IT", "market": "東証プライム"},
    "4755.T": {"name": "楽天グループ", "sector": "IT", "market": "東証プライム"},
    "3659.T": {"name": "ネクソン", "sector": "ゲーム", "market": "東証プライム"},
    "2413.T": {"name": "エムスリー", "sector": "IT", "market": "東証プライム"},
    "4689.T": {"name": "LINEヤフー", "sector": "IT", "market": "東証プライム"},
    "9449.T": {"name": "GMOインターネットグループ", "sector": "IT", "market": "東証プライム"},
    "3769.T": {"name": "GMOペイメントゲートウェイ", "sector": "IT", "market": "東証プライム"},
    "4816.T": {"name": "東映アニメーション", "sector": "メディア", "market": "東証スタンダード"},
    "2371.T": {"name": "カカクコム", "sector": "IT", "market": "東証プライム"},
    
    # ==================== 日本株_金融20（最強セクター） ====================
    "8411.T": {"name": "みずほフィナンシャルグループ", "sector": "銀行", "market": "東証プライム"},
    "8308.T": {"name": "りそなホールディングス", "sector": "銀行", "market": "東証プライム"},
    "8309.T": {"name": "三井住友トラスト・ホールディングス", "sector": "銀行", "market": "東証プライム"},
    "8604.T": {"name": "野村ホールディングス", "sector": "証券", "market": "東証プライム"},
    "8601.T": {"name": "大和証券グループ本社", "sector": "証券", "market": "東証プライム"},
    "8630.T": {"name": "SOMPOホールディングス", "sector": "保険", "market": "東証プライム"},
    "8766.T": {"name": "東京海上ホールディングス", "sector": "保険", "market": "東証プライム"},
    "8750.T": {"name": "第一生命ホールディングス", "sector": "保険", "market": "東証プライム"},
    "8725.T": {"name": "MS&ADインシュアランスグループ", "sector": "保険", "market": "東証プライム"},
    "8795.T": {"name": "T&Dホールディングス", "sector": "保険", "market": "東証プライム"},
    "8591.T": {"name": "オリックス", "sector": "リース", "market": "東証プライム"},
    "8593.T": {"name": "三菱HCキャピタル", "sector": "リース", "market": "東証プライム"},
    "8697.T": {"name": "日本取引所グループ", "sector": "取引所", "market": "東証プライム"},
    "7182.T": {"name": "ゆうちょ銀行", "sector": "銀行", "market": "東証プライム"},
    "8354.T": {"name": "ふくおかフィナンシャルグループ", "sector": "銀行", "market": "東証プライム"},
    "8331.T": {"name": "千葉銀行", "sector": "銀行", "market": "東証プライム"},
    # 8355.T (静岡銀行) - Yahoo Financeでデータ取得不可のため削除
    "7186.T": {"name": "コンコルディア・フィナンシャルグループ", "sector": "銀行", "market": "東証プライム"},
    
    # ==================== 日本株_自動車15 ====================
    "7269.T": {"name": "スズキ", "sector": "自動車", "market": "東証プライム"},
    "7201.T": {"name": "日産自動車", "sector": "自動車", "market": "東証プライム"},
    "7211.T": {"name": "三菱自動車工業", "sector": "自動車", "market": "東証プライム"},
    "7270.T": {"name": "SUBARU", "sector": "自動車", "market": "東証プライム"},
    "7259.T": {"name": "アイシン", "sector": "自動車部品", "market": "東証プライム"},
    "7202.T": {"name": "いすゞ自動車", "sector": "自動車", "market": "東証プライム"},
    "5108.T": {"name": "ブリヂストン", "sector": "タイヤ", "market": "東証プライム"},
    "7282.T": {"name": "豊田自動織機", "sector": "機械", "market": "東証プライム"},
    "7240.T": {"name": "NOK", "sector": "自動車部品", "market": "東証プライム"},
    "7261.T": {"name": "マツダ", "sector": "自動車", "market": "東証プライム"},
    "5101.T": {"name": "横浜ゴム", "sector": "タイヤ", "market": "東証プライム"},
    
    # ==================== 日本株_消費財20 ====================
    "8267.T": {"name": "イオン", "sector": "小売", "market": "東証プライム"},
    "9843.T": {"name": "ニトリホールディングス", "sector": "小売", "market": "東証プライム"},
    "3099.T": {"name": "三越伊勢丹ホールディングス", "sector": "小売", "market": "東証プライム"},
    "2802.T": {"name": "味の素", "sector": "食品", "market": "東証プライム"},
    "2801.T": {"name": "キッコーマン", "sector": "食品", "market": "東証プライム"},
    "2269.T": {"name": "明治ホールディングス", "sector": "食品", "market": "東証プライム"},
    "2503.T": {"name": "キリンホールディングス", "sector": "飲料", "market": "東証プライム"},
    "2502.T": {"name": "アサヒグループホールディングス", "sector": "飲料", "market": "東証プライム"},
    "4922.T": {"name": "コーセー", "sector": "化粧品", "market": "東証プライム"},
    "4912.T": {"name": "ライオン", "sector": "化学", "market": "東証プライム"},
    "7453.T": {"name": "良品計画", "sector": "小売", "market": "東証プライム"},
    "3086.T": {"name": "J.フロント リテイリング", "sector": "小売", "market": "東証プライム"},
    "8252.T": {"name": "丸井グループ", "sector": "小売", "market": "東証プライム"},
    "2670.T": {"name": "ABCマート", "sector": "小売", "market": "東証プライム"},
    "7532.T": {"name": "パン・パシフィック・インターナショナル", "sector": "小売", "market": "東証プライム"},
    # 2651.T (ローソン) - 2024年7月上場廃止のため削除
    
    # ==================== 日本株_医薬15 ====================
    "4519.T": {"name": "中外製薬", "sector": "医薬品", "market": "東証プライム"},
    "4578.T": {"name": "大塚ホールディングス", "sector": "医薬品", "market": "東証プライム"},
    "4506.T": {"name": "住友ファーマ", "sector": "医薬品", "market": "東証プライム"},
    "4507.T": {"name": "塩野義製薬", "sector": "医薬品", "market": "東証プライム"},
    "4151.T": {"name": "協和キリン", "sector": "医薬品", "market": "東証プライム"},
    "4528.T": {"name": "小野薬品工業", "sector": "医薬品", "market": "東証プライム"},
    "7733.T": {"name": "オリンパス", "sector": "医療機器", "market": "東証プライム"},
    "4543.T": {"name": "テルモ", "sector": "医療機器", "market": "東証プライム"},
    "6869.T": {"name": "シスメックス", "sector": "医療機器", "market": "東証プライム"},
    "7747.T": {"name": "朝日インテック", "sector": "医療機器", "market": "東証プライム"},
    "4974.T": {"name": "タカラバイオ", "sector": "バイオ", "market": "東証プライム"},
    
    # ==================== 日本株_インフラ15 ====================
    "9001.T": {"name": "東武鉄道", "sector": "鉄道", "market": "東証プライム"},
    "9005.T": {"name": "東急", "sector": "鉄道", "market": "東証プライム"},
    "9007.T": {"name": "小田急電鉄", "sector": "鉄道", "market": "東証プライム"},
    "9008.T": {"name": "京王電鉄", "sector": "鉄道", "market": "東証プライム"},
    "9501.T": {"name": "東京電力ホールディングス", "sector": "電力", "market": "東証プライム"},
    "9502.T": {"name": "中部電力", "sector": "電力", "market": "東証プライム"},
    "9503.T": {"name": "関西電力", "sector": "電力", "market": "東証プライム"},
    "9531.T": {"name": "東京ガス", "sector": "ガス", "market": "東証プライム"},
    "9532.T": {"name": "大阪ガス", "sector": "ガス", "market": "東証プライム"},
    
    # ==================== 日本株_不動産建設15（高パフォーマンス） ====================
    "8830.T": {"name": "住友不動産", "sector": "不動産", "market": "東証プライム"},
    "3289.T": {"name": "東急不動産ホールディングス", "sector": "不動産", "market": "東証プライム"},
    "8804.T": {"name": "東京建物", "sector": "不動産", "market": "東証プライム"},
    "1925.T": {"name": "大和ハウス工業", "sector": "建設", "market": "東証プライム"},
    "1928.T": {"name": "積水ハウス", "sector": "建設", "market": "東証プライム"},
    "1802.T": {"name": "大林組", "sector": "建設", "market": "東証プライム"},
    "1803.T": {"name": "清水建設", "sector": "建設", "market": "東証プライム"},
    "1801.T": {"name": "大成建設", "sector": "建設", "market": "東証プライム"},
    "1812.T": {"name": "鹿島建設", "sector": "建設", "market": "東証プライム"},
    "1808.T": {"name": "長谷工コーポレーション", "sector": "建設", "market": "東証プライム"},
    "5232.T": {"name": "住友大阪セメント", "sector": "セメント", "market": "東証プライム"},
    "5233.T": {"name": "太平洋セメント", "sector": "セメント", "market": "東証プライム"},
    "1878.T": {"name": "大東建託", "sector": "建設", "market": "東証プライム"},
    
    # ==================== 日本株_素材15 ====================
    "5406.T": {"name": "神戸製鋼所", "sector": "鉄鋼", "market": "東証プライム"},
    "4188.T": {"name": "三菱ケミカルグループ", "sector": "化学", "market": "東証プライム"},
    "4005.T": {"name": "住友化学", "sector": "化学", "market": "東証プライム"},
    "4042.T": {"name": "東ソー", "sector": "化学", "market": "東証プライム"},
    "4183.T": {"name": "三井化学", "sector": "化学", "market": "東証プライム"},
    "4631.T": {"name": "DIC", "sector": "化学", "market": "東証プライム"},
    "4021.T": {"name": "日産化学", "sector": "化学", "market": "東証プライム"},
    "5713.T": {"name": "住友金属鉱山", "sector": "非鉄金属", "market": "東証プライム"},
    "5711.T": {"name": "三菱マテリアル", "sector": "非鉄金属", "market": "東証プライム"},
    "5706.T": {"name": "三井金属鉱業", "sector": "非鉄金属", "market": "東証プライム"},
    "5714.T": {"name": "DOWAホールディングス", "sector": "非鉄金属", "market": "東証プライム"},
    "5801.T": {"name": "古河電気工業", "sector": "非鉄金属", "market": "東証プライム"},
    
    # ==================== 米国株_大型30 ====================
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "market": "NASDAQ"},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "market": "NASDAQ"},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "market": "NASDAQ"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer", "market": "NASDAQ"},
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Semiconductor", "market": "NASDAQ"},
    "META": {"name": "Meta Platforms Inc.", "sector": "Technology", "market": "NASDAQ"},
    "TSLA": {"name": "Tesla Inc.", "sector": "Automotive", "market": "NASDAQ"},
    "BRK-B": {"name": "Berkshire Hathaway Inc.", "sector": "Financials", "market": "NYSE"},
    "UNH": {"name": "UnitedHealth Group Inc.", "sector": "Healthcare", "market": "NYSE"},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "market": "NYSE"},
    "V": {"name": "Visa Inc.", "sector": "Financials", "market": "NYSE"},
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financials", "market": "NYSE"},
    "XOM": {"name": "Exxon Mobil Corporation", "sector": "Energy", "market": "NYSE"},
    "PG": {"name": "Procter & Gamble Co.", "sector": "Consumer", "market": "NYSE"},
    "MA": {"name": "Mastercard Incorporated", "sector": "Financials", "market": "NYSE"},
    "HD": {"name": "The Home Depot Inc.", "sector": "Consumer", "market": "NYSE"},
    "CVX": {"name": "Chevron Corporation", "sector": "Energy", "market": "NYSE"},
    "MRK": {"name": "Merck & Co. Inc.", "sector": "Healthcare", "market": "NYSE"},
    "ABBV": {"name": "AbbVie Inc.", "sector": "Healthcare", "market": "NYSE"},
    "PFE": {"name": "Pfizer Inc.", "sector": "Healthcare", "market": "NYSE"},
    "KO": {"name": "The Coca-Cola Company", "sector": "Consumer", "market": "NYSE"},
    "PEP": {"name": "PepsiCo Inc.", "sector": "Consumer", "market": "NASDAQ"},
    "COST": {"name": "Costco Wholesale Corporation", "sector": "Consumer", "market": "NASDAQ"},
    "TMO": {"name": "Thermo Fisher Scientific", "sector": "Healthcare", "market": "NYSE"},
    "AVGO": {"name": "Broadcom Inc.", "sector": "Semiconductor", "market": "NASDAQ"},
    "WMT": {"name": "Walmart Inc.", "sector": "Consumer", "market": "NYSE"},
    "MCD": {"name": "McDonald's Corporation", "sector": "Consumer", "market": "NYSE"},
    "CSCO": {"name": "Cisco Systems Inc.", "sector": "Technology", "market": "NASDAQ"},
    "ACN": {"name": "Accenture plc", "sector": "Technology", "market": "NYSE"},
    "DHR": {"name": "Danaher Corporation", "sector": "Healthcare", "market": "NYSE"},
    
    # ==================== 米国株_テック30 ====================
    "CRM": {"name": "Salesforce Inc.", "sector": "Technology", "market": "NYSE"},
    "ADBE": {"name": "Adobe Inc.", "sector": "Technology", "market": "NASDAQ"},
    "ORCL": {"name": "Oracle Corporation", "sector": "Technology", "market": "NYSE"},
    "INTC": {"name": "Intel Corporation", "sector": "Semiconductor", "market": "NASDAQ"},
    "AMD": {"name": "Advanced Micro Devices", "sector": "Semiconductor", "market": "NASDAQ"},
    "QCOM": {"name": "Qualcomm Inc.", "sector": "Semiconductor", "market": "NASDAQ"},
    "TXN": {"name": "Texas Instruments", "sector": "Semiconductor", "market": "NASDAQ"},
    "MU": {"name": "Micron Technology", "sector": "Semiconductor", "market": "NASDAQ"},
    "NOW": {"name": "ServiceNow Inc.", "sector": "Technology", "market": "NYSE"},
    "SNOW": {"name": "Snowflake Inc.", "sector": "Technology", "market": "NYSE"},
    "PANW": {"name": "Palo Alto Networks", "sector": "Cybersecurity", "market": "NASDAQ"},
    "CRWD": {"name": "CrowdStrike Holdings", "sector": "Cybersecurity", "market": "NASDAQ"},
    "ZS": {"name": "Zscaler Inc.", "sector": "Cybersecurity", "market": "NASDAQ"},
    "NET": {"name": "Cloudflare Inc.", "sector": "Technology", "market": "NYSE"},
    "DDOG": {"name": "Datadog Inc.", "sector": "Technology", "market": "NASDAQ"},
    "TEAM": {"name": "Atlassian Corporation", "sector": "Technology", "market": "NASDAQ"},
    "SHOP": {"name": "Shopify Inc.", "sector": "Technology", "market": "NYSE"},
    "XYZ": {"name": "Block Inc. (旧SQ)", "sector": "Fintech", "market": "NYSE"},
    "UBER": {"name": "Uber Technologies", "sector": "Technology", "market": "NYSE"},
    "ABNB": {"name": "Airbnb Inc.", "sector": "Technology", "market": "NASDAQ"},
    "DASH": {"name": "DoorDash Inc.", "sector": "Technology", "market": "NASDAQ"},
    "COIN": {"name": "Coinbase Global", "sector": "Fintech", "market": "NASDAQ"},
    "RBLX": {"name": "Roblox Corporation", "sector": "Gaming", "market": "NYSE"},
    "PLTR": {"name": "Palantir Technologies", "sector": "Technology", "market": "NYSE"},
    "U": {"name": "Unity Software", "sector": "Gaming", "market": "NYSE"},
    "MRVL": {"name": "Marvell Technology", "sector": "Semiconductor", "market": "NASDAQ"},
    "AMAT": {"name": "Applied Materials", "sector": "Semiconductor", "market": "NASDAQ"},
    "LRCX": {"name": "Lam Research", "sector": "Semiconductor", "market": "NASDAQ"},
    
    # ==================== 米国株_金融20 ====================
    "BAC": {"name": "Bank of America Corp.", "sector": "Financials", "market": "NYSE"},
    "WFC": {"name": "Wells Fargo & Company", "sector": "Financials", "market": "NYSE"},
    "C": {"name": "Citigroup Inc.", "sector": "Financials", "market": "NYSE"},
    "GS": {"name": "Goldman Sachs Group", "sector": "Financials", "market": "NYSE"},
    "MS": {"name": "Morgan Stanley", "sector": "Financials", "market": "NYSE"},
    "SCHW": {"name": "Charles Schwab Corp.", "sector": "Financials", "market": "NYSE"},
    "BLK": {"name": "BlackRock Inc.", "sector": "Financials", "market": "NYSE"},
    "AXP": {"name": "American Express Co.", "sector": "Financials", "market": "NYSE"},
    "COF": {"name": "Capital One Financial", "sector": "Financials", "market": "NYSE"},
    "PYPL": {"name": "PayPal Holdings", "sector": "Fintech", "market": "NASDAQ"},
    "ADP": {"name": "Automatic Data Processing", "sector": "Technology", "market": "NASDAQ"},
    "FIS": {"name": "Fidelity National Info", "sector": "Fintech", "market": "NYSE"},
    "MET": {"name": "MetLife Inc.", "sector": "Insurance", "market": "NYSE"},
    "PRU": {"name": "Prudential Financial", "sector": "Insurance", "market": "NYSE"},
    "ALL": {"name": "The Allstate Corporation", "sector": "Insurance", "market": "NYSE"},
    "TRV": {"name": "The Travelers Companies", "sector": "Insurance", "market": "NYSE"},
    "AFL": {"name": "Aflac Incorporated", "sector": "Insurance", "market": "NYSE"},
    
    # ==================== 米国株_工業20 ====================
    "CAT": {"name": "Caterpillar Inc.", "sector": "Industrial", "market": "NYSE"},
    "DE": {"name": "Deere & Company", "sector": "Industrial", "market": "NYSE"},
    "HON": {"name": "Honeywell International", "sector": "Industrial", "market": "NASDAQ"},
    "UNP": {"name": "Union Pacific Corp.", "sector": "Transportation", "market": "NYSE"},
    "UPS": {"name": "United Parcel Service", "sector": "Transportation", "market": "NYSE"},
    "BA": {"name": "The Boeing Company", "sector": "Aerospace", "market": "NYSE"},
    "LMT": {"name": "Lockheed Martin Corp.", "sector": "Defense", "market": "NYSE"},
    "RTX": {"name": "RTX Corporation", "sector": "Defense", "market": "NYSE"},
    "GD": {"name": "General Dynamics Corp.", "sector": "Defense", "market": "NYSE"},
    "NOC": {"name": "Northrop Grumman Corp.", "sector": "Defense", "market": "NYSE"},
    "GE": {"name": "General Electric Company", "sector": "Industrial", "market": "NYSE"},
    "MMM": {"name": "3M Company", "sector": "Industrial", "market": "NYSE"},
    "EMR": {"name": "Emerson Electric Co.", "sector": "Industrial", "market": "NYSE"},
    "ITW": {"name": "Illinois Tool Works", "sector": "Industrial", "market": "NYSE"},
    "ETN": {"name": "Eaton Corporation", "sector": "Industrial", "market": "NYSE"},
    "FDX": {"name": "FedEx Corporation", "sector": "Transportation", "market": "NYSE"},
    "CSX": {"name": "CSX Corporation", "sector": "Transportation", "market": "NASDAQ"},
    "NSC": {"name": "Norfolk Southern Corp.", "sector": "Transportation", "market": "NYSE"},
    "DAL": {"name": "Delta Air Lines Inc.", "sector": "Airlines", "market": "NYSE"},
    "UAL": {"name": "United Airlines Holdings", "sector": "Airlines", "market": "NASDAQ"},
    
    # ==================== 米国株_エネルギー15 ====================
    "COP": {"name": "ConocoPhillips", "sector": "Energy", "market": "NYSE"},
    "SLB": {"name": "Schlumberger Limited", "sector": "Energy", "market": "NYSE"},
    "EOG": {"name": "EOG Resources Inc.", "sector": "Energy", "market": "NYSE"},
    "MPC": {"name": "Marathon Petroleum Corp.", "sector": "Energy", "market": "NYSE"},
    "VLO": {"name": "Valero Energy Corp.", "sector": "Energy", "market": "NYSE"},
    "PSX": {"name": "Phillips 66", "sector": "Energy", "market": "NYSE"},
    "OXY": {"name": "Occidental Petroleum", "sector": "Energy", "market": "NYSE"},
    "KMI": {"name": "Kinder Morgan Inc.", "sector": "Energy", "market": "NYSE"},
    "ENPH": {"name": "Enphase Energy Inc.", "sector": "Renewables", "market": "NASDAQ"},
    "SEDG": {"name": "SolarEdge Technologies", "sector": "Renewables", "market": "NASDAQ"},
    "FSLR": {"name": "First Solar Inc.", "sector": "Renewables", "market": "NASDAQ"},
    "NEE": {"name": "NextEra Energy Inc.", "sector": "Utilities", "market": "NYSE"},
    "DUK": {"name": "Duke Energy Corporation", "sector": "Utilities", "market": "NYSE"},
    
    # ==================== 欧州株_主力20 ====================
    "ASML": {"name": "ASML Holding N.V.", "sector": "Semiconductor", "market": "NASDAQ"},
    "NVO": {"name": "Novo Nordisk A/S", "sector": "Healthcare", "market": "NYSE"},
    "SAP": {"name": "SAP SE", "sector": "Technology", "market": "NYSE"},
    "TM": {"name": "Toyota Motor Corp. (ADR)", "sector": "Automotive", "market": "NYSE"},
    "SHEL": {"name": "Shell plc", "sector": "Energy", "market": "NYSE"},
    "AZN": {"name": "AstraZeneca PLC", "sector": "Healthcare", "market": "NASDAQ"},
    "NVS": {"name": "Novartis AG", "sector": "Healthcare", "market": "NYSE"},
    "HSBC": {"name": "HSBC Holdings plc", "sector": "Financials", "market": "NYSE"},
    "UL": {"name": "Unilever PLC", "sector": "Consumer", "market": "NYSE"},
    "BP": {"name": "BP p.l.c.", "sector": "Energy", "market": "NYSE"},
    "SNY": {"name": "Sanofi", "sector": "Healthcare", "market": "NASDAQ"},
    "GSK": {"name": "GSK plc", "sector": "Healthcare", "market": "NYSE"},
    "DEO": {"name": "Diageo plc", "sector": "Consumer", "market": "NYSE"},
    "BUD": {"name": "Anheuser-Busch InBev", "sector": "Consumer", "market": "NYSE"},
    "RIO": {"name": "Rio Tinto Group", "sector": "Mining", "market": "NYSE"},
    "BHP": {"name": "BHP Group Limited", "sector": "Mining", "market": "NYSE"},
    "VALE": {"name": "Vale S.A.", "sector": "Mining", "market": "NYSE"},
    "LYG": {"name": "Lloyds Banking Group", "sector": "Financials", "market": "NYSE"},
    "BCS": {"name": "Barclays PLC", "sector": "Financials", "market": "NYSE"},
    "BTI": {"name": "British American Tobacco", "sector": "Consumer", "market": "NYSE"},
}


def main():
    db = DatabaseManager()
    
    print("=" * 60)
    print("📊 ウォッチリストへの銘柄一括追加")
    print("=" * 60)
    
    # 現在の銘柄数を確認
    current_watchlist = db.get_watchlist()
    current_count = len(current_watchlist)
    current_tickers = {w['ticker'] for w in current_watchlist}
    
    print(f"📌 現在の登録銘柄数: {current_count}")
    
    # 新規追加する銘柄をカウント
    new_tickers = {ticker for ticker in TICKERS_TO_ADD.keys() if ticker not in current_tickers}
    print(f"➕ 新規追加予定: {len(new_tickers)}銘柄")
    print(f"⏭️ 既存スキップ: {len(TICKERS_TO_ADD) - len(new_tickers)}銘柄")
    
    # 追加実行
    added_count = 0
    updated_count = 0
    
    for ticker, info in TICKERS_TO_ADD.items():
        result = db.add_to_watchlist(
            ticker=ticker,
            name=info.get("name"),
            sector=info.get("sector"),
            market=info.get("market")
        )
        if result:
            if ticker in new_tickers:
                added_count += 1
            else:
                updated_count += 1
    
    # 結果表示
    final_watchlist = db.get_watchlist()
    final_count = len(final_watchlist)
    
    print()
    print("-" * 60)
    print("✅ 完了!")
    print(f"   新規追加: {added_count}銘柄")
    print(f"   情報更新: {updated_count}銘柄")
    print(f"   最終登録数: {final_count}銘柄")
    print("=" * 60)
    
    # セクター別統計
    sectors = {}
    for w in final_watchlist:
        sector = w.get('sector') or 'その他'
        sectors[sector] = sectors.get(sector, 0) + 1
    
    print("\n📊 セクター別銘柄数:")
    for sector, count in sorted(sectors.items(), key=lambda x: -x[1]):
        print(f"   {sector}: {count}銘柄")


if __name__ == "__main__":
    main()
