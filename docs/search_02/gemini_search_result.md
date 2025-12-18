# **高頻度取引およびAI駆動型アルゴリズム取引における次世代アーキテクチャ：市場微細構造の制約と統計的頑健性の確立に関する包括的提言**

## **1\. エグゼクティブ・サマリー：驚異的なパフォーマンス指標の批判的検証と次なるステップ**

貴殿が共有された「勝率84%、プロフィットファクター（PF）5.0」という成果は、表層的な数値だけを見れば、クオンツ業界における「聖杯」に近い水準です。機関投資家の世界では、PF 2.0を超えれば極めて優秀、3.0を超えればまず「バグ」か「過学習（Overfitting）」、あるいは「未来情報の漏洩（Look-ahead Bias）」を疑うのが通例です 1。したがって、本レポートでは、貴殿の成果を称賛しつつも、あえて「懐疑的な監査者」としての立場から、この数値をライブ環境で再現するために必要な技術的・構造的なハードルを徹底的に洗い出します。

シミュレーション環境と実市場（Live Trading）の間には、\*\*「実装のギャップ（Implementation Gap）」\*\*と呼ばれる深い溝が存在します。特に東京証券取引所（TSE）は、特別気配（Special Quote）や板寄せ（Itayose）、ストップ高・安（Price Limits）といった独自の市場微細構造（Microstructure）を持っており、これらを厳密にモデリングしないバックテストは、現実の収益性を著しく歪めます。

本提言書は、以下の4つの柱を中心に、貴殿のシステムを「リテール（個人）レベル」から「インスティテューショナル（機関）レベル」へと昇華させるためのロードマップを提示します。

1. **データインフラの刷新**：yfinance等のスクレイピングベースのデータソースからの脱却と、取引所公式データ（J-Quants）への移行。  
2. **市場微細構造の完全なシミュレーション**：板寄せ、特別気配、流動性枯渇を考慮した厳密な約定ロジックの実装。  
3. **AIモデルの高度化と検証**：時系列データの自己相関を排除した「Purged Cross-Validation」の導入と、Transformer系モデルへの進化。  
4. **リスク管理と執行戦略**：ケリー基準の危険性と、ボラティリティ・ターゲティングやCPPI（Constant Proportion Portfolio Insurance）によるダウンサイド防御。

## ---

**2\. データインフラストラクチャの整合性と信頼性**

定量的取引戦略の根幹はデータです。「Garbage In, Garbage Out（ゴミを入れればゴミが出る）」の原則は絶対であり、PF 5.0という異常値は、入力データの品質に起因する可能性があります。

### **2.1 オープンソースデータ（yfinance）の構造的欠陥**

現在使用されているPythonライブラリ yfinance は、プロトタイピングには有用ですが、日本の株式市場におけるアルゴリズム取引のバックボーンとしては致命的な脆弱性を抱えています。

#### **2.1.1 データの欠落と連続性の喪失**

複数の開発者コミュニティやIssue Trackerで報告されている通り、yfinance には日本株の日中足データ（Intraday Data）において、深刻な品質問題が存在します 2。具体的には、「昨日のデータ」が新しい取引日の開始とともに消失したり、特定の時間帯（例えば大引け前の5分間）のデータが欠落する現象が確認されています 4。  
これらの欠損値（Missing Values）は、移動平均線やボリンジャーバンドなどのテクニカル指標の計算に断絶をもたらします。バックテストエンジンが欠損を「直前の値で埋める（Forward Fill）」処理を行った場合、ボラティリティが過小評価され、リスクが見かけ上低くなるため、シャープレシオやPFが人工的に押し上げられる原因となります。

#### **2.1.2 修正株価（Adjusted Close）の計算ロジックと日本特有の事情**

米国株と異なり、日本株のコーポレートアクションは複雑です。株式分割（Stock Splits）や配当落ちだけでなく、ライツオファリングや第三者割当増資などが頻繁に行われます。yfinance の auto\_adjust パラメータは、これらのイベントを自動的に処理しようと試みますが、その調整ロジックは米国市場向けに最適化されており、日本市場の複雑な資本異動に対して不正確な調整を行うケースが報告されています 5。  
例えば、株式併合（Reverse Split）が行われた際、調整が適切になされないと、価格が突如として10倍に跳ね上がったかのように記録され、AIモデルがこれを「異常な収益機会」として学習してしまいます。これが「勝率84%」の正体である可能性を排除できません。正確なバックテストには、分割比率（Split Ratio）と配当額を正確に把握し、戦略に応じて「遡及修正（Back-adjusted）」または「未来調整（Forward-adjusted）」系列を自前で構築する必要があります。

#### **2.1.3 API制限と法的リスク**

yfinance は公式APIではなく、Yahoo Financeのウェブサイトをスクレイピングするラッパーに過ぎません。したがって、Yahoo側のUI変更やレートリミット（アクセス制限）の対象となりやすく、運用中に突如としてデータ取得が不能になる「RateLimitError」が発生するリスクが常につきまといます 7。また、スクレイピング行為自体が、サーバーへの過度な負荷をかける場合、業務妨害として法的リスクを招く可能性も否定できません 9。

### **2.2 推奨される移行先：J-Quants APIの導入**

貴殿のシステムをプロフェッショナルな水準に引き上げるための最優先事項は、**日本取引所グループ（JPX）が提供する公式データ配信サービス「J-Quants API」への移行**です 11。

#### **2.2.1 クオンツ分析におけるJ-Quantsの優位性**

J-Quantsは、取引所のマッチングエンジンから生成された「一次情報」を提供するため、データの信頼性が圧倒的に高いのが特徴です。

* **公式OHLCV:** yfinance のようなサードパーティによる加工が入っていない、真正な始値・高値・安値・終値・出来高データ。  
* **コーポレートアクション・マスター:** 株式分割や配当の権利落ち日、効力発生日、比率が正確に提供されるため、独自の修正株価系列を構築可能です。  
* **セクター別・投資主体別データ:** 個別の株価だけでなく、業種別の空売り比率 12 や、信用取引残高（週次） 12 など、アルファ（超過収益）の源泉となるオルタナティブデータが取得可能です。

#### **2.2.2 コストとアクセシビリティ**

J-Quantsは個人投資家向けに設計されており、機関投資家向けのBloombergyやRefinitiv端末（月額数十万円）と比較して、極めてリーズナブルです。

* **Freeプラン:** 過去2年分のヒストリカルデータ（12週間遅延）が無償で利用可能であり、初期検証には十分です 13。  
* **Premiumプラン:** 月額16,500円で、すべての日次データ・財務データ・オプションデータ等にアクセス可能です 12。これは、信頼性の低いデータで運用して損失を出すリスク（Cost of Bad Data）と比較すれば、無視できるコストです。

### **2.3 データパイプラインの構築戦略**

単にAPIを叩くのではなく、ローカル環境に堅牢なデータベースを構築することを推奨します。

| データソース | 推奨用途 | 長所 | 短所 |
| :---- | :---- | :---- | :---- |
| **J-Quants API** | **学習・バックテスト** | 公式データ、高品質、豊富な属性情報 | リアルタイム性には欠ける（日次更新が主） |
| **auカブコム (kabusapi)** | **リアルタイム執行・板情報** | Push配信、フル板情報、Python親和性高 | ヒストリカルデータの取得期間が短い |
| **Rakuten RSS** | 非推奨 | 無料 | Windows/Excel依存、Python連携が不安定 |
| **yfinance** | プロトタイプのみ | 手軽、無料 | 信頼性低、欠損、調整ミス、IPBANリスク |

**推奨アーキテクチャ:**

1. **データベース:** 時系列データに特化した **TimescaleDB** (PostgreSQL拡張) を採用。  
2. **ETL処理:** 毎夜、J-Quantsから日次データを取得し、異常値検知（Spike Filter）を通した上でDBに格納。  
   * *検知ロジック例:* Close が前日比±20%を超え、かつコーポレートアクションがない場合、アラートを発報。  
3. **ユニバース管理:** 上場廃止銘柄（Survivorship Bias対策）を含めたマスタ管理。yfinance では取得困難な「倒産した企業」のデータもJ-Quantsアーカイブには含まれており、生存バイアスを排除した真正なバックテストが可能になります。

## ---

**3\. 市場微細構造の現実とシミュレーションの乖離**

PF 5.0という数値は、シミュレーターが「摩擦のない市場（Frictionless Market）」を仮定している可能性を強く示唆しています。しかし、東京証券取引所は、世界でも特殊な「板寄せ方式」や「ストップ高・安」のメカニズムを持っており、これらは流動性を劇的に変化させます。

### **3.1 「板寄せ（Itayose）」と「ザラバ（Zaraba）」の区別**

多くの簡易シミュレーターは、すべての約定を「ザラバ（連続約定）」として処理しますが、寄付（Open）と大引け（Close）は「板寄せ」というオークション方式で行われます 14。

#### **3.1.1 始値（Open）の罠**

もし貴殿のモデルが「始値でエントリーする」戦略を採用している場合、シミュレーション上で「始値＝直前の気配値」と仮定するのは危険です。

* **メカニズム:** 9:00の寄付前には、売り注文と買い注文が蓄積され、需給が一致する単一の価格（合致点）で全注文が約定します 16。  
* **リスク:** 8:59分時点の気配値を見て「買い」と判断しても、9:00の板寄せで大口の成行売りが入れば、想定より遥かに低い価格で約定、あるいは「買い気配（Untraded Ask）」のまま値がつかない可能性があります。  
* **改善案:** バックテストロジックにおいて、Openでの約定は「直前の板情報（Full Depth）」に基づいた需給シミュレーションを行うか、スリッページ（Slippage）を保守的（例：ボラティリティの0.5倍等）に見積もる必要があります。

### **3.2 「特別気配（Tokubetsu Kehai）」による機会損失**

米国市場のLULD（Limit Up Limit Down）とは異なり、TSEは急激な価格変動を抑制するために「特別気配」を表示し、売買を一時停止させます 15。

#### **3.2.1 見えない壁**

例えば、現在値が1000円の株に対し、大量の買い注文が入り、即座に1050円で約定しそうな場合でも、TSEは「特別買い気配 1020円」などを表示し、数分間（現在は3分間隔で更新 17）取引を停止させます。

* **シミュレーションの落とし穴:** 簡易的なOHLCデータのみを使ったバックテストでは、安値1000円、高値1050円のローソク足の中で、1020円で指値注文を出していれば「約定した」と判定してしまいます。しかし、実際にはその間「特別気配」が出ており、**板上の注文は一切約定していない**（＝空振り）可能性があります。  
* **対策:** ティックデータ（Tick Data）または1分足データに含まれる「気配フラグ（Quote Status）」を監視し、Special Quote フラグが立っている間は、指値注文の約定判定を無効化（Embargo）するロジックを組み込む必要があります。

### **3.3 ストップ高・ストップ安（Nehaba）の流動性ブラックホール**

日本株には、前日の終値に基づいた値幅制限（Daily Price Limits）が存在します 18。

#### **3.3.1 張り付き（Locked Limit）**

株価がストップ高（Limit Up）に達すると、売り注文が枯渇し、比例配分（Proration）のみが行われる状態になります。

* **エントリーの不可能性:** 強いモメンタム戦略が「ストップ高で買いエントリー」するシグナルを出した場合、シミュレーターは約定させがちですが、現実には数百万株の買い注文の最後尾に並ぶことになり、約定確率はほぼゼロです 15。  
* **イグジットの不可能性:** 逆に、保有株が悪材料でストップ安（Limit Down）に張り付いた場合、損切り（Stop Loss）注文を出しても約定せず、翌日まで持ち越すことになります。これは致命的なテールリスクですが、単純なバックテストでは「ストップロス価格で損切りできた」として損失を限定してしまいがちです。

**実装すべき制約ロジック:**

Python

def check\_execution\_limit(order, daily\_high, daily\_low, limit\_up\_price, limit\_down\_price):  
    if order.side \== 'BUY' and daily\_high \== limit\_up\_price and daily\_low \== limit\_up\_price:  
        \# 全日ストップ高張り付きの場合  
        return False \# 約定不可  
    if order.side \== 'SELL' and daily\_low \== limit\_down\_price and daily\_high \== limit\_down\_price:  
        \# 全日ストップ安張り付きの場合  
        return False \# 約定不可（損切り不能）  
    return True

このような「流動性制約」をコードに組み込むだけで、PF 5.0はより現実的な数値（おそらく2.0以下）に収束するでしょう。

## ---

**4\. AIモデルの妥当性検証：過学習とリークの排除**

「勝率84%」は、金融市場の効率性を考慮すると、統計的に異常な数値です。これはモデルの優秀さではなく、\*\*「未来情報の漏洩（Data Leakage）」\*\*を示唆しています。

### **4.1 時系列交差検証（Cross-Validation）の落とし穴**

通常のK-Fold交差検証（データをランダムに分割）は、時系列データに対しては無効です。なぜなら、金融データには強い自己相関（Serial Correlation）があり、ランダム分割では「未来のデータ」を使って「過去」を予測することになるからです 20。  
また、単純な時系列分割（Time Series Split：前半を学習、後半をテスト）であっても、学習データとテストデータの境界部分でリークが発生します。

#### **4.1.1 推奨手法：Combinatorial Purged Cross-Validation (CPCV)**

ファイナンシャル機械学習の権威であるMarcos Lopez de Prado氏が提唱する **Purged K-Fold CV** を導入してください 22。

1. **Embargoing（禁止期間）:** テストデータの直後の学習データを使用しないようにします。例えば、予測ラベルが「5日後のリターン」である場合、テスト期間終了後の5日間以上のデータを学習セットから削除（Embargo）し、相関の減衰を待ちます 24。  
2. **Purging（パージ）:** 学習データの中で、そのラベル計算期間がテスト期間と重複するものを削除します。これにより、テスト期間の正解ラベルに含まれる情報が学習データに混入することを完全に防ぎます。

この厳格な検証を行った場合、勝率は55%〜60%程度まで低下すると予想されますが、それこそが「真の実力」です。

### **4.2 Look-ahead Bias（先読みバイアス）の点検**

以下の点がコードに含まれていないか、至急確認してください 1。

* **マクロ指標の参照日:** GDPやインフレ率などの経済指標を「対象月」の日付で使っていませんか？実際には発表まで1〜2ヶ月のラグ（Publication Lag）があります。  
* **引け値の利用:** 日中のトレード判断を行う特徴量（Feature）に、その日の「終値（Close）」や、終値を使って計算されるテクニカル指標が含まれていませんか？  
* **スケーリング:** MinMaxScaler や StandardScaler をデータセット全体（全期間）に対して適用していませんか？スケーリングは必ず「その時点までに得られたデータ」のみを使って、ローリングウィンドウまたはExpandingウィンドウで行う必要があります。

## ---

**5\. 次世代アルファの創出：AIモデルの高度化と特徴量エンジニアリング**

検証の厳格化によってパフォーマンスが低下した後、それを再び向上させるのが「真のアルファ」の追求です。LSTMのような古典的なRNNから、より現代的なアーキテクチャへの移行を推奨します。

### **5.1 モデルアーキテクチャ：LightGBMとTransformerの適材適所**

#### **5.1.1 構造化データには決定木（Gradient Boosting）**

株価、出来高、テクニカル指標といった「表形式データ（Tabular Data）」に対しては、深層学習（Deep Learning）よりも **LightGBM** や **XGBoost** のような勾配ブースティング決定木（GBDT）の方が、ノイズへの耐性が高く、かつ学習が高速であるという研究結果が多数存在します 25。

* **改善案:** 目的関数（Objective Function）のカスタマイズ。単なる二乗誤差（MSE）の最小化ではなく、トレーディングに特化した損失関数、例えば「シャープレシオの負値を最小化する関数」などを定義し、モデルに直接「リスク調整後リターンの最大化」を学習させることが可能です。

#### **5.1.2 時系列パターンにはTemporal Fusion Transformer (TFT)**

もし深層学習を用いるなら、LSTMではなく **Temporal Fusion Transformer (TFT)** を検討してください。

* **優位性:** TFTはAttentionメカニズムを用いて、「どの時点の」「どの特徴量が」予測に寄与したかを可視化（Interpretability）できます。ブラックボックスになりがちなAIモデルにおいて、モデルが「なぜその判断をしたか」を解釈できることは、運用停止の判断（Drawdown時のデバッグ）において極めて重要です。

### **5.2 日本株特有の「アルファ」：オルタナティブデータの活用**

一般的なテクニカル指標（RSI, MACD）は市場参加者の大半が監視しており、エッジが消失しています。日本市場特有のデータを特徴量として組み込むことで、差別化を図ります。

#### **5.2.1 信用残高（Margin Balance）の変化**

日本では個人投資家の信用取引が活発であり、その残高（買い残・売り残）は需給の偏りを示す先行指標となります 12。

* **戦略:** 「信用買い残（Margin Buying Position）」が歴史的高水準にある銘柄は、将来の売り圧力（戻り売り）が強いため、上値が重くなります。J-Quantsから取得できる週次の信用残高データを日次補間し、特徴量として投入します。  
* **逆日歩（Reverse Repo/Premium）:** 貸借倍率が1倍を割り込み、売り長（Short heavy）になった銘柄は、踏み上げ（Short Squeeze）のリスクと同時にチャンスを孕んでいます。

#### **5.2.2 セクター別空売り比率**

JPXは業種別の空売り集計データを公表しています 28。

* **活用法:** 特定のセクターに対して空売り比率が急上昇した後の「巻き戻し（Covering）」を狙うセクターローテーションモデルの構築が有効です。

#### **5.2.3 日経VI（Volatility Index）のリード・ラグ効果**

日経平均VI（Volatility Index Japan）と日経平均株価、および米国のVIX指数の間には、複雑なリード・ラグ（先行・遅行）関係が存在します 30。

* **アイデア:** 米国市場のVIX先物の動きが、翌日の日本市場のボラティリティ、ひいては価格形成に先行する傾向があります 32。これをラグ特徴量としてモデルに組み込むことで、夜間の海外市場の影響を予測に反映させることができます。

### **5.3 日本語自然言語処理（NLP）の壁と突破口**

英語圏のFinBERT等は、日本語の金融テキスト（決算短信やニュース）には適用できません。

#### **5.3.1 日本語金融特化型BERTモデル**

izumi-lab/bert-small-japanese-fin や bardsai/finance-sentiment-ja-base といった、日本の有価証券報告書や経済ニュースで事前学習されたモデルを利用してください 33。

* **データソース:** TDnet（適時開示情報）の見出しやサマリー。  
* **実装:** ニューステキストを形態素解析（MeCab等）し、FinBERTに入力して「Positive/Negative/Neutral」のセンチメントスコアを算出。これを数値特徴量としてLightGBM等の入力に結合します。これにより、「好決算だが株価が下がる（織り込み済み）」といったパターンを、センチメントと価格アクションの乖離から学習できる可能性があります。

## ---

**6\. リスク管理と執行インフラの最適化**

### **6.1 ポジションサイジングの数理：ケリー基準を超えて**

「勝率84%」を信じてフルレバレッジを掛ければ、一度のブラックスワン・イベントで破産します。

#### **6.1.1 ケリー基準（Kelly Criterion）の危険性**

ケリー基準は「確率とペイオフレシオが既知かつ正確」であることを前提としています 35。市場の確率は不確実であるため、フル・ケリー（Full Kelly）を適用するとボラティリティが許容不能なレベルに達します。

* **推奨:** **ハーフ・ケリー（Half Kelly）** あるいはそれ以下の保守的な配分を採用すること。

#### **6.1.2 ボラティリティ・ターゲティング**

ポートフォリオ全体の目標ボラティリティ（例：年率15%）を設定し、直近の市場ボラティリティに応じてポジションサイズを動的に調整する手法です 36。

$$Position\\\_Size\_t \= \\frac{Target\\\_Vol}{Realized\\\_Vol\_t} \\times Capital$$

これにより、市場が荒れている局面（例：コロナショック時）では自動的にポジションを縮小し、生存確率を高めることができます。

#### **6.1.3 CPPI (Constant Proportion Portfolio Insurance)**

AIモデルの不確実性に備え、**CPPI** 戦略の導入を強く推奨します 38。

* **ロジック:** 「フロア（最低保証額）」を設定し、資産額とフロアの差額（クッション）に一定の乗数（Multiplier）を掛けた金額のみをリスク資産（AI戦略）に配分します。  
* **効果:** 資産がフロアに近づくと、自動的にリスク資産への配分がゼロになり、強制的にキャッシュポジションへ移行します。これは、AIモデルが「構造変化（Regime Shift）」に対応できずに損失を出し続けた際の安全装置（キルスイッチ）として機能します。

### **6.2 執行APIとシステム選定：ハイブリッド戦略**

日本市場において、単一のAPIですべてを賄うことは困難です。以下のハイブリッド構成が最適解となります。

| 機能要件 | J-Quants (JPX) | auカブコム (kabusapi) | 楽天証券 (MarketSpeed II/RSS) | 推奨選択 |
| :---- | :---- | :---- | :---- | :---- |
| **ヒストリカルデータ** | **最高** (公式・調整済・長期間) | 普通 (期間短い) | 悪い (Excel主体) | **J-Quants** (学習用) |
| **リアルタイム板情報** | 有料・制限あり | **最高** (WebSocket/Push配信) | 普通 (Windows限定) | **auカブコム** (監視・執行用) |
| **発注API** | なし (データ専用) | **REST API** (Python親和性高) | COM/Excel (レガシー) | **auカブコム** (発注用) |
| **板情報の深さ** | 制限あり | **フル板** (全気配) | 制限あり | **auカブコム** (流動性判断) |

**結論:**

* **バックテスト・学習:** J-Quants API  
* **本番執行・リアルタイム監視:** auカブコム証券 (kabusapi)  
  * kabusapi はREST APIであり、Pythonからの制御が容易で、Linux環境（AWS/GCP）でも動作可能です。楽天RSSはWindows上のExcel連携が前提であり、安定したサーバーサイド運用には不向きです 40。

### **6.3 レイテンシとコロケーション**

HFT（高頻度取引）を目指さない限り、コロケーション（取引所サーバーへの物理的近接）は必須ではありませんが、AWSの東京リージョン（ap-northeast-1）にサーバーを置くことは推奨されます。kabusapi のトークンリフレッシュ管理などを適切に実装し、API制限（レートリミット）に抵触しないよう、ratelimit ライブラリ等で制御を行う必要があります 42。

## ---

**7\. 結論とロードマップ**

貴殿の Stock Predictor は、その高いパフォーマンス指標ゆえに、現段階では「過学習」または「市場構造の無視」という重大なリスクを内包しています。しかし、これは失敗ではなく、洗練されたシステムへと進化するための通過点です。

**推奨されるアクションプラン:**

1. **フェーズ1（検証の厳格化）:**  
   * J-Quants API（Free/Lightプラン）を導入し、データを刷新する。  
   * Combinatorial Purged CV を実装し、リークのない真の勝率を測定する（目標：PF 1.5以上）。  
   * シミュレーターに板寄せ、特別気配、ストップ高・安の制約ロジックを追加する。  
2. **フェーズ2（モデルの高度化）:**  
   * LightGBM または TFT を導入し、信用残高やセクター別空売り比率などの日本市場特有の特徴量を追加する。  
   * FinBERTを用いたニュースセンチメント分析の統合を検討する。  
3. **フェーズ3（ペーパートレーディング）:**  
   * auカブコムのAPIを用い、資金を投じずにリアルタイムデータでシグナルを生成・記録し、スリッページや約定率を実測する。

このプロセスを経ることで、貴殿のシステムは「机上の空論」から、機関投資家レベルの堅牢性を備えた「資産運用エンジン」へと進化するでしょう。勝率が84%から60%に落ちたとしても、それが再現性のある60%であれば、複利の力によって莫大な富を生み出すことが可能です。

#### **引用文献**

1. Look-Ahead Bias In Backtests And How To Detect It | by Michael Harris | Medium, 12月 17, 2025にアクセス、 [https://mikeharrisny.medium.com/look-ahead-bias-in-backtests-and-how-to-detect-it-ad5e42d97879](https://mikeharrisny.medium.com/look-ahead-bias-in-backtests-and-how-to-detect-it-ad5e42d97879)  
2. Missing Data · Issue \#525 · ranaroussi/yfinance \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/ranaroussi/yfinance/issues/525](https://github.com/ranaroussi/yfinance/issues/525)  
3. yfinance suddenly skips yesterday : r/algotrading \- Reddit, 12月 17, 2025にアクセス、 [https://www.reddit.com/r/algotrading/comments/1oq51ux/yfinance\_suddenly\_skips\_yesterday/](https://www.reddit.com/r/algotrading/comments/1oq51ux/yfinance_suddenly_skips_yesterday/)  
4. Why the last row of my intraday yfinance data is missing? \- Stack Overflow, 12月 17, 2025にアクセス、 [https://stackoverflow.com/questions/77712354/why-the-last-row-of-my-intraday-yfinance-data-is-missing](https://stackoverflow.com/questions/77712354/why-the-last-row-of-my-intraday-yfinance-data-is-missing)  
5. Why Adj Close Disappeared in yfinance (And How to Adapt) | by JosueMonte \- Medium, 12月 17, 2025にアクセス、 [https://medium.com/@josue.monte/why-adj-close-disappeared-in-yfinance-and-how-to-adapt-6baebf1939f6](https://medium.com/@josue.monte/why-adj-close-disappeared-in-yfinance-and-how-to-adapt-6baebf1939f6)  
6. Adjusted Close from yfinance is not the same as total return from Yahoo Finance · Issue \#2070 \- GitHub, 12月 17, 2025にアクセス、 [https://github.com/ranaroussi/yfinance/issues/2070](https://github.com/ranaroussi/yfinance/issues/2070)  
7. Having Issues Downloading Adjusted Close Prices with yfinance – Constant Rate Limit Errors & Cookie Problems : r/learnpython \- Reddit, 12月 17, 2025にアクセス、 [https://www.reddit.com/r/learnpython/comments/1kgwlxx/having\_issues\_downloading\_adjusted\_close\_prices/](https://www.reddit.com/r/learnpython/comments/1kgwlxx/having_issues_downloading_adjusted_close_prices/)  
8. How to handle rate limits | OpenAI Cookbook, 12月 17, 2025にアクセス、 [https://cookbook.openai.com/examples/how\_to\_handle\_rate\_limits](https://cookbook.openai.com/examples/how_to_handle_rate_limits)  
9. Is web scraping legal? Yes, if you know the rules. \- Apify Blog, 12月 17, 2025にアクセス、 [https://blog.apify.com/is-web-scraping-legal/](https://blog.apify.com/is-web-scraping-legal/)  
10. The Legal Landscape of Web Scraping \- Quinn Emanuel, 12月 17, 2025にアクセス、 [https://www.quinnemanuel.com/the-firm/publications/the-legal-landscape-of-web-scraping/](https://www.quinnemanuel.com/the-firm/publications/the-legal-landscape-of-web-scraping/)  
11. J-Quants API | Japan Exchange Group \- JPX, 12月 17, 2025にアクセス、 [https://www.jpx.co.jp/english/markets/other-data-services/j-quants-api/index.html](https://www.jpx.co.jp/english/markets/other-data-services/j-quants-api/index.html)  
12. TOP | J-Quants, 12月 17, 2025にアクセス、 [https://jpx-jquants.com/](https://jpx-jquants.com/)  
13. Release of API Distribution Service J-Quants API (Paid Version) for Retail Investors \- JPX, 12月 17, 2025にアクセス、 [https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20230403-01.html](https://www.jpx.co.jp/english/corporate/news/news-releases/6020/20230403-01.html)  
14. Itayose Auction Method: How Japanese Markets Set Commodity Prices \- Investopedia, 12月 17, 2025にアクセス、 [https://www.investopedia.com/terms/i/itayose.asp](https://www.investopedia.com/terms/i/itayose.asp)  
15. Transaction Methods | Trading Rules of Domestic Stocks | Japan Exchange Group, 12月 17, 2025にアクセス、 [https://www.jpx.co.jp/english/equities/trading/domestic/04.html](https://www.jpx.co.jp/english/equities/trading/domestic/04.html)  
16. Itayose Conditions and Pricing Examples \- JPX, 12月 17, 2025にアクセス、 [https://www.jpx.co.jp/english/derivatives/rules/trading-methods/tvdivq0000004h12-att/tvdivq000000ueul.pdf](https://www.jpx.co.jp/english/derivatives/rules/trading-methods/tvdivq0000004h12-att/tvdivq000000ueul.pdf)  
17. TSE confirms changes to trading rules effective 9 May, 12月 17, 2025にアクセス、 [https://www.thetradenews.com/tse-confirms-changes-to-trading-rules-effective-9-may/](https://www.thetradenews.com/tse-confirms-changes-to-trading-rules-effective-9-may/)  
18. Nikkei Future: Japan Exchange Circuit Breaker Trigger Explained \- E-Housing, 12月 17, 2025にアクセス、 [https://e-housing.jp/post/nikkei-future-japan-exchange](https://e-housing.jp/post/nikkei-future-japan-exchange)  
19. Price Limits/ Circuit Breaker Rule | Derivatives | Japan Exchange Group \- JPX, 12月 17, 2025にアクセス、 [https://www.jpx.co.jp/english/derivatives/rules/price-limit-cb/index.html](https://www.jpx.co.jp/english/derivatives/rules/price-limit-cb/index.html)  
20. Machine Learning for Financial Market Prediction \- Time Series Prediction With Sklearn and Keras \- \- Alpha Architect, 12月 17, 2025にアクセス、 [https://alphaarchitect.com/machine-learning-financial-market-prediction-time-series-prediction-sklearn-keras/](https://alphaarchitect.com/machine-learning-financial-market-prediction-time-series-prediction-sklearn-keras/)  
21. The Combinatorial Purged Cross-Validation method \- Towards AI, 12月 17, 2025にアクセス、 [https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method](https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method)  
22. skfolio.model\_selection.CombinatorialPurgedCV, 12月 17, 2025にアクセス、 [https://skfolio.org/generated/skfolio.model\_selection.CombinatorialPurgedCV.html](https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html)  
23. Cross Validation in Finance: Purging, Embargoing, Combinatorial \- QuantInsti Blog, 12月 17, 2025にアクセス、 [https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)  
24. \[D\] Benefits of Purged CV in Time Series? : r/MachineLearning \- Reddit, 12月 17, 2025にアクセス、 [https://www.reddit.com/r/MachineLearning/comments/1j392dd/d\_benefits\_of\_purged\_cv\_in\_time\_series/](https://www.reddit.com/r/MachineLearning/comments/1j392dd/d_benefits_of_purged_cv_in_time_series/)  
25. Comparing LSTM and Random Forests for Stock Price Movement Forecasting \- ijrti, 12月 17, 2025にアクセス、 [https://www.ijrti.org/papers/IJRTI2401047.pdf](https://www.ijrti.org/papers/IJRTI2401047.pdf)  
26. An Economic Forecasting Method Based on the LightGBM-Optimized LSTM and Time-Series Model \- NIH, 12月 17, 2025にアクセス、 [https://pmc.ncbi.nlm.nih.gov/articles/PMC8492266/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8492266/)  
27. Outstanding Margin Trading, etc. | Japan Exchange Group \- JPX, 12月 17, 2025にアクセス、 [https://www.jpx.co.jp/english/markets/statistics-equities/margin/index.html](https://www.jpx.co.jp/english/markets/statistics-equities/margin/index.html)  
28. Information on Outstanding Short Selling Positions | Japan Exchange Group \- JPX, 12月 17, 2025にアクセス、 [https://www.jpx.co.jp/english/markets/public/short-selling/01.html](https://www.jpx.co.jp/english/markets/public/short-selling/01.html)  
29. Information on Outstanding Short Selling Positions | Japan Exchange Group \- JPX, 12月 17, 2025にアクセス、 [https://www.jpx.co.jp/english/markets/public/short-selling/index.html](https://www.jpx.co.jp/english/markets/public/short-selling/index.html)  
30. What jump effects are implicit in Nikkei 225 returns and the changes in the volatility index Japan? \- ResearchGate, 12月 17, 2025にアクセス、 [https://www.researchgate.net/publication/287350150\_What\_jump\_effects\_are\_implicit\_in\_Nikkei\_225\_returns\_and\_the\_changes\_in\_the\_volatility\_index\_Japan](https://www.researchgate.net/publication/287350150_What_jump_effects_are_implicit_in_Nikkei_225_returns_and_the_changes_in_the_volatility_index_Japan)  
31. The Lead-Lag Relationship between VIX Futures and SPX Futures \- ACFR \- AUT, 12月 17, 2025にアクセス、 [https://acfr.aut.ac.nz/\_\_data/assets/pdf\_file/0010/544519/VIX\_leadlag-2.pdf](https://acfr.aut.ac.nz/__data/assets/pdf_file/0010/544519/VIX_leadlag-2.pdf)  
32. Time-dependent lead-lag relationships between the VIX and VIX futures markets, 12月 17, 2025にアクセス、 [https://ideas.repec.org/p/arx/papers/1910.13729.html](https://ideas.repec.org/p/arx/papers/1910.13729.html)  
33. izumi-lab/bert-small-japanese-fin \- Hugging Face, 12月 17, 2025にアクセス、 [https://huggingface.co/izumi-lab/bert-small-japanese-fin](https://huggingface.co/izumi-lab/bert-small-japanese-fin)  
34. bardsai/finance-sentiment-ja-base \- Hugging Face, 12月 17, 2025にアクセス、 [https://huggingface.co/bardsai/finance-sentiment-ja-base](https://huggingface.co/bardsai/finance-sentiment-ja-base)  
35. Beware of Excessive Leverage – Introduction to Kelly and Optimal F \- QuantPedia, 12月 17, 2025にアクセス、 [https://quantpedia.com/beware-of-excessive-leverage-introduction-to-kelly-and-optimal-f/](https://quantpedia.com/beware-of-excessive-leverage-introduction-to-kelly-and-optimal-f/)  
36. Kelly Criterion in real trading \- Quantitative Finance Stack Exchange, 12月 17, 2025にアクセス、 [https://quant.stackexchange.com/questions/80498/kelly-criterion-in-real-trading](https://quant.stackexchange.com/questions/80498/kelly-criterion-in-real-trading)  
37. Volatility Targeting vs Buy & Hold: Python Backtest Reveals the Truth \- YouTube, 12月 17, 2025にアクセス、 [https://www.youtube.com/watch?v=BwvE3TzwzXs](https://www.youtube.com/watch?v=BwvE3TzwzXs)  
38. Introduction to CPPI – Constant Proportion Portfolio Insurance \- QuantPedia, 12月 17, 2025にアクセス、 [https://quantpedia.com/introduction-to-cppi-constant-proportion-portfolio-insurance/](https://quantpedia.com/introduction-to-cppi-constant-proportion-portfolio-insurance/)  
39. A Constant Proportion Portfolio Insurance Style Trading Strategy \- Alpaca, 12月 17, 2025にアクセス、 [https://alpaca.markets/learn/cppi-1](https://alpaca.markets/learn/cppi-1)  
40. Understanding Rate Limits \- Essential API Insights for RSS Developers \- MoldStud, 12月 17, 2025にアクセス、 [https://moldstud.com/articles/p-understanding-rate-limits-essential-api-insights-for-rss-developers](https://moldstud.com/articles/p-understanding-rate-limits-essential-api-insights-for-rss-developers)  
41. Python API limit exceeded despite increasing amount of second \- Stack Overflow, 12月 17, 2025にアクセス、 [https://stackoverflow.com/questions/66899480/python-api-limit-exceeded-despite-increasing-amount-of-second](https://stackoverflow.com/questions/66899480/python-api-limit-exceeded-despite-increasing-amount-of-second)  
42. ratelimit · PyPI, 12月 17, 2025にアクセス、 [https://pypi.org/project/ratelimit/](https://pypi.org/project/ratelimit/)  
43. Python API Rate Limiting \- How to Limit API Calls Globally \- Stack Overflow, 12月 17, 2025にアクセス、 [https://stackoverflow.com/questions/40748687/python-api-rate-limiting-how-to-limit-api-calls-globally](https://stackoverflow.com/questions/40748687/python-api-rate-limiting-how-to-limit-api-calls-globally)