# **アルゴリズム取引システム高度化のための包括的研究報告書：v9.0cからv10.0への進化戦略**

## **目次**

1. **エグゼクティブサマリー**  
2. **第1章 現行システム（v9.0c）の診断と課題分析**  
   * 1.1 パフォーマンス指標の批判的評価  
   * 1.2 2025年3月・4月の失敗要因：レジーム・ブラインドネス  
   * 1.3 既存シグナル（加重スコア）の限界とS\&P500フィルターの遅効性  
   * 1.4 ケリー基準によるポジション管理のリスク  
3. **第2章 市場レジーム検知の高度化：急落と急騰の早期識別**  
   * 2.1 日経平均ボラティリティー・インデックス（Nikkei 225 VI）による「恐怖」の定量化  
   * 2.2 マーケット・ブレドスの活用：Zweig Breadth Thrustによる回復期検知  
   * 2.3 信用取引残高（信用需給）と裁定買い残の分析  
   * 2.4 外国人投資家動向と先物手口情報の統合  
4. **第3章 テクニカル指標の改善と特徴量エンジニアリング**  
   * 3.1 ADX（Average Directional Index）によるトレンド/レンジ判定フィルター  
   * 3.2 出来高分析の深化：オン・バランス・ボリューム（OBV）と機関投資家の痕跡  
   * 3.3 マルチタイムフレーム分析（MTF）によるフラクタル構造の利用  
   * 3.4 ファクター投資の視点：バリュー、モメンタム、クオリティの融合  
5. **第4章 機械学習（Machine Learning）の統合戦略**  
   * 4.1 モデル選択：LightGBM（勾配ブースティング決定木）の優位性  
   * 4.2 教師データの生成：トリプル・バリア法（Triple Barrier Method）  
   * 4.3 検証手法の刷新：Purged K-Fold Cross-Validationによる先読みバイアスの排除  
   * 4.4 特徴量重要度（Feature Importance）とSHAP値による解釈性確保  
6. **第5章 リスク管理とポジションサイジングの再構築**  
   * 5.1 ケリー基準からボラティリティ・ターゲティングへの移行  
   * 5.2 ポートフォリオ最適化：階層的リスク・パリティ（HRP）の導入  
   * 5.3 出口戦略の動的化：Chandelier ExitとATRトレーリングストップ  
7. **第6章 実装ロードマップとバックテスト・プロトコル**  
   * 6.1 v10.0アーキテクチャの設計図  
   * 6.2 ウォークフォワード・テストと実運用への移行基準  
8. **結論**

## ---

**エグゼクティブサマリー**

本報告書は、日本株短期〜中期売買アルゴリズム「v9.0c」が直面している課題、特に2025年3月の急落局面での損失（-3.71%）と4月の回復局面における機会損失（+0.00%）を解決し、シャープレシオ2.0超を目指す次世代システム「v10.0」への移行戦略を詳述するものである。

現行システムは、RSIやMACDなどの古典的テクニカル指標の加重スコアに依存しており、相場の「状態（レジーム）」の変化を認識できない「静的（Static）」なモデルであることに根本的な脆弱性がある。3月のドローダウンは、ボラティリティが急拡大する「クラッシュ・レジーム」において、平時の「押し目買い」ロジックが機能し続けた結果であり、4月の機会損失は、V字回復という特殊なモメンタム環境に対して、遅効性の高い移動平均線フィルターがエントリーを阻害したことに起因する。

本調査では、これらの課題に対し、以下の4つの柱に基づく包括的な改善策を提案する。

1. **レジーム検知の導入**: 日経平均VIを用いたボラティリティ監視と、Zweig Breadth Thrustによる強力な買いシグナルの導入により、市場の「パニック」と「熱狂」を定量的に識別する。  
2. **機械学習の実装**: LightGBMを採用し、従来の線形加重スコアから、非線形な相互作用を学習可能なモデルへと移行する。教師ラベルにはトリプル・バリア法を採用し、トレードの損益構造に直接最適化する。  
3. **特徴量の多次元化**: ADXによるトレンド強度のフィルタリング、OBVや信用需給データによる需給解析、マルチタイムフレーム分析を統合し、ダマシを排除する。  
4. **リスク管理の動的化**: 破産リスクを内包するケリー基準を廃止し、ボラティリティ・ターゲティングおよび階層的リスク・パリティ（HRP）へ移行することで、ドローダウンを数学的に抑制する。

## ---

**第1章 現行システム（v9.0c）の診断と課題分析**

### **1.1 パフォーマンス指標の批判的評価**

現在のパフォーマンス指標（年間リターン+19.03%、シャープレシオ1.62、最大ドローダウン5.9%、勝率55%）は、一般的な個人投資家の基準を大きく上回る優秀な数値である。しかし、プロフェッショナルなクオンツ運用の視点、特にヘッジファンドや機関投資家の基準に照らすと、いくつかの懸念材料が浮かび上がる。

まず、**シャープレシオ 1.62** は良好だが、これが「強気相場」に依存した数値である可能性が高い。2025年3月の-3.71%という単月損失は、最大ドローダウン5.9%の過半を占めており、システムが特定の市場ストレス（テールリスクイベント）に対して脆弱であることを示唆している。リスク調整後リターンの観点からは、リターンを犠牲にしてでもドローダウンを抑制し、シャープレシオを2.0〜2.5のレンジへ引き上げることが、長期間の運用における複利効果を最大化する鍵となる。

**勝率 55%** という数値は、トレンドフォロー型戦略としては標準的だが、平均利益/平均損失（ペイオフレシオ）への依存度が高いことを意味する。勝率が50%付近まで低下した場合、一気に期待値がマイナスに転じるリスクがある。機械学習の導入により、エントリーの精度（Precision）を向上させ、勝率を60%超へ引き上げることが可能である。

### **1.2 2025年3月・4月の失敗要因：レジーム・ブラインドネス**

v9.0cが直面した最大の問題は、市場環境の変化（レジーム・シフト）を認識できなかったことにある。これを「レジーム・ブラインドネス（Regime Blindness）」と呼ぶ。

* 3月の失敗（弱気相場検知の遅れ）:  
  v9.0cはRSIやボリンジャーバンド（BB）の逆張りシグナルを含んでいると推測される。通常の調整局面では、RSIの低下は「売られすぎ」による買いシグナルとなる。しかし、3月のような急落局面（おそらくファンダメンタルズ要因や外部ショックによる構造的な崩壊）では、RSIの低下は「強い下降モメンタム」の証左となる。システムは「安くなった」と判断して買い向かい、落ちてくるナイフを掴む結果となった。これは、ボラティリティの急拡大を「リスク」としてではなく「機会」として誤認した結果である。  
* 4月の失敗（回復期の機会損失）:  
  急落後の4月、相場が+0.00%（エントリーなし）であったことは、システムが「安全確認」に時間をかけすぎたことを意味する。現行のエントリー条件である「上昇トレンド（おそらく移動平均線やMACDで判定）」は、急落後のV字回復においては反応が遅すぎる。価格が底を打ち、鋭角に上昇する初期段階では、移動平均線は依然として下向きであり、システムはこれを「下降トレンド中の戻り」と判断してエントリーを却下した可能性が高い。この「初動の取り逃がし」は、年間リターンの大幅な低下を招く。

### **1.3 既存シグナル（加重スコア）の限界とS\&P500フィルターの遅効性**

現在のシグナル生成ロジック「RSI, MA, MACD, BB, 出来高, ROCの加重スコア \>= 0.2」は、線形モデルの一種である。  
各指標に固定の重み（ウェイト）を与えて足し合わせる手法は、指標間の相互作用（Interaction）を無視している。例えば、「ボラティリティが低い時のRSI」と「ボラティリティが高い時のRSI」は全く異なる意味を持つが、加重スコアモデルではこれらを同一視してしまう。  
また、**S\&P500 \< 200MA で停止** というフィルターは、日本株運用において致命的な遅れを生む可能性がある。

1. **タイムラグ**: 米国市場の終値確定（日本時間早朝）を見てから日本株の取引判断を行うため、リアルタイムの相関乖離に対応できない。  
2. **デカップリング**: 近年、日米株式市場の相関は必ずしも1.0ではない。為替（ドル円）の影響や、日本独自の金融政策（日銀）により、S\&P500が軟調でも日本株が堅調、あるいはその逆のケースが増えている。S\&P500が200MAを下回っているという理由だけで、日本株の有望な個別銘柄のチャンスを全て放棄するのは、過剰なフィルタリング（Type II Error）である。

### **1.4 ケリー基準によるポジション管理のリスク**

ポジションサイジングに「ケリー基準」を採用している点も、修正が必要なリスク要因である。  
ケリー基準 ($f^\* \= (bp \- q) / b$) は、勝率 ($p$) とペイオフレシオ ($b$) が正確に既知であり、かつ将来も不変であるという仮定の下で、幾何平均リターンを最大化する。しかし、金融市場において勝率とペイオフレシオは常に変動する推定値に過ぎない。  
推定誤差がある中で「フル・ケリー」を適用すると、推奨されるレバレッジが過大になりやすく、破産リスク（Ruin Risk）が飛躍的に高まる。特に「最大25%/銘柄」という集中投資は、3月のような相関が高い（全銘柄が同時に下落する）局面では、ポートフォリオ全体に壊滅的なダメージを与えかねない。

## ---

**第2章 市場レジーム検知の高度化：急落と急騰の早期識別**

v10.0への改善において最優先すべきは、トレード戦略を適用する「土俵」を見極めるレジーム検知機能の実装である。

### **2.1 日経平均ボラティリティー・インデックス（Nikkei 225 VI）による「恐怖」の定量化**

米国株におけるVIX指数同様、日本株には\*\*日経平均VI（Nikkei 225 VI）\*\*が存在する。これは大阪取引所の日経225オプション価格から算出される、将来30日間の予想変動率である 1。

#### **2.1.1 暴落検知のしきい値とロジック**

通常、日経平均VIは20ポイント前後で推移する。これが急上昇する局面は、投資家がプットオプションを買い急いでいる（＝パニック）状態を示す。

* **絶対値基準**: VIが **30.0** を超えた場合、市場は「クラッシュ・レジーム」にあると定義する 3。この領域では、テクニカル指標の「売られすぎ」は無視し、すべての新規買いエントリーを凍結、あるいは既存ポジションの強制決済（ロスカットラインの引き上げ）を行うべきである。3月の損失は、このフィルターがあれば回避できた可能性が高い。  
* **相対値基準（ボリンジャーバンド）**: VI自体のボリンジャーバンド（20日、2σ）を計算し、VIが+2σを突破した瞬間を「レジーム転換点」とする。これは絶対値基準よりも早期に異常を検知できる。

#### **2.1.2 VIを用いた動的ポジション管理**

VIの水準に応じて、基本ポジションサイズを調整する「VIスケーリング」を導入する。

* $VI \< 20$: 通常サイズ（100%）  
* $20 \\le VI \< 25$: 警戒モード（サイズ 70%）  
* $25 \\le VI \< 30$: 防御モード（サイズ 40%）  
* $VI \\ge 30$: 緊急モード（サイズ 0% またはヘッジショート）

### **2.2 マーケット・ブレドスの活用：Zweig Breadth Thrustによる回復期検知**

4月の回復を取り逃がした原因は、インデックス価格や移動平均線のみを見ていたことにある。相場の転換（特にセリングクライマックス後の急騰）は、一部の大型株ではなく、市場全体の銘柄が一斉に上昇する「ブレドス（Breadth）」の変化に最初に現れる。

#### **2.2.1 Zweig Breadth Thrust Indicator (ZBT) の実装**

ZBTは、マーティン・ツバイク博士が開発した強力な買いシグナルであり、弱気相場からの立ち上がりを捉えるのに特化している 4。

* **計算式**:  
  1. 東証プライム（または日経225構成銘柄）の「値上がり銘柄数」と「値下がり銘柄数」を取得。  
  2. ブレドス比率 \= 値上がり銘柄数 / (値上がり銘柄数 \+ 値下がり銘柄数)  
  3. ZBT \= ブレドス比率の10日指数平滑移動平均 (EMA)  
* シグナル条件:  
  ZBTが 0.40（40%）以下 の状態から、10日以内に 0.615（61.5%）以上 へ急騰した場合 4。

この現象は「市場参加者のセンチメントが一気に弱気から強気へ転換した」ことを意味し、極めて信頼性の高い中期的買いシグナルとなる。4月の局面でこのシグナルが出ていれば、移動平均線が好転する前に、市場のエネルギーを感じ取ってエントリーできていたはずである。Pythonとyfinanceを用いれば、日経225構成銘柄のデータを取得し、日次でこの指標を計算することが可能である 7。

### **2.3 信用取引残高（信用需給）と裁定買い残の分析**

日本市場特有の需給要因として、**信用取引残高（Shinyo Balance）** の分析が不可欠である 10。

* **信用買い残（Margin Buying Balance）**: 個人投資家の「将来の売り圧力」を表す。信用買い残が歴史的高水準にある状態で株価が下落し始めると、追証（Margin Call）回避のための投げ売りが発生し、下落が加速する。3月の急落前に信用買い残が積み上がっていなかったかを確認し、これを「危険シグナル」としてアルゴリズムに組み込むべきである。  
* **信用評価損益率（Matsui Securitiesなど）**: これが-15%〜-20%に達すると「追証ライン」となり、セリングクライマックスが近いことを示唆する。逆に、これが底を打って回復し始めたタイミングは、4月のような反発局面での絶好のエントリー機会となる 13。  
* **裁定買い残（Arbitrage Buying Balance）**: 外国人投資家の先物買いに伴う現物買いのポジション。これが高水準にあると、先物売りによる裁定解消売りが出やすく、上値が重くなる。

### **2.4 外国人投資家動向と先物手口情報の統合**

日本株の売買シェアの約6〜7割は外国人投資家が占めており、彼らの動向がトレンドを決定する 14。

* **投資部門別売買動向**: 毎週木曜日に発表される前週のデータだが、外国人投資家が「買い越し」に転じた週は、その後数週間の上昇トレンドの起点になりやすい。これを週次のフィルターとして利用する（外国人が売り越している間はロングポジションを抑制するなど） 14。  
* **先物手口**: 日中のリアルタイム判断には、先物の手口情報（特にゴールドマン・サックスやモルガン・スタンレーなどの主要外資系証券の買い越し・売り越し）を監視することが有効である。v10.0では、データフィードが可能であれば、これを日中の高頻度シグナルとして取り入れることを検討すべきである。

## ---

**第3章 テクニカル指標の改善と特徴量エンジニアリング**

機械学習モデルの入力となる「特徴量（Feature）」の質が、AIの予測精度を決定する。既存の指標に加え、より情報の粒度が高い指標を導入する。

### **3.1 ADX（Average Directional Index）によるトレンド/レンジ判定フィルター**

RSIやストキャスティクスなどのオシレーター系指標は、レンジ相場では有効だが、強いトレンド相場では「ダマシ（False Signal）」を頻発する（上昇トレンド中に「買われすぎ」シグナルが出続けて早すぎる利確や逆張り売りを誘発する）。  
これを防ぐために、トレンドの「強さ」を測る ADX（Average Directional Index） を導入する 17。

* **フィルターロジック**:  
  * $ADX(14) \> 25$: トレンド相場と判定。**RSIの逆張りシグナル（70以上で売り）を無効化**し、順張りロジック（押し目買い、ブレイクアウト）を優先する。  
  * $ADX(14) \< 20$: レンジ相場と判定。MACDのようなトレンド追随型指標を無効化し、ボリンジャーバンドの±2σ逆張りやRSIのスイングトレードを優先する。

これにより、3月のような強い下降トレンド（ADX上昇）でオシレーターが「売られすぎ」を示しても、買い向かうのを防ぐことができる。

### **3.2 出来高分析の深化：オン・バランス・ボリューム（OBV）と機関投資家の痕跡**

出来高は価格に先行する。単なる「出来高」ではなく、価格変動と紐づいた累積指標を用いる。

* オン・バランス・ボリューム（OBV）:  
  株価が前日比プラスの日の出来高を加算し、マイナスの日の出来高を減算する。  
  * *ダイバージェンス検知*: 4月の回復局面において、株価の上昇よりも先にOBVが上昇し始めている場合、それは「スマートマネー（機関投資家）」による密かな集め（Accumulation）を示唆する。これを検知することで、価格ブレイク前の早期エントリーが可能になる 17。  
* Chaikin Money Flow (CMF):  
  高値・安値に対する終値の位置と出来高を組み合わせ、資金の流入・流出をより詳細に測定する。日中の値動きの中で「終値が高値圏で引ける」ことが多い場合、強い買い圧力を示唆する 17。

### **3.3 マルチタイムフレーム分析（MTF）によるフラクタル構造の利用**

日足のみの分析では「木を見て森を見ず」の状態になりやすい。上位足（週足）のトレンドは下位足（日足）を支配する。

* 週足フィルター:  
  週足のMACDヒストグラムが上昇中、あるいは週足移動平均線（13週、26週）がゴールデンクロスしている状態でのみ、日足の「買い」シグナルを採用する。  
* アライメント（一致）戦略:  
  最も勝率が高いのは、上位足のトレンド方向に、下位足が戻り（Pullback）から再開するタイミングである。  
  * 例：週足上昇トレンド \+ 日足RSIが一時的に低下し、再上昇を始めたポイント。  
    このロジックをPythonで実装する場合、pandasのresample機能を用いて日次データを週次データに変換し、指標を計算した後、再度日次データにマージ（ffill）して特徴量とする 20。

### **3.4 ファクター投資の視点：バリュー、モメンタム、クオリティの融合**

アカデミックなアプローチとして、ファーマ・フレンチの5ファクターモデルなどの知見を取り入れる 23。特に2024年以降の日本市場では、東証のPBR改革要請により「バリュー（低PBR）」ファクターが強く効いている。

* バリュー・モメンタム合成:  
  単なる割安（バリュー）ではなく、「割安かつ、直近のモメンタムが強い」銘柄を選定する。  
  * 特徴量として PBR、PER、配当利回り を追加。  
  * 12ヶ月モメンタム と 1ヶ月リバーサル の指標を追加 26。  
    これにより、単なるテクニカル売買から、ファンダメンタルズの裏付けのある「負けにくい」銘柄選択へと進化させる。

## ---

**第4章 機械学習（Machine Learning）の統合戦略**

v9.0cの「加重スコア」を、AIによる予測モデルに置き換える。ここでは、金融時系列データに特化した最新のパイプラインを提案する。

### **4.1 モデル選択：LightGBM（勾配ブースティング決定木）の優位性**

金融データはノイズが多く、欠損値が含まれ、特徴量間の関係が非線形である。このため、ニューラルネットワーク（LSTMなど）よりも、決定木ベースのアンサンブル学習、特に **LightGBM (Light Gradient Boosting Machine)** が実務上最も高いパフォーマンスを発揮する 28。

* **LightGBMの利点**:  
  * **高速性**: 大規模なバックテストデータを高速に学習可能。  
  * **欠損値処理**: テクニカル指標の計算過程で生じるNaNをネイティブに処理できる。  
  * **Leaf-wise Growth**: 複雑なパターン（深い相互作用）を学習しやすい。  
  * **カスタム目的関数**: トレード独自の損失関数（シャープレシオ最大化など）を定義しやすい 31。

### **4.2 教師データの生成：トリプル・バリア法（Triple Barrier Method）**

機械学習で最も重要なのは「何を予測させるか（ラベリング）」である。単純に「翌日の終値」を予測させると、ノイズばかりを学習してしまう。  
Marcos Lopez de Prado氏が提唱する トリプル・バリア法 を採用する 33。

* **仕組み**: エントリー時点から、以下の3つの「バリア（壁）」のいずれかに最初に到達した時点でラベルを決定する。  
  1. **上部バリア（利確）**: エントリー価格 $\\times (1 \+ \\text{目標リターン})$  
  2. **下部バリア（損切）**: エントリー価格 $\\times (1 \- \\text{許容リスク})$  
  3. **垂直バリア（時間切れ）**: エントリーから $N$ 日経過  
* **動的閾値**: 目標リターンと許容リスクは固定値（例：5%）ではなく、その時点のボラティリティ（ATRや標準偏差）に基づいて動的に設定する。ボラティリティが高い時はバリアを広げ、低い時は狭めることで、市場環境に適応したラベル付けが可能になる。  
* **クラス分類**:  
  * 上部バリア到達 $\\rightarrow$ クラス1（買い）  
  * 下部バリア到達 $\\rightarrow$ クラス-1（見送り、または売り）  
  * 垂直バリア到達 $\\rightarrow$ クラス0（中立）

### **4.3 検証手法の刷新：Purged K-Fold Cross-Validationによる先読みバイアスの排除**

時系列データにおいて、通常のランダムなK-Fold交差検証を行うと、未来の情報が過去の学習データに漏れる「リーク（Data Leakage）」が発生し、バックテスト結果が過剰に良く見える（過学習）。  
これを防ぐために Purged K-Fold Cross-Validation（パージ付き交差検証） を導入する 36。

* Purging（パージ）:  
  テストデータ期間の直前の学習データを「削除」する。例えば、トレードの保有期間が10日間であれば、テスト期間開始前の10日分のデータは、テスト期間のトレード結果と重複している可能性があるため、学習に使ってはならない。  
* Embargo（エンバーゴ）:  
  テスト期間終了後のデータも、シリアル相関（自己相関）の影響を断ち切るために、一定期間（例：保有期間の1%〜5%）空けてから次の学習データとして使用する。

この厳密な検証を経ることで、3月や4月のような未知の相場環境でも機能する堅牢なモデル（Robust Model）を構築できる。

### **4.4 特徴量重要度（Feature Importance）とSHAP値による解釈性確保**

AIを「ブラックボックス」にしないために、**SHAP (SHapley Additive exPlanations)** 値を用いて、モデルの判断根拠を可視化する 39。

* 「なぜこのエントリーを選んだのか？」に対し、「RSIが低いだけでなく、VIが落ち着いており、かつ外国人買い越しが増加しているため」といった複合的な理由を定量的に把握できる。  
* 特定のレジーム（例：暴落時）でどの特徴量が重要かを分析することで、特徴量の取捨選択（Feature Selection）を行い、モデルを軽量化・高速化できる。

## ---

**第5章 リスク管理とポジションサイジングの再構築**

### **5.1 ケリー基準からボラティリティ・ターゲティングへの移行**

前述の通り、ケリー基準はリスクが高すぎる。ヘッジファンド等の機関投資家が採用している **ボラティリティ・ターゲティング（Volatility Targeting）** へ移行すべきである 41。

* 基本概念:  
  ポートフォリオ全体のリスク（ボラティリティ）を一定（例：年率15%）に保つように、ポジションサイズを逆算する。  
* 計算式:

  $$\\text{Target Weight}\_t \= \\frac{\\text{Target Vol}}{\\text{Realized Vol}\_t}$$

  ここで、$\\text{Realized Vol}\_t$ は直近（例：20日）の資産変動率の標準偏差である。  
* 効果:  
  3月のように市場のボラティリティが急上昇（$\\text{Realized Vol}$ が増大）すると、分母が大きくなるため、自動的に $\\text{Target Weight}$（ポジションサイズ）が縮小される。これにより、予測モデルが「買い」シグナル出し続けていても、リスク管理モジュールが強制的にポジションを落とし、損失を限定する。逆にボラティリティが低い4月のような局面では、ポジションサイズを大きく取り、リターンを稼ぎに行く。

### **5.2 ポートフォリオ最適化：階層的リスク・パリティ（HRP）の導入**

複数銘柄を保有する場合、**階層的リスク・パリティ（Hierarchical Risk Parity: HRP）** を用いて銘柄間のウェイトを配分する 44。

* 従来の平均分散法（Mean-Variance）の欠点:  
  相関行列の逆行列計算が必要だが、暴落時には全銘柄の相関が1に近づくため計算が不安定になり、特定の銘柄に極端なウェイトを振ってしまう。  
* **HRPの仕組み**:  
  1. **クラスタリング**: 銘柄間の相関距離に基づいて、似た動きをする銘柄をグループ化（階層化）する（デンドログラム作成）。  
  2. **配分**: クラスターごとにリスクが均等になるように資金を配分し、さらにクラスター内の銘柄にも再配分する。  
* 効果:  
  例えば「輸出関連株」と「内需株」という異なるクラスターを自動認識し、分散投資効果を最大化できる。これにより、ポートフォリオ全体のドローダウン耐性が大幅に向上する。Pythonライブラリ PyPortfolioOpt を使用することで容易に実装可能である 46。

### **5.3 出口戦略の動的化：Chandelier ExitとATRトレーリングストップ**

固定の「-10%ハードストップ」は、ボラティリティの高い銘柄ではノイズで刈られやすく、低い銘柄では損失が大きすぎる。  
Chandelier Exit（シャンデリア・イグジット） を導入する 48。

* ロジック:  
  過去 $N$ 日間の最高値（Highest High）から、ATR（Average True Range）の $K$ 倍だけ下の価格をストップラインとする。

  $$\\text{Stop Price} \= \\text{Highest High}\_N \- (K \\times ATR\_N)$$

  （推奨値: $N=22$, $K=3.0$）  
* 利点:  
  価格が上昇して最高値を更新するたびに、ストップラインも自動的に切り上がる（トレーリングストップ機能）。ATRを用いることで、銘柄ごとのボラティリティに合わせた「適正な距離」を保つことができる。急落時にはボラティリティ（ATR）が拡大するため、ストップ幅が広がりそうに見えるが、最高値からの下落幅も大きくなるため、結果的に適切なポイントで利益確定または損切りが執行される。

## ---

**第6章 実装ロードマップとバックテスト・プロトコル**

### **6.1 v10.0アーキテクチャの設計図**

v9.0cからv10.0への移行は、以下の4フェーズで実施することを推奨する。

| フェーズ | 期間 | 実装内容 | 使用ツール/ライブラリ |
| :---- | :---- | :---- | :---- |
| **Phase 1: データ基盤** | 2週間 | 日経VI、信用残、外国人動向、構成銘柄データの自動取得パイプライン構築 | yfinance, pandas, JPX Data |
| **Phase 2: 特徴量・ロジック** | 3週間 | ADX, OBV, ZBTの実装。レジーム検知ロジック（VIフィルター）の組み込み | ta-lib, pandas-ta |
| **Phase 3: AIモデル構築** | 4週間 | トリプル・バリア法によるラベリング、LightGBMの学習とチューニング、Purged CVによる検証 | lightgbm, optuna, scikit-learn |
| **Phase 4: リスクエンジン** | 3週間 | ボラティリティ・ターゲティング、HRPの実装。バックテストとペーパートレーディング | pyportfolioopt, zipline / backtrader |

### **6.2 ウォークフォワード・テストと実運用への移行基準**

最終的な実運用（Live Trading）への移行可否は、単純なバックテスト結果ではなく、ウォークフォワード・テスト（Walk-Forward Analysis） の結果で判断する。  
過去データを「学習期間」と「テスト期間」のペアに分割し、窓をずらしながら（例：2020年で学習→2021年でテスト、2021年で学習→2022年でテスト...）検証を行う。これにより、カーブフィッティング（過学習）のリスクを最小化できる。  
**合格基準案**:

* ウォークフォワード期間（Out-of-Sample）でのシャープレシオ \> 2.0  
* 最大ドローダウン \< 15%  
* 3月（仮想）のような高VI局面でのポジション縮小動作が確認できること  
* 4月（仮想）のようなZBT点灯局面での買いエントリー動作が確認できること

## ---

**結論**

v9.0cのアルゴリズムは、平時の相場においては一定の優位性を持っているが、市場構造の変化（レジーム・シフト）に対する適応力が欠如している。2025年3月の損失と4月の機会損失は、この「静的モデル」の限界を露呈したものである。

本報告書で提案した **「レジーム検知（VI・Breadth）」「機械学習（LightGBM・Triple Barrier）」「動的リスク管理（Vol Target・HRP）」** の3要素を統合したv10.0システムは、市場の「恐怖」と「強欲」を定量的に捉え、自律的に戦略を変容させる能力を持つ。これにより、ダウンサイドリスクを数学的に制御しながら、回復局面のアップサイドを逃さず捉えることが可能となり、目標とする「高いリターンと安定性（高シャープレシオ）」の実現に大きく寄与するであろう。

---

参考文献・データソース:  
1 Nikkei VI methodology and thresholds.  
4 Zweig Breadth Thrust calculation.  
10 Margin Buying Balance data.  
28 LightGBM application in finance.  
33 Triple Barrier Method.  
41 Volatility Targeting logic.  
44 Hierarchical Risk Parity.  
36 Purged K-Fold Cross Validation.  
14 Foreign Investor influence.  
20 Multi-timeframe analysis strategies.

#### **引用文献**

1. S\&P/JPX JGB VIX | S\&P Dow Jones Indices \- S\&P Global, 12月 13, 2025にアクセス、 [https://www.spglobal.com/spdji/en/indices/indicators/sp-jpx-jgb-vix/](https://www.spglobal.com/spdji/en/indices/indicators/sp-jpx-jgb-vix/)  
2. Nikkei Stock Average Volatility Index, 12月 13, 2025にアクセス、 [https://indexes.nikkei.co.jp/en/nkave/index/profile?idx=nk225vi](https://indexes.nikkei.co.jp/en/nkave/index/profile?idx=nk225vi)  
3. Nikkei Stock Average Volatility Index Futures \- JPX, 12月 13, 2025にアクセス、 [https://www.jpx.co.jp/english/derivatives/products/vi/225-vi-futures/tvdivq0000003ki3-att/VIFutures\_201608\_E.pdf](https://www.jpx.co.jp/english/derivatives/products/vi/225-vi-futures/tvdivq0000003ki3-att/VIFutures_201608_E.pdf)  
4. Zweig Breadth Thrust | TrendSpider Learning Center, 12月 13, 2025にアクセス、 [https://trendspider.com/learning-center/zweig-breadth-thrust/](https://trendspider.com/learning-center/zweig-breadth-thrust/)  
5. Breadth Thrust Indicator – Can it Forecast Major Market Bottoms? \- TradingSim, 12月 13, 2025にアクセス、 [https://app.tradingsim.com/blog/breadth-thrust-indicator/](https://app.tradingsim.com/blog/breadth-thrust-indicator/)  
6. Breadth Thrust \- MarketInOut.com, 12月 13, 2025にアクセス、 [https://www.marketinout.com/technical\_analysis.php?t=Breadth\_Thrust\&id=30](https://www.marketinout.com/technical_analysis.php?t=Breadth_Thrust&id=30)  
7. Estimating SP 500 Breadth Indicator — with Python \- Medium, 12月 13, 2025にアクセス、 [https://medium.com/quant-factory/estimating-sp-500-breadth-indicator-with-python-8282468c95e8](https://medium.com/quant-factory/estimating-sp-500-breadth-indicator-with-python-8282468c95e8)  
8. Exploring the Advance-Decline (A/D) Line: Building It for BIST | by Muhammed Burak Bedir, 12月 13, 2025にアクセス、 [https://medium.com/@mburakbedir/exploring-the-advance-decline-a-d-line-building-it-for-bist-0d84f1ad6a0b](https://medium.com/@mburakbedir/exploring-the-advance-decline-a-d-line-building-it-for-bist-0d84f1ad6a0b)  
9. Calculating and graphing daily SPX market breadth from scratch. \- GitHub, 12月 13, 2025にアクセス、 [https://github.com/jamesdellinger/market\_breadth](https://github.com/jamesdellinger/market_breadth)  
10. Outline of the Margin Trading System \- JPX, 12月 13, 2025にアクセス、 [https://www.jpx.co.jp/english/equities/trading/margin/outline/tvdivq0000007szb-att/OutlineOfTheMarginTradingSystem.pdf](https://www.jpx.co.jp/english/equities/trading/margin/outline/tvdivq0000007szb-att/OutlineOfTheMarginTradingSystem.pdf)  
11. New Financial Activity Indexes: Early Warning System for Financial Imbalances in Japan, 12月 13, 2025にアクセス、 [https://www.boj.or.jp/en/research/wps\_rev/wps\_2014/data/wp14e07.pdf](https://www.boj.or.jp/en/research/wps_rev/wps_2014/data/wp14e07.pdf)  
12. Understanding Japanese margin transactions, 12月 13, 2025にアクセス、 [https://www.jsf.co.jp/en/ir/library/media/main/00/teaserItems3/0114/tableContents/02/multiFileUpload20/link/Securities%20Finance%20Times-2.pdf](https://www.jsf.co.jp/en/ir/library/media/main/00/teaserItems3/0114/tableContents/02/multiFileUpload20/link/Securities%20Finance%20Times-2.pdf)  
13. Can margin traders predict future stock returns in Japan? \- IDEAS/RePEc, 12月 13, 2025にアクセス、 [https://ideas.repec.org/a/eee/pacfin/v17y2009i1p41-57.html](https://ideas.repec.org/a/eee/pacfin/v17y2009i1p41-57.html)  
14. Japan \- Net Positions of Foreign Investors vs. Nikkei 225 | MacroMicro, 12月 13, 2025にアクセス、 [https://en.macromicro.me/charts/1068/jp-tse1-foreigners-net-purchase-nikkei225](https://en.macromicro.me/charts/1068/jp-tse1-foreigners-net-purchase-nikkei225)  
15. Investing in Japanese Markets: Why Japan? \- NCPERS, 12月 13, 2025にアクセス、 [https://www.ncpers.org/files/NCPERS%202025%20Why%20Japan%20Sumitomo%20Mitsui%20Trust%20Asset%20Management%20Americas%20Inc%20-%20May9DB.pdf](https://www.ncpers.org/files/NCPERS%202025%20Why%20Japan%20Sumitomo%20Mitsui%20Trust%20Asset%20Management%20Americas%20Inc%20-%20May9DB.pdf)  
16. Trading in Japan: Day Trading, Swing Trading, Strategy, Rules And Backtest \- QuantifiedStrategies.com, 12月 13, 2025にアクセス、 [https://www.quantifiedstrategies.com/trading-japan-day-trading-swing-trading/](https://www.quantifiedstrategies.com/trading-japan-day-trading-swing-trading/)  
17. Chaikin Money Flow & Volatility: Crypto Trading Guide \- Phemex, 12月 13, 2025にアクセス、 [https://phemex.com/academy/what-is-chaikin-money-flow-and-chaikin-volatility](https://phemex.com/academy/what-is-chaikin-money-flow-and-chaikin-volatility)  
18. Which indicators are worth using? : r/investing \- Reddit, 12月 13, 2025にアクセス、 [https://www.reddit.com/r/investing/comments/1chr93l/which\_indicators\_are\_worth\_using/](https://www.reddit.com/r/investing/comments/1chr93l/which_indicators_are_worth_using/)  
19. 5 Best Volume Indicators for Scalping \- LuxAlgo, 12月 13, 2025にアクセス、 [https://www.luxalgo.com/blog/5-best-volume-indicators-for-scalping/](https://www.luxalgo.com/blog/5-best-volume-indicators-for-scalping/)  
20. How To Perform A Multi TimeFrame Analysis \+ 5 Strategies \- Tradeciety, 12月 13, 2025にアクセス、 [https://tradeciety.com/how-to-perform-a-multiple-time-frame-analysis](https://tradeciety.com/how-to-perform-a-multiple-time-frame-analysis)  
21. Multiple Time Frames \- Backtesting.py, 12月 13, 2025にアクセス、 [https://kernc.github.io/backtesting.py/doc/examples/Multiple%20Time%20Frames.html](https://kernc.github.io/backtesting.py/doc/examples/Multiple%20Time%20Frames.html)  
22. Multiple Timeframes Trading: Build Custom Indicators in Python \- YouTube, 12月 13, 2025にアクセス、 [https://www.youtube.com/watch?v=jObikg7gfpU](https://www.youtube.com/watch?v=jObikg7gfpU)  
23. Comparison of the CAPM and Multi-Factor Fama–French Models for the Valuation of Assets in the Industries with the Highest Number of Transactions in the US Market \- MDPI, 12月 13, 2025にアクセス、 [https://www.mdpi.com/2227-7072/13/3/126](https://www.mdpi.com/2227-7072/13/3/126)  
24. (PDF) Fama-French Five Factor Model: Systematic Literature Review \- ResearchGate, 12月 13, 2025にアクセス、 [https://www.researchgate.net/publication/393556945\_Fama-French\_Five\_Factor\_Model\_Systematic\_Literature\_Review](https://www.researchgate.net/publication/393556945_Fama-French_Five_Factor_Model_Systematic_Literature_Review)  
25. The Impact of COVID-19 on the Fama-French Five-Factor Model: Unmasking Industry Dynamics \- MDPI, 12月 13, 2025にアクセス、 [https://www.mdpi.com/2227-7072/12/4/98](https://www.mdpi.com/2227-7072/12/4/98)  
26. Is Japan Different? Evidence on Momentum and Market Dynamics \- ResearchGate, 12月 13, 2025にアクセス、 [https://www.researchgate.net/publication/260994247\_Is\_Japan\_Different\_Evidence\_on\_Momentum\_and\_Market\_Dynamics](https://www.researchgate.net/publication/260994247_Is_Japan_Different_Evidence_on_Momentum_and_Market_Dynamics)  
27. Residual Momentum and Reversal Strategies Revisited \- Super.so, 12月 13, 2025にアクセス、 [https://assets.super.so/e46b77e7-ee08-445e-b43f-4ffd88ae0a0e/files/017c102d-5882-4e93-9f4c-4ef8500ef7d3.pdf](https://assets.super.so/e46b77e7-ee08-445e-b43f-4ffd88ae0a0e/files/017c102d-5882-4e93-9f4c-4ef8500ef7d3.pdf)  
28. Optimizing Stock Price Prediction with LightGBM and Engineered Features \- ResearchGate, 12月 13, 2025にアクセス、 [https://www.researchgate.net/publication/395654169\_Optimizing\_Stock\_Price\_Prediction\_with\_LightGBM\_and\_Engineered\_Features](https://www.researchgate.net/publication/395654169_Optimizing_Stock_Price_Prediction_with_LightGBM_and_Engineered_Features)  
29. A Hybrid AI Framework for Enhanced Stock Movement Prediction: Integrating ARIMA, RNN, and LightGBM Models \- MDPI, 12月 13, 2025にアクセス、 [https://www.mdpi.com/2079-8954/13/3/162](https://www.mdpi.com/2079-8954/13/3/162)  
30. \[2501.07580\] Assets Forecasting with Feature Engineering and Transformation Methods for LightGBM \- arXiv, 12月 13, 2025にアクセス、 [https://arxiv.org/abs/2501.07580](https://arxiv.org/abs/2501.07580)  
31. Parameters — LightGBM 4.6.0.99 documentation, 12月 13, 2025にアクセス、 [https://lightgbm.readthedocs.io/en/latest/Parameters.html](https://lightgbm.readthedocs.io/en/latest/Parameters.html)  
32. Custom Objective for LightGBM | Hippocampus's Garden, 12月 13, 2025にアクセス、 [https://hippocampus-garden.com/lgbm\_custom/](https://hippocampus-garden.com/lgbm_custom/)  
33. An expansion of the Triple-Barrier Method by Marcos López de Prado \- GitHub, 12月 13, 2025にアクセス、 [https://github.com/nkonts/barrier-method](https://github.com/nkonts/barrier-method)  
34. maxzager/Financial-series-and-Triple-Barrier-Method: I did this project as one of the parts from a Python test for my Master's degree. The objective was to practice the treatment of financial time series. \- GitHub, 12月 13, 2025にアクセス、 [https://github.com/maxzager/Financial-series-and-Triple-Barrier-Method](https://github.com/maxzager/Financial-series-and-Triple-Barrier-Method)  
35. Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling Method and Machine Learning Approach for Pair Trading Strategy in Cryptocurrency Markets \- MDPI, 12月 13, 2025にアクセス、 [https://www.mdpi.com/2227-7390/12/5/780](https://www.mdpi.com/2227-7390/12/5/780)  
36. mlfinlab/mlfinlab/cross\_validation/combinatorial.py at master · hudson-and-thames/mlfinlab \- GitHub, 12月 13, 2025にアクセス、 [https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/cross\_validation/combinatorial.py](https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/cross_validation/combinatorial.py)  
37. Cross Validation in Finance: Purging, Embargoing, Combinatorial \- QuantInsti Blog, 12月 13, 2025にアクセス、 [https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)  
38. The Combinatorial Purged Cross-Validation method \- Towards AI, 12月 13, 2025にアクセス、 [https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method](https://towardsai.net/p/l/the-combinatorial-purged-cross-validation-method)  
39. 12月 13, 2025にアクセス、 [https://towardsdatascience.com/shap-for-time-series-event-detection-5b4d9d0f96f4/\#:\~:text=Feature%20importance%20is%20a%20widespread,feature%20contributes%20to%20the%20prediction.](https://towardsdatascience.com/shap-for-time-series-event-detection-5b4d9d0f96f4/#:~:text=Feature%20importance%20is%20a%20widespread,feature%20contributes%20to%20the%20prediction.)  
40. SHAP Values vs Feature Importance | by Amit Yadav | Biased-Algorithms | Medium, 12月 13, 2025にアクセス、 [https://medium.com/biased-algorithms/shap-values-vs-feature-importance-ba6b91c16319](https://medium.com/biased-algorithms/shap-values-vs-feature-importance-ba6b91c16319)  
41. Volatility Target Optimization \- Python \- Quantitative Finance Stack Exchange, 12月 13, 2025にアクセス、 [https://quant.stackexchange.com/questions/38329/volatility-target-optimization-python](https://quant.stackexchange.com/questions/38329/volatility-target-optimization-python)  
42. Volatility target \- Read the Docs, 12月 13, 2025にアクセス、 [https://kundan-reads.readthedocs.io/en/latest/finance/risk\_management/volatility\_target/](https://kundan-reads.readthedocs.io/en/latest/finance/risk_management/volatility_target/)  
43. An Introduction to Volatility Targeting \- QuantPedia, 12月 13, 2025にアクセス、 [https://quantpedia.com/an-introduction-to-volatility-targeting/](https://quantpedia.com/an-introduction-to-volatility-targeting/)  
44. Mean-variance and hierarchical risk parity: An empirical study of large-cap stock portfolios \- CFE \- Columbia University, 12月 13, 2025にアクセス、 [https://cfe.columbia.edu/sites/cfe.columbia.edu/files/content/Posters/2025/Mean-variance%20and%20hierarchical%20risk%20parity%20An%20empirical%20study%20of%20large-cap%20stock%20portfolios.pdf](https://cfe.columbia.edu/sites/cfe.columbia.edu/files/content/Posters/2025/Mean-variance%20and%20hierarchical%20risk%20parity%20An%20empirical%20study%20of%20large-cap%20stock%20portfolios.pdf)  
45. TRADITIONAL VS. AI-DRIVEN PORTFOLIO OPTIMIZATION: WHICH MODEL WINS? | by Engin Sorhun | Medium, 12月 13, 2025にアクセス、 [https://medium.com/@enginsorhun/traditional-vs-ai-driven-portfolio-optimization-which-model-wins-348fdd136677](https://medium.com/@enginsorhun/traditional-vs-ai-driven-portfolio-optimization-which-model-wins-348fdd136677)  
46. PyPortfolioOpt/cookbook/5-Hierarchical-Risk-Parity.ipynb at main \- GitHub, 12月 13, 2025にアクセス、 [https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/5-Hierarchical-Risk-Parity.ipynb](https://github.com/robertmartin8/PyPortfolioOpt/blob/master/cookbook/5-Hierarchical-Risk-Parity.ipynb)  
47. Installation — PyPortfolioOpt 1.5.4 documentation, 12月 13, 2025にアクセス、 [https://pyportfolioopt.readthedocs.io/](https://pyportfolioopt.readthedocs.io/)  
48. Chandelier Exit \- Strategy (TradingView) \- 81 Backtests \- TradeSearcher, 12月 13, 2025にアクセス、 [https://tradesearcher.ai/strategies/1731-chandelier-exit-strategy](https://tradesearcher.ai/strategies/1731-chandelier-exit-strategy)  
49. SystemTrader \- Testing a Mean-Reversion System with the Chandelier Exit (SPY, QQQ, IJR) \- RSI(5) \- StockCharts.com, 12月 13, 2025にアクセス、 [https://articles.stockcharts.com/article/articles-arthurhill-2016-12-systemtrader---testing-a-mean-reverion-system-with-the-chandelier-exit-spy-qqq-ijr---rsi5/](https://articles.stockcharts.com/article/articles-arthurhill-2016-12-systemtrader---testing-a-mean-reverion-system-with-the-chandelier-exit-spy-qqq-ijr---rsi5/)  
50. Zweig Breadth Thrust Indicator Trading Strategy- How To Measure Market Breadth \- QuantifiedStrategies.com, 12月 13, 2025にアクセス、 [https://www.quantifiedstrategies.com/zweig-breadth-thrust-indicator-strategy/](https://www.quantifiedstrategies.com/zweig-breadth-thrust-indicator-strategy/)  
51. Outstanding Margin Trading, etc. | Japan Exchange Group \- JPX, 12月 13, 2025にアクセス、 [https://www.jpx.co.jp/english/markets/statistics-equities/margin/index.html](https://www.jpx.co.jp/english/markets/statistics-equities/margin/index.html)