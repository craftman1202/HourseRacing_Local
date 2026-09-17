`scikit-learn` の公式ドキュメント（前処理ユーザーガイド、スケーラー比較例、スケーリングの重要性の例、common pitfalls、SVM ガイド）と Anthropic の Agent Skills 仕様・オーサリング指針を実際に読んだうえで、そこに書かれている内容だけを根拠に構成しました。以下がそのまま配置できるファイル一式です。

ディレクトリ構成（参照は SKILL.md から一階層のみ。公式ベストプラクティスが「ネストした参照は部分読みされる」と警告しているため）:

```
scaling-numeric-features/
├── SKILL.md
├── reference/model-requirements.md
├── reference/scaler-catalog.md
├── reference/distribution-transforms.md
├── reference/pitfalls-and-leakage.md
└── scripts/profile_features.py
```

---

## SKILL.md

````markdown
---
name: scaling-numeric-features
description: Decide whether and how to scale or transform numeric features (StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer, Normalizer, log/target/cyclical transforms) for a given model family and data distribution. 数値特徴量のスケーリング・変換の要否判断と選択。Use when the user builds a preprocessing pipeline, asks "should I standardize/normalize this", tunes a model that converges poorly, has features on wildly different scales, skewed or heavy-tailed distributions, outliers, or sparse matrices.
---

# 数値特徴量のスケーリングと分布変換

## 大原則

スケーリングは「とりあえずかける」処理ではない。**必要性はモデル族が決め、種類はデータの分布が決める**。
この2軸を順に確定させてから実装する。

さらに、スケーリングは無料ではない。scikit-learn のドキュメントは、
**低スケール側の変数が非予測的（ノイズ）だった場合、スケーリング後にそれらの寄与が相対的に大きくなり
過学習が増えて性能が下がりうる**と明示している。だから常に「無変換」をベースラインとして残す。

## ワークフロー

複雑なケースでは以下をチェックリストとして応答にコピーし、進捗を消し込むこと。

- [ ] 1. モデル族を確定し、スケーリングの要否を判定（`reference/model-requirements.md`）
- [ ] 2. train/test を分割済みか確認（分割前に統計量を計算しない）
- [ ] 3. `scripts/profile_features.py` で列プロファイルを取得
- [ ] 4. プロファイルに基づき列ごとに変換を選択（`reference/scaler-catalog.md`）
- [ ] 5. 分布そのものを変える必要があるか判断（`reference/distribution-transforms.md`）
- [ ] 6. `ColumnTransformer` + `Pipeline` に組み込む（生の fit_transform を書かない）
- [ ] 7. 検証ループ：変換後に再プロファイル＋無変換ベースラインとの CV 比較
- [ ] 8. リーク要因の最終チェック（`reference/pitfalls-and-leakage.md`）

## ステップ1：要否の一次判定

```
モデルは決定木ベースのみ（DecisionTree / RandomForest / HistGradientBoosting 等）か？
├─ YES → 単調なスケーリングは不要。ここで終了してよい。
│         scikit-learn は決定木ベースの推定器を「任意のスケーリングに対してロバストな
│         注目すべき例外」と位置づけている。
│         ただし「分布変換」「周期特徴量」「カテゴリ処理」は別問題なので
│         reference/distribution-transforms.md は引き続き参照する。
│
└─ NO → 距離・内積・カーネル・正則化・勾配のいずれかを使うモデル。
         スケーリングは原則必要。ステップ2へ。
         scikit-learn は metric-based / gradient-based 推定器が
         「おおよそ標準化されたデータ（中心化＋単位分散）」を仮定することが多く、
         スケーリングされていないデータは勾配ベース推定器の収束を
         遅らせる、あるいは妨げることさえあると述べている。
```

判断に迷ったら「そのモデルの損失・距離計算に、特徴量の絶対値がそのまま入るか」を問う。
入るなら必要。分割点の大小比較しか使わないなら不要。

## ステップ2：変換の種類を選ぶ

`profile_features.py` の出力を上から順に当てはめる。最初に合致したものを採用する
（複数候補を並べて迷わない）。

| 列の性質（プロファイル指標） | 選ぶ変換 | 理由 |
|---|---|---|
| ゼロ率が高い／疎行列（CSR/CSC） | `MaxAbsScaler` | 疎データ用に設計されており、ゼロを保存する。中心化は疎性を壊すので禁止 |
| 外れ値が多い（MAD 基準で 1% 超） | `RobustScaler` | 中心・尺度を分位点で推定するため、少数の極端値に引きずられない |
| 強い歪度（\|skew\| ≥ 1）かつ正規性が欲しい | `PowerTransformer` | 分散安定化と歪度最小化。負値があれば Yeo-Johnson、厳密に正なら Box-Cox 可 |
| 外れ値が極端で、順位情報だけで十分 | `QuantileTransformer` | 順位変換。ただし相関と距離を歪めるので後述の注意を読む |
| 上記に該当せず、分布がおおむね素直 | `StandardScaler` | 既定の第一候補 |
| 出力レンジを [0,1] に固定したい要件がある | `MinMaxScaler` | ただし外れ値に非常に敏感 |
| サンプル単位の向きだけが意味を持つ（テキストの BoW/TF-IDF 等） | `Normalizer` | 列単位ではなく行単位で単位ノルム化する別カテゴリの処理 |

各変換の詳細な挙動・落とし穴・API 引数は `reference/scaler-catalog.md` を読むこと。

## ステップ3：変換すべきでない・避けるべきケース

以下に当てはまるときは、スケーリングを外すか変更する。

- **決定木ベースのモデルしか使わない**。効果がなく、パイプラインと解釈を複雑にするだけ。
- **疎行列を中心化しようとしている**。`StandardScaler` は `with_mean=False` を明示しない限り
  `ValueError` を出す。黙って中心化すると疎性が壊れ、メモリを過剰に確保して実行が落ちることがある。
- **低スケールの列がノイズだと分かっている**。スケーリングでノイズの寄与が増幅され、過学習が悪化しうる。
  この疑いがあるときは、無変換との CV 比較を必ず行う。
- **係数を元の単位で解釈する必要がある**。この場合は変換するか否かを意識的にトレードオフし、
  変換するなら逆変換して報告する。
- **すでに同一単位・同一レンジの特徴量群**（例：ピクセル値、同一センサの多チャンネル）。
  相対関係を保ちたいなら列ごとではなく全体で一つの尺度を使う判断もありうる。
- **`QuantileTransformer` を、特徴量間の相関や距離構造が本質的なタスクに使う**。
  この変換は「特徴量内および特徴量間の相関と距離を歪める」と明記されている。

## ステップ4：実装の型

必ず `Pipeline` / `ColumnTransformer` の中に入れる。`fit` は学習データのみ、
`transform` は学習・テスト両方、というルールを構造的に強制するため。

```python
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

pre = ColumnTransformer(
    transformers=[
        ("num", RobustScaler(), make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist"),
         make_column_selector(dtype_include=[object, "category"])),
    ],
    verbose_feature_names_out=False,
)
model = Pipeline([("pre", pre), ("est", estimator)])
```

列ごとに異なるスケーラーを当てる場合も、`ColumnTransformer` の transformer を増やして表現する。
`remainder` に推定器（例：`MinMaxScaler()`）を渡せば、残り列の既定処理も宣言的に書ける。

## ステップ5：検証ループ（省略しない）

1. 変換後のデータに対して `scripts/profile_features.py --validate` を再実行する。
2. 意図した性質が達成されたか確認する（歪度が縮んだか、列間の尺度差が縮んだか、
   外れ値の扱いが狙い通りか）。
3. 未達なら変換を変えて 1 に戻る。
4. 達成したら、**無変換パイプラインと同じ CV 分割で**スコアを比較する。
   改善していなければ変換を採用しない。

冪変換については、ある分布には非常にガウス的な結果を与える一方、別の分布にはまったく効かない
ことがあるため、変換前後の可視化が重要である、とドキュメントが明示している。数値指標だけで
判断せず、ヒストグラムを1枚描くこと。

## 目的変数（y）の変換

特徴量ではなく y が強く歪んでいる回帰では `TransformedTargetRegressor` を使う。
予測は自動で逆変換されて元のスケールに戻る。

```python
from sklearn.compose import TransformedTargetRegressor
regr = TransformedTargetRegressor(regressor=estimator, func=np.log, inverse_func=np.exp)
```

scikit-learn の例では、対数正規的に歪んだ y に対し
`QuantileTransformer(output_distribution="normal")` を噛ませることで
線形回帰の R² が 0.64 から 0.67 に改善している。効果は問題依存なので必ず比較する。

## 必要なパッケージ

`scikit-learn`, `numpy`, `pandas`（`profile_features.py` が使用）。
実行環境にこれらが入っていることを確認してから `scripts/profile_features.py` を実行する。
````

---

## reference/model-requirements.md

````markdown
# モデル族ごとのスケーリング要件

## 目次
- 1. 不要なモデル
- 2. 必須のモデル
- 3. 収束・数値安定性のために推奨されるモデル
- 4. 具体的な影響の大きさ（実測例）
- 5. 出典が触れていない事項

## 1. 不要なモデル

決定木ベースの推定器（`DecisionTree*`, `RandomForest*`, `HistGradientBoosting*` など）。
scikit-learn はスケーラー比較の解説で、多くの推定器が「各特徴量がゼロ付近の値を取ること、
より重要には全特徴量が比較可能な尺度で変動すること」を前提に設計されている一方、
**決定木ベースの推定器は任意のスケーリングに対してロバストな注目すべき例外**だと述べている。

補足：これらのモデルに対しては、単調なスケーリングだけでなく単調な分布変換（log など）も
効果が薄い。分割点の位置が写像されるだけだからである。代わりに投資すべきは、
周期性のエンコード、カテゴリの扱い、交互作用や集約特徴量の設計。

## 2. 必須のモデル

### SVM（`SVC`, `SVR`, `LinearSVC` ほか）
公式ガイドが「SVM アルゴリズムはスケール不変ではないので、**データをスケーリングすることを強く推奨する**」
と明言している。推奨される具体策として、入力ベクトル X の各属性を [0,1] または [-1,+1] に
スケーリングするか、平均0・分散1に標準化することを挙げ、
**同じスケーリングをテストベクトルにも適用しなければ意味のある結果は得られない**と注意している。
実装上は `make_pipeline(StandardScaler(), SVC())` が推奨形。

### 距離ベース（k-NN、k-means などの近傍・クラスタリング）
Wine データセットで `proline`（0〜1000 程度）と `hue`（1〜10 程度）の2特徴量を使った例では、
スケーリングの有無で **KNN の決定境界が完全に別のモデルになる**。距離が proline の差に
ほぼ支配され、hue は相対的に無視されるためである。標準化すると両者が概ね -3〜3 に収まり、
近傍構造に対する寄与が均等化する。

### 正則化つき線形モデル（Ridge / Lasso / ElasticNet / 正則化ロジスティック回帰）
前処理ユーザーガイドは、学習アルゴリズムの目的関数に使われる多くの要素
（SVM の RBF カーネル、線形モデルの L1/L2 正則化項など）が、
**全特徴量がゼロ中心であること、または分散が同オーダーであることを仮定している**と述べる。
ある特徴量の分散が桁違いに大きいと目的関数を支配し、他の特徴量から学習できなくなる。

### PCA およびそれを含むパイプライン
PCA は分散を最大化する方向を探すため、単に尺度が大きいという理由で変動が大きい特徴量があると、
その特徴量が主成分の方向を支配する。Wine データでは、スケーリングなしだと proline が
他より約2桁大きい重みで第一主成分を支配する。

なお、特徴量を個別に中心化・スケーリングするだけでは不十分で、下流モデルが特徴量の
線形独立性を仮定する場合がある。その場合は `PCA(whiten=True)` で特徴量間の線形相関も除ける。

## 3. 収束・数値安定性のために推奨されるモデル

勾配ベースの推定器全般。スケーラー比較の解説は、**スケーリングされていないデータは
多くの勾配ベース推定器の収束を遅らせる、あるいは妨げることさえある**と述べている。
また、正則化なしロジスティック回帰のような例について「収束を容易にするため」に
正規化が要求されると説明している。

## 4. 具体的な影響の大きさ（実測例）

scikit-learn の "Importance of Feature Scaling" 例（Wine データ、PCA 2成分 + `LogisticRegressionCV`）:

| 条件 | テスト精度 | log-loss | 選ばれた最適 C |
|---|---|---|---|
| スケーリングなし + PCA | 35.19% | 1.18 | 0.0000 |
| 標準化あり + PCA | 96.30% | 0.0739 | 6.16 |

スケーリングしない場合は必要な正則化量が大きくなる（C が小さくなる）点も重要。
つまり**スケーリングの欠如はハイパーパラメータ探索の結果まで歪める**。

## 5. 出典が触れていない事項

本スキルが参照した scikit-learn ドキュメントは、ナイーブベイズや一部のモデルについて
スケーリング要否を明示していない。該当モデルを扱う場合は「不要」と決めつけず、
無変換と変換ありを CV で比較して経験的に判断すること。
````

---

## reference/scaler-catalog.md

````markdown
# スケーラー・カタログ

## 目次
- StandardScaler
- MinMaxScaler
- MaxAbsScaler
- RobustScaler
- QuantileTransformer
- PowerTransformer
- Normalizer（行単位、他とは別カテゴリ）
- 疎データでの選択
- 外れ値耐性の比較まとめ

---

## StandardScaler

平均を引き、標準偏差で割る（非定数特徴量のみ）。既定の第一候補。

- `with_mean=False` / `with_std=False` で中心化・尺度化を個別に無効化できる。
- 学習データで `fit` し、テストデータに同じ変換を再適用するための Transformer API を持つ。
- **弱点**：外れ値が平均と標準偏差の推定に影響する。California Housing の例では、
  外れ値の大きさが特徴量ごとに異なるため、変換後の広がりが特徴量間で大きく食い違い、
  片方は概ね [-2, 4]、もう片方は [-0.2, 0.2] に圧縮された。
  **外れ値があると、StandardScaler は特徴量スケールの均衡を保証できない。**

## MinMaxScaler

`X_std = (X - X.min()) / (X.max() - X.min())`、`X_scaled = X_std * (max - min) + min`。
`feature_range` で出力レンジを指定できる。

- 動機：非常に小さい標準偏差に対するロバスト性、および疎データでゼロ要素を保つこと。
- **弱点**：外れ値に非常に敏感。California Housing の例では、全体を [0,1] に収める結果、
  内側のデータ（inlier）が [0, 0.005] という極端に狭い範囲に圧縮された。
- テストデータは学習時のレンジを外れうる（例では変換後に -1.5 や 1.67 が出ている）。
  レンジ固定を保証したいなら別途クリッピングを検討する。

## MaxAbsScaler

各特徴量を最大絶対値で割り、学習データが [-1, 1] に収まるようにする。

- 正値のみなら [0,1]、負値のみなら [-1,0]、混在なら [-1,1]。
- ゼロ中心のデータや**疎データ向け**。正値のみのデータでは MinMaxScaler と似た挙動になる。
- **弱点**：大きな外れ値の影響を受ける点は MinMaxScaler と同様。

## RobustScaler

中心と尺度を分位点（既定は IQR、`quantile_range` で指定）で推定する。

- 少数の非常に大きな外れ値に影響されないため、変換後の各特徴量の値域が
  互いに概ね揃う（California Housing の例では両特徴量とも大半が [-2, 3] に収まった）。
- **注意**：外れ値自体は変換後も残る。外れ値のクリップまで欲しいなら非線形変換が必要。
- 疎入力に `fit` はできないが、`transform` は疎入力に適用できる。

## QuantileTransformer

累積分布関数に基づく非線形変換 `G^-1(F(X))`。`output_distribution` は `"uniform"`（既定）
または `"normal"`。

- 順位変換なので特徴量内の順序は保たれ、異常な分布を滑らかにし、スケーリング手法より
  外れ値の影響を受けにくい。
- **代償**：特徴量内および特徴量間の**相関と距離を歪める**。
- 外れ値を事前に定義された範囲の境界（0 と 1）に折り畳むため、極端な値に**飽和アーティファクト**が出る。
- `output_distribution="normal"` の場合、無限大にならないよう入力の最小・最大が
  1e-7 と 1-1e-7 分位に対応するようクリップされる。
- 学習集合に外れ値を足したり引いたりしても変換がほぼ変わらない、という意味で RobustScaler と同様に頑健。

## PowerTransformer

最尤法で λ を推定する冪変換。分散安定化と歪度最小化を狙う。

- `method="yeo-johnson"`（既定）：負値を含んでよい。
- `method="box-cox"`：**厳密に正のデータにのみ適用可能**。
- 既定で `standardize=True`（変換後にゼロ平均・単位分散化）。切りたいなら明示的に `False`。
- **効き方は分布依存**。ある分布には非常にガウス的な結果を与える一方、別の分布にはまったく
  効かない。ドキュメントは「**変換前後でデータを可視化することの重要性を示している**」と述べている。

## Normalizer（行単位）

これだけ**サンプル単位**の処理で、各行を単位ノルム（`l1` / `l2` / `max`）に正規化する。
列単位のスケーリングとは目的が違うので混同しないこと。

- 内積やカーネルでサンプル間類似度を測る場合に有用。テキスト分類・クラスタリングの
  ベクトル空間モデルが典型例。
- `fit` は何もしない（ステートレス）。ただし Pipeline に入れられる。
- L2 正規化は spatial sign preprocessing とも呼ばれる。
- 密配列・`scipy.sparse` の両方を受け付ける。

## 疎データでの選択

- 疎データの**中心化は疎構造を破壊する**ため、ほぼ妥当ではない。
  一方、特徴量ごとの尺度が違うなら尺度合わせには意味がある。
- **推奨は `MaxAbsScaler`**（疎データ用に設計されている）。
- `StandardScaler` も `scipy.sparse` を受け付けるが、`with_mean=False` を明示しないと
  `ValueError` になる。黙って中心化すると疎性が壊れ、意図せず大量のメモリを確保して
  実行がクラッシュしうるため。
- スケーラーは CSR / CSC を受け付け、それ以外の疎形式は CSR に変換される。
  無駄なメモリコピーを避けるため、上流で CSR か CSC を選んでおくこと。

## 外れ値耐性の比較まとめ

| 変換 | 外れ値への耐性 | 外れ値は残るか | 備考 |
|---|---|---|---|
| StandardScaler | 低（平均・標準偏差が影響を受ける） | 残る | 特徴量間の尺度均衡を保証できない |
| MinMaxScaler | 低 | 残る | inlier が極端に圧縮されうる |
| MaxAbsScaler | 低 | 残る | 疎データ向け |
| RobustScaler | 高（分位点ベース） | 残る | クリップが要るなら非線形変換へ |
| QuantileTransformer | 高 | 境界に折り畳まれる | 相関・距離を歪める、飽和あり |
| PowerTransformer | 中（分布依存） | 縮む場合がある | 可視化必須 |
````

---

## reference/distribution-transforms.md

````markdown
# 分布・構造の変換（スケーリング以外）

## 目次
- 1. 単調変換と順位保存
- 2. 周期的特徴量（時刻・曜日・月）
- 3. ビン化
- 4. 高カーディナリティのカテゴリ
- 5. 目的変数の変換

## 1. 単調変換と順位保存

分位変換と冪変換はいずれも単調変換であり、**各特徴量内の値の順位を保存する**。
順位を保存するということは、決定木ベースのモデルにとってはほぼ無意味な変換である、
ということでもある（分割点が写像されるだけ）。単調変換に投資する価値があるのは、
線形性・正規性・分散一定を仮定する側のモデル。

## 2. 周期的特徴量（時刻・曜日・月）

Bike Sharing Demand（時間別需要）での比較。CV は `TimeSeriesSplit`、指標は MAE（最大需要に対する比）。

| 特徴量設計 + モデル | MAE |
|---|---|
| 生の時間特徴量 + `HistGradientBoostingRegressor` | 0.044 |
| MinMax スケールした序数時刻 + `RidgeCV` | 0.142 |
| 時刻を one-hot + `RidgeCV` | 0.099 |
| sin/cos 三角関数エンコード + `RidgeCV` | 0.125 |
| 周期スプライン（`SplineTransformer(extrapolation="periodic")`）+ `RidgeCV` | 0.097 |

読み取るべき教訓：

- **決定木ベースは時間特徴量を素のまま渡してよい**。序数入力と目的変数の非単調な関係を
  学習できるため。この例では前処理ほぼゼロで最良だった。
- **線形モデルには非線形項を人手で作る必要がある**。生の hour をそのまま入れると、
  朝6時→8時の増加と夕方18時→20時の増加を区別できず性能が出ない。
- one-hot は柔軟だが特徴量数が爆発する（分を使うと 24 → 1440 になる）。
  その場合は `KBinsDiscretizer` で水準数を減らしてから one-hot する手がある。
- **周期スプラインは one-hot と同等の精度をより少ない特徴量で達成した**
  （hour は 24 水準に対し n_splines=12）。`extrapolation="periodic"` により
  深夜を跨いでも滑らかに繋がる。
- sin/cos はこの例では one-hot・スプラインに劣った。安価だが万能ではない。

## 3. ビン化

細かい序数・数値変数の水準数を落としつつ、one-hot の非単調な表現力を得たいときは
`KBinsDiscretizer` を使う。one-hot による過学習リスクを抑える目的に有効。

## 4. 高カーディナリティのカテゴリ

- **低カーディナリティ**：`OneHotEncoder`。未知カテゴリには `handle_unknown="infrequent_if_exist"`
  を使うと、カテゴリを手で列挙するより堅牢なことが多い。
  非正則化の線形回帰など共線性が問題になる場合は `drop="first"`、2値だけ落とすなら `drop="if_binary"`。
- **低頻度カテゴリ**：`OneHotEncoder` / `OrdinalEncoder` の `min_frequency`・`max_categories` で
  まとめられる。
- **高カーディナリティ**：`TargetEncoder`。カテゴリ別の目的変数平均を全体平均に向けて縮小した値で
  エンコードする。`smooth="auto"` は経験ベイズ推定を使い、大きい `smooth` ほど全体平均寄りになる。
  **重要**：学習データには必ず `fit_transform` を使うこと。内部の cross fitting により
  目的変数リークと下流モデルの過学習を防ぐ設計になっており、`fit(X, y).transform(X)` は
  `fit_transform(X, y)` と一致しない。公式にも `fit` の単独使用はリークを招くため非推奨とある。
  未知カテゴリは全体平均 `target_mean_` でエンコードされる。
- `OrdinalEncoder` を数値特徴量として扱うと、順序が恣意的になるため予測性能が下がりやすい、
  と `TargetEncoder` のドキュメントが指摘している。

## 5. 目的変数の変換

`TransformedTargetRegressor` は y を変換してから回帰し、予測を逆変換して元の空間に戻す。
`transformer` を渡すか、`func` / `inverse_func` の関数ペアを渡す（両方同時は不可）。

既定では毎回 `fit` 時に両者が互いの逆関数かチェックされる。`check_inverse=False` で
無効化できるが、誤った逆関数を渡すと R² が -3.02 まで壊れる例がドキュメントにある。
チェックは基本的に有効のままにすること。
````

---

## reference/pitfalls-and-leakage.md

````markdown
# 前処理の落とし穴とリーク対策

## 目次
- 1. 変換の適用漏れ
- 2. データリーク
- 3. リークを避ける手順
- 4. 再現性と random_state

## 1. 変換の適用漏れ

学習データだけスケーリングしてテストデータに適用し忘れると、特徴量空間が変わりモデルは機能しない。
scikit-learn の例では、同じデータで MSE が 62.80 対 0.90 になった。

正しい対処は、学習と同じ Transformer で `transform` すること。推奨はそもそも `Pipeline` を使い、
変換を忘れる可能性を構造的に潰すこと。

## 2. データリーク

リークとは、**予測時には利用できない情報**をモデル構築に使ってしまうこと。CV スコアが
楽観的に出て、本番で性能が落ちる。

**一般則：テストデータに対して `fit` を決して呼ばない。**

学習・テストの両方に同じ前処理を適用する必要はあるが、その**変換は学習データからのみ学習**
されねばならない。平均で割る正規化なら、その平均は学習サブセットの平均であって
全データの平均ではない。

リークの具体例（特徴量選択）：完全にランダムな X と y（本来の精度は 0.5 相当）で、
全データに `SelectKBest` を適用してから分割すると精度 0.76 が出てしまう。
分割後に学習データだけで選択すると 0.5 に戻る。

ドキュメントは、このリークが `StandardScaler`、`SimpleImputer`、`PCA` を含む
**ほぼすべての変換に当てはまる**と注意している。

## 3. リークを避ける手順

1. **前処理より先に** train / test に分割する。
2. `fit` / `fit_transform` にテストデータを含めない。`fit(X)` のように全データを使うと
   楽観的なスコアになる。
3. `transform` は学習・テスト両方に適用する（学習に `fit_transform`、テストに `transform`）。
4. `Pipeline` を使う。正しいサブセットに正しいメソッドが適用されることを保証し、
   CV やハイパーパラメータ探索と組み合わせるのに最適。
5. `ColumnTransformer` を `Pipeline` の中で使う。pandas で事前に前処理すると、
   (a) テストデータの統計量が前処理器に混入して CV が信頼できなくなる、
   (b) 前処理器のパラメータを探索対象にできない、という問題が起きる。

## 4. 再現性と random_state

- 整数を渡すと、`fit` / `split` を何度呼んでも同じ結果になる。
- `None` や `RandomState` インスタンスを渡すと呼ぶたびに結果が変わる。
- **CV スプリッターには整数を渡すのが最も安全**で、たいていのユースケースを
  カバーする（fold 間の比較可能性が保たれる）。
- 推定器側は `RandomState` インスタンスを渡すと fold ごとに RNG が変わり、
  乱数初期化に対して頑健な CV 結果が得られるため望ましい場合がある。
- 実行間で完全に再現したいなら `random_state=None` の使用をすべて排除する。
  プログラム冒頭で `rng` を宣言して渡し回すのが推奨。
  なお `np.random.seed(0)` によるグローバルシード設定は推奨されていない。
````

---

## scripts/profile_features.py

```python
#!/usr/bin/env python3
"""数値列をプロファイルし、スケーリング/変換の候補を提案する。

使い方:
    python scripts/profile_features.py data.csv
    python scripts/profile_features.py data.parquet --exclude target id
    python scripts/profile_features.py transformed.csv --validate

出力は列ごとの診断行と、推奨変換およびその理由。
判定はあくまで出発点であり、最終判断は SKILL.md のワークフロー
(無変換ベースラインとの CV 比較) で行うこと。
"""

import argparse
import sys

import numpy as np
import pandas as pd

# --- 判定しきい値。すべて根拠つきで調整可能にしてある --------------------------
# 正規分布なら歪度は0。対数正規など右裾の長い分布では顕著に正になるため、
# |skew| >= 1 を「分布変換を検討する水準の歪み」とみなす。
SKEW_THRESHOLD = 1.0

# 外れ値は MAD ベースで判定する。正規分布では 1.4826 * MAD が標準偏差の一致推定量に
# なるので、中央値から 3 * 1.4826 * MAD を超える点は正規分布下で約0.27%しか出ない。
# 実測割合が 1% を超えるなら「正規分布から明確に逸脱した裾」と判断し、
# 分位点ベースの RobustScaler を候補にする。
MAD_SIGMA_FACTOR = 1.4826
OUTLIER_SIGMA = 3.0
OUTLIER_SHARE_THRESHOLD = 0.01

# range / IQR は裾の重さの指標。正規分布では IQR ≈ 1.349σ で、
# n=1e4 でも range はおよそ 7.7σ なので比は約 5.7、n=1e6 でも約 7.3 にしかならない。
# 20 を超える場合は正規分布では説明しにくい重い裾があると判断する。
RANGE_IQR_THRESHOLD = 20.0

# ゼロが過半を占める列は実質的に疎。中心化すると疎性を壊すため MaxAbsScaler を優先する。
SPARSE_ZERO_SHARE = 0.5

# 列間の標準偏差比。距離・正則化・PCA 系のモデルでは、この比が大きいほど
# スケーリング未実施の悪影響が大きい。1桁違えば実務上「無視できない差」とみなす。
SCALE_DISPARITY_THRESHOLD = 10.0

# 整数かつ水準数が少ない列は、連続量ではなくカテゴリ/序数の可能性が高い。
LOW_CARDINALITY_MAX = 20


def load_table(path):
    """CSV / Parquet を読み込む。失敗時は原因を明示して終了する。"""
    lower = path.lower()
    try:
        if lower.endswith((".parquet", ".pq")):
            return pd.read_parquet(path)
        if lower.endswith((".csv", ".csv.gz", ".tsv")):
            sep = "\t" if lower.endswith(".tsv") else ","
            return pd.read_csv(path, sep=sep)
    except FileNotFoundError:
        sys.exit(f"ERROR: ファイルが見つかりません: {path}")
    except Exception as exc:  # 読み込み失敗の原因をそのまま提示する
        sys.exit(f"ERROR: {path} の読み込みに失敗しました: {exc}")
    sys.exit(
        f"ERROR: 未対応の拡張子です: {path}\n"
        "対応形式: .csv, .csv.gz, .tsv, .parquet, .pq"
    )


def profile_column(s):
    """1列分の診断指標を返す。欠損は除外して計算する。"""
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    n_total = len(s)
    if x.size == 0:
        return {"n_valid": 0, "missing_share": 1.0}

    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    mad = np.median(np.abs(x - med))
    robust_sigma = MAD_SIGMA_FACTOR * mad
    if robust_sigma > 0:
        outlier_share = float(
            np.mean(np.abs(x - med) > OUTLIER_SIGMA * robust_sigma)
        )
    else:
        # MAD が 0 = 過半数が同一値。中央値と異なる値をすべて外れ値扱いすると
        # 誤解を招くため、判定不能として NaN を返す。
        outlier_share = float("nan")

    value_range = float(x.max() - x.min())
    range_over_iqr = value_range / iqr if iqr > 0 else float("inf")

    return {
        "n_valid": int(x.size),
        "missing_share": 1.0 - x.size / n_total if n_total else 0.0,
        "n_unique": int(np.unique(x).size),
        "is_integer": bool(np.all(np.equal(np.mod(x, 1), 0))),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "median": float(med),
        "iqr": float(iqr),
        "skew": float(pd.Series(x).skew()) if x.size > 2 else 0.0,
        "zero_share": float(np.mean(x == 0)),
        "all_positive": bool(x.min() > 0),
        "outlier_share": outlier_share,
        "range_over_iqr": float(range_over_iqr),
    }


def recommend(p):
    """プロファイルから推奨変換と理由を決める。最初に合致した規則を採用する。"""
    if p.get("n_valid", 0) == 0:
        return "SKIP", "有効な数値がありません（全欠損または非数値）"
    if p["n_unique"] == 1:
        return "DROP", "定数列。情報量がなく、標準化すると0除算相当になります"
    if p["n_unique"] == 2:
        return "NONE", "2値列。スケーリング不要（必要なら 0/1 に符号化）"
    if p["is_integer"] and p["n_unique"] <= LOW_CARDINALITY_MAX:
        return (
            "REVIEW",
            f"整数かつ水準数 {p['n_unique']}。連続量ではなくカテゴリ/序数の可能性。"
            "one-hot か周期エンコードを検討してください",
        )
    if p["zero_share"] >= SPARSE_ZERO_SHARE:
        return (
            "MaxAbsScaler",
            f"ゼロ率 {p['zero_share']:.0%}。疎性を壊さないため中心化は避けます",
        )

    outlier = p["outlier_share"]
    heavy = p["range_over_iqr"] > RANGE_IQR_THRESHOLD
    if (outlier == outlier and outlier > OUTLIER_SHARE_THRESHOLD) or heavy:
        detail = []
        if outlier == outlier:
            detail.append(f"MAD基準の外れ値 {outlier:.1%}")
        if heavy:
            detail.append(f"range/IQR={p['range_over_iqr']:.1f}（重い裾）")
        if abs(p["skew"]) >= SKEW_THRESHOLD:
            method = "box-cox" if p["all_positive"] else "yeo-johnson"
            return (
                f"PowerTransformer(method='{method}') または RobustScaler",
                "、".join(detail)
                + f"、かつ歪度 {p['skew']:.2f}。正規性が要るなら冪変換、"
                "順位と外れ値をそのまま残すなら RobustScaler",
            )
        return "RobustScaler", "、".join(detail) + "。分位点ベースの尺度が安全です"

    if abs(p["skew"]) >= SKEW_THRESHOLD:
        method = "box-cox" if p["all_positive"] else "yeo-johnson"
        return (
            f"PowerTransformer(method='{method}')",
            f"歪度 {p['skew']:.2f}。"
            + ("厳密に正なので Box-Cox が使えます" if p["all_positive"]
               else "非正値を含むため Yeo-Johnson を使います"),
        )

    return "StandardScaler", "外れ値・強い歪みなし。既定の標準化で十分です"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="CSV / TSV / Parquet ファイル")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="除外する列名（目的変数や ID）")
    ap.add_argument("--validate", action="store_true",
                    help="変換後データの検証モード。残存する問題のみ報告する")
    args = ap.parse_args()

    df = load_table(args.path)
    num = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in args.exclude if c in df.columns], errors="ignore"
    )
    if num.shape[1] == 0:
        sys.exit(
            "ERROR: 数値列が見つかりません。"
            f"検出された列: {list(df.columns)[:20]}"
        )

    profiles = {c: profile_column(num[c]) for c in num.columns}

    print(f"rows={len(df)}  numeric_columns={num.shape[1]}\n")
    header = f"{'column':<24}{'skew':>8}{'outlier%':>10}{'rng/IQR':>10}{'zero%':>8}  recommendation"
    print(header)
    print("-" * len(header))
    for c, p in profiles.items():
        rec, why = recommend(p)
        if args.validate and rec in ("StandardScaler", "NONE", "SKIP"):
            continue
        ol = p.get("outlier_share", float("nan"))
        print(
            f"{c[:23]:<24}"
            f"{p.get('skew', float('nan')):>8.2f}"
            f"{(ol * 100 if ol == ol else float('nan')):>10.1f}"
            f"{p.get('range_over_iqr', float('nan')):>10.1f}"
            f"{p.get('zero_share', 0) * 100:>8.1f}"
            f"  {rec}"
        )
        print(f"{'':<24}    └ {why}")

    stds = [p["std"] for p in profiles.values()
            if p.get("n_valid", 0) > 1 and p.get("std", 0) > 0]
    if len(stds) >= 2:
        disparity = max(stds) / min(stds)
        print(f"\n列間の標準偏差比 (max/min) = {disparity:.1f}")
        if disparity > SCALE_DISPARITY_THRESHOLD:
            print(
                "  → 尺度差が大きいです。距離ベース・カーネル・正則化線形・PCA・勾配法を"
                "使うならスケーリングは必須と考えてください。"
                "決定木ベースのみなら無視して構いません。"
            )
        else:
            print(
                "  → 尺度差は限定的です。スケーリングの効果は小さい可能性があるため、"
                "無変換ベースラインとの CV 比較を必ず行ってください。"
            )

    if args.validate:
        print("\n検証モード: 上に何も出ていなければ、しきい値上の問題は解消しています。")
        print("最後に変換前後のヒストグラムを目視で確認してください"
              "（冪変換は分布によって効かないことがあります）。")


if __name__ == "__main__":
    main()
```

---

いくつか補足しておきます。

`name` はベストプラクティスが推奨する動名詞形に寄せました。`description` に日本語と英語のキーワードを両方入れているのは、description がスキル選択の唯一の手がかりであり「何をするか」と「いつ使うか」の両方を含める必要があるためです。

公式指針は SKILL.md 本体を500行以内に保ち、詳細は参照ファイルへ逃がすこと、参照は SKILL.md から一階層のみにすること（入れ子にすると Claude が `head -100` などで部分読みして情報が欠ける）、100行を超える参照ファイルには目次を付けることを求めているので、その形に従っています。

意図的に入れた設計判断が一つあります。カタログの選択表で「最初に合致したものを採用する」と書いたのは、ベストプラクティスが「必要でない限り複数の選択肢を提示しない」ことをアンチパターンとして挙げているためです。逆に、要否判断のステップは高い自由度（テキストの指針）にしてあります。ここは文脈依存で複数の正解がある領域だからです。

一点、出典の限界も明示しておきます。参照した scikit-learn ドキュメントは決定木系・SVM・k-NN・PCA・正則化線形モデルについては明確に述べていますが、ナイーブベイズなど一部のモデルのスケーリング要否には触れていません。そこは「不明」として扱い、経験的に比較するよう reference に書きました。LightGBM 等の個別実装についても一次情報を確認できなかったので、scikit-learn の「決定木ベースの推定器」という一般的な記述に留めています。

最初に作るべき評価3件としては、(1) 外れ値の多い住宅価格系データで RobustScaler が選ばれるか、(2) 疎な TF-IDF 行列で中心化を回避できるか、(3) 決定木のみのパイプラインで「スケーリング不要」と正しく打ち切れるか、あたりが素直だと思います。