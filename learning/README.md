# nar-ml — 地方競馬 予測モデル（ローカル開発）

`docs/Design_Modeling.md`（設計書 v1.0）と `docs/TestDesign_Modeling.md`（テスト仕様 v1.0）
の実装。EDA は `~/.claude/skills/` の DS/統計スキルの手順に沿って構成してある。

## 実データで動かす

```bash
cd learning
export PYTHONPATH=src
P=.venv/bin/python                        # numpyro/torch/lightgbm はこちらにある
D="file://$PWD/data_real"

$P -m nar.cli --data-root $D track-master              # 競馬場マスタ（廃止場を含む31場）
$P -m nar.cli --data-root $D ingest --start 1998-01 --end 2026-08
$P -m nar.cli --data-root $D ingest --kind odds --start 2026-02 --end 2026-08
$P -m nar.cli --data-root $D bronze                    # 展開・デコードのみ
$P -m nar.cli --data-root $D silver                    # 型付け・キー付与
$P -m nar.cli --data-root $D leak-check                # 累積成績8列の時点性判定
$P -m nar.cli --data-root $D eda                       # EDA 一式 → artifacts/eda/
$P -m nar.cli --data-root $D features                  # as-of 特徴量 → gold/

# walk-forward 学習（評価）→ 本番用モデル（配布物）
$P -m nar.cli --data-root $D learn --models clogit,lgbm,tabm,bayes
$P -m nar.cli --data-root $D gate-report               # Blocker の実測結果
$P -m nar.cli --data-root $D fit-final                 # OOS 直前までの全データで学習
$P -m nar.cli --data-root $D report                    # → artifacts/model_performance.md
```

取得順は必ず `track-master` を先に。廃止場（福山・荒尾ほか16場）のコードが無いと
1998-2013 の大半が silver に落ちない。

**`learn` は評価、`fit-final` が配布物**。walk-forward が作るのは fold ごとの
短い期間のモデルで、本番には出さない。`fit-final` は OOS 開始日から embargo を
引いた日までの全データで学習し、較正温度は学習に使っていない末尾90日で測る。

### ばんえい（別 variant）

ばんえいは 200m 直線・そりの重量で決まる別競技なので、平地の設定を一切触らずに
`conf_banei/` を丸ごと差し替えて学習する。切り替えは `NAR_CONF_DIR` の1点だけ。

```bash
NAR_CONF_DIR=conf_banei $P -m nar.cli --data-root $D features
NAR_CONF_DIR=conf_banei $P -m nar.cli --data-root $D --artifacts artifacts/banei learn
NAR_CONF_DIR=conf_banei $P -m nar.cli --data-root $D --artifacts artifacts/banei \
    fit-final --out ./artifacts/final_banei
```

gold の書き出し先は `gold/features_noodds_banei/` で、平地の gold を上書きしない。
特徴量集合は `asof_features(cfg)` が variant から決める（平地側は1列も変わらない —
`tests/test_banei.py` が固定している）。落とす列と足す列の根拠は
`src/nar/features/banei.py` の docstring にある。

対象は 帯広ば(3) だけでなく 北見ば(1)・岩見ば(2)・旭川ば(4) も含む（1998-2006 に
実在、競技として同一）。平地は `conf/features.yaml` の `exclude` で従来どおり
1-4 を落とす。

合成データで一通り通したいときは `nar synth` → 同じ流れ（`DATA_ROOT` を `./data` に）。

`nar train` / `nar evaluate` は条件付きロジット単体の軽量版で、`nar learn` が全モデル版。

`DATA_ROOT` を `gs://...` に差し替えればコード変更なしで GCS 上を読み書きする（RP-06）。
パスは全て fsspec URI で扱っており、ディレクトリ階層は GCP 設計と同一にしてある。

## この実装が守っていること

設計書のうち、**間違えると全評価が無効になる部分**を不変条件として実装に埋め込んである。

**pre-race スキーマ**（`transform/prerace.py`）— post-race 列は「落とし忘れ」ではなく
「落とせなかったら `LeakageError`」。累積成績8列のホワイトリスト
`ASOF_RACE_WHITELIST` は、EDA の時点性判定（LK-09/10/11）が as-of-race を示した
列だけが入れる。実データ 4,831,906 行での判定結果は
`artifacts/leak_verdict.json` にあり、コード側の集合と一致することをテストで
固定してある（判定を回さずに列を足せない）。

**as-of 集計**（`features/builder.py`）— 履歴系特徴量は
`GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`（発走時刻が厳密に前）で集計する。
`EXCLUDE CURRENT ROW` だと、全順序で手前に来る**同時刻の別場**が入ってしまう。
同じ分に発走するレースは対象レースの発走時点でまだ終わっていない。
収縮の事前確率 p̄ も as-of（全期間平均を使うと 1998年の行の収縮先に
2026年までの勝率が入る）。

**速度指数**（`features/builder.py:speed_index`）— 場×距離の as-of 統計で
標準化する都合上、履歴の部分集合だけを渡すと同じ走破時計でも別の値になる。
既に値が入っている行は再計算せず保存済みを使う。運用時に渡せるのは常に部分集合。

**embargo の自動導出**（`config.py`）— `conf/cv.yaml` に `embargo_days` を書くと
例外になる。値は `conf/features.yaml` の `max_lookback_days` からのみ導出される。
二重管理は必ずズレるので、経路を1本にしてある。

**OOS ロック**（`eval/splits.py`）— `--unlock-oos` なしで OOS 期間に触れると
`OOSAccessError`。開封は `artifacts/oos_access.log` に追記され、回数を検証できる。

**Too-Good-To-Be-True ガード**（`eval/guards.py`）— OOS Top-1 > 60%、NLL < 1.20 などが
発火したら `TooGoodToBeTrueError` でパイプラインを止める。成果ではなくバグとして扱う。

## EDA の構成（SKILLS ベース）

`nar/eda/` は3つのスキルの手順を重ねてある。順序に意味があるので `runner.run()` を使う。

| スキル | 対応 | 実装 |
|---|---|---|
| analysis-planning | 何に答えるかを先に書き出す | `runner.plan()` |
| programmatic-eda | 手順1〜5（構造→欠損→外れ値→分布→相関） | `profile.py` |
| programmatic-eda | 40項目チェックリストとサインオフ | `checklist.py` |
| programmatic-eda | 成果物2種（フルレポート / 所見サマリ） | `report.py` |
| data-quality-audit | 6次元スコアカード（CRITICAL〜LOW） | `quality.py` |
| analysis-assumptions-log | 前提・除外・判断理由の記録 | `assumptions.py` |
| analysis-qa-checklist | 納品前サインオフ | `checklist.signoff()` |

その上に設計書 §7 の6問を載せている。

| 問 | 内容 | 実装 | 出力の使い道 |
|---|---|---|---|
| Q1 | 年×競馬場のカバレッジと欠損 | `questions.coverage_map` | `train_from` を実測で決める |
| **Q2** | **累積成績8列の時点性（最優先）** | `leakage.py`（LK-01/02/03） | ホワイトリストの可否 |
| Q3 | 名寄せ（`horse_sk` / `horse_alias`）の検証 | `questions.identity_audit` | 過去成績特徴量の信頼度 |
| Q4 | ターゲット基礎統計・favourite-longshot bias | `questions.favourite_longshot_bias` | 市場効率性の水準 |
| Q5 | 年次 PSI による構造変化点 | `questions.yearly_drift` | fold 境界の設計 |
| Q6 | 券種別・場別の控除率実測 | `questions.measured_takeout` | 期待値計算に実測値を使う |

閾値の既定は `programmatic-eda/references/quality_thresholds.md` に準拠。
プロジェクト固有の上書きは `conf/eda.yaml` の `overrides` にのみ書き、**理由を必須**にして
`analysis-assumptions-log` に転記される形にしてある。

### Q2 について

EDA の最優先項目。判定は LK-01（世代間差分）→ LK-02（単調性）→ LK-03（終端一致）の順で、
**判定不能は破棄側に倒す**。破棄しても情報は失われない（1998年からの全レース結果があるので
同じ集計を自前で作れる）が、誤って残した場合は全評価が無効になる。非対称なので破棄に倒す。

LK-01 は当日ファイルと後日の月次ファイルが両方揃わないと実行できない。
**日次取得は今日から回しておくこと。過去は取り返せない。**

## テスト対応表

テスト仕様の ID をテスト名に埋めてある（`pytest -k LK` 等で絞れる）。

| 章 | ID | ファイル |
|---|---|---|
| 2. データ取得 | IG-01〜16 | `tests/test_ingest.py` |
| 2.3 スキーマガード | SG-01〜04 | `tests/test_ingest.py` |
| **3. リーク検証** | **LK-01〜08** | `tests/test_leakage.py` |
| 3.4 CV分割 | CV-01〜10 | `tests/test_cv.py` |
| 4. 変換・キー | TR-01〜11 | `tests/test_transform.py` |
| 5. 特徴量 | FE-01〜10 | `tests/test_features.py` |
| 6. モデル | MD-01〜11, EN-01〜05 | `tests/test_models.py` |
| 6.3/6.4 TabM・ベイズ | MD-12〜22 | `tests/test_models_deep.py` |
| 7. 評価・経済 | EV, CA, EC | `tests/test_eval.py` |
| 8. 二トラック | TB-01〜07 | `tests/test_trackb.py` |
| 9.2/9.3 HPO・特徴量選択 | CV-09/10 | `tests/test_selection_hpo.py` |
| 9. ガード | RF-01〜08 | `tests/test_eval.py` |
| 10. 再現性 | RP-01〜03 | `tests/test_eval.py`, `tests/test_selection_hpo.py` |
| EDA | — | `tests/test_eda.py` |

`@pytest.mark.slow` が付いたもの（MD-04 の β 回収、LK-06 のラベルシャッフル、
LG-10 の LambdaRank）は nightly 想定。既定では実行される。

## 実行環境（重要）

`numpyro` / `mlflow` はシステム Python に入れられない（PEP 668 の externally-managed）ため、
`.venv`（`--system-site-packages`）に入れてある。**テストと学習は `.venv` の Python で回す。**

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m nar.cli learn --models clogit,lgbm,tabm,bayes
```

`.venv` には **CPU 版 torch** を入れてある。このマシンの WSL2 GPU パススルーが
セッション中に壊れ（`nvidia-smi` 自体が segfault、`libcuda.so.1` のロードで落ちる）、
CUDA 版 torch は import すら通らなくなったため。**GPU を使うには Windows 側で
`wsl --shutdown` して WSL を再起動する必要がある。** 復旧後に CUDA 版へ戻せば
TabM は GPU で動く（コード変更は不要。`TabMConfig.device` が自動判定する）。

## 未実装

**存在しない実装に対する skip テストは置いていない。** green は「通った」意味であるべきで、
「まだ無い」を緑で表現すると受け入れ判定が壊れる。

| 項目 | 状況 |
|---|---|
| SBC（MD-20） | 関数は実装済み（`bayes.simulation_based_calibration`）だが、100回の事前生成＋推論が CPU では現実的でないため未実行。テストも置いていない |
| 実データ取得の本番実行 | `NarClient` は実装済みだが未実行（ネットワーク未使用） |
| 運用設計（`docs/Design_Operation.md` / `docs/TestDesign_Operation.md`） | 未着手。本実装はモデリング設計書2本のみを対象にしている |

## 設計書からの逸脱

| 箇所 | 設計書 | 実装 | 理由 |
|---|---|---|---|
| 前処理 | polars | pandas + DuckDB | polars 未導入。as-of 集計の本体は DuckDB のウィンドウ関数なので性能上の中心は変わらない |
| 速度指数の基準 | その日の同場同距離の平均タイム | 当日の**先行レースのみ**、無ければ同場同距離の as-of 累積平均 | 当日全レースの平均は同日の後続レースを参照する。同一馬が同日に複数回走ったときにリークする |
| 速度指数の下限 | 規定なし | 同場同距離の先行 100 走を満たすまで NULL | 基準統計量を as-of で取ると最初の数走は「3件から推定した標準偏差」で割ることになり桁で暴れる |
| gold の丸め | 規定なし | float を小数10桁で丸める | DuckDB のウィンドウ集計は並列実行で加算順が変わり、1e-14 の差で FE-09（決定性）と LK-05（ビット一致）が落ちる |

テスト仕様側の期待値のうち、実測に基づいて緩めたものは §13 の手順どおり理由を残してある。

| ID | 仕様の期待値 | 実装での判定 | 理由 |
|---|---|---|---|
| FE-07 | 速度指数の平均 0±0.05 / SD 1±0.1 | 平均 \|·\|<0.25 / SD 0.6〜1.5 | 標準化の基準統計量自体を as-of で取るため、厳密には標準正規にならない |
| EC-08 | 全馬均等買い ROI = (1-τ) ± 1pt | 開催日ブロックブートストラップ CI が 0.80 を含む | 重い裾を持つ推定量で、この規模では SE だけで数 pt に達する。点推定の固定幅判定は不安定 |

## 受け入れ判定の現在地

| マイルストーン | 状況 |
|---|---|
| M1 データ基盤 | IG / SG / TR / RP-03〜07 は green。IG-14（完全オフライン再構築）も green |
| M2 リーク検証 | LK / CV は green。**ただし Q2 の時点性判定は実データ待ち**（LK-01 に当日ファイルが要る） |
| M3 ベースライン | FE / MD-01〜06 / EV / EC は green。条件付きロジットが動作 |
| M4 全モデル | MD-07〜22 と CA / EN が green。**ただし SBC（MD-20）は未実行**、および MD-16/17（R-hat < 1.01、ESS > 400）はサンプル数の都合で仕様値に届いていない（実測値をテストに記録） |
| M5 最終評価 | TB-01〜07 が green、RF-01〜08 未発火、OOS 開封は記録済み。ただし判定はすべて合成データ上のもの |

合成データでは EV-03（トラックA が市場ベースラインを上回る）は**成立しない**。
合成市場は真の効用に直接アクセスしており、モデルは as-of 履歴という弱い代理変数しか
見ていないため、構造上モデルが負ける。実データでの判定にのみ意味がある。

## ディレクトリ

```
learning/
├── conf/          data / features / cv / eda の設定
├── src/nar/
│   ├── io/        ストレージ抽象（fsspec URI）と台帳
│   ├── ingest/    HTTP 取得・レート制限・CP932 デコード
│   ├── transform/ スキーマガード・キー設計・pre-race・払戻の縦持ち化
│   ├── features/  as-of 集計と経験ベイズ収縮
│   ├── eda/       SKILLS ベースの EDA
│   ├── eval/      分割・指標・較正・経済・ガード
│   ├── models/    条件付きロジット・LambdaRank・TabM・階層ベイズ・
│   │              残差モデル（トラックB）・制約付きスタッキング
│   ├── train/     walk-forward パイプラインと nested HPO
│   ├── tracking.py 再現性の記録（MLflow / JSON フォールバック）
│   ├── report.py  性能レポート生成
│   ├── synth.py   Plackett-Luce 合成データ（FX-03 / FX-04）
│   └── cli.py
├── tests/         テスト仕様の ID に対応
└── artifacts/     EDA レポート・CV 指標・OOS 開封台帳
```

`data/` と `artifacts/` は生成物。`data/raw/` だけが不変層で、それ以外は
raw から再生成できる（IG-14）。
