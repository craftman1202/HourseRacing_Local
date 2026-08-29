# nar-ops — 地方競馬 予測システム 運用パイプライン

`docs/Design_Operation.md`（オペレーション設計書 v1.0）と
`docs/TestDesign_Operation.md`（検証テスト仕様書 v1.0）の実装。

学習は `learning/`（ローカル・`nar` パッケージ）、推論と配信はここ（`narops`）。

## GCP（sample-335613）の状態

**作成済み**（2026-08-26）。`asia-northeast1`。

| 種別 | 名前 | 備考 |
|---|---|---|
| BigQuery | `nar_ops`（12テーブル） | 大きい6テーブルは日次パーティション＋`require_partition_filter` |
| GCS | `nar-raw-sample-335613` | 30日 Nearline / 365日 Coldline。公開防止＋UBLA |
| GCS | `nar-model-sample-335613` | 90日 Nearline。同上 |
| SA | `nar-ops` / `nar-api` / `nar-web` | 設計どおりの最小権限のみ |
| Cloud Tasks | `nar-queue` | 最大3回・30〜270秒バックオフ |
| Secret Manager | `DISCORD_WEBHOOK_{PREDICTION,ALERT,DAILY}` | asia-northeast1 に配置 |

実 GCP の状態は `pytest tests/test_gcp_live.py -m costly` で検証できる（11件・読み取りのみ／課金なし）。

## 実データで動かす手順

```bash
cd operation
export PYTHONPATH=src:../learning/src

# 1. 確定層に月次データを入れる（学習側の silver から）
python -m narops.cli --db data/nar_ops.duckdb load-history \
    --silver ../learning/data_real/silver

# 2. 評価用モデルを作り、OOS で測る（開封は1回だけ）
cd ../learning
python -m nar.cli --data-root file://$PWD/data_real gate-report
python -m nar.cli --data-root file://$PWD/data_real fit-final
python -m nar.cli --data-root file://$PWD/data_real evaluate-final --unlock-oos \
    --reason "配布モデルの最終評価"

# 3. 出荷用モデルを全データで作り直す（評価が済んだら OOS 期間も使う）
python -m nar.cli --data-root file://$PWD/data_real fit-final \
    --through 2026-08-26 --out ./artifacts/final_prod

# 4. リリースを組み立て、GCS へ上げる
cd ../operation
python -m narops.cli publish-release --release-id v2026.08.28-A \
    --final-dir ../learning/artifacts/final_prod \
    --eval-final-dir ../learning/artifacts/final \
    --artifacts ../learning/artifacts
python -m narops.cli upload-release --release-id v2026.08.28-A --set-current

# 5. 状態確認
python -m narops.cli health
```

**`fit-final` は評価用と出荷用で2回走らせる**。`--through` を付けない既定は
OOS 境界の手前で打ち切った評価用で、これを OOS で測る。測り終えたら
`--through` に最新データ日を渡して全データで作り直したものを配る。
順序を逆にすると、健全性チェックが「学習日が古い」と警告する（実際にした）。

`load-history` は速度指数を**全履歴**で計算してから投入する。`--since` は
「投入する行」を絞るだけで、計算範囲は絞らない。1か月分だけで計算すると
場×距離の基準統計量が揃わず、ほぼ全行 NaN になって正しい値を上書きする。

確定層の `start_ts` は **UTC の naive**（DB-10）。NAR のファイルは JST の naive
なので、投入時に必ず変換する。ずれると `start_ts < 発走時刻` の比較で
**同日の先行レースが丸ごと履歴から落ちる**。`assert_utc_start_ts` が分布で
検出して止める。
BQ スキーマは `infra/bq/*.json` にあり、**`db/schema.py` の DDL から生成**しているので
ローカルと本番が乖離しない。

### 稼働中（2026-08-28）

| 種別 | 名前 | 状態 |
|---|---|---|
| Artifact Registry | `nar-ops` | `nar-ops:v2026.08.28e` |
| Cloud Run | `nar-ops` / `nar-api` / `nar-web` | min-instances=0、shadow 固定 |
| Cloud Scheduler | 3ジョブ | 02:40 取込 / 08:00 当日計画 / 月曜04:00 週次 |
| GCS モデル | `v2026.08.28-A` | current 設定済み |

`/health` `/plan-day` `/ingest-and-refresh` `/weekly-report` は実 BigQuery
に対して 200 を返すことを確認済み。

```bash
# ビルドとデプロイ（コンテキストはリポジトリのルート）
IMG=asia-northeast1-docker.pkg.dev/sample-335613/nar-ops/nar-ops:vYYYY.MM.DD
docker build -f operation/Dockerfile -t $IMG . && docker push $IMG
python -m narops.cli deploy --image $IMG \
    --service-url https://nar-ops-bwl3vrgzdq-an.a.run.app --apply --confirm
```

### 実 GCP でしか出なかった落とし穴

ローカルのテストが全部通っていても、次は本番で初めて出た。同種の再発は
テストで固定してある（`tests/test_deploy.py`）。

- **2つのバックエンドでインターフェースが違った** — `wh.con` / `register` /
  `execute` は DuckDB 実装にしか無く、BigQuery 構成では属性ごと存在しない。
  書き込みは `insert_frame`、読み取りは `query` に寄せた。
- **`allow_full_scan` は BigQuery では逃げ道にならない** —
  `require_partition_filter` は実行そのものを拒否する。免除をやめた。
- **パーティションは1テーブル 10,000 個が上限** — 1998年からの日次は
  10,210 個で投入が拒否される。履歴を持つ表は月次にする。
- **IAM は「データ権限」と「実行権限」が別** — `bigquery.dataEditor` だけでは
  クエリを投げられない（`bigquery.jobUser` が要る）。`storage.objectCreator`
  は書き込み専用で、配布物を読むには `objectViewer` が要る。
  Scheduler → Cloud Run は**サービス単位**の `run.invoker` が要る。
- **実行時にしか要らない依存は忘れる** — `beautifulsoup4`（出馬表の解析）と
  `db-dtypes`（BigQuery の DATE 列変換）。

### このプロジェクト固有の注意

共用プロジェクトで、既存システム（`chx-*` / `jra-*` / `hourse-racing-webui` /
`finml-*`、および**別の競馬システム** `hourse_racing` データセットと
`hourse-racing-inference-sample-335613` バケット）が稼働している。

- 本システムは `nar_ops` / `nar-*` 名前空間のみを作成・変更する。
  `gcp.assert_owned()` が書き込み経路の入口で強制する。
- **Cloud Scheduler の無料枠（請求先アカウント単位で3ジョブ）は既存17ジョブで
  消費済み**。本システムの3ジョブは全額課金対象（+$0.30/月）。
- **Cloud Run の無料枠も既存7サービスと共有**。設計書 §6.1 の「合計 $0.5〜1.1」は
  本システム単独の前提なので、実際の増分はこれより上振れる。
- **CO-05（組織ポリシーで Vertex AI / Cloud SQL を作成不可）は適用していない**。
  両 API は既存ワークロードが有効化済みで、プロジェクト単位で禁止すると他システムが
  壊れる。運用規約としての deny 一覧に留め、実防御は予算アラートで代替する。
  この逸脱は `test_co05_high_cost_services_are_listed_as_denied` に状態ごと固定してある。

## 動かす

```bash
cd operation
P=../learning/.venv/bin/python          # numpyro/mlflow 入りの venv を共用

$P -m pytest tests/ -q                  # 199 件（costly は既定で除外）
PYTHONPATH=src:../learning/src $P -m narops.cli init
PYTHONPATH=src:../learning/src $P -m narops.cli health
PYTHONPATH=src:../learning/src $P -m narops.cli estimate-cost
PYTHONPATH=src:../learning/src $P -m narops.cli check-secrets --env .env
```

`.env` には実際の webhook URL が入るので `.gitignore` 済み。雛形は `.env.example`。
`check-secrets` は解決可否だけを表示し、**値は必ず伏せる**。

## この実装が守っていること

設計のうち「静かに間違った金額を賭ける」経路に直結する部分を、不変条件として
コードに埋め込んである。テスト仕様が Blocker に指定した項目がこれに当たる。

**学習コードの共有**（`shared.py`）— 期待値・Kelly・自己インパクト補正・`to_prerace`
は運用側に**再実装を持たない**。`nar` の実体を re-export するだけで、
`skew.assert_no_duplicate_implementation()` が関数オブジェクトの同一性を検証する。
TypeScript で再実装すると Web と Discord の数値が乖離するので、API も Python のまま
OpenAPI から型を生成する方針にしてある（SK-02）。

**二層 DB と as-of**（`db/`）— 確定層とライブ層の UNION を `start_ts < 発走時刻` で
厳密に絞る。発走後のレコードを注入しても特徴量がビット単位で変わらないことを
テストで固定している（DB-03）。

**鮮度ゲート**（`db/freshness.py`）— 確定層が前日分に届かなければ推論しない。
古い履歴で計算した特徴量で賭けるのは「推論が出ない」よりはるかに悪い（DB-04）。

**skew 監視**（`skew.py`）— 前日分の snapshot と確定層のみからの再計算を突き合わせ、
許容されるのは日内更新に依存する3系統だけ。それ以外の乖離は配信を止める（SK-01/03）。

**ゼロ埋め禁止**（`inference.py`）— オッズ欠損時はトラックA 単独へ縮退する。
埋めて推論を続けると学習時の分布から外れる（IN-05）。

**確率の総和**（`inference.py`）— レース内総和が 1±1e-9 でなければ例外を投げて
配信を止める。間違った推論を配信するより、届かないほうがまし（IN-01）。

**秘密情報**（`config.py` / `discord/client.py`）— webhook URL は `repr`・例外文・
ログのいずれにも出さない。既知の値の置換だけでなく URL 形状の正規表現でも潰す（DC-01）。

**課金ガード**（`db/backend.py`）— `maximum_bytes_billed` 未設定のクエリは実行できず、
パーティションフィルタの無いクエリは例外になる（CO-01 / DB-09）。

## テスト対応表

テスト仕様の ID をテスト名に埋めてある（`pytest -k SK` 等で絞れる）。

| 章 | ID | ファイル | 件数 |
|---|---|---|---|
| 2. モデル配布 | MP-01〜08 | `tests/test_model_artifacts.py` | 26 |
| 3. DB 更新戦略 | DB-01〜10 | `tests/test_db.py` | 20 |
| 4. 学習/推論 skew | SK-01〜07 | `tests/test_skew.py` | 16 |
| 5. 推論エンドポイント | IN-01〜11 | `tests/test_inference.py` | 26 |
| 6. スケジューリング | SC-01〜07 | `tests/test_scheduling.py` | 19 |
| 7. Discord 配信 | DC-01〜10 | `tests/test_discord.py` | 31 |
| 9. コスト・監視 | CO-01〜05 / MO-01〜05 | `tests/test_cost_monitoring.py` | 22 |
| 10. リリース・障害注入 | RL-01〜06 / DR-01〜06 | `tests/test_release_dr.py` | 34 |
| 8. Web API 契約 | WB-01〜07 / AU-01〜02 | `tests/test_release_dr.py` | （同上） |
| 結合 | — | `tests/test_end_to_end.py` | 4 |
| 実 GCP 検証 | DB-09 / CO-03 / RL-06 / DC-02 | `tests/test_gcp_live.py` | 11（`-m costly`） |

マーカーは仕様 §1.3 どおり `unit` / `component` / `integration` / `e2e` /
`synthetic_monitor` / `slow` / `costly`。既定で走るのは L0/L1/L2 相当のみで、
実 GCP・実 Discord に打つものは `costly` でオプトインする。

## 外部依存の代替実装

仕様 §1.3 の表に対応させてある。本番へは各モジュールの1関数を差し替えるだけで移る。

| 依存 | L1 代替（実装済み） | 本番 |
|---|---|---|
| BigQuery | DuckDB（`db/backend.py`）。パーティション必須と bytes_billed を実際に強制 | BigQuery |
| Cloud Tasks | インプロセスキュー（`tasks.py`）。タスク名の一意制約を再現 | Cloud Tasks |
| GCS | ローカル FS（`model/registry.py`）。ディレクトリ構造は `gs://nar-model/` と同一 | GCS |
| Discord | `httpx.MockTransport`（レスポンスコード注入可） | 実 webhook |
| Secret Manager | `.env`（`config.py`） | Secret Manager |
| Scheduler | 静的検証（`scheduling.validate_scheduler_jobs`） | Cloud Scheduler（未作成） |

## 未実装

**存在しない実装に対する skip テストは置いていない。**

| 項目 | 状況 |
|---|---|
| Web フロントエンド（Next.js 16 / 画面5種） | 未実装。API 契約（`api.py`）と認可・課金の境界のみ実装。UI 操作は仕様どおり Playwright に委譲する想定で、そのケースも未作成 |
| ONNX 変換の実処理 | 等価性検証（`model/onnx_check.py`）は実装済みだが、TabM → ONNX の変換自体は未実装 |
| NAR HTTP 取得（当日ファイル・スケジュール） | 学習側 `nar.ingest` の `NarClient` を流用する前提。運用側からの呼び出しは未配線 |
| Terraform 本体 | `infra/services.json` に設計値を置き、CO-02 / RL-06 がそれを検査する。実資源は gcloud で作成済みなので、`.tf` を書く際は `terraform import` が必要 |
| L3/L4（実 Cloud Run での E2E・合成監視） | 未実施。Cloud Run 未デプロイのため |
| 収支確定・`pnl_daily` の集計ジョブ | テーブル定義と Discord 表示は実装済み。日次集計処理は未実装 |

## 受け入れ判定（DoD）の現在地

| ゲート | 必要 ID | 状況 |
|---|---|---|
| G1 シャドー運用開始 | MP / DB / SK / SC / DR / CO / RL-02,04,06 | L0/L1 は green。実 BigQuery でも DB-09 / CO-03 / RL-06 を確認済み。**Cloud Run 未デプロイのため未達** |
| G2 ペーパートレード | G1 ＋ IN / DC / WB / MO | IN・DC・MO は green。WB は API 契約のみで画面が無いため未達 |
| G3 小額実運用 | G2 ＋ SLO の30日連続 GREEN | 未達（運用実績が無い） |
| G4 本格運用 | G3 ＋ 90日 SLO | 未達 |

## 設計書からの逸脱

| 箇所 | 設計書 | 実装 | 理由 |
|---|---|---|---|
| 前処理・DWH | BigQuery | ローカル検証は DuckDB、本番は BigQuery（`gcp.BigQueryWarehouse`） | 制約（パーティション必須・bytes_billed）は両方で同じチェックを通す。実 BigQuery でもフィルタ無しクエリが拒否されることを `test_gcp_live.py` で確認済み |
| タイムスタンプ | 記載なし | **DB 内の naive timestamp は UTC** という規約を `db/types.py` で強制 | DuckDB / BigQuery のドライバが tz-aware 値をローカル時刻へ変換して tz を落とすため。往復で JST-naive が残ると as-of フィルタが 9 時間ずれる |
| プール総額 | `nar.pool_size_model` テーブル | `PoolSizeModel`（既定値 + 安全係数 0.7） | 過去実績が無いため。保守的に小さく見積もり、期待値を過大評価しない側へ倒す方針は設計どおり |
