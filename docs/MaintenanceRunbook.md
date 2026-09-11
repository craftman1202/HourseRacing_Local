# 定期メンテナンス手順書 v1.0

`docs/Design_Modeling.md` / `docs/Design_Operation.md` の実装が既に持っている手順・
閾値・カレンダーを、**作業単位（データ再収集／学習／モデル選択／評価／デプロイ／テスト）**で
横断的にまとめ直したもの。個々の根拠・設計判断は上記2文書と `learning/README.md` /
`operation/README.md` にあるので、ここでは「いつ・どこで・何を叩くか」だけに絞る。

対象システムは平地モデルとばんえいモデルの2系統（`learning/README.md` の該当節参照）。
コマンド例は平地を基本形とし、ばんえい差分がある箇所だけ明示する。

## 0. 全体カレンダー

| 頻度 | 作業 | 実行方式 |
|---|---|---|
| 毎日 02:40 JST | 月次ファイル取り込み → 確定層 MERGE → skew 検証 | 自動（Cloud Scheduler → `POST /ingest-and-refresh`） |
| 毎日 08:00 JST | 当日スケジュール取得 → 推論タスク積み込み | 自動（Cloud Scheduler → `POST /plan-day`） |
| 毎日（コード変更のたび） | unit / component テスト | 自動（CI 相当、全 push） |
| 日次〜PRマージ時 | integration テスト | 手動 or CI |
| 毎週月曜 04:00 JST | モデル鮮度チェック・RF ガード・週次レポート | 自動（Cloud Scheduler → `POST /weekly-report`） |
| リリース前・週次 | e2e テスト（実 GCP・実 Discord） | 手動、`-m e2e` |
| 常時 | 合成監視（synthetic_monitor） | 自動 |
| 四半期ごと（基本） | 再学習 → 評価 → （指標が上回れば）デプロイ | 手動 |
| 劣化検知時（随時） | 再学習 → 評価 → デプロイ | 手動、週次レポートのアラートが起点 |
| 必要になったとき | costly テスト（実 GCP・実課金） | 手動オプトイン、`-m costly` |

再学習の頻度が「四半期ごとを基本、劣化検知時は随時」であることは `docs/Design_Operation.md`
に明記されている（週次の ECE 劣化・PSI 監視が劣化検知のトリガー）。

## 1. データ再収集

### 1.1 自動（日次、本番）

`POST /ingest-and-refresh`（02:40 JST）が当月・前月分の月次ファイルを再取得し、確定層へ
MERGE する。手動で叩く必要は通常ない。障害時のみ手動再実行を検討する。

### 1.2 手動（ローカル開発・バックフィル・スキーマドリフト対応時）

```bash
cd learning
export PYTHONPATH=src
P=.venv/bin/python
D="file://$PWD/data_real"

$P -m nar.cli --data-root $D track-master              # 場コードマスタ（廃止場含む）を先に
$P -m nar.cli --data-root $D ingest --start 1998-01 --end $(date +%Y-%m)
$P -m nar.cli --data-root $D ingest --kind odds --start 2026-03 --end $(date +%Y-%m)
$P -m nar.cli --data-root $D bronze
$P -m nar.cli --data-root $D silver
$P -m nar.cli --data-root $D leak-check                # Q2: 累積成績8列の時点性を都度確認
```

- `track-master` を必ず先に実行する。廃止場（福山・荒尾ほか16場）のコードが無いと
  1998-2013 の大半が silver に落ちない。
- 日次取得（当日オッズ・当日ファイル）は止めない。過去のオッズは取り返せない
  （`learning/README.md`）。取得が数日でも止まると、その期間の当日ファイルベースの
  検証（LK-01 の世代間差分など）ができなくなる。
- 月次ファイルは「月末から45日以上経過し、直近取得で `sha256` が一致」した月のみ
  再取得対象から外れる（`is_final`）。スキーマドリフト（`status='schema_drift'`）が
  出たら silver への昇格は自動停止するので、`meta/schema_hash.json` の期待値を
  見直してから再実行する。

### 1.3 本番確定層への反映（学習側 → 運用側）

```bash
cd operation
export PYTHONPATH=src:../learning/src
python -m narops.cli load-history --silver ../learning/data_real/silver
```

`load-history` は速度指数を**全履歴**で計算してから投入する。`--since` は投入行を
絞るだけで、計算範囲は絞らない点に注意（1か月分だけで計算すると場×距離の基準統計量が
揃わず、ほぼ全行 NaN になる）。

### 1.4 確定層への列追加（スキーマ移行）

特徴量を追加して確定層に新しい列が必要になった場合、`bq update` で列を足すだけでは
**既存行が NULL のまま**になり、学習時は値があり推論時だけ NULL という train-serving
skew になる。列追加と既存行の後追い投入は必ずセットで行う。

```bash
cd operation
export PYTHONPATH=src:../learning/src
bq --project_id=sample-335613 update nar_ops.entry_result_final infra/bq/entry_result_final.json
bq --project_id=sample-335613 update nar_ops.entry_result_live  infra/bq/entry_result_live.json
python scripts/backfill_last3f.py --silver ../learning/data_real/silver           # 計画
python scripts/backfill_last3f.py --silver ../learning/data_real/silver --apply   # 実行
```

`load-history` を全期間で流し直す方法は使えない。`db/merge.py::merge_final` の更新経路は
変更行を `(race_id, horse_no)` のタプル列挙で DELETE する実装で、日次の訂正（数千行）を
前提にしている。数百万行の一括更新ではクエリ長上限を超える。列ごとの一度きりの移行は
`scripts/backfill_last3f.py` と同じ形（ステージング表へロード → 1本の UPDATE）で書く。

順序は **列追加 → 既存行の後追い投入 → コンテナのデプロイ → skew 検証 → `promote`**。
特に「後追い投入はコンテナのデプロイより前」を守る。理由は2つある。

1. 逆にすると、新モデルが全行 NULL の特徴量を見ながら推論する時間帯ができる。
2. 新コードを先に入れると、翌朝の `ingest-and-refresh` が「既存行は NULL、投入行は
   値あり」を**訂正**とみなし、当月・前月の全行（十数万行）を DELETE + INSERT しようと
   する。`merge_final` の DELETE はタプル列挙なのでクエリ長上限に当たり、日次更新が
   丸ごと落ちる。先に埋めてあれば差分ゼロで素通りする。

列追加そのものは既存モデルに影響しない。`feature_spec` は**モデルが宣言した列だけ**で
ハッシュを取るので（`features.feature_spec_of` の `names` 引数）、ビルダーが列を増やしても
旧リリースの契約ハッシュは変わらない。実際に本番 `v2026.08.28-A` の
`feature_spec_hash` が新コードでも一致することを確認してから移行すること。

## 2. モデル学習

### 2.1 EDA（学習前に必ず）

```bash
cd learning
$P -m nar.cli --data-root $D eda
```

Q2（累積成績8列の時点性）が最優先。判定不能は破棄側に倒す実装になっている。

### 2.2 特徴量構築 → walk-forward 学習（評価用）

```bash
$P -m nar.cli --data-root $D features
$P -m nar.cli --data-root $D learn --models clogit,lgbm,tabm,bayes
```

`learn` は**評価用**（walk-forward の fold ごとの短期間モデル）。本番配布物はここでは
作らない。出力は `artifacts/oof_predictions.parquet`（キャリブレーション検証などに使う）、
`artifacts/fold_metrics.csv`、`artifacts/ensemble_weights.csv` など。

### 2.3 ばんえい（別モデル系統）

```bash
NAR_CONF_DIR=conf_banei $P -m nar.cli --data-root $D features
NAR_CONF_DIR=conf_banei $P -m nar.cli --data-root $D --artifacts artifacts/banei learn
```

平地の設定・gold は一切触らない。切替は `NAR_CONF_DIR` の1点のみ。

## 3. モデル選択

「モデル選択」は3段階ある。

1. **アルゴリズム間**: `learn` の出力 `fold_metrics.csv` をレース内 NLL（主指標）で比較する
   （`artifacts/model_performance.md` §1）。Top-1/NDCG では較正が改善しないため NLL を優先。
2. **アンサンブル重み**: `learn` が OOF のみから制約付きスタッキング重みを推定し、
   `artifacts/ensemble_weights.csv` に書く。訓練内予測を混ぜていないことは
   `ConstrainedStacker` が例外で強制する。
3. **新モデル vs 現行本番モデル**: `narops promote` の二段確認がこの判断そのもの
   （§6.2 参照）。shadow 実測 NLL/ECE が既存の production NLL/ECE を上回って初めて
   切り替えが通る。

```bash
cd learning
$P -m nar.cli --data-root $D gate-report               # Blocker の実測結果
cat artifacts/gate_report.json                          # can_publish が true か確認
```

## 4. モデル評価

### 4.1 出荷候補の OOS 最終評価（ロック区間、1回だけ開封）

```bash
cd learning
$P -m nar.cli --data-root $D fit-final                              # 評価用（OOS 直前まで）
$P -m nar.cli --data-root $D evaluate-final --unlock-oos \
    --reason "配布モデルの最終評価"
```

- OOS への直接アクセスはコードでロックされている。`--unlock-oos` なしで触れると
  `OOSAccessError`。開封は `artifacts/oos_access.log` に追記され、回数を検証できる。
  **むやみに繰り返さない**（開封回数そのものが評価の健全性の指標）。
- Too-Good-To-Be-True ガード（RF-01〜03、`operation/conf/ops.yaml` の
  `rf_top1_max=0.60` / `rf_nll_min=1.20` / `rf_roi_max=1.30`（3か月継続））が発火したら
  publish しない。良すぎる結果は成果ではなくバグとして扱う。
- 結果は `artifacts/oos_metrics.json` に出る。`順序を逆にしない`
  （`fit-final` の既定 → `evaluate-final` → 出荷用 `fit-final --through` の順を守る。
  逆にすると健全性チェックが「学習日が古い」と警告する）。

### 4.2 較正（キャリブレーション）の確認

`learning/notebooks/win_place_calibration.ipynb` に一着確率と複勝（3着以内）確率の
リライアビリティ図・ECE・年次安定性をまとめてある。四半期ごとの再学習後、および
週次レポートで ECE 劣化アラートが出たときに実行する。

```bash
cd learning
# nbformat/nbclient/ipykernel は .venv に未導入なら追加する（pyproject.toml の
# notebook extra、pyproject 本体の依存とは分離してある）
.venv/bin/pip install --quiet -e ".[notebook]"
.venv/bin/python -m ipykernel install --user --name nar-venv --display-name "Python 3 (nar .venv)"

.venv/bin/python -c "
import nbformat
from nbclient import NotebookClient
path = 'notebooks/win_place_calibration.ipynb'
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=900, kernel_name='nar-venv',
               resources={'metadata': {'path': 'notebooks'}}).execute()
nbformat.write(nb, path)
"
# もしくは VS Code / JupyterLab で「nar-venv」カーネルを選んで開き、再実行する
```

Harville 複勝確率の計算（LightGBM・アンサンブルの2モデル分、実測 約3分/実行）が
律速する。全モデルに広げると1モデルあたり実測 約85秒（61,513 レース）かかるので、
むやみに対象モデルを増やさない。

このノートブックは walk-forward の OOF（2019-2023）を使う。ロック区間（OOS、
2024-02〜現在）そのものの較正を見たい場合は `artifacts/oos_predictions.parquet` を読む。
`evaluate-final` が集計指標（`oos_metrics.json`）と一緒に行単位の予測も書き出すので、
OOF と同じ列構成（`race_id` / `horse_no` / `is_win` / `finish_pos` / モデル名）で扱える。
**OOS の開封は1回きり**なので、較正を見るためだけに再開封しないこと。

### 4.3 本番監視（自動、週次）

`operation/conf/ops.yaml` の閾値:

| 指標 | 閾値 | 意味 |
|---|---|---|
| `model_stale_days` | 90日 | `current` の学習期間末からこれ以上経つと鮮度アラート |
| `ece_degradation` | 0.02 | 30日移動窓の ECE が OOS 評価時より悪化した量 |
| `psi_threshold` | 0.25 | 特徴量分布 PSI（月次計算） |
| `coverage_min` | 0.90 | 推論対象レースに対するカバレッジ |
| `rf_top1_max` / `rf_nll_min` / `rf_roi_max`（3か月） | 0.60 / 1.20 / 1.30 | Too-Good-To-Be-True ガード |

いずれかが発火したら§2〜§6（再学習〜デプロイ）を随時サイクルで回す。

## 5. モデルデプロイ

### 5.1 出荷用モデルの作成とリリース組み立て

```bash
cd learning
$P -m nar.cli --data-root $D fit-final --through $(date +%Y-%m-%d) --out ./artifacts/final_prod

cd ../operation
python -m narops.cli publish-release --release-id vYYYY.MM.DD-A \
    --final-dir ../learning/artifacts/final_prod \
    --eval-final-dir ../learning/artifacts/final \
    --artifacts ../learning/artifacts
python -m narops.cli upload-release --release-id vYYYY.MM.DD-A --set-current
```

`operation/scripts/train_and_publish.py --release-id vYYYY.MM.DD-A` は
`gate-report → fit-final → publish-release` の3ステップを順序保証だけして薄く束ねたもの。
個別に叩くか、このスクリプトを使うかはどちらでもよい（中身は同じ）。

ばんえいは `--family banei` を付け、`bootstrap-current` または `promote` も
`--family banei` で行う（`current_banei.json` が別に管理される）。

### 5.2 本番切り替え（二段確認）

```bash
cd operation
python -m narops.cli promote vYYYY.MM.DD-A --actor <name> --confirm \
    --shadow-days 14 --shadow-nll <shadow実測> --shadow-ece <shadow実測> \
    --production-nll <現行実測> --production-ece <現行実測>
```

shadow 実測が現行の production 実測を上回って初めて通る。初回リリースで `current` が
まだ無いときだけ `bootstrap-current` を使う。

問題が出たら:

```bash
python -m narops.cli rollback --actor <name>
```

### 5.3 コンテナのビルドとデプロイ（Cloud Run）

```bash
IMG=asia-northeast1-docker.pkg.dev/sample-335613/nar-ops/nar-ops:vYYYY.MM.DD
docker build -f operation/Dockerfile -t $IMG . && docker push $IMG
```

**`narops deploy --apply --confirm` は使わない。** 実装が `$IMG` を nar-ops・nar-api・
nar-web の3サービス全部に配ってしまう（per-service イメージ指定が未実装）。nar-api と
nar-web は別 Dockerfile（`operation/Dockerfile.api` / `web/Dockerfile`）由来の別イメージを
本番で動かしているため、このコマンドを使うと api と web が nar-ops のイメージで
再デプロイされ両方落ちる。

nar-ops だけを更新するときは、現在の設定を引き継いで `gcloud run deploy` を直接叩く
（`--set-env-vars` は既存の環境変数を丸ごと置き換えるので、事前に
`gcloud run services describe nar-ops --format=json` で現在値を確認する）。
具体コマンドは `operation/README.md` の該当節を参照。

デプロイ後は必ず:

```bash
python -m narops.cli verify-release          # 配布物のSHA-256一致とfeature_specを確認
python -m narops.cli health                  # DB鮮度・current・学習期間末を確認
curl <service-url>/health                    # model_release / banei_model_release を確認
```

## 6. テスト

### 6.1 マーカーと実行頻度（`operation/pyproject.toml` / `learning/pyproject.toml` に既定済み）

| マーカー | 内容 | 頻度 |
|---|---|---|
| `unit` | 全モック・純関数 | 全 push、60秒以内 |
| `component` | HTTP/GCS/BQ を代替実装 | 全 push、5分以内 |
| `integration` | エミュレータ＋一時 BQ | PR マージ時・日次 |
| `e2e` | 実 GCP・実 Discord | リリース前・週次 |
| `synthetic_monitor` | 本番合成監視 | 常時 |
| `slow`（learning） | モデル正当性・全期間リーク検証 | nightly |
| `costly` | 実課金・実送信を伴う | 明示的オプトインのみ |

### 6.2 実行コマンド

```bash
# learning（199件超、slow込みで既定実行）
cd learning
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m pytest tests/ -k LK -q     # リーク検証だけ絞る、等

# operation（既定で costly を除外）
cd operation
P=../learning/.venv/bin/python
$P -m pytest tests/ -q
$P -m pytest tests/ -m costly -q               # 実 GCP・実課金。月次の健全性確認などで明示実行
```

`costly` は 11件・読み取りのみ（課金なし設計）だが、実 GCP 資源に依存するため
デプロイ直後やモデル差し替え直後に手動で回すのが実務的。

### 6.3 デプロイ前チェックリスト（最低限）

1. `pytest tests/ -q`（learning・operation とも green）
2. `narops check-secrets --env .env`（Discord webhook 等が解決できるか。値は表示しない）
3. `narops estimate-cost`（月額見積が目標以下か）
4. `gate-report` の `can_publish: true`
5. OOS ガード（RF-01〜03）未発火
6. `verify-release` で配布物のハッシュ一致

## 7. 申し送り事項（未対応の改善項目）

- `narops deploy --apply --confirm` は per-service イメージ指定が未実装（§5.3）。
  3サービス構成を Terraform 化する際に合わせて直す。
- `nar-ops` の `POST /ingest-and-refresh` エンドポイントと、それを 02:40 に叩く
  旧 Cloud Scheduler ジョブ（`nar-ingest-and-refresh`）が、実処理を `nar-refresh`
  Job（2026-08-29 追加）に移した後も本番に残っている。実処理をしないまま
  "0 rows"/"ok" を記録し続けるだけで実害はないが、Scheduler の無料枠3個を
  無駄に1個消費している（`operation/src/narops/deploy.py` の `SCHEDULE_CRON`
  からこのエントリを削り、`nar-refresh-daily` の Scheduler/Job 定義を
  `services.json`/`deploy.py` に正式に取り込むこと。`services.json` の
  `cloud_run_jobs.nar-refresh._scheduler` に現状の作成コマンドが残してある）。
- 2026-09-10、コスト最適化として `nar-refresh` Job に bronze の確定月キャッシュを
  導入し（`operation/src/narops/refresh.py`、`docs/Design_Operation.md` §2.2）、
  本番へデプロイ・実測検証済み。デプロイ直後の検証で「`is_final` が本番
  manifest に1件も立っていない」別の不具合（IG-16 の日次窓の外に落ちた過去分は
  二度目の不変観測を得る機会が永久に無い、という構造的な行き詰まり）が見つかり、
  一度きりの確定化スイープ（`operation/scripts/finalize_history_backlog.py --apply`、
  実行23分16秒）で解消した。実測は
  「キャッシュ未着時 13分36秒・14分52秒」→「スイープ後 11分32秒」（約20%減）。
  `docs/Design_Operation.md` §6.1・`narops.cost.default_usage()` の
  `refresh_avg_seconds=700` は実測反映済み。
  **残っている改善余地**: 実測11分32秒の内訳を見ると、削減の主要因は当初期待した
  「ZIP 再展開の省略」よりも小さく、残り時間の大半は (a) `sync_dir` が bronze を
  月×表ごとの小さな parquet ファイル単位で個別に GCS 往復コピーしているオーバー
  ヘッドと、(b) `build_silver_frames` の全履歴 speed_index 再計算（仕様上削れない）
  が占める。(a) を `gcloud storage cp -r` 相当の一括コピーや月単位の tar 化に
  置き換えればさらに縮む可能性があるが、本タスクの範囲では未着手。
  なお `finalize_history_backlog.py` は一度実行すればよく、`is_final` になった
  月は以降どこからも再取得されない（毎月・毎年繰り返す必要はない）。
- 2026-09-11、コスト最適化として entity キャッシュを導入した
  （`operation/src/narops/refresh.py` の `write_entity_cache`/`read_entity_cache`、
  `features.py::history_before` の `entity_cache` 引数、`docs/Design_Operation.md`
  §2.5）。本番実測で `SELECT * FROM entry_result_final`（馬・騎手・調教師・種牡馬の
  全キャリア取得）が1日 150〜275GB のスキャンになっており（月額 $30 規模）、
  `entry_result_final` が日次 MERGE でしか更新されない事実を使って、その MERGE
  直後のコピーを GCS へ1日1回書き出し、`/infer` はそこを読んで BigQuery への
  問い合わせを避けるようにした。
  **デプロイに2つのコンポーネントが要る**: (1) `nar-refresh` Job（キャッシュを
  書く側、`refresh.py` の変更を含む新イメージ）、(2) `nar-ops` サービス
  （キャッシュを読む側、`features.py`/`service.py`/`app.py` の変更を含む新
  イメージ、かつ `NAROPS_INGEST_STORE` 環境変数が要る — `deploy.py` の
  env var 一覧に追加済み）。**`nar-ops` は実際の推論・配信を担う本番サービス**
  なので、`nar-refresh`（内部バッチ）よりデプロイの影響範囲が大きい。
  `operation/tests/test_entity_cache.py` に history_before/build_for_race の
  完全一致テスト（キャッシュ有無で特徴量出力が1バイトも変わらないこと）を
  用意してあり、実装中にこのテストが実際に1件の不具合（tz-aware/naive の
  混在で `pd.to_datetime` が一部行を静かに NaT にする）を検出・修正させた
  （§2.5参照）。デプロイ後は `nar-refresh` を1回実行して当日分のキャッシュを
  作り、`gcloud storage ls` で `entity_cache/{当日}.parquet` の存在を確認し、
  `/infer` の実行ログに `entry_result_final` への `SELECT *`（下限日付なし）が
  出なくなっていることを `INFORMATION_SCHEMA.JOBS_BY_PROJECT` で確認すること。
  GCS ライフサイクル（`deploy.py::LIFECYCLE`、7日で削除）は `gcloud storage
  buckets update --lifecycle-file` を別途適用しないと反映されない
  （`narops deploy --apply` の gcs_lifecycle ステップ、または直接 gcloud）。

  **本番障害（2026-09-11、デプロイ当日）と再修正**: `nar-ops` デプロイ直後、
  笠松のレース（1R・2R・3R）が無音で推論欠測した（利用者からの報告で発覚。
  Cloud Tasks の queue にタスクが残らず、`prediction`/`notification_log` にも
  何も無く、ログにも該当 race_id の痕跡が一切無いという「静かな」壊れ方）。
  原因はメモリ: `entity_cache`（本番実測 480万行・43列）を `pd.read_parquet()`
  で全件 pandas 化する実装だったため、列を DDL 相当の27列に絞ってもなお
  5GB を超え、`nar-ops`（4GiB 上限）を OOM Kill していた（Cloud Run は
  OOM を 503 として扱い、Cloud Tasks が3回リトライして同じ理由で失敗し続けた
  末にタスクを諦めて消す — この経路は成功時と同じくログにほぼ残らない）。
  単体テスト（フィクスチャは数百行）ではこの規模の問題は原理的に再現しない。
  **等価性は単体テストで検証できても、本番データ規模でのメモリは実データで
  しか分からない。** 新しいモデル・データパイプラインを本番規模データに
  向けて初めてデプロイするときは、実データでのメモリ実測を必須にすること。

  対応: `entity_cache` を DataFrame ではなくローカル parquet ファイルパスに変更し、
  絞り込みは `narops.refresh.query_entity_cache`（`pyarrow.parquet.read_table`
  のフィルタ pushdown）が行毎に行うようにした。実測: 旧実装 5.1GB →
  新実装 1レースあたり約440MB（一時的、リクエスト終了で解放）。
  `operation/tests/test_entity_cache.py::test_history_before_cache_avoids_loading_full_table_into_pandas`
  が `pd.read_parquet` を一切呼ばないことを直接検証する回帰テストとして残っている。
  再デプロイ後、笠松4Rで `prediction`/`notification_log` への正常書き込みと
  エラーログ0件を確認済み（2026-09-11 12:43 JST）。1R・2R・3Rの3レースぶんは
  障害中に発走時刻を過ぎ、恒久的に欠測扱いとなった（後から推論しても意味が
  無いため再実行はしない）。
