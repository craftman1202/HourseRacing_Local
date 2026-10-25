# 地方競馬の予測運用の停止と再開手順（2026-10-25）

2026-10-25 に、地方競馬（NAR）の予測運用と自動発注を止め、GCP の費用を（Secret Manager を除いて）ゼロにした。
このメモは、**止めた内容**と、**再開するときに必要な作業**（資源の作り直し・データベースの更新）をまとめたもの。
共用プロジェクト `sample-335613` には他の系統（`chx-*` / `jra-*` / `hourse-racing-webui` / `stockapp` など）が同居している。
**下の手順では `nar-*` / `nar_ops` 以外を触らない。**

## 1. 止めたもの

方針（ユーザー指示）: **止めておいても費用がかからないものは削除せず停止、費用がかかるものは削除**。

| 区分 | 対象 | 状態 | 理由 |
|---|---|---|---|
| PC | `nar-orders@main.service`（楽天の自動発注ランナー） | 停止＋自動起動を無効化 | |
| Cloud Scheduler | `nar-plan-day` / `nar-ingest-and-refresh` / `nar-refresh-daily` / `nar-weekly-report` | **削除**（定義は `gcp_snapshot_2026-10-25/`） | 一時停止中でもジョブ数に課金される（無料は 3 件まで） |
| Cloud Tasks | キュー `nar-queue` | 空にして**一時停止**（残す） | 費用なし |
| Cloud Run サービス | `nar-ops` / `nar-api` / `nar-web` | **残す**（最小インスタンス 0 で待機）。ただしイメージは消したので起動できない | 費用なし |
| Cloud Run ジョブ | `nar-refresh` | **残す**（実行しない）。イメージは消した | 費用なし |
| Artifact Registry | リポジトリ `nar-ops` / `nar-api` / `nar-web`（イメージ約 2.5GB） | **削除** | 保管料 |
| Cloud Storage | `nar-raw-sample-335613`（約 3GB）、`nar-model-sample-335613`（約 0.1GB） | **削除（ユーザーが実行）** | 保管料 |
| BigQuery | データセット `nar_ops`（12 表、約 1.2GB） | **削除（ユーザーが実行）** | 無料枠（請求先ごと 10GB）を他系統で超えているか確認できない |
| 残すもの | サービスアカウント `nar-ops@` / `nar-api@` / `nar-web@` と IAM | 残す | 費用なし |
| 残すもの | Secret Manager の `AUTH_SECRET` / `AUTH_GOOGLE_ID` / `AUTH_GOOGLE_SECRET`（nar-web）、`DISCORD_WEBHOOK_ALERT` / `_DAILY` / `_PREDICTION` | **残す** | 値を控えられなかったので消せない。有効バージョン 6 個までが無料枠だが、他系統の分と合わせて超えている可能性が高く、1 バージョン月 $0.06（計 約 $0.36） |

費用をさらにゼロにするなら、上の Secret Manager の 6 個を消す。再作成が必要になるもの:
Discord の Webhook URL は `operation/.env`（`DISCORD_WEBHOOK_*`）に写しがある。`AUTH_SECRET` は乱数（`openssl rand -base64 32`）で作り直せる。
`AUTH_GOOGLE_ID` / `AUTH_GOOGLE_SECRET` は Google Cloud の「API とサービス → 認証情報」の OAuth クライアントから取得する。
消したら、(D) の前に `gcloud secrets create <名前> --data-file=-` で作り直す。

## 2. バックアップ（消す前に取ったもの）

`~/nar-gcp-backup-2026-10-25/`（この PC のみ。**git 管理外**。別の場所にも写しておくこと）

| 場所 | 中身 |
|---|---|
| `gcs/nar-model-sample-335613/` | モデルのリリース全部（`releases/<id>/`）、`current.json`、`current_banei.json`、`promotions.jsonl` |
| `gcs/nar-raw-sample-335613/` | `ingest-store/`（元データの ZIP・manifest・bronze・entity キャッシュ）、`orders/`（発注計画の JSON） |
| `bq/_bq_export/<表>/*.parquet` | `nar_ops` の全表（`entry_result_final` = 確定層の全履歴、`prediction`、`bet_candidate`、`odds_snapshot`、`notification_log` など） |

リポジトリ内の記録:

- `docs/gcp_snapshot_2026-10-25/` … Cloud Run サービス・ジョブ、Scheduler、キューの設定（`gcloud ... describe --format=yaml` の出力。秘密の値は含まない）
- `operation/infra/services.json`、`operation/infra/bq/*.json` … 資源とテーブルの定義
- コード … 最終のコミットは `git log` で確認（停止時点の main）

`odds_snapshot`（発走前のオッズの記録）は**再取得できない**。バックアップの parquet が唯一の写し。

## 3. 再開の手順

全体の順序: **(A) 資源を作る → (B) データを戻す → (C) 確定層を最新にする → (D) デプロイ → (E) スケジューラを戻す → (F) PC のランナー**。

前提: `gcloud auth login` 済み、プロジェクト `sample-335613`、リージョン `asia-northeast1`。

```bash
P=sample-335613; R=asia-northeast1; B=$HOME/nar-gcp-backup-2026-10-25
gcloud services enable run.googleapis.com cloudscheduler.googleapis.com cloudtasks.googleapis.com \
  artifactregistry.googleapis.com bigquery.googleapis.com secretmanager.googleapis.com --project=$P
```

### (A) 資源を作る

削除していない資源（Cloud Run のサービス・ジョブ、Cloud Tasks のキュー、サービスアカウント、Secret Manager）は作り直し不要。
キューは一時停止中なので `gcloud tasks queues resume nar-queue --location=$R --project=$P` で戻す。

```bash
# Artifact Registry（3 つ）
for r in nar-ops nar-api nar-web; do
  gcloud artifacts repositories create $r --repository-format=docker --location=$R --project=$P
done

# バケット（ライフサイクルは operation/src/narops/deploy.py の LIFECYCLE と同じ。raw: 30 日で Nearline、365 日で Coldline、
# ingest-store/entity_cache/ は 7 日で削除。model: 90 日で Nearline）
gcloud storage buckets create gs://nar-raw-sample-335613 gs://nar-model-sample-335613 --location=$R --project=$P

# Cloud Tasks のキューを消してしまっていた場合のみ（設定は gcp_snapshot_2026-10-25/queue_nar-queue.yaml）
# gcloud tasks queues create nar-queue --location=$R --project=$P --max-attempts=3 --min-backoff=30s --max-backoff=270s --max-doublings=16

# BigQuery のデータセットと表（スキーマ・パーティションは operation/infra/bq/*.json、deploy.py の PARTITION_*）
cd operation && PYTHONPATH=src:../learning/src python -m narops.cli init   # DWH を初期化
```

> `narops deploy --apply --confirm` は使わない。nar-ops・nar-api・nar-web の 3 つに**同じイメージ**を配る実装で、api と web が壊れる
> （メモリ `narops-deploy-tool-shares-one-image-across-services`）。下の `gcloud run deploy` を直接使う。

IAM はサービスアカウントを残しているので、そのまま使える。消えていた場合は次を付け直す:
`nar-ops@` … bigquery.dataEditor / bigquery.jobUser / cloudtasks.enqueuer / cloudtasks.taskDeleter / **cloudtasks.viewer**（無いと `/plan-day` が 500 を返す）/ run.invoker / secretmanager.secretAccessor / storage.objectAdmin。
`nar-api@` … bigquery.dataViewer / bigquery.jobUser / storage.objectViewer。`nar-web@` … run.invoker。

### (B) データを戻す

1. **モデル**（これが無いと推論できない）

   ```bash
   gcloud storage cp -r $B/gcs/nar-model-sample-335613/* gs://nar-model-sample-335613/
   ```

   現行は `v2026.10.18-A`（平地）と `v2026.09.17-B-banei`（ばんえい）。`current.json` / `current_banei.json` が指す。

2. **元データの ZIP と manifest**（任意。省略しても (C) で全期間を取り直せる）

   ```bash
   gcloud storage cp -r $B/gcs/nar-raw-sample-335613/ingest-store gs://nar-raw-sample-335613/
   ```

   戻すと、(C) で再取得が必要なのは「止めていた期間」だけになる（約 3GB・数分）。戻さないと 1998 年からの全月を取り直す（約 345 か月 × 3 秒間隔で約 20 分、ジョブは 16Gi で 1 時間以内）。

3. **BigQuery の表**（任意）

   ```bash
   for t in entry_result_final prediction bet_candidate notification_log odds_snapshot race_schedule feature_snapshot job_run skew_check entry_result_audit entry_result_live; do
     bq load --project_id=$P --source_format=PARQUET --replace nar_ops.$t "$B/bq/_bq_export/$t/*.parquet"
   done
   ```

   必須なのは `entry_result_final`（確定層。推論の特徴量の元）。**戻さなくても (C) で作り直せる**（元データから）。
   `prediction` / `bet_candidate` / `notification_log` / `odds_snapshot` は履歴の記録で、推論には要らない（成績の分析や週次レポートに使う）。
   パーティションの設定は `bq load` で失われるので、`init` で作った表に追加する場合は `--noreplace` か、先に `bq load` した表を `init` の定義で作り直すこと。

### (C) 確定層（データベース）を最新にする ← 再開でいちばん大事な作業

確定層 `entry_result_final` は、毎日の `nar-refresh` ジョブが「NAR の月次データ（keiba.go.jp）→ bronze → silver → MERGE」で更新する。
**日次ジョブの既定は「当月＋前月だけ再取得」「直近 7 日分だけ MERGE」**なので、**長く止めたあとは既定のままだと穴が空く。**

停止日数に応じて次を選ぶ:

| 止めていた期間 | やること |
|---|---|
| 〜1 週間 | ジョブを 1 回実行するだけ（既定で足りる） |
| 1 週間〜1 か月 | `NAROPS_REFRESH_SINCE_DAYS`（止めていた日数 + 余裕）を付けてジョブを 1 回実行 |
| 1 か月〜 | 上に加えて、`operation/scripts/finalize_history_backlog.py --apply` を先に実行（再取得の対象を全月に広げる。`--apply` なしは計画だけ） |
| 全部作り直す（(B)-2・(B)-3 を戻していない） | ジョブを 1 回実行。manifest が空なので全月を取得し、確定層も全履歴を入れる |

```bash
# ジョブは残っている（イメージが無いだけ）。先に (D) でイメージをビルド・push し、`gcloud run jobs update nar-refresh --image=$IMG ...` で差し替える。
# 消えていた場合は次で作る（設定は gcp_snapshot_2026-10-25/job_nar-refresh.yaml）
IMG=asia-northeast1-docker.pkg.dev/$P/nar-ops/nar-ops:vYYYY.MM.DD
gcloud run jobs create nar-refresh --project=$P --region=$R --image=$IMG \
  --command=python --args=-m,narops.refresh --cpu=4 --memory=16Gi --task-timeout=3600s --max-retries=1 \
  --service-account=nar-ops@$P.iam.gserviceaccount.com \
  --set-env-vars=NAROPS_INGEST_STORE=gs://nar-raw-sample-335613/ingest-store,NAROPS_BACKEND=bigquery,NAROPS_MODE=paper,MODEL_PROVENANCE=real,NAROPS_PROJECT=$P,NAROPS_REFRESH_SINCE_DAYS=60

gcloud run jobs execute nar-refresh --project=$P --region=$R --wait
```

確認: ログに `daily_refresh 完了: N 行 merge / 確定層の最新 <昨日>`、`/health` が `status: ok`（`latest_final_date` が昨日）。
`取得に失敗` と出たら、keiba.go.jp が一時的に 302 などを返している。時間をおいて再実行する（2026-10-22 に実際にあった。再試行とジョブの失敗扱いは入れてある）。
終わったら `NAROPS_REFRESH_SINCE_DAYS` は外す（`gcloud run jobs update nar-refresh --remove-env-vars=NAROPS_REFRESH_SINCE_DAYS`）。

**止めている間のレース結果**: 確定層には、次回の MERGE で入る。ただし、止めている間に NAR の月次データの形式が変わると `schema drift` で止まる。
その場合は `docs/MaintenanceRunbook.md` §1.4 を参照。

### (D) イメージのビルドとデプロイ

コンテキストはリポジトリのルート。

```bash
IMG_OPS=asia-northeast1-docker.pkg.dev/$P/nar-ops/nar-ops:vYYYY.MM.DD
docker build -f operation/Dockerfile -t $IMG_OPS . && docker push $IMG_OPS

# nar-ops（推論・通知・計画）。設定は gcp_snapshot_2026-10-25/svc_nar-ops.yaml
gcloud run deploy nar-ops --project=$P --region=$R --image=$IMG_OPS \
  --service-account=nar-ops@$P.iam.gserviceaccount.com --memory=4Gi --cpu=2 --min-instances=0 --max-instances=3 \
  --no-allow-unauthenticated --quiet \
  --set-env-vars=NAROPS_PROJECT=$P,NAROPS_DATASET=nar_ops,NAROPS_MODEL_BUCKET=nar-model-sample-335613,NAROPS_RAW_BUCKET=nar-raw-sample-335613,NAROPS_ROLE=ops,NAROPS_SERVICE_URL=<デプロイ後のURL>,MODEL_PROVENANCE=real,NAROPS_MODE=paper,NAROPS_INGEST_STORE=gs://nar-raw-sample-335613/ingest-store,NAROPS_ORDER_BUCKET=nar-raw-sample-335613
```

- **メモリは 4Gi 必須**（1Gi は推論中に落ちる）。`NAROPS_ORDER_BUCKET` が無いと、自動発注の計画 JSON が GCS に置かれない。
- `NAROPS_SERVICE_URL` は、サービスを作り直すとドメイン（`nar-ops-bwl3vrgzdq-an.run.app`）が**変わる可能性がある**。デプロイ後の URL で環境変数と Scheduler の URL を更新する。
- ダッシュボード（任意）: `nar-api`（`operation/Dockerfile.api`）と `nar-web`（`web/Dockerfile`）は別イメージ。設定は `svc_nar-api.yaml` / `svc_nar-web.yaml`。
  nar-web は `AUTH_URL`（nar-web の公開 URL）が必須で、秘密情報 3 つ（`AUTH_SECRET` など）を Secret Manager から渡す
  （メモリ `nar-web-auth-url-required-on-cloud-run`）。URL が変わったら Google のサインインの承認済みリダイレクト URI も更新する。

### (E) スケジューラとキュー

```bash
URL=<nar-ops の URL>
mk() { gcloud scheduler jobs create http "$1" --project=$P --location=$R --schedule="$2" --time-zone=Asia/Tokyo \
        --uri="$URL$3" --http-method=POST --oidc-service-account-email=nar-ops@$P.iam.gserviceaccount.com --oidc-token-audience="$URL$3" \
        --attempt-deadline="$4"; }
mk nar-plan-day            "0 8 * * *"  /plan-day            540s
mk nar-ingest-and-refresh  "40 2 * * *" /ingest-and-refresh  540s
mk nar-weekly-report       "0 4 * * 1"  /weekly-report       540s
# 日次の確定層更新（Cloud Run ジョブを起動する）
gcloud scheduler jobs create http nar-refresh-daily --project=$P --location=$R --schedule="0 5 * * *" --time-zone=Asia/Tokyo \
  --uri="https://$R-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$P/jobs/nar-refresh:run" --http-method=POST \
  --oauth-service-account-email=nar-ops@$P.iam.gserviceaccount.com --oauth-token-scope=https://www.googleapis.com/auth/cloud-platform \
  --attempt-deadline=180s
```

- **日中に再開したら、当日の `/plan-day` は自動では走らない**（毎朝 8:00 JST に走るため）。その日から推論を始めたいときは、
  `/plan-day` を手動で叩く（メモリ `mid-day-deploy-misses-todays-plan-day`）。
- `nar-ingest-and-refresh`（毎日 2:40）は、実際の確定層の更新を `nar-refresh` ジョブ（5:00）に任せており、二重になっている。

### (F) PC の自動発注ランナー

1. **先に (A)〜(E) と動作確認を済ませる**（推論が動いて、発注計画が `gs://nar-raw-sample-335613/orders/<日付>/` に置かれることを確認）。
2. 楽天のログインが切れている。`operation/.env` の認証情報で入り直す（二要素認証のコードはメールで届く）。手順は `docs/AutoOrders.md` §6。
3. 設定 `~/.cache/nar-orders/orders_live.yaml`（実発注。券種・上限は `docs/AutoOrders.md` §4）を確認する。
4. `cd operation && scripts/orders.sh enable && scripts/orders.sh start`。**最初は `pause` してから起動し、挙動を確認して `resume`** が安全。

## 4. 再開前に見直すこと

- **モデルが古くなる。** 現行 `v2026.10.18-A` は 2026-10 までのデータで学習している。止めた期間が長いほど、再学習を検討する（`docs/MaintenanceRunbook.md` §2〜§5）。
- **買い目の規則**（`operation/conf/bet_rules.yaml`）は、2020〜2023 の検証と OOS（2024-02〜、一度開封済み）で選んだもの。
  実運用に乗せていた 4 券種（単勝・複勝・馬連・ワイド）は、OOS で ROI の CI が 1 をまたいでおり、優位とは言えなかった。再開前に実績（`bet_candidate`・台帳）で見直すこと。
- **keiba.go.jp の仕様変更**（URL・形式）があると、取得が止まる。最初の `nar-refresh` の結果を必ず見る。
- 楽天の画面の構造が変わっていると、ランナーは「見送り」「エラー」を出す（`rakuten_probe.py` は購入しない調査用）。
