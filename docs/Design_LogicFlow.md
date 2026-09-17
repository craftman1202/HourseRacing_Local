# 地方競馬 予測システム 全体ロジックフロー設計書（UML）

対象範囲: 学習（`learning/src/nar`）と運用（`operation/src/narops`）の処理の流れ全体
位置づけ: `Design_Modeling.md`（学習の設計判断）と `Design_Operation.md`（運用の設計判断）が「なぜそうするか」を書くのに対し、本書は**現在のコードが実際にどう動くか**を図で示す。列単位の詳細（特徴量エンジニアリング前後のスキーマ）は `Design_FeatureSchema.md` を参照。
基準: 2026-09-16 時点の `main`（d1b181a）。図中の関数名・ファイル名はすべてコード上に実在するもの。
記法: 図は Mermaid（GitHub / VS Code のプレビューで描画される）。

---

## 0. 全体を一枚で

```mermaid
flowchart LR
  subgraph NAR["NAR（keiba.go.jp）"]
    DL["DataDownload<br/>月次 ZIP（レース・払戻・出馬表・オッズ）"]
    TODAY["TodayRaceInfo<br/>TodayRaceInfoTop / DebaTable /<br/>OddsTanFuku / RaceMarkTable"]
  end

  subgraph LOCAL["ローカル（学習）learning/"]
    RAW[("raw ZIP<br/>不変層")] --> BRONZE[("bronze<br/>展開＋デコードのみ")] --> SILVER[("silver<br/>race / entry / payout / odds")]
    SILVER --> GOLD[("gold<br/>features.parquet")]
    GOLD --> WF["nar learn<br/>walk-forward 評価"]
    GOLD --> FF["nar fit-final<br/>配布用モデル"]
    WF -- "ensemble_weights.csv" --> PUB
    FF -- "final_meta.json / モデル本体" --> PUB["narops publish-release"]
  end

  subgraph GCP["GCP（運用）"]
    REG[("GCS nar-model<br/>releases/ + current.json")]
    JOB["Cloud Run Job<br/>nar-refresh（05:00）"]
    OPS["Cloud Run nar-ops<br/>/plan-day /snapshot-odds<br/>/infer /refresh-live /weekly-report"]
    BQ[("BigQuery nar_ops<br/>entry_result_final / live<br/>race_schedule / prediction ...")]
    GCSRAW[("GCS nar-raw<br/>raw・bronze・manifest<br/>entity_cache")]
    API["Cloud Run nar-api"] --> WEB["Cloud Run nar-web<br/>Next.js"]
    DISCORD["Discord Webhook"]
  end

  DL --> RAW
  DL --> JOB
  PUB --> REG
  REG --> OPS
  JOB <--> GCSRAW
  JOB --> BQ
  TODAY --> OPS
  OPS <--> BQ
  GCSRAW -- "entity_cache" --> OPS
  OPS --> DISCORD
  BQ --> API
```

学習と運用をつなぐのは「モデルアーティファクトの契約」だけ（`manifest.json` + `feature_spec.json` + `standardizer.json` + モデル本体）。**特徴量の計算コードは共有**しており、運用側は `nar.features.builder.build()` と `nar.transform.keys` / `nar.transform.prerace` を直接 import する（書き直さない。SK-02）。

---

## 1. データ系譜（レイヤ図）

```mermaid
flowchart TB
  subgraph L0["raw（不変）"]
    Z["monthly/race/ym=YYYY-MM/YYYYMM_race.zip<br/>monthly/odds/ym=YYYY-MM/YYYYMM_odds.zip"]
    M[("meta/manifest.duckdb<br/>sha256・is_final")]
  end
  subgraph L1["bronze（全列 str）"]
    B1["race/ym=*/part-00.parquet（66列）"]
    B2["entry/ym=*/part-00.parquet（36列）"]
    B3["payout/ym=*/part-00.parquet（54列）"]
    B4["odds/ym=*/part-01..03.parquet（10列）"]
  end
  subgraph L2["silver（型付け・キー付与）"]
    S1["race.parquet<br/>487,095 行 × 16 列"]
    S2["entry.parquet<br/>4,835,353 行 × 41 列"]
    S3["payout.parquet（縦持ち）<br/>4,848,425 行 × 9 列"]
    S4["odds.parquet（単勝のみ）<br/>90,737 行 × 5 列"]
  end
  subgraph L3["gold（as-of 特徴量）"]
    G1["features_noodds/features.parquet<br/>4,367,821 行 × 49 列（平地）"]
    G2["features_noodds_banei/features.parquet<br/>467,049 行 × 50 列（ばんえい）"]
  end
  subgraph L4["配布物（リリース）"]
    R["manifest.json / feature_spec.json / standardizer.json<br/>clogit_beta.json / lgbm_rank.txt / tabm.onnx(+.data)"]
  end
  subgraph L5["運用 DWH（BigQuery）"]
    F["entry_result_final（確定層）"]
    LV["entry_result_live（ライブ層）"]
    SN["feature_snapshot / prediction / bet_candidate"]
  end

  Z -- "ingest.unzip.extract<br/>transform.bronze.build_from_zip" --> L1
  B1 -- "silver.build_race" --> S1
  B2 -- "silver.build_entry<br/>+ attach_start_ts" --> S2
  B3 -- "silver.build_payout" --> S3
  B4 -- "silver.build_odds" --> S4
  S1 & S2 -- "features.builder.build" --> G1 & G2
  G1 -- "train.final.fit_and_export" --> R
  G2 -- "（NAR_CONF_DIR=conf_banei）" --> R
  S1 & S2 -- "refresh._to_bq_payload<br/>（speed_index を全履歴で計算）<br/>db.merge.merge_final" --> F
  F & LV -- "features.history_before<br/>→ builder.build（推論時）" --> SN
```

行数は `learning/data_real` の実測（2026-09-03 生成、gold の `content_hash` = `b50ea94e…` / `b49cf81f…` は現行リリースの `dataset_version` と一致）。

---

## 2. 学習系

### 2.1 CLI ステージのアクティビティ図

```mermaid
flowchart TD
  A0([開始]) --> A1["nar track-master<br/>競馬場名→コード（廃止場含む31場）<br/>meta/track_master.json"]
  A1 --> A2["nar ingest --start 1998-01<br/>fetch_month: should_fetch → HTTP → sha256 比較 → raw 書き込み"]
  A2 --> A2b["nar ingest --kind odds --start 2026-02"]
  A2b --> A3["nar bronze<br/>ZIP 展開・BOM判定デコード・全列 str"]
  A3 --> A4["nar silver<br/>型付け・race_id・horse_sk・払戻縦持ち"]
  A4 --> A5{"nar leak-check<br/>累積成績8列は as-of-race か"}
  A5 -- "判定不能・as-of-download" --> A5x["ASOF_RACE_WHITELIST から外す<br/>（d_* 特徴量を作らない）"]
  A5 -- "as-of-race（2026-08 に 8列とも確定）" --> A6
  A5x --> A6["nar eda<br/>Q1〜Q6・品質スコアカード"]
  A6 --> A7["nar features<br/>gold/features_noodds[_banei]/features.parquet<br/>+ content_hash.txt"]
  A7 --> A8["nar learn --models clogit,lgbm,tabm,bayes<br/>walk-forward 5 fold → OOF → 制約付きスタッキング"]
  A8 --> A9["nar gate-report<br/>pytest（Blocker ID）+ OOS ガード → gate_report.json"]
  A9 --> A10["nar fit-final<br/>OOS 境界 − embargo まで（評価用）<br/>--through で全期間（出荷用）"]
  A10 --> A11{"nar evaluate-final --unlock-oos<br/>（OOS 開封は1回）"}
  A11 -- "RF ガード発火" --> AX([publish 中止])
  A11 -- "未発火" --> A12["narops publish-release<br/>staging/ に組み立て → registry.publish"]
  A12 --> A13["narops promote（二段確認）<br/>current.json / current_banei.json"]
  A13 --> A14([運用へ])
```

ばんえいは同じ流れを `NAR_CONF_DIR=conf_banei` で回す（`include.baba_codes=[1,2,3,4]`、`variant: banei`）。平地側は `exclude.baba_codes=[1,2,3,4]`。

### 2.2 ingest の確定判定（状態機械）

`nar.io.manifest.should_fetch` / `is_finalizable` / `finalize_pass` の挙動。

```mermaid
stateDiagram-v2
  [*] --> 未取得
  未取得 --> 取得済み_未確定: fetch 成功（is_final=False）
  未取得 --> parse_error: HTTP 例外 / ZIP を開けない
  parse_error --> 取得済み_未確定: 次回再取得で成功
  取得済み_未確定 --> 取得済み_未確定: 再取得・sha256 変化（raw 上書き）
  取得済み_未確定 --> 確定: 再取得・sha256 不変 かつ 月末+45日経過
  取得済み_未確定 --> 取得済み_未確定: 当月・前月以外（daily_refresh_months=2 の窓外）→ skipped
  確定 --> 確定: should_fetch=False（ネットワークに触れない）
```

窓外に落ちた過去月は「不変の観測」が二度と起きないため、初回バックフィル分は `operation/scripts/finalize_history_backlog.py` で一度だけ全月を再確認して確定させた（`Design_Operation.md` §2.2）。

### 2.3 特徴量ビルダーの内部（`nar.features.builder.build`）

```mermaid
flowchart TD
  I["入力: entry（silver, 結果列込み）+ race（silver）+ FeatureConfig"] --> F1["baba_code で exclude / include フィルタ"]
  F1 --> F2["last3f が無ければ NaN 列を追加（列集合を固定）"]
  F2 --> F3["speed_index(entry)<br/>pandas の groupby 累積和で as-of 標準化<br/>（既に列があれば再計算しない）"]
  F3 --> SQL
  subgraph SQL["DuckDB 1クエリ"]
    E["CTE e: n_runners = COUNT(*) OVER race_id<br/>pace_balance（1走ごとの算術）"]
    E --> H["CTE horse: PARTITION BY horse_sk<br/>GROUPS UNBOUNDED PRECEDING AND 1 PRECEDING"]
    E --> J["CTE jockey: PARTITION BY jockey_sk"]
    E --> T["CTE trainer: PARTITION BY trainer_sk"]
    E --> JT["CTE jt: PARTITION BY jockey_sk, trainer_sk"]
    E --> S["CTE sire: PARTITION BY sire_sk"]
    H & J & T & JT & S --> SEL["SELECT: 収縮（事前確率 1/n_runners）<br/>Wilson 下限・field_size・draw_rel・log_prize<br/>JOIN race_raw USING race_id"]
  end
  SQL --> P1["初出走行の h_winrate / h_top3rate / h_si_last3 /<br/>h_best_si / h_pace_bal を NULL に"]
  P1 --> P2{"entry に累積成績8列があるか"}
  P2 -- "ある" --> P3["assert_prerace → declared.build<br/>d_* 16列（d_extra_starts は自前集計との差）"]
  P2 -- "ない" --> P4["警告のみ（d_* を作らない）"]
  P3 & P4 --> P5{"variant == banei"}
  P5 -- "yes" --> P6["banei.build → b_* 10列を追加<br/>BANEI_DROPPED 8列を削除"]
  P5 -- "no" --> P7
  P6 --> P7{"track == A"}
  P7 -- "yes" --> P8["assert_no_market_info<br/>（odds / 人気 / payout … を含む列名で LeakageError）"]
  P7 -- "no" --> P9
  P8 --> P9["finalize: 特徴量を float64 に固定・小数10桁で丸め"]
  P9 --> O["出力: gold（識別列11 + 特徴量）"]
```

時点の正しさを担保する3点:

1. 集計フレームは `ORDER BY start_ts GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`。**同時刻の別場のレースも除外**する（`ROWS … EXCLUDE CURRENT ROW` ではそれが混ざる）。
2. `speed_index` の基準統計量も「同じ (場, 距離) で発走時刻が厳密に前」の行だけ（`peer` を引いて同時刻を外す）。
3. 当該レースの結果列（`finish_pos`, `time_sec`, `last3f`）は入力に含まれるが、ウィンドウから当該行が外れるので出力には漏れない（LK-05 が未来行を改ざんしてビット一致を検証）。

### 2.4 walk-forward 評価（`nar learn` → `train.pipeline`）のシーケンス

```mermaid
sequenceDiagram
  autonumber
  participant CLI as cli.cmd_learn
  participant G as OOSGuard
  participant PL as pipeline.run_walkforward
  participant F as run_fold（fold 1..5）
  participant SEL as selection.select
  participant M as モデル（clogit/lgbm/tabm/bayes）
  participant CAL as TemperatureScaler
  participant ST as ConstrainedStacker

  CLI->>CLI: build_features(silver) → feat（gold と同じ関数）
  CLI->>G: filter(feat)（2024-02-01 以降を落とす）
  G-->>PL: feat（OOS 除外済み）
  PL->>PL: trainable(): 着順 NULL 行・勝者≠1頭のレースを除外
  PL->>PL: make_folds（train_end = valid_start − 181日）→ validate_folds
  loop 各 fold
    PL->>F: fold
    F->>SEL: prepare(train_raw) の上で3段選択（Null Importance → (VIF) → RFE）
    SEL-->>F: selected
    F->>F: prepare(train)・prepare(valid)（**それぞれ自分の統計量で**補完・標準化）
    loop 各モデル
      F->>M: fit(train) → predict(valid)
      M-->>F: スコア / 確率
      F->>F: normalize_within_race
      F->>CAL: fit(valid) → transform(valid)
    end
    F->>F: baseline_uniform / baseline_market を追加・metrics.summary
  end
  PL-->>CLI: FoldOutput × 5
  CLI->>ST: fit(OOF, is_win, race_id, fold_ids)（SLSQP, w≥0, Σw=1）
  ST-->>CLI: 重み → artifacts/ensemble_weights.csv
  CLI->>CLI: oof_predictions.parquet / fold_metrics.csv / selected_features.csv
```

fold の境界（`conf/cv.yaml`、embargo は `max_lookback_days=180` から自動導出）:

| fold | train | valid |
|---|---|---|
| 1 | 1998-01-01 〜 2018-08-04 | 2019-02-01 〜 2019-12-31 |
| 2 | 1998-01-01 〜 2019-08-04 | 2020-02-01 〜 2020-12-31 |
| 3 | 1998-01-01 〜 2020-08-04 | 2021-02-01 〜 2021-12-31 |
| 4 | 1998-01-01 〜 2021-08-04 | 2022-02-01 〜 2022-12-31 |
| 5 | 1998-01-01 〜 2022-08-04 | 2023-02-01 〜 2023-12-31 |
| OOS | 1998-01-01 〜 2023-08-04 | 2024-02-01 〜 現在（施錠） |

（`train_end = valid_start − (embargo_days + 1)` 日。設計書 §9.1 の表は embargo 30日時代のもので、実装は 181 日前で切る。）

### 2.5 配布用モデルの学習（`nar fit-final` → `train.final.fit_and_export`）

```mermaid
sequenceDiagram
  autonumber
  participant CLI as cli.cmd_fit_final
  participant FF as final.fit_and_export
  participant SEL as selection.select
  participant M as clogit / lgbm / tabm
  participant ONNX as narops.publish（export_tabm_onnx / verify_onnx）

  CLI->>FF: gold features.parquet, asof_features(cfg), content_hash
  FF->>FF: training_window: [train_start, OOS開始−181日]（--through 指定時はその日まで）
  FF->>FF: trainable(window)
  FF->>FF: 末尾90日を cal_raw、それ以前を fit_raw に分割
  FF->>SEL: select(apply_stats(fit_raw))（n_null_runs=5）
  SEL-->>FF: selected（平地の現行出荷版で 20 列）
  FF->>FF: stats = fit_stats(fit_raw, selected)（median / mean / std）
  loop clogit, lgbm, tabm
    FF->>M: fit(apply_stats(fit_raw)) → predict(apply_stats(cal_raw))
    M-->>FF: 確率
    FF->>FF: TemperatureScaler.fit(cal) → temperatures[name]
    alt tabm
      FF->>ONNX: export（内部標準化 μ/σ を埋め込む）→ verify（最大差・Top-1 一致）
      ONNX-->>FF: 不一致ならファイルを消して配布物から外す
    end
  end
  FF-->>CLI: feature_names.json / standardizer.json / final_meta.json<br/>clogit_beta.json / lgbm_rank.txt / tabm.onnx(.data)
```

### 2.6 リリースと昇格（状態機械）

```mermaid
stateDiagram-v2
  [*] --> 学習成果物: nar fit-final
  学習成果物 --> 拒否: gate_report が GREEN でない / OOS ガード発火
  学習成果物 --> staging: narops publish-release<br/>（build_release: feature_spec 生成・重みを実体のあるモデルで再正規化・sha256）
  staging --> releases: registry.publish（--stage なし）
  releases --> current: narops promote（confirmed=True・監査ログ promotions.jsonl）
  current --> releases: 次のリリースへ切り替え / rollback
  拒否 --> [*]
```

`manifest.json` に入る値の出どころ:

| manifest キー | 出どころ |
|---|---|
| `feature_spec.names` | `final_meta.json.feature_names`（fit-final の選択結果） |
| `ensemble_weights` | `artifacts/ensemble_weights.csv`（**walk-forward OOF** で推定。fit-final のモデルで再推定はしていない） |
| `calibration.temperature` | `final_meta.json.temperatures` のうち**重み最大のモデル**の温度（1つだけ） |
| `oos_metrics` | `artifacts/oos_metrics.json` の `ensemble`（評価用モデルで測った値） |
| `dataset_version` | gold の `content_hash` |
| `lookback_days` | `conf*/features.yaml` の `max_lookback_days` |
| `purpose` / `oos_evaluated_on` | `final_meta.json`（`--through` なら `production`） |

現行ポインタ（`operation/data/nar-model`、2026-09-16）: 平地 `current.json` = `v2026.09.05-B`（production、学習 1998-01-01〜2026-09-05、重み lgbm 0.059 / tabm 0.941、温度 0.9775）、ばんえい `current_banei.json` = `v2026.08.30-A-banei`（production、〜2026-08-24、lgbm 0.237 / tabm 0.763、温度 0.9853）。

---

## 3. 運用系

### 3.1 コンポーネント図

```mermaid
flowchart LR
  subgraph Scheduler["Cloud Scheduler"]
    SC1["nar-refresh-daily 05:00"]
    SC2["nar-plan-day 08:00"]
    SC3["nar-weekly-report 月 04:00"]
    SC0["nar-ingest-and-refresh 02:40<br/>（旧・実処理なし）"]
  end
  subgraph Tasks["Cloud Tasks nar-queue"]
    T1["infer-{race_id}-{epoch}<br/>発走 −13分"]
    T2["odds-all-{epoch}<br/>初回発走−35分〜最終発走、10分／直前帯5分"]
    T3["refresh-live-{epoch}<br/>12/16/20時（開催時間帯のみ）"]
  end
  subgraph OPS["nar-ops（app.py / service.py）"]
    E1["/plan-day"]
    E2["/snapshot-odds"]
    E3["/infer"]
    E4["/refresh-live"]
    E5["/weekly-report"]
    E6["/ingest-and-refresh（旧）"]
  end
  JOB["nar-refresh Job<br/>python -m narops.refresh"]

  SC1 --> JOB
  SC2 --> E1
  SC3 --> E5
  SC0 --> E6
  E1 -- "enqueue" --> T1 & T2 & T3
  E2 -- "reconcile（5分以上の変更で再登録）" --> T1
  T1 --> E3
  T2 --> E2
  T3 --> E4
```

### 3.2 日次タイムライン

```mermaid
gantt
  title 開催日の1日（JST）
  dateFormat HH:mm
  axisFormat %H:%M
  section 確定層
  nar-refresh（実測 約11分半）       :done, r1, 05:00, 12m
  section 計画
  plan-day                          :milestone, p1, 08:00, 0m
  section 開催中（例: 初回 14:30, 最終 20:50）
  snapshot-odds（10分／直前5分）      :active, o1, 13:55, 415m
  infer（各レース 発走−13分）          :crit, i1, 14:17, 393m
  refresh-live 16:00                :milestone, l2, 16:00, 0m
  refresh-live 20:00                :milestone, l3, 20:00, 0m
```

`refresh-live` の 12:00 は、初回発走より前なら積まれない（`plan_live_refreshes` が `first ≤ fire ≤ last+30分` で絞る）。

### 3.3 `nar-refresh` Job（`narops.refresh.daily_refresh`）

```mermaid
sequenceDiagram
  autonumber
  participant SCH as Cloud Scheduler
  participant JOB as refresh.main
  participant GCS as GCS persist_root
  participant NAR as NAR DataDownload
  participant BQ as BigQuery
  participant DC as Discord（alert）

  SCH->>JOB: jobs/nar-refresh:run（OIDC）
  JOB->>GCS: sync_dir → workdir（raw, bronze, manifest, track_master）
  JOB->>NAR: backfill(1998-01 〜 当月, kind=race)<br/>（未確定の当月・前月だけ実際に取得）
  JOB->>JOB: finalize_pass → run_bronze（is_final かつ bronze 既存の月は展開しない）
  JOB->>JOB: build_silver_frames（**全履歴**の bronze から race/entry）
  JOB->>JOB: _to_bq_payload: turn/baba_condition 付与・last3f 数値化<br/>speed_index（全履歴）・列名 ASCII 化・start_ts JST→UTC
  JOB->>BQ: merge_final（since = 7日前以降の行、年ごと）+ audit
  JOB->>BQ: freshness.check
  JOB->>GCS: write_entity_cache（full_payload → entity_cache/{当日}.parquet）
  JOB->>GCS: sync_dir workdir → persist_root
  alt 例外
    JOB->>BQ: job_run(failed)
    JOB->>DC: alert fetch_failure（Blocker）
  end
```

### 3.4 `/plan-day` と `/snapshot-odds`

```mermaid
sequenceDiagram
  autonumber
  participant SCH as Scheduler / Tasks
  participant SVC as service
  participant SRC as NarSource
  participant BQ as BigQuery
  participant Q as Cloud Tasks

  SCH->>SVC: POST /plan-day
  SVC->>SRC: fetch_schedule(day)（TodayRaceInfoTop を解析）
  alt 取得・解析失敗
    SVC-->>SCH: aborted + Blocker alert（部分結果で走らせない）
  end
  SVC->>BQ: race_schedule UPSERT（race_id, race_date, baba_code, race_no, start_ts, status）
  SVC->>Q: infer タスク（ばんえいは専用モデルがあるときだけ）
  SVC->>Q: odds タスク群 / refresh-live タスク群

  SCH->>SVC: POST /snapshot-odds（数分おき）
  SVC->>SRC: fetch_schedule(day)
  SVC->>SVC: reconcile(current, stored)<br/>5分以上の変更 → 旧タスク削除・新名で再登録／消えたレース → 削除
  loop 発走まで 0〜35 分のレース
    SVC->>SRC: fetch_odds（OddsTanFuku）
  end
  SVC->>BQ: odds_snapshot INSERT
```

### 3.5 `/infer`（1レースの推論）

```mermaid
sequenceDiagram
  autonumber
  participant T as Cloud Tasks
  participant S as service.infer_endpoint
  participant BQ as BigQuery
  participant SRC as NarSource
  participant FE as features.build_for_race
  participant B as nar.features.builder.build
  participant RT as runtime（Standardizer / Scorers）
  participant INF as inference.run_inference
  participant DC as Discord

  T->>S: POST /infer {race_id}
  S->>BQ: race_schedule 行（無い列は distance=1200 等で仮埋め）
  S->>S: assert_within_window（発走 13分+60秒前より早い / 発走後 → expired）
  S->>S: bundle_for(baba_code)（1-4 → ばんえい束、他 → 平地束）
  S->>BQ: freshness.require_fresh（確定層が古ければ Blocker で停止）
  S->>SRC: fetch_entry_card（DebaTable 解析 + parse_race_header）
  S->>SRC: fetch_odds（失敗しても続行）
  S->>S: 当日ページ値で distance / class_level / prize_yen / turn / baba_condition を上書き
  S->>FE: build_for_race(card, row, manifest, feature_config, entity_cache)
  FE->>FE: to_prerace(card) → assert_prerace
  FE->>BQ: history_before: 馬・騎手・調教師・種牡馬の全キャリア<br/>（確定層は entity_cache があれば GCS の parquet から、ライブ層は常に BQ）<br/>+ 直近180日の全レース
  FE->>FE: start_ts < 発走時刻 で切る・列名を日本語へ戻す（RECORD_COLUMN_MAP）
  FE->>FE: 対象レース行を「結果 NaN」で履歴に連結・race_meta を組み立て
  FE->>B: build(combined, race_meta, cfg)（**学習と同一関数**）
  B-->>FE: 全行の特徴量 → 対象レースだけ抽出
  FE->>FE: assert_serving_inputs（ばんえい: 重量の欠測・値域外で停止）<br/>feature_spec_of(declared) → verify_feature_spec（hash 不一致で停止）
  FE-->>S: FeatureResult
  S->>BQ: feature_snapshot（DELETE→INSERT、features は JSON）
  S->>RT: standardizer.apply（学習時の median で補完 → z 値）
  S->>INF: run_inference(models, weights, temperature, odds)
  INF->>INF: 各モデル score → race_softmax → blend（log 空間の重み付き和）<br/>→ 温度 → Σp=1 検証 → Harville で p_top3
  INF->>INF: オッズ揃い → p_market・EV・1/4 Kelly・自己インパクト補正・予算按分
  INF-->>S: InferenceResult
  S->>BQ: prediction（is_shadow = mode==shadow）
  alt mode が paper / live
    S->>BQ: bet_candidate
    S->>BQ: notification_log を確認（dedupe_key）
    S->>DC: race_embed（EV 閾値未満は送らない）
  end
```

失敗時の分岐（`InferenceOutcome`）:

| 例外・状況 | status | リトライ | 通知 |
|---|---|---|---|
| スケジュールに無い | failed | — | — |
| 推論窓の外 / 発走後 | expired | しない | — |
| 確定層が古い（`StaleDataError`） | failed | — | Blocker |
| NAR 取得失敗（一般例外） | failed | **する**（retryable） | — |
| `InsufficientData` / `ZeroFillForbidden` / `NormalizationError` | insufficient_data | しない | Critical |

### 3.6 `/refresh-live` と `/weekly-report`

```mermaid
flowchart TD
  L0["POST /refresh-live"] --> L1{"本日3回を超えたか（budget）"}
  L1 -- "yes" --> LN["noop"]
  L1 -- "no" --> L2["race_schedule から発走済みレースを列挙"]
  L2 --> L3["NarSource.fetch_results<br/>RaceMarkTable の着順 + DebaTable のキーを horse_no で結合"]
  L3 --> L4["speed_index = NaN で固定<br/>（当日分だけでは as-of 統計を再現できない）"]
  L4 --> L5["refresh_live → entry_result_live"]

  W0["POST /weekly-report"] --> W1["系統ごと（flat / banei）にモデル鮮度<br/>train_period.end から 90 日超でアラート"]
  W1 --> W2["直近30日のカバレッジ<br/>prediction レース数 / race_schedule レース数 < 0.90"]
  W2 --> W3["系統ごとに RF ガード<br/>prediction × entry_result_final の Top-1 > 0.60 / NLL < 1.20"]
  W3 --> W4{"blocks_delivery のアラートがあるか"}
  W4 -- "yes" --> W5["state.blocked（配信停止）"]
  W4 -- "no" --> W6["job_run 記録"]
```

### 3.7 運用モード（状態機械、`narops.mode`）

```mermaid
stateDiagram-v2
  [*] --> shadow: 既定（NAROPS_MODE 未設定）
  shadow --> paper: 人手で NAROPS_MODE=paper
  paper --> live: 人手で NAROPS_MODE=live（ModelProvenance=real が条件）
  shadow: shadow<br/>prediction(is_shadow) のみ
  paper: paper<br/>bet_candidate + Discord（仮想）
  live: live<br/>実投票前提
  paper --> 配信停止: skew 不一致 / RF ガード発火
  live --> 配信停止: 同上
  配信停止: delivery_blocked<br/>推論は続くが配信しない
```

`nar-refresh` Job の環境変数は `NAROPS_MODE=paper`（`infra/services.json`）。

---

## 4. 学習と運用で共有するコード（クラス図）

```mermaid
classDiagram
  direction LR
  class nar_transform_keys {
    +make_race_id(baba_code, date, race_no)
    +horse_sk(name, birth, sire)
    +add_horse_sk(df)
    +person_keys(df, name, affil, prefix)
  }
  class nar_transform_prerace {
    +POST_RACE_ALL
    +ASOF_RACE_WHITELIST
    +to_prerace(df)
    +assert_prerace(df)
  }
  class nar_features_builder {
    +ASOF_FEATURES
    +asof_features(cfg)
    +speed_index(entry)
    +build(entry, race, cfg)
  }
  class nar_eval_metrics {
    +race_softmax()
    +normalize_within_race()
    +harville_place_probability()
  }
  class ConstrainedStacker {
    +weights
    +predict_proba(preds, race_ids)
  }
  class narops_deba_table {
    +parse(html, race_id, date)
    +parse_race_header(html)
    +attach_keys(card)
  }
  class narops_features {
    +history_before(wh, as_of_ts, ...)
    +build_for_race(wh, card, row, manifest, cfg)
    +save_snapshot()
  }
  class narops_runtime {
    +Standardizer
    +LinearScorer
    +LgbmScorer
    +OnnxScorer
    +load_models(release)
  }
  class narops_inference {
    +run_inference()
    +blend()
    +PoolSizeModel
  }
  class narops_refresh {
    +daily_refresh()
    +_to_bq_payload()
    +write_entity_cache()
    +query_entity_cache()
  }
  narops_deba_table ..> nar_transform_keys : キー生成を共有
  narops_features ..> nar_transform_prerace : to_prerace
  narops_features ..> nar_features_builder : build
  narops_refresh ..> nar_features_builder : speed_index
  narops_inference ..> ConstrainedStacker : 重み付き幾何平均
  narops_inference ..> nar_eval_metrics : softmax / Harville
  narops_features --> narops_runtime : 特徴量 → 標準化
  narops_runtime --> narops_inference : スコア
```

---

## 5. コードを追って分かった設計とのずれ（2026-09-16）

図を起こす過程で、既存の設計書の記述と実装が食い違っている箇所を確認した。#3 以外は**この文書の作成では修正していない**（挙動を変えるため、判断が要る）。

| # | 重要度 | 内容 | 根拠 |
|---|---|---|---|
| 1 | 高 | **本番の skew 検証は実質的に働いていない。** `Design_Operation.md` §2.4 は「前日分を確定層のみから再計算して snapshot と比較」と書くが、`service.ingest_and_refresh_endpoint` は `compare(snap, snap, …)` と**同じフレーム同士**を比較しており常に一致する。しかもこのエンドポイントは旧経路で、実処理の `nar-refresh` Job（`refresh.daily_refresh`）は skew 検証を呼ばない。`narops skew-check` CLI も `--recomputed` を渡さなければ snap 同士の比較になり、再計算値を作るコードは存在しない。 | `operation/src/narops/service.py:403`、`cli.py:428`、`refresh.py::daily_refresh` |
| 2 | 中 | skew の許容列 `j_wins_today` / `t_wins_today` / `track_speed_bias` は特徴量として存在しない（騎手・調教師の「当日成績」や馬場差の特徴量は未実装）。許容リストは空振りしている。 | `conf/ops.yaml` の `skew.tolerated_columns`、`builder.ASOF_FEATURES` |
| 3 | ~~中~~ **修正済み（2026-09-17）** | ~~較正の適用順が学習と推論で違う。walk-forward では**モデルごとに**温度をかけてから OOF で重みを推定するが、推論は**温度なしのモデル別 softmax を合成してから**、重み最大モデルの温度を1回だけかける。現行の温度は 0.95〜0.99 なので数値差は小さいが、手順は一致していない。~~ `Manifest` に `model_temperatures`（モデルごとの温度）を追加し、`narops/inference.py::run_inference` がこれを使って学習と同じ「モデルごとに較正 → アンサンブル」の順序に変更した。この dict が無い旧リリースは従来どおり合成後に1回だけ較正する経路にフォールバックする。本番（flat: v2026.09.17-C、banei: v2026.09.17-B-banei）に反映済み。 | `train/pipeline.py::run_fold`、`narops/inference.py::run_inference`、`narops/model/manifest.py::Manifest.model_temperatures`、`narops/publish.py::build_release` |
| 4 | 中 | 特徴量選択の第3段（RFE）の内部分割は `race_id` 順の 75/25。`race_id` は競馬場コード始まりなので**時系列ではなく場コードでの分割**になっている（同じ問題を `_fit_predict_bayes` は `start_ts` で並べ直して回避している）。 | `features/selection.py::select` |
| 5 | 低 | 申告値の収縮 `d_*_winrate` の事前確率は `0.1` 固定（`declared.build` に `prior_win=None` が渡る）。docstring の「出走頭数の逆数相当」とは違い、自前集計の収縮（`1/n_runners`）とも揃っていない。 | `features/builder.py:383`、`features/declared.py:80` |
| 6 | 低 | `class_level` は `_class_level` が 1〜5 を定義するが、実データの `class_name` は 普通/一般/特別/重賞/準重賞 の5値だけで、値は **2・4・5 しか出ない**。またレース内で定数なので、条件付きロジットではレース内 softmax で相殺される（配布物の係数は約 1e-17）。 | `silver.race` の実測、`clogit_beta.json` |
| 7 | 低 | 推論時の `history_before` は「直近 `lookback_days` 日の全レース」も取得するが、事前確率を `1/n_runners`（対象レースの頭数）に変えて以降、対象行の特徴量はこの断片に依存しない。取得の必要性は再確認の余地がある（削る前に `test_entity_cache.py` の等価性テストで確認すること）。 | `narops/features.py::history_before` の docstring 2番 |
| 8 | 低 | `Design_Modeling.md` §9.1 の fold 表は embargo 30 日の時代のもので、実装は 181 日前で train を切る（本書 §2.4 の表が現行値）。 | `eval/splits.py::make_folds` |
