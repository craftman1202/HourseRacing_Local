# 自動発注（楽天競馬）運用メモ

クラウドの推論が終わると、PC 上のランナーが楽天競馬で自動発注し、結果を Discord に出す。
**実資金が動く。** 迷ったら `pause`（一時停止）してから調べる。

## 1. 構成

```
Cloud Run (nar-ops)  推論 → Discord に買い目を通知
        │              └→ GCS に発注計画 JSON  gs://nar-raw-sample-335613/orders/<日付>/<race_id>.json
        ▼
このPC  nar-orders.service（systemd）  20 秒ごとに GCS を見る
        │   各買い目: 上限・券種・時間窓を確認 → 台帳に「処理中」を原子的に宣言 → 楽天で発注
        ▼
楽天競馬  カゴを空にする → 入力 → カゴを読み戻して照合 → 確認画面を照合 → 「投票する」を1回だけ
        ▼
Discord に結果（受付 / 見送り / エラー）、受付画面を ~/.cache/nar-orders/receipts/ に保存
```

## 2. 毎日使うコマンド

すべて `operation/` で実行（`cd /home/machine_learner/HourseRacing_Local/operation`）。引数なしで実行すると一覧が出る。

| やりたいこと | コマンド |
|---|---|
| 今の状態と、今日の発注の一覧・合計 | `scripts/orders.sh status` |
| 今日の発注だけ見る | `scripts/orders.sh today` |
| ログをリアルタイムで見る | `scripts/orders.sh logs`（Ctrl+C で抜ける） |
| クラウドが置いた今日の計画 | `scripts/orders.sh plans` |
| **一時停止**（新しい発注だけ止める。ランナーは動いたまま） | `scripts/orders.sh pause` |
| 一時停止を解除 | `scripts/orders.sh resume` |
| **完全に止める** | `scripts/orders.sh stop` |
| 起動する | `scripts/orders.sh start` |
| 再起動（設定を変えたあとは必須） | `scripts/orders.sh restart` |
| PC 起動時の自動起動を切る / 戻す | `scripts/orders.sh disable` / `enable` |
| 今の設定を見る | `scripts/orders.sh config` |

Claude に「止めて」と言えば `pause` / `stop` を試みる（自動判定で拒否されることがある。その場合は上のコマンドを自分で実行する）。

## 3. 設定を変える（上限・券種・実発注/ドライラン）

実際に使われる設定は **`~/.cache/nar-orders/orders_live.yaml`**。リポジトリの `operation/conf/orders.yaml` は
その元（`live: false` の既定値）。**変えたら `scripts/orders.sh restart`。**

| 項目 | 意味 | 今の値 |
|---|---|---|
| `live` | `true` で実発注、`false` でドライラン（確認画面の手前まで。送信しない） | true |
| `bet_types` | 発注してよい券種。ここに無い券種は計画があっても見送り | 枠連・枠単以外の全券種 |
| `line_cap_yen` | 1 券種（1 回の買い目）の上限。超えたら点数を削らず見送り | ¥1,200 |
| `race_cap_yen` | 1 レースの上限 | ¥2,000 |
| `daily_cap_yen` | 1 日の合計上限（楽天の購入限度額 ¥3,690 以下にする） | ¥3,600 |
| `open_min_before` / `close_min_before` | 発走の何分前から / 何分前まで発注するか（楽天の締切は発走の 2 分前） | 10 / 3 |
| `max_consecutive_errors` | 連続でエラーになったら停止する回数 | 3 |

- 楽天側にも**購入限度額**がある（アカウント設定。2026-10-18 時点 ¥3,690）。超える注文は楽天に拒否される。
- 枠連・枠単は画面の構造を確認していないので、`bet_types` に入れていても発注しない（見送り）。

## 4. 何が起きたか確認する

- **Discord:** 発注ごとに「✅ 発注済み / ⏭ 見送り（理由つき）/ ❌ エラー」。ランナー起動時にも通知が出る。
- **台帳:** `~/.cache/nar-orders/ledger.sqlite`（`scripts/orders.sh today` で見られる）。状態は
  `placed`（受付済み）/ `skipped`（見送り）/ `rejected`（楽天が拒否）/ `error`（失敗）/ `pending`（処理中）/ `dry_run`（試験）。
- **受付画面:** `~/.cache/nar-orders/receipts/` に `<時刻>_<race_id>_<券種>_done.png`。受付番号・買い目・金額が写っている。
- 実際に買えたかの最終確認は楽天競馬の「履歴」。

## 5. 困ったとき

| 症状 | 原因と対処 |
|---|---|
| Discord に「❌ 楽天にログインしていません」 | ログインが切れた。`scripts/orders.sh pause` → 楽天に入り直す（下記）→ `resume`。メールで認証コードが届く |
| 台帳に `pending` が残っている | 発注の途中でプロセスが落ちた。**自動では再試行しない**（二重購入の防止）。楽天の「履歴」で買えているか確認し、買えていなければ台帳の行を直すか放置（その買い目は買われない） |
| 「送信後に完了画面を確認できません」 | 送信したが完了画面が出なかった。再送しない。必ず楽天の「履歴」で確認 |
| 「連続エラーで停止」 | 原因（Discord のエラー文）を直してから `scripts/orders.sh restart` |
| 発注されない | `status` で稼働中か → `plans` でクラウドが計画を置いたか → `config` の券種・上限 → 一時停止中でないか（`~/.cache/nar-orders/STOP` があれば停止中） |
| PC を再起動したら動いていない | WSL が止まっている。WSL を起動すれば自動で起動する（`enable` 済みなら）。ログアウト後も動かすには `loginctl enable-linger $USER` |

**楽天に入り直す手順:** 認証コードがメールで届く。Claude に依頼すれば、Gmail からコードを読んで入力できる
（Chrome のプロファイル `~/.cache/nar-orders/chrome-profile` を使うので、先にランナーを `stop` する）。

## 6. 中央競馬など、ランナーを並べて動かす

排他ロックは使っていない。同じ買い目を二重に買わないのは、台帳への**原子的な宣言**（`claim`）で保証している
（同じ台帳を見るランナーが何個動いても、同じレース・同じ券種は 1 つのランナーしか処理できない）。

別のランナー（例: 中央競馬用）を並べるときの注意:

- **ブラウザのプロファイルは別にする。** Chrome は 1 つのプロファイルを同時に 1 つのプロセスしか開けない。
  `NAR_ORDERS_PROFILE=~/.cache/nar-orders/chrome-profile-jra` のように指定する（初回はそのプロファイルで別途ログインが必要）。
- 台帳を分けたいなら `--ledger` か `NAR_ORDERS_LEDGER`。日次の上限は台帳ごとに数えるので、**合計の上限はランナーごとに分けて決める**
  （楽天の購入限度額は 1 アカウントで共通）。
- GCS の計画は同じ場所に置かれる。中央競馬の計画を同じバケットに置くなら、`race_id` が重ならないこと。
- サービスは別名（例: `nar-orders-jra.service`）にして、`infra/nar-orders.service` を複製する。

## 7. ファイルの場所

| もの | 場所 |
|---|---|
| ランナー本体 | `operation/src/narops/orders/`（`runner.py` / `guard.py` / `ledger.py` / `rakuten.py` / `report.py`） |
| 操作コマンド | `operation/scripts/orders.sh` |
| サービス定義 | `operation/infra/nar-orders.service`（実体 `~/.config/systemd/user/nar-orders.service`） |
| 設定（元） | `operation/conf/orders.yaml`（実際に使うのは `~/.cache/nar-orders/orders_live.yaml`） |
| 台帳・受付画面・プロファイル | `~/.cache/nar-orders/` |
| 認証情報 | `operation/.env`（`RAKUTEN_ID` / `RAKUTEN_PASS`。git 管理外。ログや通知には出さない） |
| 投票画面の調査用 | `operation/scripts/rakuten_probe.py`（購入しない） |
| 手動で計画を作る | `operation/scripts/plan_race.py <race_id>` |
