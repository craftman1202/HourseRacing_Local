# 自動発注（楽天競馬）運用メモ

クラウドの推論が終わると、PC 上のランナーが楽天競馬で自動発注し、結果を Discord に出す。
**実資金が動く。** 迷ったら `pause`（一時停止）してから調べる。

> **現在の状態（2026-10-19）: 一時停止中。** 損失ベースの上限（下記）はまだ実レースで検証していないため、
> 来週のレースで挙動を確認してから再開する（再開は `scripts/orders.sh resume-all`、指示があるまで解除しない）。

## 1. 構成

```
Cloud Run (nar-ops)  推論 → Discord に買い目を通知
        │              └→ GCS に発注計画 JSON  gs://nar-raw-sample-335613/orders/<日付>/<race_id>.json
        ▼
このPC  nar-orders@<名前>.service（systemd のテンプレート。ログイン無しでも自動起動）
        │   20 秒ごとに GCS を見て、各買い目を:
        │   時間窓・券種・上限を確認 → 台帳に「処理中」を原子的に宣言 → 楽天で発注
        ▼
楽天競馬  カゴを空にする → 入力 → カゴを読み戻して照合 → 確認画面を照合 → 「投票する」を1回だけ
        ▼
Discord に結果（受付 / 見送り / エラー）、受付画面を ~/.cache/nar-orders/receipts/ に保存
```

## 2. 毎日使うコマンド

`operation/` で実行（`cd /home/machine_learner/HourseRacing_Local/operation`）。引数なしで使い方が出る。
ランナーは名前で区別する（既定は `main`）。他の名前を操作するときは `-i 名前` を先頭に付ける（例: `scripts/orders.sh -i jra pause`）。

| やりたいこと | コマンド |
|---|---|
| **動いているランナーの一覧**（名前・状態・自動起動・一時停止中か） | `scripts/orders.sh list` |
| 状態と、今日の発注の一覧・合計・基準残高 | `scripts/orders.sh status` |
| 今日の発注だけ見る（台帳。全ランナー共通） | `scripts/orders.sh today` |
| ログをリアルタイムで見る | `scripts/orders.sh logs`（Ctrl+C で抜ける） |
| クラウドが置いた今日の計画 | `scripts/orders.sh plans` |
| **一時停止**（このランナーだけ。動いたまま、新しい発注だけ止める。通知も出さず静かに待つ） | `scripts/orders.sh pause` |
| 一時停止の解除（再開） | `scripts/orders.sh resume` |
| **全ランナーを一時停止 / 解除** | `scripts/orders.sh pause-all` / `resume-all` |
| 完全に止める / 起動する | `scripts/orders.sh stop` / `start` |
| 再起動（設定ファイルを変えたあとは必須） | `scripts/orders.sh restart` |
| PC 起動時の自動起動を切る / 戻す | `scripts/orders.sh disable` / `enable` |
| 今の設定を見る | `scripts/orders.sh config` |
| **別のランナーを追加する**（雛形を作る。起動はしない） | `scripts/orders.sh new <名前>` |

Claude に「止めて」と言えば `pause` / `stop` を試みる（自動判定で拒否されることがある。その場合は上のコマンドを自分で実行する）。
一時停止は「停止ファイル」で行う（全体: `~/.cache/nar-orders/STOP`、個別: `STOP.<名前>`）。ファイルを作る/消すだけでよい。

## 3. 1 日の上限 = 損失 ¥3,000（賭け額ではない）

```
損失 = その日の基準残高 − 今の精算可能額 + 処理中の買い目の額
買い目を足してよい  ⇔  損失 + この買い目の額 ≤ 損失上限（¥3,000）
```

- **精算可能額**は楽天の投票画面の見出しにある値。購入すると減り、払戻があると増える。発注の直前に毎回読む。
- **基準残高**はその日の最初に記録する（台帳の `baseline` 表）。途中から動き始めたときは「今の残高 + 今日すでに受け付けた額」とする
  （払戻が既にあれば損失を多めに見積もる＝安全側）。
- **未確定の注文は全部外れると見なす**（買った額は残高から引かれているので、自然にそうなる）。
- **利益が出ても上限は広げない**（損失は 0 を下限にする）。入金で残高が増えたのを利益と取り違えないため。
- 残高を読めないときは発注しない（見送り）。
- 賭け額の合計上限は無くなった（1 券種・1 レースの上限は残っている）。楽天側の購入限度額（アカウント設定）は別に効く。

## 3.5 どの戦略を発注するか（券種ごとの採用基準）

買い目の規則は **`operation/conf/bet_rules.yaml`**（クラウドが読む。変えたら nar-ops の再デプロイが必要）。2026-10-20 に次の基準で選び直した:

- 候補: EV 系（EV の下限 1.0〜10.0 を 0.1 刻み × **的中確率の足切り `p_min` 0〜60% を 1% 刻み** × 全部買う/レース最大1点）と、無オッズ系（買い方 × 人気との混合 × 軸の確率の下限 0〜80% を 1% 刻み）
- 選ぶ基準: 2020〜2023 の検証期間で、**ブートストラップ（開催日単位）の ROI 中央値が 1 以上**、かつ **CI 下限が最大**。ベット 200 レース未満は除外
- 条件を満たす戦略が無い券種（今は枠連）は**採用なし＝買い目を出さない**
- 実運用に乗せているのは **単勝・複勝・馬連・ワイド** だけ（`bet_rules.yaml` の規則も `bet_types` もこの 4 券種）。馬単・3連複・3連単・枠単は、探索で選ばれても OOS が 1 未満だったので外した
- 候補が多い（1 券種あたり数千〜1 万）ほど、最良を選ぶことで検証期間に過適合する。OOS の確認（`strategy_ci_oos.py`）を必ず見る
- 選び直すとき: `learning/scripts/strategy_ci_select.py` → `strategy_ci_oos.py`（結果は `learning/artifacts/ci_select/`）→ `bet_rules.yaml` を更新 → 再デプロイ
- OOS（2024-02〜）は選択に使っていない確認用。10/17 に一度開封済みなので、完全な未見データではない

## 4. 設定を変える

実際に使われる設定は **`~/.cache/nar-orders/orders_live.yaml`**（`main` の場合。`scripts/orders.sh config` で場所が出る）。
リポジトリの `operation/conf/orders.yaml` はその元（`live: false` の既定値）。**変えたら `scripts/orders.sh restart`。**

| 項目 | 意味 | 今の値 |
|---|---|---|
| `live` | `true` で実発注、`false` でドライラン（確認画面の手前まで。送信しない） | true |
| `bet_types` | 発注してよい券種。ここに無い券種は計画があっても見送り | **単勝・複勝・馬連・ワイド**（2026-10-20 ユーザー指示） |
| `line_cap_yen` | 1 券種（1 回の買い目）の賭け額上限。超えたら点数を削らず見送り | ¥1,200 |
| `race_cap_yen` | 1 レースの賭け額上限 | ¥2,000 |
| `daily_loss_cap_yen` | **1 日の損失の上限**（3 章） | ¥3,000 |
| `open_min_before` / `close_min_before` | 発走の何分前から / 何分前まで発注するか（楽天の締切は発走の 2 分前） | 10 / 3 |
| `max_consecutive_errors` | 連続でエラーになったら停止する回数 | 3 |

古い `daily_cap_yen`（賭け額の上限）が設定に残っていると、ランナーは起動時にエラーで止まる（取り違え防止）。

## 5. 何が起きたか確認する

- **Discord:** 発注ごとに「✅ 発注済み / ⏭ 見送り（理由つき）/ ❌ エラー」。ランナー起動時にも通知（名前つき）。
- **台帳:** `~/.cache/nar-orders/ledger.sqlite`（`scripts/orders.sh today`）。状態は
  `placed`（受付済み）/ `skipped`（見送り）/ `rejected`（楽天が拒否・未対応券種）/ `error`（失敗）/ `pending`（処理中）/ `dry_run`（試験）。
- **受付画面:** `~/.cache/nar-orders/receipts/` に `<時刻>_<race_id>_<券種>_done.png`。受付番号・買い目・金額が写っている。
- 実際に買えたかの最終確認は楽天競馬の「履歴」。

## 6. 困ったとき

| 症状 | 原因と対処 |
|---|---|
| Discord に「精算可能額を読めず…見送り」 | ログイン切れの可能性が高い。`pause` → 楽天に入り直す（下記）→ `resume` |
| 台帳に `pending` が残っている | 発注の途中でプロセスが落ちた。**自動では再試行しない**（二重購入の防止）。楽天の「履歴」で買えているか確認 |
| 「送信後に完了画面を確認できません」 | 送信したが完了画面が出なかった。再送しない。必ず楽天の「履歴」で確認 |
| 「連続エラーで停止」 | 原因（Discord のエラー文）を直してから `scripts/orders.sh restart` |
| 発注されない | `list` で稼働中・一時停止中でないか → `plans` でクラウドが計画を置いたか → `config` の券種・上限 → 損失が上限に達していないか（`status`） |
| PC を再起動したら動いていない | WSL が止まっている。WSL を起動すれば自動で起動する（`enable` 済みなら）。ログアウト後も動かすには `loginctl enable-linger $USER` |

**楽天に入り直す手順:** 認証コードがメールで届く。Claude に依頼すれば Gmail からコードを読んで入力できる
（Chrome のプロファイルを使うので、先にそのランナーを `stop` する）。

## 7. 複数のランナーを同時に動かす（地方競馬向けの別サービス・中央競馬など）

ランナーは systemd のテンプレート `nar-orders@<名前>.service`。名前ごとに環境ファイル `~/.config/nar-orders/<名前>.env` を持つ。
**排他ロックは使わない。** 同じ買い目を二重に買わないのは、台帳への原子的な宣言（`claim`）で保証している
（同じ台帳を見るランナーが何個動いても、同じレース・同じ券種は 1 つのランナーしか処理できない）。

追加する手順:

1. `scripts/orders.sh new <名前>` … 環境ファイル・設定（`live: false`）・プロファイル置き場を作り、**一時停止状態**にする
2. 設定 `~/.cache/nar-orders/<名前>/orders.yaml` を直す（担当する券種・上限など）
3. そのプロファイル（`~/.cache/nar-orders/<名前>/chrome-profile`）で楽天にログイン（認証コードが届く）
4. `systemctl --user enable --now nar-orders@<名前>.service`
5. 挙動を確認してから `scripts/orders.sh -i <名前> resume`

注意:

- **Chrome のプロファイルは別々にする。** 1 つのプロファイルは同時に 1 つのプロセスしか開けない（雛形は別にしてある）。
- **台帳は共有する**（雛形も共有）。1 日の損失は楽天アカウントの残高で測るので、ランナーが何個あっても**アカウント全体で ¥3,000**。
  ランナーごとに上限を分けたいなら台帳も分けるが、その場合は各ランナーが同じ残高を見て損失を数えるので、合計は各上限の和を超えうる。
- **同じ種類のランナーを 2 つ動かさない。** 同じ計画を 2 つが拾うと、先に宣言した方だけが発注し、もう一方は何もしない（無駄だが安全）。
- 計画（GCS）の置き場は共通。別サービスが計画を置くなら `race_id` が重ならないこと。

## 8. ファイルの場所

| もの | 場所 |
|---|---|
| ランナー本体 | `operation/src/narops/orders/`（`runner.py` / `guard.py` / `ledger.py` / `rakuten.py` / `report.py`） |
| 操作コマンド | `operation/scripts/orders.sh` |
| サービス定義（テンプレート） | `operation/infra/nar-orders@.service`（実体 `~/.config/systemd/user/nar-orders@.service`） |
| ランナーごとの環境 | `~/.config/nar-orders/<名前>.env`（設定・プロファイル・台帳の場所） |
| 設定（元） | `operation/conf/orders.yaml`（実際に使うのは `main` なら `~/.cache/nar-orders/orders_live.yaml`） |
| 台帳・受付画面・プロファイル・停止ファイル | `~/.cache/nar-orders/` |
| 認証情報 | `operation/.env`（`RAKUTEN_ID` / `RAKUTEN_PASS`。git 管理外。ログや通知には出さない） |
| 投票画面の調査用（購入しない） | `operation/scripts/rakuten_probe.py` |
| 手動で計画を作る | `operation/scripts/plan_race.py <race_id>` |
