#!/usr/bin/env bash
# 自動発注ランナー（楽天競馬）の操作をまとめたコマンド。使い方は引数なしで実行すると出る。
# 詳しい説明: docs/AutoOrders.md
SERVICE=nar-orders.service
STATE="$HOME/.cache/nar-orders"
LEDGER="${NAR_ORDERS_LEDGER:-$STATE/ledger.sqlite}"
CONF="$STATE/orders_live.yaml"
BUCKET="gs://nar-raw-sample-335613/orders"

usage() {
  cat <<'EOF'
使い方: scripts/orders.sh <コマンド>

  status      ランナーの状態と、今日の発注の一覧・合計
  logs        ランナーのログを流す（Ctrl+C で抜ける。ランナーは止まらない）
  today       今日の発注の一覧と合計（台帳）
  plans       今日クラウドが置いた発注計画の一覧（GCS）

  pause       発注を一時停止（ランナーは動いたまま、新しい発注だけ止める）  ← まずこれ
  resume      一時停止を解除
  stop        ランナーを止める（もう発注しない。自動起動もしないまま）
  start       ランナーを起動する
  restart     ランナーを再起動する（設定ファイルを変えたあとに必要）
  disable     PC を起動しても自動では動かさない      enable  自動起動を戻す

  config      今の設定（上限・券種・実発注かドライランか）を見る
  receipts    保存してある受付画面（スクリーンショット）の置き場を開く場所を表示
EOF
}

today_summary() {
  python3 - "$LEDGER" <<'PY'
import sqlite3, sys, datetime
db = sqlite3.connect(sys.argv[1])
d = datetime.date.today().isoformat()
rows = list(db.execute("select race_id, bet_type, stake_yen, status, substr(detail,1,48), at from orders "
                       "where race_date=? order by at", (d,)))
icon = {"placed": "✅受付", "dry_run": "🧪試験", "skipped": "⏭見送", "rejected": "⛔拒否", "error": "❌エラー", "pending": "⏳処理中"}
for r in rows:
    print(f"{r[0]}  {r[1]:<5} ¥{r[2]:>5,}  {icon.get(r[3], r[3]):<7} {r[4]}")
tot = lambda s: sum(r[2] for r in rows if r[3] == s)
print(f"--- {d}  受付 ¥{tot('placed'):,}（{sum(1 for r in rows if r[3]=='placed')}件）"
      f" / 見送り {sum(1 for r in rows if r[3]=='skipped')}件 / エラー・拒否 {sum(1 for r in rows if r[3] in ('error','rejected'))}件"
      f" / 処理中 {sum(1 for r in rows if r[3]=='pending')}件")
if any(r[3] == "pending" for r in rows):
    print("⚠ 処理中のまま残っている行は、途中で落ちた注文です。二重購入を避けるため自動では再試行しません。"
          "楽天の「履歴」で買えているか確認してください。")
PY
}

case "$1" in
  status)   systemctl --user status "$SERVICE" --no-pager | head -8; echo; today_summary ;;
  logs)     journalctl --user -u "$SERVICE" -f -n 30 ;;
  today)    today_summary ;;
  plans)    gcloud storage ls -l "$BUCKET/$(date +%F)/" 2>&1 | tail -30 ;;
  pause)    mkdir -p "$STATE" && touch "$STATE/STOP" && echo "一時停止しました（$STATE/STOP）。解除は: scripts/orders.sh resume" ;;
  resume)   [ -e "$STATE/STOP" ] && unlink "$STATE/STOP"; echo "一時停止を解除しました" ;;
  stop)     systemctl --user stop "$SERVICE" && echo "停止しました。再開は: scripts/orders.sh start" ;;
  start)    systemctl --user start "$SERVICE" && echo "起動しました" ;;
  restart)  systemctl --user restart "$SERVICE" && echo "再起動しました" ;;
  disable)  systemctl --user disable "$SERVICE" ;;
  enable)   systemctl --user enable "$SERVICE" ;;
  config)   grep -v '^\s*#' "$CONF" | grep -v '^\s*$'; echo; echo "（設定ファイル: $CONF）" ;;
  receipts) echo "$STATE/receipts/   （受付画面のスクリーンショットと HTML）" ;;
  *)        usage ;;
esac
