#!/usr/bin/env bash
# 自動発注ランナー（楽天競馬）の操作をまとめたコマンド。引数なしで実行すると使い方が出る。
# 詳しい説明: docs/AutoOrders.md
set -u
STATE="$HOME/.cache/nar-orders"
CONFD="$HOME/.config/nar-orders"
UNITS="$HOME/.config/systemd/user"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NAME="main"
if [ "${1:-}" = "-i" ]; then NAME="${2:?-i の後にランナーの名前}"; shift 2; fi
SERVICE="nar-orders@${NAME}.service"
ENVFILE="$CONFD/$NAME.env"
BUCKET="gs://nar-raw-sample-335613/orders"

env_value() { grep -E "^$1=" "$ENVFILE" 2>/dev/null | head -1 | cut -d= -f2-; }
LEDGER="$(env_value NAR_ORDERS_LEDGER)"; LEDGER="${LEDGER:-$STATE/ledger.sqlite}"
CONF="$(env_value NAR_ORDERS_CONF)";     CONF="${CONF:-$STATE/orders_live.yaml}"

usage() {
  cat <<'EOF'
使い方: scripts/orders.sh [-i 名前] <コマンド>
        -i を付けないときは名前 = main（nar-orders@main.service）

  list              動いているランナーを全部表示（名前・状態・一時停止中か）
  status            このランナーの状態と、今日の発注の一覧・合計・損失
  logs              このランナーのログを流す（Ctrl+C で抜ける。ランナーは止まらない）
  today             今日の発注の一覧と合計（台帳。ランナー共通）
  plans             今日クラウドが置いた発注計画の一覧（GCS）

  pause             このランナーを一時停止（動いたまま、新しい発注だけ止める。静かに待つ）  ← まずこれ
  resume            一時停止を解除（再開）
  pause-all         全ランナーを一時停止      resume-all  全ランナーの一時停止を解除
  stop              ランナーを止める（もう発注しない）      start    起動する
  restart           再起動（設定ファイルを変えたあとは必須）
  disable / enable  PC を起動したときの自動起動を切る / 戻す

  config            このランナーの設定（損失上限・券種・実発注かドライランか）
  new <名前>        別のランナーを追加する（設定・プロファイル・台帳の雛形を作る。起動はしない）
  receipts          保存してある受付画面（スクリーンショット）の置き場
EOF
}

paused_state() {
  if [ -e "$STATE/STOP" ]; then echo "一時停止中（全ランナー）"
  elif [ -e "$STATE/STOP.$1" ]; then echo "一時停止中"
  else echo "発注中"; fi
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
base = db.execute("select balance_yen from baseline where race_date=?", (d,)).fetchone()
print(f"--- {d}  受付 ¥{tot('placed'):,}（{sum(1 for r in rows if r[3]=='placed')}件）"
      f" / 見送り {sum(1 for r in rows if r[3]=='skipped')}件 / エラー・拒否 {sum(1 for r in rows if r[3] in ('error','rejected'))}件"
      f" / 処理中 {sum(1 for r in rows if r[3]=='pending')}件")
print("    今日の基準残高:", f"¥{base[0]:,}" if base else "（まだ記録なし）",
      " ※損失 = 基準残高 − 今の精算可能額。今の残高は楽天の画面か受付画面で見る")
if any(r[3] == "pending" for r in rows):
    print("⚠ 処理中のまま残っている行は、途中で落ちた注文です。二重購入を避けるため自動では再試行しません。"
          "楽天の「履歴」で買えているか確認してください。")
PY
}

case "${1:-}" in
  list)
    for f in "$CONFD"/*.env; do
      [ -e "$f" ] || continue
      n="$(basename "$f" .env)"
      printf '%-12s %-9s %-10s %s\n' "$n" "$(systemctl --user is-active "nar-orders@$n.service")" \
        "$(systemctl --user is-enabled "nar-orders@$n.service" 2>/dev/null)" "$(paused_state "$n")"
    done ;;
  status)   systemctl --user status "$SERVICE" --no-pager | head -8; echo "発注: $(paused_state "$NAME")"; echo; today_summary ;;
  logs)     journalctl --user -u "$SERVICE" -f -n 30 ;;
  today)    today_summary ;;
  plans)    gcloud storage ls -l "$BUCKET/$(date +%F)/" 2>&1 | tail -30 ;;
  pause)    mkdir -p "$STATE" && touch "$STATE/STOP.$NAME" && echo "[$NAME] 一時停止しました。解除: scripts/orders.sh -i $NAME resume" ;;
  resume)   [ -e "$STATE/STOP.$NAME" ] && unlink "$STATE/STOP.$NAME"
            [ -e "$STATE/STOP" ] && echo "※全ランナー一時停止（$STATE/STOP）が残っています。resume-all で解除"
            echo "[$NAME] 一時停止を解除しました" ;;
  pause-all)  mkdir -p "$STATE" && touch "$STATE/STOP" && echo "全ランナーを一時停止しました。解除: scripts/orders.sh resume-all" ;;
  resume-all) [ -e "$STATE/STOP" ] && unlink "$STATE/STOP"; echo "全ランナーの一時停止を解除しました" ;;
  stop)     systemctl --user stop "$SERVICE" && echo "[$NAME] 停止しました。再開: scripts/orders.sh -i $NAME start" ;;
  start)    systemctl --user start "$SERVICE" && echo "[$NAME] 起動しました（発注は: $(paused_state "$NAME")）" ;;
  restart)  systemctl --user restart "$SERVICE" && echo "[$NAME] 再起動しました（発注は: $(paused_state "$NAME")）" ;;
  disable)  systemctl --user disable "$SERVICE" ;;
  enable)   systemctl --user enable "$SERVICE" ;;
  config)   [ -f "$CONF" ] && grep -v '^\s*#' "$CONF" | grep -v '^\s*$'; echo; echo "（設定: $CONF / 台帳: $LEDGER / 環境: $ENVFILE）" ;;
  new)
    N="${2:?new の後に名前}"
    [ -e "$CONFD/$N.env" ] && { echo "$N は既にあります（$CONFD/$N.env）"; exit 1; }
    mkdir -p "$CONFD" "$STATE/$N"
    sed 's/^live: true/live: false/' "$HERE/conf/orders.yaml" > "$STATE/$N/orders.yaml"
    cat > "$CONFD/$N.env" <<EOT
NAR_ORDERS_NAME=$N
NAR_ORDERS_CONF=$STATE/$N/orders.yaml
NAR_ORDERS_PROFILE=$STATE/$N/chrome-profile
NAR_ORDERS_LEDGER=$STATE/ledger.sqlite
EOT
    touch "$STATE/STOP.$N"
    echo "作成: $CONFD/$N.env と $STATE/$N/orders.yaml（live: false、一時停止状態）"
    echo "次: 設定（$STATE/$N/orders.yaml）を直す → 楽天にログイン（プロファイル $STATE/$N/chrome-profile）→"
    echo "    systemctl --user enable --now nar-orders@$N.service → scripts/orders.sh -i $N resume" ;;
  receipts) echo "$STATE/receipts/   （受付画面のスクリーンショットと HTML）" ;;
  *)        usage ;;
esac
