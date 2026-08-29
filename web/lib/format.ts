/**
 * 表示フォーマットは operation/src/narops/discord/format.py の丸め規則と揃える。
 * Discord に届く数値と Web に出す数値の見え方が違うと信頼を損なう（DC-05/DC-06）。
 */

export function fmtPct(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

export function fmtEv(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(2);
}

export function fmtYen(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x) || x <= 0) return "—";
  return `¥${Math.trunc(x).toLocaleString("ja-JP")}`;
}

export function fmtNum(x: number | null | undefined, digits = 3): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

/**
 * EV の緑/黄/グレー3段しきい値。discord/format.py::ev_color と同じ数値
 * （green=1.20, yellow=1.05）を UI の色分けとしてミラーする。計算そのものは
 * サーバー側（narops.shared / nar.eval.economic）の単一実装のみが行い、
 * ここは表示の色分けだけを担う。
 */
export const EV_COLOR_THRESHOLDS = { green: 1.2, yellow: 1.05 } as const;

export type EvTone = "success" | "warning" | "neutral";

export function evTone(
  ev: number | null | undefined,
  green = EV_COLOR_THRESHOLDS.green,
  yellow = EV_COLOR_THRESHOLDS.yellow,
): EvTone {
  if (ev === null || ev === undefined || Number.isNaN(ev)) return "neutral";
  if (ev >= green) return "success";
  if (ev >= yellow) return "warning";
  return "neutral";
}

/** 緑/黄/グレーの3段。EV バッジ・アラート表示の色分けを Discord と揃える。 */
export function evColorClass(
  ev: number | null | undefined,
  green = EV_COLOR_THRESHOLDS.green,
  yellow = EV_COLOR_THRESHOLDS.yellow,
): string {
  if (ev === null || ev === undefined || Number.isNaN(ev)) return "text-muted-foreground";
  if (ev >= green) return "text-emerald-500";
  if (ev >= yellow) return "text-amber-500";
  return "text-muted-foreground";
}
