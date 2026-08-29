import type { DailyPnL } from "@/lib/nar-api";

/**
 * サーバー・クライアント両方から呼ぶ純粋関数はここに置く。
 * "use client" ファイルの export はサーバーから直接呼べない
 * （クライアント参照になるため）。
 */
export function toEquitySeries(data: DailyPnL[]): { date: string; equity: number }[] {
  const ordered = [...data].sort(
    (a, b) => new Date(a.business_date).getTime() - new Date(b.business_date).getTime(),
  );
  const series: { date: string; equity: number }[] = [];
  for (const row of ordered) {
    const prevEquity = series.length > 0 ? series[series.length - 1].equity : 0;
    series.push({ date: row.business_date, equity: prevEquity + row.return_yen - row.stake_yen });
  }
  return series;
}

export function computeDrawdown(rows: DailyPnL[]): { maxDrawdown: number; current: number } {
  const series = toEquitySeries(rows);
  let peak = 0;
  let maxDrawdown = 0;
  for (const { equity } of series) {
    peak = Math.max(peak, equity);
    maxDrawdown = Math.min(maxDrawdown, equity - peak);
  }
  const last = series.length > 0 ? series[series.length - 1].equity : 0;
  return { maxDrawdown, current: last - peak };
}
