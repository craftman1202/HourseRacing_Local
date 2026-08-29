import type { HorsePrediction } from "@/lib/nar-api";
import { fmtPct } from "@/lib/format";

/** 予測確率と市場確率の乖離を横棒で可視化する（設計書 §5.2(2)）。 */
export function ProbabilityGapBar({ horses }: { horses: HorsePrediction[] }) {
  const withMarket = horses.filter((h) => h.p_market !== null && h.p_market !== undefined);
  if (withMarket.length === 0) {
    return <p className="text-sm text-muted-foreground">市場確率が未取得です</p>;
  }
  const maxAbs = Math.max(
    ...withMarket.map((h) => Math.abs(h.p_win - (h.p_market ?? 0))),
    0.01,
  );

  return (
    <div className="flex flex-col gap-1.5">
      {withMarket.map((h) => {
        const gap = h.p_win - (h.p_market ?? 0);
        const widthPct = (Math.abs(gap) / maxAbs) * 50;
        const positive = gap >= 0;
        return (
          <div key={h.horse_no} className="flex items-center gap-2 text-xs">
            <span className="w-10 shrink-0 text-muted-foreground">#{h.horse_no}</span>
            <div className="relative h-4 flex-1 bg-muted">
              <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
              <div
                className={`absolute inset-y-0 ${positive ? "bg-success" : "bg-danger"}`}
                style={{
                  width: `${widthPct}%`,
                  left: positive ? "50%" : `${50 - widthPct}%`,
                }}
              />
            </div>
            <span className="w-28 shrink-0 tabular-nums">
              {fmtPct(h.p_win)} / {fmtPct(h.p_market)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
