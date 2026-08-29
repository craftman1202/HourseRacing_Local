import Link from "next/link";

import type { TodayRace } from "@/lib/nar-api";
import { evTone, fmtEv } from "@/lib/format";
import { Badge } from "@/components/ui/badge";

const TONE_BLOCK_CLASS: Record<ReturnType<typeof evTone>, string> = {
  success: "border-success bg-success/10 hover:bg-success/20",
  warning: "border-warning bg-warning/10 hover:bg-warning/20",
  neutral: "border-border bg-muted hover:bg-border",
};

const STATUS_LABEL: Record<TodayRace["status"], string> = {
  ok: "配信済み",
  insufficient_data: "データ不足",
  expired: "未算出のまま発走",
  not_computed: "未算出",
};

function timeLabel(iso: string): string {
  return new Date(iso).toLocaleTimeString("ja-JP", { hour: "2-digit", minute: "2-digit" });
}

/** 場ごとに時系列で並べたタイムライン（設計書 §5.2(1)）。 */
export function TodayTimeline({ races }: { races: TodayRace[] }) {
  if (races.length === 0) {
    return <p className="text-sm text-muted-foreground">本日の開催データがありません。</p>;
  }

  const byTrack = new Map<string, TodayRace[]>();
  for (const r of races) {
    const list = byTrack.get(r.track_name) ?? [];
    list.push(r);
    byTrack.set(r.track_name, list);
  }
  for (const list of byTrack.values()) {
    list.sort((a, b) => new Date(a.start_ts).getTime() - new Date(b.start_ts).getTime());
  }

  return (
    <div className="flex flex-col gap-3">
      {[...byTrack.entries()].map(([track, list]) => (
        <div key={track} className="flex items-center gap-3">
          <span className="w-16 shrink-0 text-xs text-muted-foreground">{track}</span>
          <div className="flex flex-1 flex-wrap gap-2">
            {list.map((r) => {
              const tone = r.status === "ok" ? evTone(r.top_ev_adjusted) : "neutral";
              return (
                <Link
                  key={r.race_id}
                  href={`/race/${r.race_id}`}
                  className={`flex min-w-20 flex-col items-center gap-0.5 rounded-md border px-2 py-1.5 text-xs transition-colors ${TONE_BLOCK_CLASS[tone]}`}
                  title={r.status === "ok" ? undefined : STATUS_LABEL[r.status]}
                >
                  <span className="font-medium">{r.race_no}R</span>
                  <span className="text-muted-foreground">{timeLabel(r.start_ts)}</span>
                  {r.status === "ok" ? (
                    <span>{fmtEv(r.top_ev_adjusted)}</span>
                  ) : (
                    <Badge tone="neutral" className="px-1.5 py-0 text-[10px]">
                      {STATUS_LABEL[r.status]}
                    </Badge>
                  )}
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
