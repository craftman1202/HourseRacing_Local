"use client";

import { useEffect, useRef, useState } from "react";

import { fmtPct } from "@/lib/format";

type Tick = { horse_no: number; p_market: number | null }[];

/** 発走10分前のみ SSE 接続を張る。それ以外は何もしない（課金対策）。 */
export function LiveOdds({ raceId, startTsIso }: { raceId: string; startTsIso: string }) {
  const [withinWindow, setWithinWindow] = useState(false);
  const [tick, setTick] = useState<Tick | null>(null);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const check = () => {
      const deltaMin = (new Date(startTsIso).getTime() - Date.now()) / 60000;
      setWithinWindow(deltaMin > 0 && deltaMin <= 10);
    };
    check();
    const id = setInterval(check, 15000);
    return () => clearInterval(id);
  }, [startTsIso]);

  useEffect(() => {
    if (!withinWindow) {
      esRef.current?.close();
      esRef.current = null;
      return;
    }
    const es = new EventSource(`/api/sse/odds/${encodeURIComponent(raceId)}`);
    es.onmessage = (e) => {
      try {
        setTick(JSON.parse(e.data) as Tick);
        setUpdatedAt(new Date());
      } catch {
        // 壊れたイベントは無視
      }
    };
    es.onerror = () => es.close();
    esRef.current = es;
    return () => es.close();
  }, [withinWindow, raceId]);

  if (!withinWindow || !tick) return null;

  return (
    <div className="rounded-md border border-border bg-muted p-3 text-xs">
      <p className="mb-2 font-medium text-muted-foreground">
        ライブオッズ（発走10分前〜） {updatedAt && `更新: ${updatedAt.toLocaleTimeString("ja-JP")}`}
      </p>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {tick.map((t) => (
          <span key={t.horse_no}>
            #{t.horse_no} {fmtPct(t.p_market)}
          </span>
        ))}
      </div>
    </div>
  );
}
