"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { formatDelta } from "@/components/race/countdown";

/**
 * 「次の推奨レースまでのカウントダウン」（設計書 §5.2(1)、最も見られる要素）。
 * サーバー負荷ゼロ: 対象レースは呼び出し側（サーバー側）で選定済みで、ここでは
 * setInterval によるクライアント側の秒刻み表示だけを行う。
 */
export function NextRaceCountdown({
  raceId,
  trackName,
  raceNo,
  startTsIso,
}: {
  raceId: string;
  trackName: string;
  raceNo: number;
  startTsIso: string;
}) {
  const [now, setNow] = useState<number>(() => Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const delta = new Date(startTsIso).getTime() - now;

  return (
    <Link
      href={`/race/${raceId}`}
      className="flex flex-col gap-1 rounded-lg border border-border bg-card p-4 hover:border-accent"
    >
      <span className="text-xs text-muted-foreground">次の推奨レース</span>
      <span className="text-sm">
        {trackName} {raceNo}R
      </span>
      <span className="text-3xl font-semibold tabular-nums text-accent" suppressHydrationWarning>
        {formatDelta(delta)}
      </span>
    </Link>
  );
}
