"use client";

import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";

export function formatDelta(ms: number): string {
  if (ms <= 0) return "発走済み";
  const totalSec = Math.floor(ms / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  return h > 0
    ? `${h}時間${m}分${s}秒`
    : `${m}分${s}秒`;
}

/**
 * サーバー負荷ゼロ: setInterval はクライアント側のみで動く（設計書 §5.3）。
 * 初期値は useState の遅延初期化子で作る（effect 内での同期 setState を避ける）。
 * SSR 時刻とハイドレーション時刻は必ず1秒未満ずれるため、この秒刻み表示だけは
 * suppressHydrationWarning で許容する（React 公式が挙げる正当なケース）。
 */
export function RaceCountdown({ startTsIso }: { startTsIso: string }) {
  const [now, setNow] = useState<number>(() => Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const delta = new Date(startTsIso).getTime() - now;
  const started = delta <= 0;

  return (
    <div className="flex flex-col items-end gap-1">
      <span className="text-2xl font-semibold tabular-nums" suppressHydrationWarning>
        {formatDelta(delta)}
      </span>
      {started && <Badge tone="warning">発走後 — 参考表示</Badge>}
    </div>
  );
}

/** 発走後は投票補助 UI を無効化する（WB-05 と同じ判定、表示制御のみ）。 */
export function useBettingUiEnabled(startTsIso: string): boolean {
  const [enabled, setEnabled] = useState(() => Date.now() < new Date(startTsIso).getTime());
  useEffect(() => {
    const id = setInterval(() => {
      setEnabled(Date.now() < new Date(startTsIso).getTime());
    }, 1000);
    return () => clearInterval(id);
  }, [startTsIso]);
  return enabled;
}
