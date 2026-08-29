"use client";

import { useEffect, useState } from "react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import { useBettingUiEnabled } from "@/components/race/countdown";

type Site = "rakuten" | "spat4";

export function BetslipPanel({ raceId, startTsIso }: { raceId: string; startTsIso: string }) {
  const [site, setSite] = useState<Site>("rakuten");
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const toast = useToast();
  const enabled = useBettingUiEnabled(startTsIso);

  useEffect(() => {
    let cancelled = false;
    Promise.resolve()
      .then(() => {
        setLoading(true);
        setError(null);
      })
      .then(() => fetch(`/api/races/${encodeURIComponent(raceId)}/betslip?site=${site}`))
      .then(async (res) => {
        if (!res.ok) throw new Error(await res.text());
        return res.json() as Promise<{ text: string }>;
      })
      .then((data) => {
        if (!cancelled) setText(data.text);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : "取得に失敗しました");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [raceId, site]);

  async function copy() {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      toast.push("コピーしました");
    } catch {
      toast.push("コピーに失敗しました", "danger");
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>投票フォーマットをコピー</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {!enabled && (
          <p className="text-xs text-warning">
            発走後のため参考表示です。投票操作は行えません。
          </p>
        )}
        <div className="flex gap-2">
          {(["rakuten", "spat4"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSite(s)}
              className={`min-h-11 rounded-md px-4 text-sm ${
                site === s ? "bg-accent text-accent-foreground" : "bg-muted text-muted-foreground"
              }`}
            >
              {s === "rakuten" ? "楽天競馬形式" : "SPAT4形式"}
            </button>
          ))}
        </div>
        {error && <p className="text-sm text-danger">{error}</p>}
        {!error && (
          <pre className="max-h-40 overflow-auto rounded-md bg-muted p-3 text-xs whitespace-pre-wrap">
            {loading ? "読み込み中…" : text || "推奨投資額が0のためコピーする内容がありません"}
          </pre>
        )}
        <Button onClick={copy} disabled={!enabled || !text} className="self-start">
          クリップボードにコピー
        </Button>
      </CardContent>
    </Card>
  );
}
