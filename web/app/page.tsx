import { fetchHealth, fetchLivePnl, fetchTodayRaces, NarApiError } from "@/lib/nar-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { EV_COLOR_THRESHOLDS, fmtNum, fmtPct, fmtYen } from "@/lib/format";
import { EquitySparkline } from "@/components/dashboard/equity-sparkline";
import { TodayTimeline } from "@/components/dashboard/today-timeline";
import { NextRaceCountdown } from "@/components/dashboard/next-race-countdown";
import type { TodayRace } from "@/lib/nar-api";

function verdictTone(v: string): "success" | "danger" | "neutral" {
  if (v === "PASS") return "success";
  if (v === "FAIL") return "danger";
  return "neutral";
}

/**
 * 「次の推奨レース」を選ぶ。現在時刻に依存する非純粋な計算なので、
 * コンポーネント本体（react-hooks/purity の対象）の外に出す。
 */
function pickNextRace(races: TodayRace[]): TodayRace | undefined {
  const now = Date.now();
  return races
    .filter(
      (r) =>
        r.status === "ok" &&
        (r.top_ev_adjusted ?? 0) >= EV_COLOR_THRESHOLDS.yellow &&
        new Date(r.start_ts).getTime() > now,
    )
    .sort((a, b) => new Date(a.start_ts).getTime() - new Date(b.start_ts).getTime())[0];
}

export default async function DashboardPage() {
  let health;
  let healthError: string | null = null;
  try {
    health = await fetchHealth();
  } catch (e) {
    healthError = e instanceof NarApiError ? e.detail : "nar-api に接続できません";
  }

  let pnl: Awaited<ReturnType<typeof fetchLivePnl>> = [];
  let pnlError: string | null = null;
  try {
    pnl = await fetchLivePnl();
  } catch (e) {
    pnlError = e instanceof NarApiError ? e.detail : "nar-api に接続できません";
  }

  const latest = pnl[0];

  let todayRaces: Awaited<ReturnType<typeof fetchTodayRaces>> = [];
  let todayError: string | null = null;
  try {
    todayRaces = await fetchTodayRaces();
  } catch (e) {
    todayError = e instanceof NarApiError ? e.detail : "nar-api に接続できません";
  }

  const nextRace = pickNextRace(todayRaces);

  return (
    <div className="flex flex-col gap-6">
      {nextRace && (
        <NextRaceCountdown
          raceId={nextRace.race_id}
          trackName={nextRace.track_name}
          raceNo={nextRace.race_no}
          startTsIso={nextRace.start_ts}
        />
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader>
            <CardTitle>DB 鮮度</CardTitle>
          </CardHeader>
          <CardContent>
            {healthError ? (
              <p className="text-sm text-danger">{healthError}</p>
            ) : (
              <p className="text-sm">
                {health?.db_watermark
                  ? new Date(health.db_watermark).toLocaleString("ja-JP")
                  : "—"}
              </p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>モデルリリース</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">{health?.model_release ?? "—"}</p>
            <p className="mt-1 text-xs text-muted-foreground">
              学習: {health?.model_trained_on ?? "—"}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>skew 検証</CardTitle>
          </CardHeader>
          <CardContent>
            {health && <Badge tone={verdictTone(health.skew_verdict)}>{health.skew_verdict}</Badge>}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>配信状態</CardTitle>
          </CardHeader>
          <CardContent>
            {health?.delivery_blocked ? (
              <>
                <Badge tone="danger">停止中</Badge>
                {health.blocked_reason && (
                  <p className="mt-1 text-xs text-muted-foreground">{health.blocked_reason}</p>
                )}
              </>
            ) : (
              <Badge tone="success">配信中</Badge>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>直近30日の資金曲線</CardTitle>
        </CardHeader>
        <CardContent>
          {pnlError ? (
            <p className="text-sm text-danger">{pnlError}</p>
          ) : pnl.length === 0 ? (
            <p className="text-sm text-muted-foreground">実績データがありません</p>
          ) : (
            <>
              <EquitySparkline data={pnl} />
              {latest && (
                <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                  <div>
                    <p className="text-xs text-muted-foreground">直近ROI</p>
                    <p>{fmtPct(latest.roi - 1)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">的中率</p>
                    <p>{fmtPct(latest.hit_rate)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">投資額</p>
                    <p>{fmtYen(latest.stake_yen)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">区分</p>
                    <p>{latest.is_paper ? "ペーパー" : "実運用"}</p>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>本日の開催</CardTitle>
        </CardHeader>
        <CardContent>
          {todayError ? (
            <p className="text-sm text-danger">{todayError}</p>
          ) : (
            <TodayTimeline races={todayRaces} />
          )}
        </CardContent>
      </Card>
      <p className="text-xs text-muted-foreground">
        カバレッジ(直近30日): {fmtNum(health?.coverage_today, 3)}
      </p>
    </div>
  );
}
