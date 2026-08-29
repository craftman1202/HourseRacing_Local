import { fetchRace, NarApiError } from "@/lib/nar-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fmtEv, fmtPct, fmtYen, evColorClass } from "@/lib/format";
import { Table, Thead, Tbody, Tr, Th, Td } from "@/components/ui/table";
import { ProbabilityGapBar } from "@/components/race/probability-gap-bar";
import { RaceCountdown } from "@/components/race/countdown";
import { BetslipPanel } from "@/components/race/betslip-panel";
import { LiveOdds } from "@/components/race/live-odds";

export default async function RacePage({ params }: PageProps<"/race/[raceId]">) {
  const { raceId } = await params;

  let race;
  try {
    race = await fetchRace(raceId);
  } catch (e) {
    if (e instanceof NarApiError && e.status === 404) {
      return (
        <Card>
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            レース {raceId} の推奨データが見つかりません。
          </CardContent>
        </Card>
      );
    }
    const message = e instanceof NarApiError ? e.detail : "nar-api に接続できません";
    return (
      <Card>
        <CardContent className="py-10 text-center text-sm text-danger">{message}</CardContent>
      </Card>
    );
  }

  const horses = [...(race.horses ?? [])].sort((a, b) => b.p_win - a.p_win);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold">
            {race.track_name} {race.race_no}R
          </h1>
          <p className="text-sm text-muted-foreground">
            {new Date(race.start_ts).toLocaleString("ja-JP")} ・ {race.model_release} ・{" "}
            {race.track_used}
            {race.is_paper && (
              <span className="ml-2">
                <Badge tone="warning">ペーパー</Badge>
              </span>
            )}
          </p>
        </div>
        <RaceCountdown startTsIso={race.start_ts} />
      </div>

      <LiveOdds raceId={raceId} startTsIso={race.start_ts} />

      {race.status !== "ok" && (
        <Card>
          <CardContent className="py-3 text-sm">
            <Badge tone={race.status === "expired" ? "neutral" : "warning"}>{race.status}</Badge>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>出馬表（予測確率降順）</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <Table>
            <Thead>
              <Tr>
                <Th>馬番</Th>
                <Th>馬名</Th>
                <Th>予測確率</Th>
                <Th>市場確率</Th>
                <Th>EV</Th>
                <Th>補正後EV</Th>
                <Th>推奨投資額</Th>
              </Tr>
            </Thead>
            <Tbody>
              {horses.map((h) => (
                <Tr key={h.horse_no}>
                  <Td>{h.horse_no}</Td>
                  <Td>{h.horse_name ?? "—"}</Td>
                  <Td>{fmtPct(h.p_win)}</Td>
                  <Td>{fmtPct(h.p_market)}</Td>
                  <Td className={evColorClass(h.ev ?? undefined)}>{fmtEv(h.ev)}</Td>
                  <Td className={evColorClass(h.ev_adjusted ?? undefined)}>
                    {fmtEv(h.ev_adjusted)}
                  </Td>
                  <Td>{fmtYen(h.stake_yen)}</Td>
                </Tr>
              ))}
            </Tbody>
          </Table>

          <div>
            <h4 className="mb-2 text-xs font-medium text-muted-foreground">
              予測確率 vs 市場確率の乖離
            </h4>
            <ProbabilityGapBar horses={horses} />
          </div>
        </CardContent>
      </Card>

      <p className="text-xs text-muted-foreground">
        4モデル（条件付きロジット/LightGBM/TabM/ベイズ）の個別内訳とベイズ信用区間は
        現在の推論パイプラインが per-model 出力を永続化していないため未提供です。
      </p>

      <BetslipPanel raceId={raceId} startTsIso={race.start_ts} />
    </div>
  );
}
