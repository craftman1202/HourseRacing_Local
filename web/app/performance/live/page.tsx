import { fetchLivePnl, NarApiError } from "@/lib/nar-api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, Thead, Tbody, Tr, Th, Td } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { fmtPct, fmtYen } from "@/lib/format";
import { EquityCurve } from "@/components/performance/equity-curve";
import { computeDrawdown } from "@/lib/pnl";

export default async function LivePerformancePage() {
  let rows;
  try {
    rows = await fetchLivePnl();
  } catch (e) {
    const message = e instanceof NarApiError ? e.detail : "nar-api に接続できません";
    return (
      <Card>
        <CardContent className="py-10 text-center text-sm text-danger">{message}</CardContent>
      </Card>
    );
  }

  const drawdown = rows.length > 0 ? computeDrawdown(rows) : null;

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-xl font-semibold">運用実績</h1>

      {rows.length === 0 ? (
        <Card>
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            実績データがありません
          </CardContent>
        </Card>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle>累積収支</CardTitle>
            </CardHeader>
            <CardContent>
              <EquityCurve data={rows} />
              {drawdown && (
                <div className="mt-4 flex gap-6 text-sm">
                  <div>
                    <p className="text-xs text-muted-foreground">最大ドローダウン</p>
                    <p className="text-danger">{fmtYen(Math.abs(drawdown.maxDrawdown))}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">現在のドローダウン深度</p>
                    <p>{fmtYen(Math.abs(drawdown.current))}</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>日別実績</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <Thead>
                  <Tr>
                    <Th>日付</Th>
                    <Th>区分</Th>
                    <Th>レース数</Th>
                    <Th>ベット数</Th>
                    <Th>投資額</Th>
                    <Th>払戻</Th>
                    <Th>ROI</Th>
                    <Th>的中率</Th>
                  </Tr>
                </Thead>
                <Tbody>
                  {rows.map((r) => (
                    <Tr key={r.business_date}>
                      <Td>{r.business_date}</Td>
                      <Td>
                        <Badge tone={r.is_paper ? "warning" : "success"}>
                          {r.is_paper ? "ペーパー" : "実運用"}
                        </Badge>
                      </Td>
                      <Td>{r.n_races}</Td>
                      <Td>{r.n_bets}</Td>
                      <Td>{fmtYen(r.stake_yen)}</Td>
                      <Td>{fmtYen(r.return_yen)}</Td>
                      <Td>{fmtPct(r.roi - 1)}</Td>
                      <Td>{fmtPct(r.hit_rate)}</Td>
                    </Tr>
                  ))}
                </Tbody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
