import { fetchOosMetrics, NarApiError } from "@/lib/nar-api";
import { Card, CardContent } from "@/components/ui/card";
import { Table, Thead, Tbody, Tr, Th, Td } from "@/components/ui/table";
import { fmtNum, fmtPct } from "@/lib/format";

export default async function OosPerformancePage() {
  let rows;
  try {
    rows = await fetchOosMetrics();
  } catch (e) {
    const message = e instanceof NarApiError ? e.detail : "nar-api に接続できません";
    return (
      <Card>
        <CardContent className="py-10 text-center text-sm text-danger">{message}</CardContent>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold">OOS パフォーマンス</h1>
        <p className="mt-1 text-xs text-muted-foreground">
          保存済みの評価結果のみを表示します（オンザフライ再計算はしません／WB-03）。
          現在の manifest にはモデル横断の集計値しか保存されておらず、
          条件付きロジット/LightGBM/TabM/ベイズの個別内訳・人気帯別較正曲線は
          推論パイプライン側の拡張が必要なため未提供です。
        </p>
      </div>

      <Card>
        <CardContent className="pt-4">
          {rows.length === 0 ? (
            <p className="text-sm text-muted-foreground">評価データがありません</p>
          ) : (
            <Table>
              <Thead>
                <Tr>
                  <Th>モデル</Th>
                  <Th>トラック</Th>
                  <Th>race NLL</Th>
                  <Th>Top-1</Th>
                  <Th>Top-3</Th>
                  <Th>Brier</Th>
                  <Th>ECE</Th>
                  <Th>評価日時</Th>
                </Tr>
              </Thead>
              <Tbody>
                {rows.map((r, i) => (
                  <Tr key={`${r.model}-${r.track}-${i}`}>
                    <Td>{r.model}</Td>
                    <Td>{r.track}</Td>
                    <Td>{fmtNum(r.race_nll)}</Td>
                    <Td>{fmtPct(r.top1)}</Td>
                    <Td>{fmtPct(r.top3)}</Td>
                    <Td>{fmtNum(r.brier)}</Td>
                    <Td>{fmtNum(r.ece)}</Td>
                    <Td>{new Date(r.evaluated_at).toLocaleDateString("ja-JP")}</Td>
                  </Tr>
                ))}
              </Tbody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
