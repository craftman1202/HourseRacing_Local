import { auth } from "@/lib/auth";
import { fetchRace } from "@/lib/nar-api";

/**
 * 発走10分前だけクライアントが接続する想定の SSE。
 * 30秒だけ接続を保持してポーリング的に配信し、切断する（Cloud Run の課金対策、
 * 設計書 §5.1）。オッズの時系列そのものは nar-api にエンドポイントが無いため、
 * ここでは `/races/{race_id}` の最新値を数秒おきに再取得して差分配信する
 * （履歴チャートではなく最新値のライブ更新）。
 */
export async function GET(
  _req: Request,
  ctx: RouteContext<"/api/sse/odds/[raceId]">,
) {
  const session = await auth();
  if (!session?.user) return new Response(null, { status: 401 });

  const { raceId } = await ctx.params;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const encoder = new TextEncoder();
      const deadline = Date.now() + 30_000;

      const tick = async () => {
        try {
          const race = await fetchRace(raceId);
          const payload = (race.horses ?? []).map((h) => ({
            horse_no: h.horse_no,
            p_market: h.p_market,
          }));
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(payload)}\n\n`));
        } catch {
          // 一時的な取得失敗はスキップし、次の tick で再試行する
        }
      };

      await tick();
      const interval = setInterval(async () => {
        if (Date.now() >= deadline) {
          clearInterval(interval);
          controller.close();
          return;
        }
        await tick();
      }, 5000);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}
