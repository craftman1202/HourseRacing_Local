import { NextRequest, NextResponse } from "next/server";

import { auth } from "@/lib/auth";
import { fetchBetslipText } from "@/lib/nar-api";

export async function GET(
  req: NextRequest,
  ctx: RouteContext<"/api/races/[raceId]/betslip">,
) {
  const session = await auth();
  if (!session?.user) return new NextResponse(null, { status: 401 });

  const { raceId } = await ctx.params;
  const site = req.nextUrl.searchParams.get("site") === "spat4" ? "spat4" : "rakuten";

  try {
    const text = await fetchBetslipText(raceId, site);
    return NextResponse.json({ text });
  } catch (e) {
    const message = e instanceof Error ? e.message : "betslip 取得に失敗しました";
    return new NextResponse(message, { status: 502 });
  }
}
