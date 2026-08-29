import { NextResponse } from "next/server";

import { auth, isAllowedEmail } from "@/lib/auth";
import { promoteModel, NarApiError } from "@/lib/nar-api";

export async function POST(
  _req: Request,
  ctx: RouteContext<"/api/models/[id]/promote">,
) {
  const session = await auth();
  // WB-06: 昇格は admin のみ。単一テナントなので許可リスト在籍 = admin として扱う
  if (!session?.user || !isAllowedEmail(session.user.email)) {
    return new NextResponse(null, { status: 403 });
  }

  const { id } = await ctx.params;
  try {
    await promoteModel(id, true);
    return NextResponse.json({ status: "promoted" });
  } catch (e) {
    if (e instanceof NarApiError) {
      return NextResponse.json({ detail: e.detail }, { status: e.status });
    }
    return NextResponse.json({ detail: "promote に失敗しました" }, { status: 502 });
  }
}
