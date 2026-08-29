/**
 * Next.js 16: middleware は proxy に改名された（挙動は同じ）。
 * Auth.js v5 の `auth` を認可ラッパーとしてそのまま使う
 * （lib/auth.ts の callbacks.authorized が可否を判定する）。
 */
export { auth as proxy } from "@/lib/auth";

export const config = {
  matcher: ["/((?!api/auth|_next/static|_next/image|favicon.ico).*)"],
};
