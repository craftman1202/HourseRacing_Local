import NextAuth from "next-auth";
import Google from "next-auth/providers/google";

/**
 * 単一テナントの許可リスト認証（設計書 §5.1: Auth.js v5 + allowlist）。
 * billing は今回スコープ外なので Plan は allowlist に載っていれば常に "admin"。
 *
 * AUTH_GOOGLE_ID / AUTH_GOOGLE_SECRET / AUTH_SECRET は Cloud Run の
 * --set-secrets で Secret Manager から注入される想定（未設定でもビルド/起動は
 * 落とさないが、実際のサインインはできない — ユーザーが OAuth クライアントを
 * 作成するまでの既知の制約）。
 */

function allowlist(): string[] {
  return (process.env.NAR_ADMIN_EMAILS ?? "")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
}

export function isAllowedEmail(email: string | null | undefined): boolean {
  if (!email) return false;
  return allowlist().includes(email.toLowerCase());
}

export const { handlers, signIn, signOut, auth } = NextAuth({
  // Cloud Run はリバースプロキシ経由でリクエストを転送するため、Auth.js は
  // 既定では Host ヘッダを信頼せず UntrustedHost で全リクエストを拒否する。
  // AUTH_TRUST_HOST=true でも同じ効果だが、ここにも明示しておく
  // （env 未設定のまま別環境にデプロイしても壊れないように）。
  trustHost: true,
  providers: [Google],
  session: { strategy: "jwt" },
  pages: {
    signIn: "/signin",
  },
  callbacks: {
    async signIn({ user }) {
      return isAllowedEmail(user.email);
    },
    authorized({ auth: session, request }) {
      const path = request.nextUrl.pathname;
      if (path.startsWith("/api/auth") || path === "/signin") return true;
      return !!session?.user;
    },
    session({ session }) {
      return session;
    },
  },
});
