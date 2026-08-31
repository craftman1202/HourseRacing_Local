import NextAuth from "next-auth";
import Google from "next-auth/providers/google";

/**
 * 2026-08-30 の調査メモ（誤診断の記録）。
 *
 * 一時的に「Google の discovery（`.well-known/openid-configuration`）が
 * `authorization_response_iss_parameter_supported: true` を返すのに実際の
 * コールバックには `iss` が付かない」と誤診断し、discovery を迂回する独自
 * プロバイダ設定を入れたことがある。
 *
 * 実際の原因は別にあった: `AUTH_GOOGLE_ID` がプレースホルダ値のまま
 * （実在しない Client ID）だった時期に、その異常な入力に対する Google の
 * 応答がたまたま `iss` を欠いていただけだった。正しい Client ID と正しい
 * リダイレクト URI を設定すると、Google は `iss=https://accounts.google.com`
 * を確実に返す（実測で確認、2026-08-30）。discovery を迂回する独自設定は
 * `issuer` を明示していなかったため `as.issuer` が既定のプレースホルダ
 * `https://authjs.dev` のままになり、今度は `unexpected "iss" (issuer)
 * response parameter value` という**別の**例外を生んだ。
 *
 * 標準の `Google` プロバイダ（discovery 経由）に戻す。問題は最初から
 * discovery ではなく、認証情報とリダイレクト URI の設定漏れだった。
 */

/**
 * 会員登録 + 管理者許可リストの2段構成（設計書 §5.1 の許可リスト方式から、
 * 2026-08-30 に開放登録へ変更）。
 *
 * - 会員（member）: Google アカウントさえあれば誰でもサインインできる。
 *   Auth.js は DB アダプタを持たない JWT セッションなので、「初回サインイン」
 *   がそのまま「登録」を兼ねる — 別途サインアップ画面や会員テーブルは無い。
 * - 管理者（admin）: `NAR_ADMIN_EMAILS`（カンマ区切り）に載っているメール
 *   アドレスのみ。モデル昇格など影響が大きい操作はこちらでしか出さない
 *   （`isAllowedEmail` は従来どおり "admin かどうか" の判定として残す）。
 *
 * billing は今回スコープ外なので Plan の概念はまだ無い。将来サブスク化する
 * ときは、ここに「会員だが未課金」を弾く判定を足す形になる。
 *
 * AUTH_GOOGLE_ID / AUTH_GOOGLE_SECRET / AUTH_SECRET は Cloud Run の
 * --set-secrets で Secret Manager から注入される想定（未設定でもビルド/起動は
 * 落とさないが、実際のサインインはできない — ユーザーが OAuth クライアントを
 * 作成するまでの既知の制約）。
 *
 * 注意: これはこのアプリ内の認可の話で、Google 側の OAuth 同意画面が
 * 「テスト」公開ステータスのままだと、そちらは別レイヤーで事前登録した
 * テストユーザー以外のサインインを拒否する。「誰でも登録できる」を実際に
 * 有効にするには、GCP コンソールの OAuth 同意画面を「本番環境」に
 * 公開する必要がある（このリポジトリのコードからは変更できない）。
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
      // 会員登録の唯一の条件は「Google アカウントでメールアドレスが取れること」。
      // 管理者かどうかはここでは判定しない — admin 限定の操作は isAllowedEmail を
      // 呼び出し側（例: app/models/page.tsx）で個別にチェックする。
      return !!user.email;
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
