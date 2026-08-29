import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Cloud Run 向け: 最小ランタイム (.next/standalone) を生成する
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
  // サーバー専用ライブラリをクライアントバンドルへ含めない
  serverExternalPackages: ["google-auth-library"],
};

export default nextConfig;
