import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

import { auth, signOut } from "@/lib/auth";
import { Providers } from "@/components/providers";
import { ThemeToggle } from "@/components/theme-toggle";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "nar-web",
  description: "地方競馬 予測ダッシュボード",
};

const NAV = [
  { href: "/", label: "ダッシュボード" },
  { href: "/performance/oos", label: "OOS" },
  { href: "/performance/live", label: "運用実績" },
  { href: "/models", label: "モデル" },
];

export default async function RootLayout({ children }: LayoutProps<"/">) {
  const session = await auth();

  return (
    <html
      lang="ja"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <Providers>
          {session?.user && (
            <header className="border-b border-border">
              <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-4 px-4 py-3">
                <Link href="/" className="text-sm font-semibold">
                  nar-web
                </Link>
                <nav className="flex flex-1 flex-wrap gap-1">
                  {NAV.map((item) => (
                    <Link
                      key={item.href}
                      href={item.href}
                      className="rounded-md px-3 py-1.5 text-sm text-muted-foreground hover:bg-muted hover:text-foreground"
                    >
                      {item.label}
                    </Link>
                  ))}
                </nav>
                <ThemeToggle />
                <span className="hidden text-xs text-muted-foreground sm:inline">
                  {session.user.email}
                </span>
                <form
                  action={async () => {
                    "use server";
                    await signOut({ redirectTo: "/signin" });
                  }}
                >
                  <button className="text-xs text-muted-foreground hover:text-foreground">
                    サインアウト
                  </button>
                </form>
              </div>
            </header>
          )}
          <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
