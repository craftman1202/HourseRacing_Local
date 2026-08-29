"use client";

import { useEffect, useState } from "react";

type Theme = "dark" | "light" | "system";

function apply(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") {
    root.removeAttribute("data-theme");
  } else {
    root.setAttribute("data-theme", theme);
  }
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");

  useEffect(() => {
    // localStorage の読み取りは外部システムとの同期なので、コールバック内で
    // setState する形にする（effect 本体で直接 setState しない）。
    void Promise.resolve().then(() => {
      try {
        const stored = localStorage.getItem("nar-theme") as Theme | null;
        if (stored) {
          setTheme(stored);
          apply(stored);
        }
      } catch {
        // localStorage が使えない環境ではシステム設定のまま
      }
    });
  }, []);

  function cycle() {
    const next: Theme = theme === "dark" ? "light" : theme === "light" ? "system" : "dark";
    setTheme(next);
    apply(next);
    try {
      localStorage.setItem("nar-theme", next);
    } catch {
      // 保存できなくても表示は切り替わっているので無視
    }
  }

  const label = theme === "dark" ? "ダーク" : theme === "light" ? "ライト" : "自動";

  return (
    <button
      onClick={cycle}
      className="rounded-md border border-border px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground"
      aria-label="テーマ切り替え"
      suppressHydrationWarning
    >
      {label}
    </button>
  );
}
