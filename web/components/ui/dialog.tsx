"use client";

import { useEffect, useImperativeHandle, useRef, forwardRef } from "react";

export type DialogHandle = {
  open: () => void;
  close: () => void;
};

/**
 * ネイティブ <dialog> ベースの軽量モーダル。Radix 等を足さずに
 * フォーカストラップ・Esc クローズ・backdrop を無料で得る。
 */
export const Dialog = forwardRef<
  DialogHandle,
  { title: string; children: React.ReactNode; className?: string }
>(function Dialog({ title, children, className = "" }, ref) {
  const dialogRef = useRef<HTMLDialogElement>(null);

  useImperativeHandle(ref, () => ({
    open: () => dialogRef.current?.showModal(),
    close: () => dialogRef.current?.close(),
  }));

  useEffect(() => {
    const el = dialogRef.current;
    if (!el) return;
    const onClickOutside = (e: MouseEvent) => {
      if (e.target === el) el.close();
    };
    el.addEventListener("click", onClickOutside);
    return () => el.removeEventListener("click", onClickOutside);
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className={`rounded-lg border border-border bg-card p-0 text-card-foreground backdrop:bg-black/50 ${className}`}
    >
      <div className="w-[min(90vw,28rem)] p-5">
        <h2 className="mb-3 text-base font-semibold">{title}</h2>
        {children}
      </div>
    </dialog>
  );
});
