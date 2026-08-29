"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { Dialog, type DialogHandle } from "@/components/ui/dialog";
import { useToast } from "@/components/ui/toast";

/** WB-06: 昇格は二段確認必須。ダイアログでの明示的な確認をUI側でも要求する。 */
export function PromoteButton({ releaseId }: { releaseId: string }) {
  const dialogRef = useRef<DialogHandle>(null);
  const [pending, setPending] = useState(false);
  const toast = useToast();
  const router = useRouter();

  async function confirmPromote() {
    setPending(true);
    try {
      const res = await fetch(`/api/models/${encodeURIComponent(releaseId)}/promote`, {
        method: "POST",
      });
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        toast.push(body.detail ?? `昇格に失敗しました (${res.status})`, "danger");
        return;
      }
      toast.push(`${releaseId} を current に切り替えました`);
      dialogRef.current?.close();
      router.refresh();
    } finally {
      setPending(false);
    }
  }

  return (
    <>
      <Button variant="secondary" onClick={() => dialogRef.current?.open()}>
        current に切り替え
      </Button>
      <Dialog ref={dialogRef} title="昇格の確認">
        <p className="mb-4 text-sm text-muted-foreground">
          {releaseId} を current に切り替えます。OOS 指標が現行版を上回っていない場合は
          nar-api 側のゲートで拒否されます。承認なしでの本番切り替えは行いません。
        </p>
        <div className="flex justify-end gap-2">
          <Button variant="ghost" onClick={() => dialogRef.current?.close()}>
            キャンセル
          </Button>
          <Button variant="danger" onClick={confirmPromote} disabled={pending}>
            {pending ? "切り替え中…" : "切り替えを実行"}
          </Button>
        </div>
      </Dialog>
    </>
  );
}
