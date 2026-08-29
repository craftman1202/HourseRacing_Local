import { fetchModels, NarApiError } from "@/lib/nar-api";
import { auth, isAllowedEmail } from "@/lib/auth";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { fmtNum } from "@/lib/format";
import { PromoteButton } from "@/components/models/promote-button";

export default async function ModelsPage() {
  const session = await auth();
  const isAdmin = isAllowedEmail(session?.user?.email);

  let releases;
  try {
    releases = await fetchModels();
  } catch (e) {
    const message = e instanceof NarApiError ? e.detail : "nar-api に接続できません";
    return (
      <Card>
        <CardContent className="py-10 text-center text-sm text-danger">{message}</CardContent>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-xl font-semibold">モデル管理</h1>
      {releases.length === 0 ? (
        <p className="text-sm text-muted-foreground">リリースがありません</p>
      ) : (
        releases.map((r) => (
          <Card key={r.release_id}>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base text-foreground">
                {r.release_id}
                {r.is_current && (
                  <span className="ml-2">
                    <Badge tone="success">current</Badge>
                  </span>
                )}
              </CardTitle>
              {isAdmin && !r.is_current && <PromoteButton releaseId={r.release_id} />}
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
                <div>
                  <p className="text-xs text-muted-foreground">model_id</p>
                  <p>{r.model_id}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">dataset_version</p>
                  <p>{r.dataset_version}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">学習期間</p>
                  <p>
                    {r.train_period_start} 〜 {r.train_period_end}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">track / purpose</p>
                  <p>
                    {r.track} / {r.purpose}
                  </p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground">git commit</p>
                  <p className="truncate font-mono text-xs">{r.git_commit || "—"}</p>
                </div>
                <div className="col-span-2 sm:col-span-2">
                  <p className="text-xs text-muted-foreground">OOS metrics</p>
                  <p className="text-xs">
                    {Object.entries(r.oos_metrics)
                      .map(([k, v]) => `${k}=${fmtNum(v)}`)
                      .join(" / ") || "—"}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        ))
      )}
    </div>
  );
}
