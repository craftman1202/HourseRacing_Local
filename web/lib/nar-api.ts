import "server-only";

import type { components } from "./api-types.gen";

/**
 * nar-api への唯一の入口。ブラウザからは絶対に直接叩かない設計
 * (infra/services.json: nar-web の SA は roles/run.invoker のみで
 * BigQuery 権限を持たない — nar-api を呼べるのはサーバー側だけ)。
 * Client Component からこのファイルを import すると `server-only`
 * によりビルドエラーになる。
 */

export type SystemHealth = components["schemas"]["SystemHealth"];
export type RacePrediction = components["schemas"]["RacePrediction"];
export type HorsePrediction = components["schemas"]["HorsePrediction"];
export type OOSMetrics = components["schemas"]["OOSMetrics"];
export type DailyPnL = components["schemas"]["DailyPnL"];
export type ModelRelease = components["schemas"]["ModelRelease"];
export type TodayRace = components["schemas"]["TodayRace"];

export class NarApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(`nar-api ${status}: ${detail}`);
  }
}

function apiUrl(): string {
  const url = process.env.NAR_API_URL;
  if (!url) {
    throw new Error(
      "NAR_API_URL が設定されていません。nar-api の URL を環境変数で渡してください。",
    );
  }
  return url;
}

let idTokenClientPromise: Promise<{
  fetch(input: string, init?: RequestInit): Promise<Response>;
}> | null = null;

/**
 * Cloud Run 上 (K_SERVICE が設定されている) でのみ OIDC ID トークンを取得する。
 * ローカル開発では nar-api を --allow-unauthenticated で動かす前提で素通りする。
 */
async function authorizedFetch(path: string, init?: RequestInit): Promise<Response> {
  const base = apiUrl();
  const url = `${base}${path}`;

  if (!process.env.K_SERVICE) {
    return fetch(url, { ...init, cache: "no-store" });
  }

  if (!idTokenClientPromise) {
    idTokenClientPromise = (async () => {
      const { GoogleAuth } = await import("google-auth-library");
      const auth = new GoogleAuth();
      const client = await auth.getIdTokenClient(base);
      return {
        fetch: async (input: string, reqInit?: RequestInit) => {
          const headers = await client.getRequestHeaders(input);
          const merged = new Headers(reqInit?.headers);
          headers.forEach((value, key) => merged.set(key, value));
          return fetch(input, { ...reqInit, headers: merged, cache: "no-store" });
        },
      };
    })();
  }
  const client = await idTokenClientPromise;
  return client.fetch(url, init);
}

async function getJson<T>(path: string): Promise<T> {
  const res = await authorizedFetch(path);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    if (res.status === 404) throw new NarApiError(404, detail);
    throw new NarApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export function fetchHealth(): Promise<SystemHealth> {
  return getJson<SystemHealth>("/health");
}

export function fetchRace(raceId: string): Promise<RacePrediction> {
  return getJson<RacePrediction>(`/races/${encodeURIComponent(raceId)}`);
}

export function fetchOosMetrics(): Promise<OOSMetrics[]> {
  return getJson<OOSMetrics[]>("/performance/oos");
}

export function fetchLivePnl(): Promise<DailyPnL[]> {
  return getJson<DailyPnL[]>("/performance/live");
}

export function fetchModels(): Promise<ModelRelease[]> {
  return getJson<ModelRelease[]>("/models");
}

export function fetchTodayRaces(): Promise<TodayRace[]> {
  return getJson<TodayRace[]>("/races/today");
}

export async function fetchBetslipText(
  raceId: string,
  site: "rakuten" | "spat4",
): Promise<string> {
  const data = await getJson<{ site: string; text: string }>(
    `/races/${encodeURIComponent(raceId)}/betslip?site=${site}`,
  );
  return data.text;
}

export async function promoteModel(releaseId: string, confirmed: boolean): Promise<void> {
  const res = await authorizedFetch(`/models/${encodeURIComponent(releaseId)}/promote`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confirmed }),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    let message = detail;
    try {
      message = JSON.parse(detail).detail ?? detail;
    } catch {
      // detail がプレーンテキストの場合はそのまま使う
    }
    throw new NarApiError(res.status, message);
  }
}
