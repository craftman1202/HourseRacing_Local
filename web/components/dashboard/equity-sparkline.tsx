"use client";

import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { DailyPnL } from "@/lib/nar-api";
import { toEquitySeries } from "@/lib/pnl";

export function EquitySparkline({ data }: { data: DailyPnL[] }) {
  const series = toEquitySeries(data);

  return (
    <div className="h-40 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={series} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <XAxis dataKey="date" hide />
          <YAxis hide domain={["auto", "auto"]} />
          <Tooltip
            formatter={(value) => [
              `¥${Number(value).toLocaleString("ja-JP")}`,
              "累積損益",
            ]}
            labelFormatter={(label) => label}
            contentStyle={{
              background: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              fontSize: 12,
            }}
          />
          <Line
            type="monotone"
            dataKey="equity"
            stroke="var(--accent)"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
