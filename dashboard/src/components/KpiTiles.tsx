"use client";

import type { DashboardStats } from "@/lib/api";

function Tile({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: "danger" | "default";
}) {
  return (
    <div className="flex flex-col gap-1 rounded-xl border border-zinc-800 bg-zinc-950 px-5 py-4">
      <span className="text-xs font-medium uppercase tracking-wide text-zinc-500">
        {label}
      </span>
      <span
        className={
          "text-2xl font-semibold tabular-nums " +
          (accent === "danger" ? "text-red-400" : "text-zinc-50")
        }
      >
        {value}
      </span>
    </div>
  );
}

export function KpiTiles({ stats }: { stats: DashboardStats | null }) {
  const anomalyPct = stats ? (stats.anomaly_rate * 100).toFixed(1) : "—";
  const avgScore = stats ? stats.average_max_score.toFixed(4) : "—";

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <Tile label="Incidents analyzed" value={stats ? String(stats.total_incidents) : "—"} />
      <Tile label="Frames processed" value={stats ? String(stats.total_frames_analyzed) : "—"} />
      <Tile
        label="Anomaly rate"
        value={stats ? `${anomalyPct}%` : "—"}
        accent={stats && stats.anomaly_rate > 0 ? "danger" : "default"}
      />
      <Tile label="Avg. max score" value={avgScore} />
    </div>
  );
}
