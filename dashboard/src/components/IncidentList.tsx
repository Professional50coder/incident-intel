"use client";

import type { IncidentSummary } from "@/lib/api";

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function IncidentList({
  incidents,
  selectedId,
  onSelect,
}: {
  incidents: IncidentSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-zinc-800 bg-zinc-950 p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
          Recent incidents
        </h2>
        <span className="flex items-center gap-1.5 text-xs text-zinc-600">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
          live
        </span>
      </div>
      {incidents.length === 0 ? (
        <p className="py-6 text-center text-sm text-zinc-600">
          No incidents analyzed yet — upload a clip above.
        </p>
      ) : (
        <ul className="flex max-h-80 flex-col gap-1 overflow-y-auto">
          {incidents.map((incident) => (
            <li key={incident.incident_id}>
              <button
                onClick={() => onSelect(incident.incident_id)}
                className={
                  "flex w-full items-center justify-between rounded-lg px-3 py-2 text-left text-sm transition-colors " +
                  (incident.incident_id === selectedId
                    ? "bg-zinc-800 text-zinc-50"
                    : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200")
                }
              >
                <span className="flex flex-col">
                  <span className="font-medium">{incident.source_filename}</span>
                  <span className="text-xs text-zinc-600">
                    {formatTime(incident.created_at)} · {incident.frame_count} frames sampled
                  </span>
                </span>
                <span
                  className={
                    "rounded-full px-2 py-0.5 text-xs font-semibold " +
                    (incident.is_anomalous
                      ? "bg-red-500/10 text-red-400"
                      : "bg-emerald-500/10 text-emerald-400")
                  }
                >
                  {incident.is_anomalous ? "anomaly" : "normal"}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
