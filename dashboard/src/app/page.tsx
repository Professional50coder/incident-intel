"use client";

import { useCallback, useEffect, useState } from "react";
import { KpiTiles } from "@/components/KpiTiles";
import { UploadPanel } from "@/components/UploadPanel";
import { IncidentList } from "@/components/IncidentList";
import { ScoreScene } from "@/components/ScoreScene";
import {
  fetchStats,
  fetchIncidents,
  fetchIncident,
  type DashboardStats,
  type IncidentSummary,
  type IncidentResult,
} from "@/lib/api";

const POLL_INTERVAL_MS = 5000;

export default function Home() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [incidents, setIncidents] = useState<IncidentSummary[]>([]);
  const [selected, setSelected] = useState<IncidentResult | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [statsData, incidentsData] = await Promise.all([fetchStats(), fetchIncidents()]);
      setStats(statsData);
      setIncidents(incidentsData);
      setApiError(null);
    } catch (err) {
      setApiError(
        err instanceof Error
          ? `Can't reach the API: ${err.message}`
          : "Can't reach the API"
      );
    }
  }, []);

  useEffect(() => {
    // Poll immediately on mount, then on an interval - this fetches from an external system
    // (the API), it doesn't set state synchronously in the effect body itself.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
    const interval = setInterval(refresh, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [refresh]);

  async function selectIncident(id: string) {
    try {
      const detail = await fetchIncident(id);
      setSelected(detail);
    } catch {
      // leave the previous selection in place - a transient fetch failure isn't worth
      // clearing what's already on screen.
    }
  }

  function handleAnalyzed(result: IncidentResult) {
    setSelected(result);
    void refresh();
  }

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-6 py-10">
        <header className="flex flex-col gap-1">
          <h1 className="text-xl font-semibold tracking-tight">Incident Intelligence</h1>
          <p className="text-sm text-zinc-500">
            Video anomaly detection — data engineering, model selection, and a served model,
            end to end.
          </p>
          {apiError && (
            <p className="mt-1 rounded-lg border border-amber-900/50 bg-amber-500/10 px-3 py-2 text-xs text-amber-400">
              {apiError} — is the backend running at{" "}
              <code className="font-mono">NEXT_PUBLIC_API_URL</code>?
            </p>
          )}
        </header>

        <KpiTiles stats={stats} />

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1fr_1fr]">
          <div className="flex flex-col gap-6">
            <UploadPanel onAnalyzed={handleAnalyzed} />
            <IncidentList
              incidents={incidents}
              selectedId={selected?.incident_id ?? null}
              onSelect={(id) => void selectIncident(id)}
            />
          </div>
          <div className="flex flex-col gap-3">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
              Score timeline
              {selected && (
                <span className="ml-2 font-normal normal-case text-zinc-600">
                  {selected.source_filename} · max score {selected.max_score.toFixed(4)}
                </span>
              )}
            </h2>
            <ScoreScene frameScores={selected?.frame_scores ?? []} />
          </div>
        </div>
      </div>
    </div>
  );
}
