"use client";

import { useRef, useState } from "react";
import { analyzeVideo, type IncidentResult } from "@/lib/api";

export function UploadPanel({
  onAnalyzed,
}: {
  onAnalyzed: (result: IncidentResult) => void;
}) {
  const [status, setStatus] = useState<"idle" | "analyzing" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    setStatus("analyzing");
    setError(null);
    try {
      const result = await analyzeVideo(file);
      onAnalyzed(result);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-zinc-800 bg-zinc-950 p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
          Analyze footage
        </h2>
        {status === "analyzing" && (
          <span className="text-xs text-cyan-400">running inference…</span>
        )}
      </div>
      <label className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-zinc-700 px-6 py-8 text-center text-sm text-zinc-400 transition-colors hover:border-cyan-500 hover:text-cyan-400">
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          disabled={status === "analyzing"}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) void handleFile(file);
          }}
        />
        <span>Drop or select a video clip to score for anomalies</span>
        <span className="text-xs text-zinc-600">mp4, avi, mov — sampled every 5th frame</span>
      </label>
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}
