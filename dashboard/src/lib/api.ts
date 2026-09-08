export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface FrameScore {
  frame_index: number;
  score: number;
  is_anomalous: boolean;
}

export interface IncidentResult {
  incident_id: string;
  created_at: string;
  source_filename: string;
  frame_scores: FrameScore[];
  max_score: number;
}

export interface IncidentSummary {
  incident_id: string;
  created_at: string;
  source_filename: string;
  max_score: number;
  is_anomalous: boolean;
  frame_count: number;
}

export interface DashboardStats {
  total_incidents: number;
  total_frames_analyzed: number;
  anomalous_incident_count: number;
  anomaly_rate: number;
  average_max_score: number;
}

async function parseOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const detail = await response.text().catch(() => response.statusText);
    throw new Error(`${response.status} ${detail}`);
  }
  return response.json() as Promise<T>;
}

export async function fetchStats(): Promise<DashboardStats> {
  const response = await fetch(`${API_BASE_URL}/stats`, { cache: "no-store" });
  return parseOrThrow<DashboardStats>(response);
}

export async function fetchIncidents(limit = 50): Promise<IncidentSummary[]> {
  const response = await fetch(`${API_BASE_URL}/incidents?limit=${limit}`, {
    cache: "no-store",
  });
  return parseOrThrow<IncidentSummary[]>(response);
}

export async function fetchIncident(id: string): Promise<IncidentResult> {
  const response = await fetch(`${API_BASE_URL}/incidents/${id}`, {
    cache: "no-store",
  });
  return parseOrThrow<IncidentResult>(response);
}

export async function analyzeVideo(file: File): Promise<IncidentResult> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: "POST",
    body: formData,
  });
  return parseOrThrow<IncidentResult>(response);
}
