/**
 * Typed API client for NAT research endpoints.
 *
 * All endpoints proxy through Next.js rewrites to the Axum backend.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Hypothesis {
  id: string;
  agent: string;
  generator: string;
  claim: string;
  math: string;
  status: string;
  failure_reason: string | null;
  gates: Gate[];
  features: string[];
  regime_gate: string | null;
  horizon_s: number | null;
  thresholds: Record<string, unknown>;
  parent_id: string | null;
  timestamps: { created: string; completed: string | null };
}

export interface Gate {
  name: string;
  passed: boolean;
  message: string;
  metric: number | null;
  threshold: number | null;
  p_value: number | null;
}

export interface CycleSummary {
  cycle_id: string;
  agent: string;
  started: string;
  completed: string;
  duration_s: number;
  n_tested: number;
  n_registered: number;
  n_fdr_rejected: number;
  n_chained: number;
  fdr_q: number;
  hypotheses: { id: string; generator: string; claim: string; status: string }[];
  generator_stats: Record<string, { attempts: number; successes: number; hit_rate: number; weight: number }>;
}

export interface Signal {
  name: string;
  features: string[];
  regime_gate: string | null;
  extraction: string;
  horizon_s: number;
  expected_ic: number;
  symbols: string[];
  discovery_date: string;
  status: string;
  ic_history: number[];
}

export interface ResearchStats {
  total_hypotheses: number;
  total_registered: number;
  total_graveyard: number;
  agents: Record<string, {
    phase: string;
    cycle_count: number;
    last_cycle_at: string | null;
    queue_depth: number;
  }>;
}

export interface HeatmapEntry {
  feature: string;
  horizon: string;
  ic: number;
  p_value: number | null;
}

export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  per_page: number;
}

// ---------------------------------------------------------------------------
// Fetch helpers
// ---------------------------------------------------------------------------

const BASE = "";  // proxied through Next.js rewrites

async function fetchJson<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(path, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  }
  const res = await fetch(url.toString());
  if (!res.ok) {
    throw new Error(`API ${res.status}: ${res.statusText}`);
  }
  return res.json();
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

export async function getStats(): Promise<ResearchStats> {
  return fetchJson(`${BASE}/api/research/stats`);
}

export async function listHypotheses(params?: {
  page?: number;
  per_page?: number;
  agent?: string;
  status?: string;
}): Promise<PaginatedResponse<Hypothesis>> {
  const p: Record<string, string> = {};
  if (params?.page) p.page = String(params.page);
  if (params?.per_page) p.per_page = String(params.per_page);
  if (params?.agent) p.agent = params.agent;
  if (params?.status) p.status = params.status;
  return fetchJson(`${BASE}/api/research/hypotheses`, p);
}

export async function getHypothesis(id: string): Promise<Hypothesis> {
  return fetchJson(`${BASE}/api/research/hypotheses/${id}`);
}

export async function listCycles(params?: {
  page?: number;
  per_page?: number;
}): Promise<PaginatedResponse<CycleSummary>> {
  const p: Record<string, string> = {};
  if (params?.page) p.page = String(params.page);
  if (params?.per_page) p.per_page = String(params.per_page);
  return fetchJson(`${BASE}/api/research/cycles`, p);
}

export async function listSignals(): Promise<Signal[]> {
  return fetchJson(`${BASE}/api/research/signals`);
}

export async function getHeatmap(): Promise<HeatmapEntry[]> {
  return fetchJson(`${BASE}/api/research/heatmap`);
}
