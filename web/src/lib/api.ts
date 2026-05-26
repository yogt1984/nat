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
  total_cycles: number;
  by_status: Record<string, number>;
  by_agent: Record<string, number>;
  by_generator: Record<string, number>;
}

export interface HeatmapEntry {
  feature: string;
  horizon_s: number;
  ic: number;
  status: string;
}

export interface HeatmapResponse {
  entries: HeatmapEntry[];
  features: string[];
  horizons: number[];
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
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
  offset?: number;
  limit?: number;
  agent?: string;
  status?: string;
  generator?: string;
}): Promise<PaginatedResponse<Hypothesis>> {
  const p: Record<string, string> = {};
  if (params?.offset != null) p.offset = String(params.offset);
  if (params?.limit != null) p.limit = String(params.limit);
  if (params?.agent) p.agent = params.agent;
  if (params?.status) p.status = params.status;
  if (params?.generator) p.generator = params.generator;
  return fetchJson(`${BASE}/api/research/hypotheses`, p);
}

export async function getHypothesis(id: string): Promise<Hypothesis> {
  return fetchJson(`${BASE}/api/research/hypotheses/${id}`);
}

export async function listCycles(params?: {
  offset?: number;
  limit?: number;
}): Promise<PaginatedResponse<CycleSummary>> {
  const p: Record<string, string> = {};
  if (params?.offset != null) p.offset = String(params.offset);
  if (params?.limit != null) p.limit = String(params.limit);
  return fetchJson(`${BASE}/api/research/cycles`, p);
}

export async function listSignals(): Promise<Signal[]> {
  return fetchJson(`${BASE}/api/research/signals`);
}

export async function getHeatmap(): Promise<HeatmapResponse> {
  return fetchJson(`${BASE}/api/research/heatmap`);
}
