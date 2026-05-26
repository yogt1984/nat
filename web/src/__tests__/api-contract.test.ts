/**
 * API Contract Tests
 *
 * Validates that mock data fixtures match the TypeScript interfaces in lib/api.ts.
 * If a field is renamed, added, or removed in the Rust API, the corresponding
 * fixture here must be updated — catching contract drift at test time.
 *
 * These fixtures mirror the JSON produced by rust/api/src/routes/research.rs.
 */
import { describe, it, expect } from "vitest";
import type {
  Hypothesis,
  Gate,
  CycleSummary,
  ResearchStats,
  HeatmapEntry,
  HeatmapResponse,
  NetworkNode,
  NetworkEdge,
  NetworkMeta,
  NetworkResponse,
  PaginatedResponse,
} from "@/lib/api";

// ---------------------------------------------------------------------------
// Fixtures — must match Rust response shapes exactly
// ---------------------------------------------------------------------------

const GATE_FIXTURE: Gate = {
  name: "IC",
  passed: true,
  message: "PASS IC=0.08 vs min=0.03 p=0.001",
  metric: 0.08,
  threshold: 0.03,
  p_value: 0.001,
};

const HYPOTHESIS_FIXTURE: Hypothesis = {
  id: "HYP-SYS-042",
  agent: "micro",
  generator: "systematic",
  claim: "ent_book_shape gated by ent_book_shape<0.4 predicts 5s returns",
  math: "IC = Spearman(signal_t, r_{t+h})",
  status: "replicated",
  failure_reason: null,
  gates: [GATE_FIXTURE],
  features: ["ent_book_shape"],
  regime_gate: "ent_book_shape<0.4",
  horizon_s: 5.0,
  thresholds: { horizon_s: 5.0, min_ic: 0.03 },
  parent_id: null,
  timestamps: {
    created: "2026-05-25T10:00:00+00:00",
    completed: "2026-05-25T10:01:00+00:00",
  },
};

const FAILED_HYPOTHESIS_FIXTURE: Hypothesis = {
  ...HYPOTHESIS_FIXTURE,
  id: "HYP-SYS-099",
  status: "failed",
  failure_reason: "no_effect",
  gates: [{ ...GATE_FIXTURE, passed: false }],
};

const CYCLE_FIXTURE: CycleSummary = {
  cycle_id: "CYC-MICRO-001",
  agent: "micro",
  started: "2026-05-25T10:00:00+00:00",
  completed: "2026-05-25T10:05:00+00:00",
  duration_s: 300.0,
  n_tested: 10,
  n_registered: 2,
  n_fdr_rejected: 1,
  n_chained: 0,
  fdr_q: 0.05,
  hypotheses: [
    { id: "HYP-SYS-042", generator: "systematic", claim: "Test claim", status: "replicated" },
  ],
  generator_stats: {
    systematic: { attempts: 5, successes: 2, hit_rate: 0.4, weight: 1.0 },
  },
};

const STATS_FIXTURE: ResearchStats = {
  total_hypotheses: 42,
  total_cycles: 8,
  by_status: { replicated: 10, failed: 30, pending: 2 },
  by_agent: { micro: 30, macro: 12 },
  by_generator: { systematic: 20, spectral: 15, regime: 7 },
};

const HEATMAP_ENTRY_FIXTURE: HeatmapEntry = {
  feature: "ent_book_shape",
  horizon_s: 5.0,
  ic: 0.08,
  status: "replicated",
};

const HEATMAP_FIXTURE: HeatmapResponse = {
  entries: [HEATMAP_ENTRY_FIXTURE],
  features: ["ent_book_shape"],
  horizons: [5.0],
};

const NETWORK_NODE_FIXTURE: NetworkNode = {
  id: "spread_ba",
  category: "spread",
  mi: { "10t": 0.05, "50t": 0.03 },
  cmi: { "10t": 0.04, "50t": 0.02 },
  interaction: 0.003,
  cost_viable: true,
  hypothesis_count: 3,
  selected: true,
};

const NETWORK_EDGE_FIXTURE: NetworkEdge = {
  source: "spread_ba",
  target: "depth_bid",
  weight: 2,
};

const NETWORK_META_FIXTURE: NetworkMeta = {
  symbol: "BTC",
  n_samples: 6000,
  last_updated: "2026-05-21T11:00:00",
  total_features: 15,
};

const NETWORK_FIXTURE: NetworkResponse = {
  nodes: [NETWORK_NODE_FIXTURE],
  edges: [NETWORK_EDGE_FIXTURE],
  meta: NETWORK_META_FIXTURE,
};

const PAGINATED_FIXTURE: PaginatedResponse<Hypothesis> = {
  items: [HYPOTHESIS_FIXTURE],
  total: 42,
  offset: 0,
  limit: 20,
};

// ---------------------------------------------------------------------------
// Tests — type-check + runtime shape assertions
// ---------------------------------------------------------------------------

describe("API contract: Hypothesis", () => {
  it("has all required fields with correct types", () => {
    const h = HYPOTHESIS_FIXTURE;
    expect(typeof h.id).toBe("string");
    expect(typeof h.agent).toBe("string");
    expect(typeof h.generator).toBe("string");
    expect(typeof h.claim).toBe("string");
    expect(typeof h.math).toBe("string");
    expect(typeof h.status).toBe("string");
    expect(h.failure_reason).toBeNull();
    expect(Array.isArray(h.gates)).toBe(true);
    expect(Array.isArray(h.features)).toBe(true);
    expect(typeof h.thresholds).toBe("object");
    expect(typeof h.timestamps).toBe("object");
    expect(typeof h.timestamps.created).toBe("string");
  });

  it("failure_reason can be string for failed hypotheses", () => {
    expect(typeof FAILED_HYPOTHESIS_FIXTURE.failure_reason).toBe("string");
  });

  it("nullable fields accept null", () => {
    const h = HYPOTHESIS_FIXTURE;
    expect(h.failure_reason).toBeNull();
    expect(h.parent_id).toBeNull();
    // horizon_s and regime_gate can be null; verify string/number when present
    expect(typeof h.horizon_s).toBe("number");
    expect(typeof h.regime_gate).toBe("string");
  });
});

describe("API contract: Gate", () => {
  it("has all required fields", () => {
    const g = GATE_FIXTURE;
    expect(typeof g.name).toBe("string");
    expect(typeof g.passed).toBe("boolean");
    expect(typeof g.message).toBe("string");
    expect(typeof g.metric).toBe("number");
    expect(typeof g.threshold).toBe("number");
    expect(typeof g.p_value).toBe("number");
  });

  it("metric/threshold/p_value can be null", () => {
    const g: Gate = { ...GATE_FIXTURE, metric: null, threshold: null, p_value: null };
    expect(g.metric).toBeNull();
    expect(g.threshold).toBeNull();
    expect(g.p_value).toBeNull();
  });
});

describe("API contract: CycleSummary", () => {
  it("has all required fields with correct types", () => {
    const c = CYCLE_FIXTURE;
    expect(typeof c.cycle_id).toBe("string");
    expect(typeof c.agent).toBe("string");
    expect(typeof c.started).toBe("string");
    expect(typeof c.completed).toBe("string");
    expect(typeof c.duration_s).toBe("number");
    expect(typeof c.n_tested).toBe("number");
    expect(typeof c.n_registered).toBe("number");
    expect(typeof c.n_fdr_rejected).toBe("number");
    expect(typeof c.n_chained).toBe("number");
    expect(typeof c.fdr_q).toBe("number");
    expect(Array.isArray(c.hypotheses)).toBe(true);
    expect(typeof c.generator_stats).toBe("object");
  });

  it("hypotheses array items have id, generator, claim, status", () => {
    const h = CYCLE_FIXTURE.hypotheses[0];
    expect(typeof h.id).toBe("string");
    expect(typeof h.generator).toBe("string");
    expect(typeof h.claim).toBe("string");
    expect(typeof h.status).toBe("string");
  });

  it("generator_stats values have attempts, successes, hit_rate, weight", () => {
    const gs = CYCLE_FIXTURE.generator_stats["systematic"];
    expect(typeof gs.attempts).toBe("number");
    expect(typeof gs.successes).toBe("number");
    expect(typeof gs.hit_rate).toBe("number");
    expect(typeof gs.weight).toBe("number");
  });
});

describe("API contract: ResearchStats", () => {
  it("has all required fields", () => {
    const s = STATS_FIXTURE;
    expect(typeof s.total_hypotheses).toBe("number");
    expect(typeof s.total_cycles).toBe("number");
    expect(typeof s.by_status).toBe("object");
    expect(typeof s.by_agent).toBe("object");
    expect(typeof s.by_generator).toBe("object");
  });

  it("map values are numbers", () => {
    for (const v of Object.values(STATS_FIXTURE.by_status)) {
      expect(typeof v).toBe("number");
    }
    for (const v of Object.values(STATS_FIXTURE.by_agent)) {
      expect(typeof v).toBe("number");
    }
  });
});

describe("API contract: HeatmapResponse", () => {
  it("has entries, features, and horizons arrays", () => {
    expect(Array.isArray(HEATMAP_FIXTURE.entries)).toBe(true);
    expect(Array.isArray(HEATMAP_FIXTURE.features)).toBe(true);
    expect(Array.isArray(HEATMAP_FIXTURE.horizons)).toBe(true);
  });

  it("entry has feature, horizon_s, ic, status", () => {
    const e = HEATMAP_ENTRY_FIXTURE;
    expect(typeof e.feature).toBe("string");
    expect(typeof e.horizon_s).toBe("number");
    expect(typeof e.ic).toBe("number");
    expect(typeof e.status).toBe("string");
  });
});

describe("API contract: NetworkResponse", () => {
  it("has nodes, edges, meta", () => {
    expect(Array.isArray(NETWORK_FIXTURE.nodes)).toBe(true);
    expect(Array.isArray(NETWORK_FIXTURE.edges)).toBe(true);
    expect(typeof NETWORK_FIXTURE.meta).toBe("object");
  });

  it("node has all required fields", () => {
    const n = NETWORK_NODE_FIXTURE;
    expect(typeof n.id).toBe("string");
    expect(typeof n.category).toBe("string");
    expect(typeof n.mi).toBe("object");
    expect(typeof n.cmi).toBe("object");
    expect(typeof n.interaction).toBe("number");
    expect(typeof n.cost_viable).toBe("boolean");
    expect(typeof n.hypothesis_count).toBe("number");
    expect(typeof n.selected).toBe("boolean");
  });

  it("edge has source, target, weight", () => {
    const e = NETWORK_EDGE_FIXTURE;
    expect(typeof e.source).toBe("string");
    expect(typeof e.target).toBe("string");
    expect(typeof e.weight).toBe("number");
  });

  it("meta has symbol, n_samples, last_updated, total_features", () => {
    const m = NETWORK_META_FIXTURE;
    expect(typeof m.symbol).toBe("string");
    expect(typeof m.n_samples).toBe("number");
    expect(typeof m.last_updated).toBe("string");
    expect(typeof m.total_features).toBe("number");
  });
});

describe("API contract: PaginatedResponse", () => {
  it("has items, total, offset, limit", () => {
    const p = PAGINATED_FIXTURE;
    expect(Array.isArray(p.items)).toBe(true);
    expect(typeof p.total).toBe("number");
    expect(typeof p.offset).toBe("number");
    expect(typeof p.limit).toBe("number");
  });

  it("items contain valid Hypothesis objects", () => {
    const h = PAGINATED_FIXTURE.items[0];
    expect(typeof h.id).toBe("string");
    expect(typeof h.agent).toBe("string");
    expect(Array.isArray(h.gates)).toBe(true);
  });
});
