import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { Hypothesis, Gate } from "@/lib/api";
import { FailurePie } from "@/components/failure-pie";
import { GeneratorBars } from "@/components/generator-bars";
import {
  NearMissesTable,
  RecyclableTable,
  filterNearMisses,
  filterRecyclable,
} from "@/components/near-misses";

// Mock next/link
vi.mock("next/link", () => ({
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

// Mock next/dynamic for Plotly components
vi.mock("next/dynamic", () => ({
  default: () => {
    const Stub = (props: Record<string, unknown>) => {
      const data = props.data as Array<{ type: string; labels?: string[]; values?: number[] }>;
      const type = data?.[0]?.type;
      const count = data?.length ?? 0;
      return (
        <div
          data-testid="plotly-chart"
          data-type={type}
          data-trace-count={count}
          data-labels={JSON.stringify(data?.[0]?.labels ?? [])}
        />
      );
    };
    Stub.displayName = "DynamicMock";
    return Stub;
  },
}));

function makeGate(name: string, passed: boolean, metric: number | null = null): Gate {
  return {
    name,
    passed,
    message: passed ? `${name} passed` : `${name} failed`,
    metric,
    threshold: metric != null ? 0.02 : null,
    p_value: null,
  };
}

function makeHypothesis(overrides: Partial<Hypothesis> = {}): Hypothesis {
  return {
    id: `hyp_${Math.random().toString(36).slice(2, 10)}`,
    agent: "microstructure",
    generator: "systematic",
    claim: "Test claim",
    math: "",
    status: "no_effect",
    failure_reason: null,
    gates: [
      makeGate("discovery", true, 0.045),
      makeGate("cost", true, 0.03),
      makeGate("temporal", false),
      makeGate("symbol", false),
    ],
    features: ["spread_bps", "depth_imbalance"],
    regime_gate: null,
    horizon_s: 300,
    thresholds: {},
    parent_id: null,
    timestamps: { created: "2025-05-20T10:00:00Z", completed: "2025-05-20T10:05:00Z" },
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// FailurePie
// ---------------------------------------------------------------------------

describe("FailurePie", () => {
  it("renders empty state when no hypotheses", () => {
    render(<FailurePie hypotheses={[]} />);
    expect(screen.getByText("No failure data.")).toBeInTheDocument();
  });

  it("renders donut chart with gate segments", () => {
    const hyps = [
      makeHypothesis({
        gates: [makeGate("discovery", false, 0.01)],
      }),
      makeHypothesis({
        gates: [makeGate("discovery", true, 0.05), makeGate("cost", false, -0.01)],
      }),
      makeHypothesis({
        gates: [
          makeGate("discovery", true, 0.04),
          makeGate("cost", true, 0.02),
          makeGate("temporal", false),
        ],
      }),
    ];
    render(<FailurePie hypotheses={hyps} />);
    const chart = screen.getByTestId("plotly-chart");
    expect(chart).toBeInTheDocument();
    expect(chart.dataset.type).toBe("pie");

    // Should have labels for each failing gate
    const labels = JSON.parse(chart.dataset.labels ?? "[]") as string[];
    expect(labels).toContain("discovery");
    expect(labels).toContain("cost");
    expect(labels).toContain("temporal");
  });
});

// ---------------------------------------------------------------------------
// GeneratorBars
// ---------------------------------------------------------------------------

describe("GeneratorBars", () => {
  it("renders empty state when no hypotheses", () => {
    render(<GeneratorBars hypotheses={[]} />);
    expect(screen.getByText("No generator data.")).toBeInTheDocument();
  });

  it("renders stacked bar chart with pass/fail traces", () => {
    const hyps = [
      makeHypothesis({ generator: "systematic", status: "replicated" }),
      makeHypothesis({ generator: "systematic", status: "no_effect" }),
      makeHypothesis({ generator: "spectral", status: "no_replication" }),
      makeHypothesis({ generator: "regime", status: "replicated" }),
    ];
    render(<GeneratorBars hypotheses={hyps} />);
    const chart = screen.getByTestId("plotly-chart");
    expect(chart).toBeInTheDocument();
    expect(chart.dataset.type).toBe("bar");
    // 2 traces: pass + fail
    expect(chart.dataset.traceCount).toBe("2");
  });
});

// ---------------------------------------------------------------------------
// Near Misses (logic)
// ---------------------------------------------------------------------------

describe("filterNearMisses", () => {
  it("returns hypotheses that failed exactly 1 gate", () => {
    const nearMiss = makeHypothesis({
      id: "hyp_near1",
      gates: [
        makeGate("discovery", true, 0.05),
        makeGate("cost", true, 0.03),
        makeGate("temporal", true, 0.8),
        makeGate("symbol", false, 0.4),
      ],
    });
    const notNearMiss = makeHypothesis({
      id: "hyp_far1",
      gates: [
        makeGate("discovery", true, 0.05),
        makeGate("cost", false, -0.01),
        makeGate("temporal", false, 0.3),
      ],
    });
    const result = filterNearMisses([nearMiss, notNearMiss]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("hyp_near1");
  });

  it("excludes hypotheses with no gates", () => {
    const noGates = makeHypothesis({ id: "hyp_nogates", gates: [] });
    expect(filterNearMisses([noGates])).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// NearMissesTable (render)
// ---------------------------------------------------------------------------

describe("NearMissesTable", () => {
  it("renders empty state when no near misses", () => {
    const hyps = [
      makeHypothesis({
        gates: [makeGate("discovery", false), makeGate("cost", false)],
      }),
    ];
    render(<NearMissesTable hypotheses={hyps} />);
    expect(screen.getByText("No near misses found.")).toBeInTheDocument();
  });

  it("renders table sorted by IC descending", () => {
    const hyps = [
      makeHypothesis({
        id: "hyp_nm_low01",
        gates: [
          makeGate("discovery", true, 0.03),
          makeGate("cost", true, 0.01),
          makeGate("temporal", false, 0.4),
        ],
      }),
      makeHypothesis({
        id: "hyp_nm_high1",
        gates: [
          makeGate("discovery", true, 0.08),
          makeGate("cost", true, 0.05),
          makeGate("symbol", false, 0.3),
        ],
      }),
    ];
    render(<NearMissesTable hypotheses={hyps} />);

    const rows = screen.getAllByRole("row");
    // header + 2 data rows
    expect(rows).toHaveLength(3);

    // First data row should be the higher IC
    const firstLink = rows[1].querySelector("a");
    expect(firstLink?.textContent).toBe("hyp_nm_high1");
  });

  it("shows the failed gate name", () => {
    const hyps = [
      makeHypothesis({
        id: "hyp_nm_gate1",
        gates: [
          makeGate("discovery", true, 0.05),
          makeGate("temporal", false, 0.3),
        ],
      }),
    ];
    render(<NearMissesTable hypotheses={hyps} />);
    expect(screen.getByText("temporal")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Recyclable (logic)
// ---------------------------------------------------------------------------

describe("filterRecyclable", () => {
  it("returns hypotheses that failed only replication gates with IC > 0.02", () => {
    const recyclable = makeHypothesis({
      id: "hyp_recycle1",
      gates: [
        makeGate("discovery", true, 0.05),
        makeGate("cost", true, 0.03),
        makeGate("temporal", false, 0.4),
      ],
    });
    const notRecyclable = makeHypothesis({
      id: "hyp_norecycl",
      gates: [
        makeGate("discovery", false, 0.01),  // failed at discovery, not replication
      ],
    });
    const result = filterRecyclable([recyclable, notRecyclable]);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("hyp_recycle1");
  });

  it("excludes if IC is below threshold", () => {
    const lowIc = makeHypothesis({
      id: "hyp_lowic001",
      gates: [
        makeGate("discovery", true, 0.015),  // IC below 0.02
        makeGate("temporal", false, 0.3),
      ],
    });
    expect(filterRecyclable([lowIc])).toHaveLength(0);
  });

  it("includes symbol gate failures as recyclable", () => {
    const symbolFail = makeHypothesis({
      id: "hyp_symfail1",
      gates: [
        makeGate("discovery", true, 0.06),
        makeGate("cost", true, 0.04),
        makeGate("temporal", true, 0.8),
        makeGate("symbol", false, 0.4),
      ],
    });
    expect(filterRecyclable([symbolFail])).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// RecyclableTable (render)
// ---------------------------------------------------------------------------

describe("RecyclableTable", () => {
  it("renders empty state when no recyclable candidates", () => {
    const hyps = [
      makeHypothesis({
        gates: [makeGate("discovery", false, 0.01)],
      }),
    ];
    render(<RecyclableTable hypotheses={hyps} />);
    expect(screen.getByText("No recyclable candidates.")).toBeInTheDocument();
  });

  it("renders recyclable candidates with IC in green", () => {
    const hyps = [
      makeHypothesis({
        id: "hyp_rc_test1",
        gates: [
          makeGate("discovery", true, 0.07),
          makeGate("cost", true, 0.05),
          makeGate("temporal", false, 0.4),
        ],
      }),
    ];
    render(<RecyclableTable hypotheses={hyps} />);
    expect(screen.getByText("0.0700")).toBeInTheDocument();
    // The IC cell should have emerald color
    const icCell = screen.getByText("0.0700");
    expect(icCell.className).toContain("emerald");
  });
});
