import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { Hypothesis } from "@/lib/api";
import { SignalTable } from "@/components/signal-table";

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  useParams: () => ({ id: "test" }),
}));

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

// Mock next/dynamic — returns a component that renders the factory synchronously
vi.mock("next/dynamic", () => ({
  default: (factory: () => Promise<{ default: React.ComponentType }>) => {
    // For Plotly, return a stub that captures data-type from props
    const Stub = (props: Record<string, unknown>) => {
      const type = (props.data as Array<{ type: string }>)?.[0]?.type;
      return <div data-testid="plotly-chart" data-type={type} />;
    };
    Stub.displayName = "DynamicMock";
    return Stub;
  },
}));

function makeSignal(overrides: Partial<Hypothesis> = {}): Hypothesis {
  return {
    id: `hyp_${Math.random().toString(36).slice(2, 10)}`,
    agent: "microstructure",
    generator: "systematic",
    claim: "Test hypothesis claim",
    math: "",
    status: "replicated",
    failure_reason: null,
    gates: [
      {
        name: "discovery",
        passed: true,
        message: "IC above threshold",
        metric: 0.045,
        threshold: 0.02,
        p_value: 0.001,
      },
      {
        name: "cost",
        passed: true,
        message: "Cost adjusted IC positive",
        metric: 0.03,
        threshold: 0.0,
        p_value: null,
      },
      {
        name: "temporal",
        passed: true,
        message: "Replicated across time splits",
        metric: 0.8,
        threshold: 0.6,
        p_value: 0.01,
      },
      {
        name: "symbol",
        passed: true,
        message: "Replicated across symbols",
        metric: 0.7,
        threshold: 0.5,
        p_value: 0.02,
      },
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
// SignalTable tests
// ---------------------------------------------------------------------------

describe("SignalTable", () => {
  it("renders empty state when no signals", () => {
    render(<SignalTable signals={[]} />);
    expect(screen.getByText("No registered signals yet.")).toBeInTheDocument();
  });

  it("renders table with correct number of rows", () => {
    const signals = [
      makeSignal({ id: "hyp_aaa" }),
      makeSignal({ id: "hyp_bbb" }),
      makeSignal({ id: "hyp_ccc" }),
      makeSignal({ id: "hyp_ddd" }),
      makeSignal({ id: "hyp_eee" }),
    ];
    render(<SignalTable signals={signals} />);

    // Table headers present
    expect(screen.getByText("ID")).toBeInTheDocument();
    expect(screen.getByText("Agent")).toBeInTheDocument();
    expect(screen.getByText("IC")).toBeInTheDocument();

    // 5 rows — each ID truncated to 12 chars
    expect(screen.getByText("hyp_aaa")).toBeInTheDocument();
    expect(screen.getByText("hyp_eee")).toBeInTheDocument();
  });

  it("shows IC values from discovery gate", () => {
    const signal = makeSignal({
      id: "hyp_ic_test1",
      gates: [
        {
          name: "discovery",
          passed: true,
          message: "ok",
          metric: 0.0678,
          threshold: 0.02,
          p_value: null,
        },
      ],
    });
    render(<SignalTable signals={[signal]} />);
    expect(screen.getByText("0.0678")).toBeInTheDocument();
  });

  it("shows dash when no IC metric available", () => {
    const signal = makeSignal({
      id: "hyp_no_ic001",
      gates: [],
    });
    const { container } = render(<SignalTable signals={[signal]} />);
    // The IC column should have a dash
    const cells = container.querySelectorAll("td");
    const icCell = cells[3]; // 4th column is IC
    expect(icCell.textContent).toBe("—");
  });

  it("displays agent and generator correctly", () => {
    const signal = makeSignal({
      id: "hyp_agent_tst",
      agent: "medium_freq",
      generator: "spectral",
    });
    render(<SignalTable signals={[signal]} />);
    expect(screen.getByText("medium_freq")).toBeInTheDocument();
    expect(screen.getByText("spectral")).toBeInTheDocument();
  });

  it("shows formatted horizon", () => {
    const signal = makeSignal({ id: "hyp_horizon01", horizon_s: 3600 });
    render(<SignalTable signals={[signal]} />);
    expect(screen.getByText("1h")).toBeInTheDocument();
  });

  it("shows feature count", () => {
    const signal = makeSignal({
      id: "hyp_feat_cnt1",
      features: ["a", "b", "c"],
    });
    render(<SignalTable signals={[signal]} />);
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("renders status badge with correct style for replicated", () => {
    const signal = makeSignal({ id: "hyp_badge_tst" });
    render(<SignalTable signals={[signal]} />);
    const badge = screen.getByText("replicated");
    expect(badge).toBeInTheDocument();
    expect(badge.className).toContain("emerald");
  });

  it("links signal ID to detail page", () => {
    const signal = makeSignal({ id: "hyp_link_test" });
    render(<SignalTable signals={[signal]} />);
    const link = screen.getByText("hyp_link_tes"); // truncated to 12
    expect(link.closest("a")).toHaveAttribute("href", "/explorer/hyp_link_test");
  });
});

// ---------------------------------------------------------------------------
// IcBarChart tests (Plotly mock)
// ---------------------------------------------------------------------------

import { IcBarChart } from "@/components/ic-bar-chart";

describe("IcBarChart", () => {
  it("renders empty state when no signals", () => {
    render(<IcBarChart signals={[]} />);
    expect(screen.getByText("No IC data available.")).toBeInTheDocument();
  });

  it("renders plotly chart when signals have IC", () => {
    const signals = [
      makeSignal({ id: "hyp_ic1" }),
      makeSignal({ id: "hyp_ic2" }),
      makeSignal({ id: "hyp_ic3" }),
    ];
    render(<IcBarChart signals={signals} />);
    expect(screen.getByTestId("plotly-chart")).toBeInTheDocument();
    expect(screen.getByTestId("plotly-chart").dataset.type).toBe("bar");
  });

  it("renders empty state when signals have no IC metric", () => {
    const signals = [
      makeSignal({ id: "hyp_noic1", gates: [] }),
      makeSignal({ id: "hyp_noic2", gates: [] }),
    ];
    render(<IcBarChart signals={signals} />);
    expect(screen.getByText("No IC data available.")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// WeightTreemap tests (Plotly mock)
// ---------------------------------------------------------------------------

import { WeightTreemap } from "@/components/weight-treemap";

describe("WeightTreemap", () => {
  it("renders empty state when no signals", () => {
    render(<WeightTreemap signals={[]} />);
    expect(screen.getByText("No signals for treemap.")).toBeInTheDocument();
  });

  it("renders treemap chart for signals", () => {
    const signals = [
      makeSignal({ id: "hyp_tm1", agent: "microstructure" }),
      makeSignal({ id: "hyp_tm2", agent: "medium_freq" }),
      makeSignal({ id: "hyp_tm3", agent: "macro" }),
    ];
    render(<WeightTreemap signals={signals} />);
    expect(screen.getByTestId("plotly-chart")).toBeInTheDocument();
    expect(screen.getByTestId("plotly-chart").dataset.type).toBe("treemap");
  });

  it("renders with single agent type", () => {
    const signals = [
      makeSignal({ id: "hyp_sa1", agent: "microstructure" }),
      makeSignal({ id: "hyp_sa2", agent: "microstructure" }),
    ];
    render(<WeightTreemap signals={signals} />);
    expect(screen.getByTestId("plotly-chart")).toBeInTheDocument();
  });
});
