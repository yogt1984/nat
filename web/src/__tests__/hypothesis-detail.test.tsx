import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { Hypothesis } from "@/lib/api";

// Mock API
const mockGetHypothesis = vi.fn();
vi.mock("@/lib/api", () => ({
  getHypothesis: (...args: unknown[]) => mockGetHypothesis(...args),
}));

// Mock next/navigation
vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "HYP-SYS-042" }),
}));

// Mock next/link
vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    className,
  }: {
    children: React.ReactNode;
    href: string;
    className?: string;
  }) => (
    <a href={href} className={className}>
      {children}
    </a>
  ),
}));

// Mock GateWaterfall — Plotly needs browser APIs
vi.mock("@/components/gate-waterfall", () => ({
  GateWaterfall: ({ gates }: { gates: { name: string }[] }) => (
    <div data-testid="gate-waterfall">{gates.length} gates</div>
  ),
}));

// Mock MathPanel — KaTeX needs DOM APIs
vi.mock("@/components/math-panel", () => ({
  MathPanel: ({ latex }: { latex: string }) => (
    <div data-testid="math-panel">{latex.slice(0, 30)}</div>
  ),
}));

import HypothesisDetailPage from "@/app/explorer/[id]/page";

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

const MOCK_HYPOTHESIS: Hypothesis = {
  id: "HYP-SYS-042",
  agent: "microstructure",
  generator: "systematic",
  claim: "Spread compression predicts short-term momentum",
  math: "IC = \\text{corr}(\\hat{y}, y)",
  status: "replicated",
  failure_reason: null,
  gates: [
    {
      name: "IC",
      passed: true,
      message: "PASS IC=0.08 vs min=0.03",
      metric: 0.08,
      threshold: 0.03,
      p_value: 0.001,
    },
    {
      name: "Cost",
      passed: true,
      message: "PASS net Sharpe=1.2",
      metric: 1.2,
      threshold: 0.5,
      p_value: null,
    },
  ],
  features: ["spread_ba", "depth_total_bid"],
  regime_gate: "ent_book_shape<0.4",
  horizon_s: 5,
  thresholds: { horizon_s: 5, min_ic: 0.03 },
  parent_id: null,
  timestamps: {
    created: "2026-05-25T10:00:00+00:00",
    completed: "2026-05-25T10:01:00+00:00",
  },
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("HypothesisDetailPage", () => {
  beforeEach(() => {
    mockGetHypothesis.mockReset();
  });

  it("shows loading skeleton initially", () => {
    mockGetHypothesis.mockReturnValue(new Promise(() => {}));
    render(<HypothesisDetailPage />);
    expect(document.querySelector(".animate-pulse")).toBeTruthy();
  });

  it("renders hypothesis details", async () => {
    mockGetHypothesis.mockResolvedValue(MOCK_HYPOTHESIS);
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("HYP-SYS-042")).toBeInTheDocument();
    });
    expect(screen.getByText("replicated")).toBeInTheDocument();
    expect(screen.getByText("microstructure")).toBeInTheDocument();
    expect(screen.getByText("systematic")).toBeInTheDocument();
    expect(
      screen.getByText("Spread compression predicts short-term momentum")
    ).toBeInTheDocument();
  });

  it("renders Export PDF button", async () => {
    mockGetHypothesis.mockResolvedValue(MOCK_HYPOTHESIS);
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Export PDF")).toBeInTheDocument();
    });
    const btn = screen.getByText("Export PDF");
    expect(btn.tagName).toBe("BUTTON");
    expect(btn.className).toContain("no-print");
  });

  it("Export PDF button calls window.print()", async () => {
    mockGetHypothesis.mockResolvedValue(MOCK_HYPOTHESIS);
    const printSpy = vi.spyOn(window, "print").mockImplementation(() => {});
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Export PDF")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Export PDF"));
    expect(printSpy).toHaveBeenCalledOnce();
    printSpy.mockRestore();
  });

  it("Export PDF sets document title temporarily", async () => {
    mockGetHypothesis.mockResolvedValue(MOCK_HYPOTHESIS);
    const originalTitle = document.title;
    const printSpy = vi.spyOn(window, "print").mockImplementation(() => {});
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Export PDF")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Export PDF"));
    // Title should be restored after print
    expect(document.title).toBe(originalTitle);
    printSpy.mockRestore();
  });

  it("renders gate results", async () => {
    mockGetHypothesis.mockResolvedValue(MOCK_HYPOTHESIS);
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Gate Results")).toBeInTheDocument();
    });
    expect(screen.getByText("IC")).toBeInTheDocument();
    expect(screen.getByText("Cost")).toBeInTheDocument();
    expect(screen.getByText("p=0.0010")).toBeInTheDocument();
  });

  it("renders features list", async () => {
    mockGetHypothesis.mockResolvedValue(MOCK_HYPOTHESIS);
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("spread_ba")).toBeInTheDocument();
    });
    expect(screen.getByText("depth_total_bid")).toBeInTheDocument();
  });

  it("renders math derivation", async () => {
    mockGetHypothesis.mockResolvedValue(MOCK_HYPOTHESIS);
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByTestId("math-panel")).toBeInTheDocument();
    });
  });

  it("renders print header and footer (hidden by default)", async () => {
    mockGetHypothesis.mockResolvedValue(MOCK_HYPOTHESIS);
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("NAT Research Report")).toBeInTheDocument();
    });
    // Both print-only elements exist in DOM but hidden via CSS
    const header = screen.getByText("NAT Research Report").closest("div");
    expect(header?.className).toContain("print-header");
    expect(header?.className).toContain("hidden");

    const footer = screen.getByText(/NAT Research Platform/).closest("div");
    expect(footer?.className).toContain("print-footer");
    expect(footer?.className).toContain("hidden");
  });

  it("back link has no-print class", async () => {
    mockGetHypothesis.mockResolvedValue(MOCK_HYPOTHESIS);
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByText(/Back to Explorer/)).toBeInTheDocument();
    });
    expect(screen.getByText(/Back to Explorer/).className).toContain(
      "no-print"
    );
  });

  it("shows error state", async () => {
    mockGetHypothesis.mockRejectedValue(new Error("Not found"));
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("Not found")).toBeInTheDocument();
    });
  });

  it("renders metadata chips", async () => {
    mockGetHypothesis.mockResolvedValue(MOCK_HYPOTHESIS);
    render(<HypothesisDetailPage />);
    await waitFor(() => {
      expect(screen.getByText("5s")).toBeInTheDocument();
    });
    expect(screen.getByText("Horizon")).toBeInTheDocument();
    expect(screen.getByText("ent_book_shape<0.4")).toBeInTheDocument();
  });
});
