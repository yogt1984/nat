import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import type { NetworkResponse } from "@/lib/api";

// Mock the API module
const mockGetNetwork = vi.fn();
vi.mock("@/lib/api", () => ({
  getNetwork: (...args: unknown[]) => mockGetNetwork(...args),
}));

// Mock the FeatureNetwork component — D3 needs real DOM APIs
vi.mock("@/components/feature-network", () => ({
  FeatureNetwork: ({
    nodes,
    edges,
    miThreshold,
    activeCategories,
  }: {
    nodes: { id: string; category: string }[];
    edges: { source: string; target: string }[];
    miThreshold: number;
    activeCategories: Set<string>;
  }) => (
    <div data-testid="feature-network">
      <span data-testid="node-count">{nodes.length}</span>
      <span data-testid="edge-count">{edges.length}</span>
      <span data-testid="mi-threshold">{miThreshold}</span>
      <span data-testid="active-categories">
        {Array.from(activeCategories).sort().join(",")}
      </span>
    </div>
  ),
}));

import NetworkPage from "@/app/network/page";

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

const MOCK_NETWORK: NetworkResponse = {
  nodes: [
    {
      id: "spread_ba",
      category: "spread",
      mi: { "10t": 0.05, "50t": 0.03 },
      cmi: { "10t": 0.04, "50t": 0.02 },
      interaction: 0.003,
      cost_viable: true,
      hypothesis_count: 3,
      selected: true,
    },
    {
      id: "depth_bid",
      category: "depth",
      mi: { "10t": 0.02, "50t": 0.01 },
      cmi: { "10t": 0.01 },
      interaction: -0.001,
      cost_viable: false,
      hypothesis_count: 1,
      selected: false,
    },
    {
      id: "ent_shape",
      category: "entropy",
      mi: { "10t": 0.0 },
      cmi: {},
      interaction: 0.0,
      cost_viable: false,
      hypothesis_count: 0,
      selected: false,
    },
    {
      id: "flow_vwap",
      category: "flow",
      mi: { "10t": 0.01 },
      cmi: {},
      interaction: 0.0,
      cost_viable: false,
      hypothesis_count: 2,
      selected: false,
    },
  ],
  edges: [
    { source: "spread_ba", target: "depth_bid", weight: 2 },
    { source: "spread_ba", target: "flow_vwap", weight: 1 },
  ],
  meta: {
    symbol: "BTC",
    n_samples: 6000,
    last_updated: "2026-05-21T11:03:29",
    total_features: 4,
  },
};

const EMPTY_NETWORK: NetworkResponse = {
  nodes: [],
  edges: [],
  meta: { symbol: "BTC", n_samples: 0, last_updated: "", total_features: 0 },
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("NetworkPage", () => {
  beforeEach(() => {
    mockGetNetwork.mockReset();
  });

  it("shows loading state initially", () => {
    mockGetNetwork.mockReturnValue(new Promise(() => {}));
    render(<NetworkPage />);
    expect(screen.getByText("Loading network data...")).toBeInTheDocument();
  });

  it("renders page header and metadata", async () => {
    mockGetNetwork.mockResolvedValue(MOCK_NETWORK);
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByText("Feature Network")).toBeInTheDocument();
    });
    expect(screen.getByText(/BTC/)).toBeInTheDocument();
    expect(screen.getByText(/4 features/)).toBeInTheDocument();
    expect(screen.getByText(/2 co-occurrence edges/)).toBeInTheDocument();
  });

  it("renders stats badges", async () => {
    mockGetNetwork.mockResolvedValue(MOCK_NETWORK);
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByText(/3 with MI > 0/)).toBeInTheDocument();
    });
    expect(screen.getByText(/1 cost-viable/)).toBeInTheDocument();
    expect(screen.getByText(/3 tested/)).toBeInTheDocument();
  });

  it("renders category filter buttons", async () => {
    mockGetNetwork.mockResolvedValue(MOCK_NETWORK);
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByText("Spread")).toBeInTheDocument();
    });
    expect(screen.getByText("Depth")).toBeInTheDocument();
    expect(screen.getByText("Entropy")).toBeInTheDocument();
    expect(screen.getByText("Flow")).toBeInTheDocument();
  });

  it("renders MI threshold slider", async () => {
    mockGetNetwork.mockResolvedValue(MOCK_NETWORK);
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByTestId("mi-slider")).toBeInTheDocument();
    });
    expect(screen.getByText("0.000")).toBeInTheDocument();
  });

  it("passes data to FeatureNetwork component", async () => {
    mockGetNetwork.mockResolvedValue(MOCK_NETWORK);
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByTestId("feature-network")).toBeInTheDocument();
    });
    expect(screen.getByTestId("node-count").textContent).toBe("4");
    expect(screen.getByTestId("edge-count").textContent).toBe("2");
  });

  it("updates MI threshold on slider change", async () => {
    mockGetNetwork.mockResolvedValue(MOCK_NETWORK);
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByTestId("mi-slider")).toBeInTheDocument();
    });
    fireEvent.change(screen.getByTestId("mi-slider"), {
      target: { value: "0.025" },
    });
    // The threshold value shows in the label span and is passed to the component
    expect(screen.getByTestId("mi-threshold").textContent).toBe("0.025");
  });

  it("toggles category filter on click", async () => {
    mockGetNetwork.mockResolvedValue(MOCK_NETWORK);
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByText("Spread")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Spread"));
    expect(screen.getByTestId("active-categories").textContent).toBe("spread");
    expect(screen.getByText("Clear")).toBeInTheDocument();
  });

  it("clears category filters", async () => {
    mockGetNetwork.mockResolvedValue(MOCK_NETWORK);
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByText("Spread")).toBeInTheDocument();
    });
    fireEvent.click(screen.getByText("Spread"));
    expect(screen.getByText("Clear")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Clear"));
    expect(screen.getByTestId("active-categories").textContent).toBe("");
  });

  it("shows error state", async () => {
    mockGetNetwork.mockRejectedValue(new Error("Connection refused"));
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByText("Connection refused")).toBeInTheDocument();
    });
  });

  it("shows empty state when no data", async () => {
    mockGetNetwork.mockResolvedValue(EMPTY_NETWORK);
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByText(/No IT engine data/)).toBeInTheDocument();
    });
  });

  it("renders legend", async () => {
    mockGetNetwork.mockResolvedValue(MOCK_NETWORK);
    render(<NetworkPage />);
    await waitFor(() => {
      expect(screen.getByText("IT-selected")).toBeInTheDocument();
    });
    expect(screen.getByText("Cost-viable")).toBeInTheDocument();
    expect(screen.getByText(/Node size = max MI/)).toBeInTheDocument();
  });
});
