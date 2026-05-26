import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import MathLabPage from "@/app/math/page";

// Mock MathPanel — KaTeX needs DOM APIs not available in jsdom
vi.mock("@/components/math-panel", () => ({
  MathPanel: ({ latex }: { latex: string }) => (
    <div data-testid="math-panel">{latex.slice(0, 40)}</div>
  ),
}));

describe("MathLabPage", () => {
  it("renders page header", () => {
    render(<MathLabPage />);
    expect(screen.getByText("Math Lab")).toBeInTheDocument();
    expect(screen.getByText(/formulations across/)).toBeInTheDocument();
  });

  it("renders all category headings", () => {
    render(<MathLabPage />);
    expect(screen.getAllByText("Feature Definitions").length).toBe(2); // button + heading
    expect(screen.getAllByText("5-Gate Protocol").length).toBe(2);
    expect(screen.getAllByText("Position Sizing & Promotion").length).toBe(2);
    expect(screen.getAllByText("Analytical Methods").length).toBe(2);
  });

  it("renders section titles collapsed by default", () => {
    render(<MathLabPage />);
    // Sections should be present
    expect(screen.getByText("Raw Order Book (10)")).toBeInTheDocument();
    expect(screen.getByText("Gate 1: Discovery")).toBeInTheDocument();
    expect(screen.getByText("Heatmap Convolver")).toBeInTheDocument();
    // But math panels should not be visible
    expect(screen.queryAllByTestId("math-panel")).toHaveLength(0);
  });

  it("expands a section on click", () => {
    render(<MathLabPage />);
    fireEvent.click(screen.getByText("Volatility (9)"));
    const panels = screen.getAllByTestId("math-panel");
    expect(panels.length).toBe(1);
    expect(panels[0].textContent).toContain("sigma");
  });

  it("collapses a section on second click", () => {
    render(<MathLabPage />);
    fireEvent.click(screen.getByText("Volatility (9)"));
    expect(screen.getAllByTestId("math-panel")).toHaveLength(1);
    fireEvent.click(screen.getByText("Volatility (9)"));
    expect(screen.queryAllByTestId("math-panel")).toHaveLength(0);
  });

  it("expand all / collapse all", () => {
    render(<MathLabPage />);
    fireEvent.click(screen.getByText("Expand all"));
    const panels = screen.getAllByTestId("math-panel");
    expect(panels.length).toBeGreaterThan(10);

    fireEvent.click(screen.getByText("Collapse all"));
    expect(screen.queryAllByTestId("math-panel")).toHaveLength(0);
  });

  it("filters by search query", () => {
    render(<MathLabPage />);
    const input = screen.getByPlaceholderText("Search formulas...");
    fireEvent.change(input, { target: { value: "entropy" } });
    expect(screen.getByText("Entropy (24)")).toBeInTheDocument();
    // Sections that don't match "entropy" should be gone
    expect(screen.queryByText("Raw Order Book (10)")).not.toBeInTheDocument();
  });

  it("filters by category button", () => {
    render(<MathLabPage />);
    fireEvent.click(screen.getByRole("button", { name: "5-Gate Protocol" }));
    expect(screen.getByText("Gate 1: Discovery")).toBeInTheDocument();
    expect(screen.getByText("Gate 5: Correlation Dedup")).toBeInTheDocument();
    // Feature sections should be hidden
    expect(screen.queryByText("Raw Order Book (10)")).not.toBeInTheDocument();
  });

  it("toggles category filter off on second click", () => {
    render(<MathLabPage />);
    fireEvent.click(screen.getByRole("button", { name: "5-Gate Protocol" }));
    expect(screen.queryByText("Raw Order Book (10)")).not.toBeInTheDocument();
    // Click again to deselect
    fireEvent.click(screen.getByRole("button", { name: "5-Gate Protocol" }));
    expect(screen.getByText("Raw Order Book (10)")).toBeInTheDocument();
  });

  it("shows empty state when search matches nothing", () => {
    render(<MathLabPage />);
    const input = screen.getByPlaceholderText("Search formulas...");
    fireEvent.change(input, { target: { value: "xyznonexistent" } });
    expect(screen.getByText(/No formulas match/)).toBeInTheDocument();
  });

  it("combined search + category filter", () => {
    render(<MathLabPage />);
    // Select gates category
    fireEvent.click(screen.getByRole("button", { name: "5-Gate Protocol" }));
    // Then search for "cost"
    const input = screen.getByPlaceholderText("Search formulas...");
    fireEvent.change(input, { target: { value: "cost" } });
    // Only the cost gate should match
    expect(screen.getByText("Gate 2: Cost")).toBeInTheDocument();
    expect(screen.queryByText("Gate 1: Discovery")).not.toBeInTheDocument();
  });

  it("all category has active styling by default", () => {
    render(<MathLabPage />);
    const allBtn = screen.getByText("All");
    expect(allBtn.className).toContain("bg-blue-600");
  });

  it("renders descriptions for each section", () => {
    render(<MathLabPage />);
    expect(
      screen.getByText(/Direct L2 order book measurements/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Benjamini–Hochberg procedure/)
    ).toBeInTheDocument();
  });
});
