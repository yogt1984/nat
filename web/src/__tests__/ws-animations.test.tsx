import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";

// ---------------------------------------------------------------------------
// Mock WebSocket globally (jsdom doesn't have it)
// ---------------------------------------------------------------------------

const WS_OPEN = 1;
const WS_CLOSED = 3;
const WS_CONNECTING = 0;

if (typeof globalThis.WebSocket === "undefined") {
  Object.defineProperty(globalThis, "WebSocket", {
    value: { OPEN: WS_OPEN, CLOSED: WS_CLOSED, CONNECTING: WS_CONNECTING },
    writable: true,
  });
}

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

// Mock useResearchWs — control readyState per test
let mockReadyState = WS_CLOSED;
vi.mock("@/lib/ws", () => ({
  useResearchWs: () => ({
    readyState: mockReadyState,
    lastEvent: null,
  }),
}));

// Mock next/navigation
vi.mock("next/navigation", () => ({
  usePathname: () => "/",
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

import { Sidebar } from "@/components/sidebar";
import { AgentCard } from "@/components/agent-card";

// ---------------------------------------------------------------------------
// Sidebar connection status tests
// ---------------------------------------------------------------------------

describe("Sidebar connection status", () => {
  it("shows Disconnected when WS is closed", () => {
    mockReadyState = WS_CLOSED;
    render(<Sidebar />);
    expect(screen.getByTestId("ws-status-label").textContent).toBe(
      "Disconnected"
    );
    expect(screen.getByTestId("ws-status-dot").className).toContain(
      "bg-red-500"
    );
  });

  it("shows Connected when WS is open", () => {
    mockReadyState = WS_OPEN;
    render(<Sidebar />);
    expect(screen.getByTestId("ws-status-label").textContent).toBe(
      "Connected"
    );
    expect(screen.getByTestId("ws-status-dot").className).toContain(
      "bg-emerald-500"
    );
    expect(screen.getByTestId("ws-status-dot").className).toContain(
      "animate-pulse"
    );
  });

  it("shows Connecting when WS is connecting", () => {
    mockReadyState = WS_CONNECTING;
    render(<Sidebar />);
    expect(screen.getByTestId("ws-status-label").textContent).toBe(
      "Connecting..."
    );
    expect(screen.getByTestId("ws-status-dot").className).toContain(
      "bg-amber-500"
    );
  });

  it("renders nav items", () => {
    mockReadyState = WS_OPEN;
    render(<Sidebar />);
    expect(screen.getByText("Dashboard")).toBeInTheDocument();
    expect(screen.getByText("Explorer")).toBeInTheDocument();
    expect(screen.getByText("Network")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// AgentCard flash tests
// ---------------------------------------------------------------------------

describe("AgentCard", () => {
  it("renders agent name and phase", () => {
    render(
      <AgentCard
        name="microstructure"
        phase="EXECUTE"
        cycleCount={5}
        lastCycleAt={null}
        tested={42}
      />
    );
    expect(screen.getByText("Microstructure")).toBeInTheDocument();
    expect(screen.getByText("EXECUTE")).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("applies animate-flash-border when flash is true", () => {
    render(
      <AgentCard
        name="microstructure"
        phase="EXECUTE"
        cycleCount={0}
        lastCycleAt={null}
        tested={0}
        flash={true}
      />
    );
    const card = screen.getByTestId("agent-card-microstructure");
    expect(card.className).toContain("animate-flash-border");
  });

  it("does not apply flash animation when flash is false", () => {
    render(
      <AgentCard
        name="microstructure"
        phase="IDLE"
        cycleCount={0}
        lastCycleAt={null}
        tested={0}
        flash={false}
      />
    );
    const card = screen.getByTestId("agent-card-microstructure");
    expect(card.className).not.toContain("animate-flash-border");
  });

  it("applies phase-specific border color", () => {
    render(
      <AgentCard
        name="medium_freq"
        phase="SLEEPING"
        cycleCount={3}
        lastCycleAt={null}
        tested={10}
      />
    );
    const card = screen.getByTestId("agent-card-medium_freq");
    expect(card.className).toContain("border-emerald-500");
  });

  it("shows pulsing dot when executing", () => {
    render(
      <AgentCard
        name="macro"
        phase="EXECUTE"
        cycleCount={0}
        lastCycleAt={null}
        tested={0}
      />
    );
    const dots = document.querySelectorAll(".animate-pulse");
    expect(dots.length).toBeGreaterThan(0);
  });

  it("formats time ago correctly", () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60_000).toISOString();
    render(
      <AgentCard
        name="microstructure"
        phase="SLEEPING"
        cycleCount={1}
        lastCycleAt={fiveMinAgo}
        tested={3}
      />
    );
    expect(screen.getByText("5m ago")).toBeInTheDocument();
  });
});
