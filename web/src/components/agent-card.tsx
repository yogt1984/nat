"use client";

const PHASE_COLORS: Record<string, string> = {
  SLEEPING: "border-emerald-500",
  SLEEP: "border-emerald-500",
  IDLE: "border-emerald-500",
  EXECUTE: "border-blue-500",
  RUNNING: "border-blue-500",
  GENERATE: "border-amber-500",
  MANIFEST: "border-amber-500",
  MONITOR: "border-violet-500",
  ERROR: "border-red-500",
  STOPPED: "border-zinc-600",
};

const PHASE_DOTS: Record<string, string> = {
  SLEEPING: "bg-emerald-500",
  SLEEP: "bg-emerald-500",
  IDLE: "bg-emerald-500",
  EXECUTE: "bg-blue-500",
  RUNNING: "bg-blue-500",
  GENERATE: "bg-amber-500",
  MANIFEST: "bg-amber-500",
  MONITOR: "bg-violet-500",
  ERROR: "bg-red-500",
  STOPPED: "bg-zinc-600",
};

const AGENT_LABELS: Record<string, string> = {
  microstructure: "Microstructure",
  medium_freq: "Medium Freq",
  macro: "Macro",
};

interface AgentCardProps {
  name: string;
  phase: string;
  cycleCount: number;
  lastCycleAt: string | null;
  tested: number;
  flash?: boolean;
}

export function AgentCard({ name, phase, cycleCount, lastCycleAt, tested, flash }: AgentCardProps) {
  const borderColor = PHASE_COLORS[phase] || "border-zinc-700";
  const dotColor = PHASE_DOTS[phase] || "bg-zinc-600";
  const label = AGENT_LABELS[name] || name;

  const timeAgo = lastCycleAt ? formatTimeAgo(lastCycleAt) : "never";

  return (
    <div
      className={`rounded-lg border-l-4 ${borderColor} bg-zinc-900 p-4 transition-[border-color] duration-500 ${flash ? "animate-flash-border" : ""}`}
      data-testid={`agent-card-${name}`}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold">{label}</h3>
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full ${dotColor} ${phase === "EXECUTE" || phase === "RUNNING" ? "animate-pulse" : ""}`} />
          <span className="text-xs text-zinc-400">{phase}</span>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-zinc-500">Cycles</span>
          <p className="text-zinc-200 font-medium">{cycleCount}</p>
        </div>
        <div>
          <span className="text-zinc-500">Tested</span>
          <p className="text-zinc-200 font-medium">{tested}</p>
        </div>
        <div className="col-span-2">
          <span className="text-zinc-500">Last cycle</span>
          <p className="text-zinc-200 font-medium">{timeAgo}</p>
        </div>
      </div>
    </div>
  );
}

function formatTimeAgo(iso: string): string {
  const date = new Date(iso);
  const now = Date.now();
  const diffMs = now - date.getTime();
  if (diffMs < 0) return "just now";
  const mins = Math.floor(diffMs / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}
