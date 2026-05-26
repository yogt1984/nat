"use client";

import type { ResearchStats } from "@/lib/api";

interface StatsBarProps {
  stats: ResearchStats | null;
}

export function StatsBar({ stats }: StatsBarProps) {
  if (!stats) {
    return (
      <div className="grid grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="bg-zinc-900 rounded-lg p-4 animate-pulse h-20" />
        ))}
      </div>
    );
  }

  const registered = stats.by_status?.replicated || 0;
  const graveyard = (stats.by_status?.no_effect || 0) +
    (stats.by_status?.no_replication || 0) +
    (stats.by_status?.redundant || 0) +
    (stats.by_status?.fdr_rejected || 0) +
    (stats.by_status?.command_error || 0);

  const items = [
    { label: "Total Tested", value: stats.total_hypotheses, color: "text-zinc-100" },
    { label: "Registered", value: registered, color: "text-emerald-400" },
    { label: "Graveyard", value: graveyard, color: "text-red-400" },
    { label: "Cycles", value: stats.total_cycles, color: "text-blue-400" },
  ];

  return (
    <div className="grid grid-cols-4 gap-4">
      {items.map(({ label, value, color }) => (
        <div key={label} className="bg-zinc-900 rounded-lg p-4 border border-zinc-800">
          <p className="text-xs text-zinc-500 mb-1">{label}</p>
          <p className={`text-2xl font-bold ${color}`}>{value}</p>
        </div>
      ))}
    </div>
  );
}
