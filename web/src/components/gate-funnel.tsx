"use client";

import type { Hypothesis } from "@/lib/api";

interface GateFunnelProps {
  hypotheses: Hypothesis[];
}

const GATE_NAMES = ["Tested", "G1 Discovery", "G2 Temporal", "G3 Symbol", "G4 Correlation", "Registered"];
const BAR_COLORS = ["#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#c084fc", "#10b981"];

export function GateFunnel({ hypotheses }: GateFunnelProps) {
  const counts = computeFunnelCounts(hypotheses);

  if (counts[0] === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
        <h3 className="text-sm font-semibold mb-3">Gate Funnel</h3>
        <p className="text-xs text-zinc-500 text-center py-4">No data</p>
      </div>
    );
  }

  const maxCount = counts[0];

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
      <h3 className="text-sm font-semibold mb-3">Gate Funnel</h3>
      <div className="space-y-1.5">
        {GATE_NAMES.map((name, i) => {
          const count = counts[i];
          const pct = maxCount > 0 ? (count / maxCount) * 100 : 0;
          return (
            <div key={name} className="flex items-center gap-2 text-xs">
              <span className="w-24 text-zinc-500 text-right shrink-0">{name}</span>
              <div className="flex-1 h-5 bg-zinc-800 rounded overflow-hidden">
                <div
                  className="h-full rounded transition-all duration-500"
                  style={{ width: `${pct}%`, backgroundColor: BAR_COLORS[i] }}
                />
              </div>
              <span className="w-10 text-zinc-400 text-right font-mono">{count}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function computeFunnelCounts(hypotheses: Hypothesis[]): number[] {
  let tested = hypotheses.length;
  let passG1 = 0;
  let passG2 = 0;
  let passG3 = 0;
  let passG4 = 0;
  let registered = 0;

  for (const h of hypotheses) {
    const gates = h.gates || [];
    const gatesPassed = gates.filter((g) => g.passed).length;

    if (gatesPassed >= 1) passG1++;
    if (gatesPassed >= 2) passG2++;
    if (gatesPassed >= 3) passG3++;
    if (gatesPassed >= 4) passG4++;
    if (h.status === "replicated" || h.status === "registered") registered++;
  }

  return [tested, passG1, passG2, passG3, passG4, registered];
}
