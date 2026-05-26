"use client";

import dynamic from "next/dynamic";
import type { Hypothesis } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface FailurePieProps {
  hypotheses: Hypothesis[];
}

/** Count how many hypotheses failed at each gate (first failing gate). */
function failureByGate(hypotheses: Hypothesis[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const h of hypotheses) {
    const failed = h.gates.find((g) => !g.passed);
    const gate = failed?.name ?? "unknown";
    counts.set(gate, (counts.get(gate) ?? 0) + 1);
  }
  return counts;
}

const GATE_COLOR: Record<string, string> = {
  discovery: "#ef4444",
  cost: "#f97316",
  temporal: "#eab308",
  symbol: "#8b5cf6",
  correlation: "#3b82f6",
  unknown: "#71717a",
};

export function FailurePie({ hypotheses }: FailurePieProps) {
  const counts = failureByGate(hypotheses);

  if (counts.size === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 text-center">
        <p className="text-zinc-500 text-sm">No failure data.</p>
      </div>
    );
  }

  const labels = [...counts.keys()];
  const values = labels.map((l) => counts.get(l)!);
  const colors = labels.map((l) => GATE_COLOR[l] ?? "#71717a");

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
      <h3 className="text-sm font-semibold px-4 pt-4">Failure by Gate</h3>
      <Plot
        data={[
          {
            type: "pie",
            labels,
            values,
            hole: 0.45,
            marker: { colors },
            textfont: { color: "#e4e4e7", size: 11, family: "monospace" },
            hovertemplate: "<b>%{label}</b><br>%{value} (%{percent})<extra></extra>",
          } as Plotly.Data,
        ]}
        layout={{
          height: 280,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { family: "monospace", color: "#a1a1aa", size: 10 },
          margin: { l: 20, r: 20, t: 10, b: 20 },
          showlegend: true,
          legend: { font: { size: 10 }, bgcolor: "rgba(0,0,0,0)" },
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
