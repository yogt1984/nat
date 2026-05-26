"use client";

import dynamic from "next/dynamic";
import type { Hypothesis } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface WeightTreemapProps {
  signals: Hypothesis[];
}

const AGENT_COLOR: Record<string, string> = {
  microstructure: "#8b5cf6",
  medium_freq: "#3b82f6",
  macro: "#06b6d4",
};

function extractIc(h: Hypothesis): number {
  const g = h.gates.find(
    (g) => g.name === "discovery" || g.name === "ic_check"
  );
  return Math.abs(g?.metric ?? 0.01);
}

export function WeightTreemap({ signals }: WeightTreemapProps) {
  if (signals.length === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 text-center">
        <p className="text-zinc-500 text-sm">No signals for treemap.</p>
      </div>
    );
  }

  // Group by agent, size by |IC|
  const labels: string[] = [];
  const parents: string[] = [];
  const values: number[] = [];
  const colors: string[] = [];

  // Root
  labels.push("Portfolio");
  parents.push("");
  values.push(0);
  colors.push("#18181b");

  // Agent groups
  const agents = [...new Set(signals.map((s) => s.agent))];
  for (const agent of agents) {
    labels.push(agent);
    parents.push("Portfolio");
    values.push(0);
    colors.push(AGENT_COLOR[agent] || "#71717a");
  }

  // Individual signals
  for (const s of signals) {
    const ic = extractIc(s);
    labels.push(s.id.slice(0, 12));
    parents.push(s.agent);
    values.push(ic);
    colors.push(AGENT_COLOR[s.agent] || "#71717a");
  }

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
      <h3 className="text-sm font-semibold px-4 pt-4">Portfolio Weights by Agent</h3>
      <Plot
        data={[
          {
            type: "treemap",
            labels,
            parents,
            values,
            marker: { colors },
            textfont: { color: "#e4e4e7", size: 10, family: "monospace" },
            hovertemplate:
              "<b>%{label}</b><br>|IC|: %{value:.4f}<extra></extra>",
            branchvalues: "total",
          } as Plotly.Data,
        ]}
        layout={{
          height: 320,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { family: "monospace", color: "#a1a1aa", size: 10 },
          margin: { l: 10, r: 10, t: 10, b: 10 },
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
