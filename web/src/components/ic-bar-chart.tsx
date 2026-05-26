"use client";

import dynamic from "next/dynamic";
import type { Hypothesis } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface IcBarChartProps {
  signals: Hypothesis[];
}

function extractIc(h: Hypothesis): number | null {
  const g = h.gates.find(
    (g) => g.name === "discovery" || g.name === "ic_check"
  );
  return g?.metric ?? null;
}

export function IcBarChart({ signals }: IcBarChartProps) {
  const withIc = signals
    .map((s) => ({ id: s.id.slice(0, 12), agent: s.agent, ic: extractIc(s) }))
    .filter((s): s is { id: string; agent: string; ic: number } => s.ic != null)
    .sort((a, b) => b.ic - a.ic);

  if (withIc.length === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 text-center">
        <p className="text-zinc-500 text-sm">No IC data available.</p>
      </div>
    );
  }

  const AGENT_COLOR: Record<string, string> = {
    microstructure: "#8b5cf6",
    medium_freq: "#3b82f6",
    macro: "#06b6d4",
  };

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
      <h3 className="text-sm font-semibold px-4 pt-4">Discovery IC by Signal</h3>
      <Plot
        data={[
          {
            type: "bar",
            x: withIc.map((s) => s.id),
            y: withIc.map((s) => s.ic),
            marker: {
              color: withIc.map(
                (s) => AGENT_COLOR[s.agent] || "#71717a"
              ),
            },
            hovertemplate:
              "<b>%{x}</b><br>IC: %{y:.4f}<extra></extra>",
          },
        ]}
        layout={{
          height: 280,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { family: "monospace", color: "#a1a1aa", size: 10 },
          margin: { l: 50, r: 20, t: 10, b: 70 },
          xaxis: {
            tickangle: -45,
            tickfont: { size: 9 },
            gridcolor: "#27272a",
          },
          yaxis: {
            title: { text: "IC", font: { size: 11 } },
            gridcolor: "#27272a",
            zeroline: true,
            zerolinecolor: "#3f3f46",
          },
          bargap: 0.3,
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
