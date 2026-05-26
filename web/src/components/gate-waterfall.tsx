"use client";

import dynamic from "next/dynamic";
import type { Gate } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface GateWaterfallProps {
  gates: Gate[];
}

export function GateWaterfall({ gates }: GateWaterfallProps) {
  if (gates.length === 0) return null;

  // Only show gates that have numeric metric + threshold
  const measurable = gates.filter((g) => g.metric != null && g.threshold != null);

  if (measurable.length === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
        <h3 className="text-sm font-semibold mb-3">Gate Waterfall</h3>
        <p className="text-xs text-zinc-500">No numeric gate metrics available.</p>
      </div>
    );
  }

  const names = measurable.map((g) => g.name);
  const metrics = measurable.map((g) => g.metric!);
  const thresholds = measurable.map((g) => g.threshold!);
  const colors = measurable.map((g) => (g.passed ? "#10b981" : "#ef4444"));

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
      <h3 className="text-sm font-semibold px-4 pt-4">Gate Waterfall</h3>
      <Plot
        data={[
          {
            type: "bar",
            orientation: "h",
            y: names,
            x: metrics,
            name: "Metric",
            marker: { color: colors },
            hovertemplate: "<b>%{y}</b><br>Metric: %{x:.4f}<extra></extra>",
          },
          {
            type: "scatter",
            mode: "markers",
            y: names,
            x: thresholds,
            name: "Threshold",
            marker: {
              symbol: "line-ns",
              size: 14,
              color: "#a1a1aa",
              line: { width: 2, color: "#a1a1aa" },
            },
            hovertemplate: "<b>%{y}</b><br>Threshold: %{x:.4f}<extra></extra>",
          },
        ]}
        layout={{
          height: Math.max(200, measurable.length * 40 + 80),
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { family: "monospace", color: "#a1a1aa", size: 10 },
          margin: { l: 140, r: 30, t: 10, b: 40 },
          xaxis: {
            title: { text: "Value", font: { size: 11 } },
            gridcolor: "#27272a",
            zeroline: true,
            zerolinecolor: "#3f3f46",
          },
          yaxis: {
            autorange: "reversed",
          },
          showlegend: true,
          legend: {
            x: 1,
            xanchor: "right",
            y: 1,
            font: { size: 10 },
            bgcolor: "rgba(0,0,0,0)",
          },
          bargap: 0.3,
        }}
        config={{
          responsive: true,
          displayModeBar: false,
        }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
