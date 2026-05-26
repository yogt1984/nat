"use client";

import dynamic from "next/dynamic";
import type { Hypothesis } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

interface GeneratorBarsProps {
  hypotheses: Hypothesis[];
}

export function GeneratorBars({ hypotheses }: GeneratorBarsProps) {
  // Count successes (replicated) and failures per generator
  const stats = new Map<string, { pass: number; fail: number }>();
  for (const h of hypotheses) {
    const gen = h.generator;
    if (!stats.has(gen)) stats.set(gen, { pass: 0, fail: 0 });
    const s = stats.get(gen)!;
    if (h.status === "replicated" || h.status === "discovery_passed") {
      s.pass++;
    } else {
      s.fail++;
    }
  }

  if (stats.size === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 text-center">
        <p className="text-zinc-500 text-sm">No generator data.</p>
      </div>
    );
  }

  const generators = [...stats.keys()].sort();
  const passes = generators.map((g) => stats.get(g)!.pass);
  const fails = generators.map((g) => stats.get(g)!.fail);

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
      <h3 className="text-sm font-semibold px-4 pt-4">Generator Success / Failure</h3>
      <Plot
        data={[
          {
            type: "bar",
            name: "Pass",
            x: generators,
            y: passes,
            marker: { color: "#10b981" },
          },
          {
            type: "bar",
            name: "Fail",
            x: generators,
            y: fails,
            marker: { color: "#ef4444" },
          },
        ]}
        layout={{
          height: 280,
          barmode: "stack",
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { family: "monospace", color: "#a1a1aa", size: 10 },
          margin: { l: 40, r: 20, t: 10, b: 80 },
          xaxis: { tickangle: -45, tickfont: { size: 9 }, gridcolor: "#27272a" },
          yaxis: { title: { text: "Count", font: { size: 11 } }, gridcolor: "#27272a" },
          showlegend: true,
          legend: { x: 1, xanchor: "right", y: 1, font: { size: 10 }, bgcolor: "rgba(0,0,0,0)" },
        }}
        config={{ responsive: true, displayModeBar: false }}
        style={{ width: "100%" }}
      />
    </div>
  );
}
