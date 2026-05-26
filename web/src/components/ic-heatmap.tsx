"use client";

import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useMemo, useCallback } from "react";
import type { HeatmapResponse } from "@/lib/api";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

// Feature category prefixes for row grouping
const CATEGORY_ORDER = [
  "spread", "depth", "imbalance", "flow", "whale", "ent", "trend",
  "vol", "liq", "concentration", "regime", "gmm", "cross", "momentum",
  "funding", "oi",
];

function formatHorizon(s: number): string {
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.round(s / 60)}min`;
  if (s < 86400) return `${Math.round(s / 3600)}h`;
  return `${Math.round(s / 86400)}d`;
}

function categorize(feature: string): string {
  for (const prefix of CATEGORY_ORDER) {
    if (feature.startsWith(prefix)) return prefix;
  }
  return "other";
}

interface IcHeatmapProps {
  data: HeatmapResponse;
}

export function IcHeatmap({ data }: IcHeatmapProps) {
  const router = useRouter();

  const { z, features, horizons } = useMemo(() => {
    const { entries, features: rawFeatures, horizons: rawHorizons } = data;

    // Sort features by category then name
    const sorted = [...rawFeatures].sort((a, b) => {
      const ca = categorize(a);
      const cb = categorize(b);
      if (ca !== cb) {
        const ia = CATEGORY_ORDER.indexOf(ca);
        const ib = CATEGORY_ORDER.indexOf(cb);
        return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
      }
      return a.localeCompare(b);
    });

    const sortedHorizons = [...rawHorizons].sort((a, b) => a - b);
    const horizonLabels = sortedHorizons.map(formatHorizon);

    // Build lookup: (feature, horizon_s) → ic values for averaging
    const lookup = new Map<string, number[]>();
    for (const e of entries) {
      const key = `${e.feature}|${e.horizon_s}`;
      if (!lookup.has(key)) lookup.set(key, []);
      lookup.get(key)!.push(e.ic);
    }

    // Build z matrix: rows = features, cols = horizons
    const zMatrix: (number | null)[][] = sorted.map((feat) =>
      sortedHorizons.map((h) => {
        const vals = lookup.get(`${feat}|${h}`);
        if (!vals || vals.length === 0) return null;
        return vals.reduce((a, b) => a + b, 0) / vals.length;
      })
    );

    return { z: zMatrix, features: sorted, horizons: horizonLabels };
  }, [data]);

  const handleClick = useCallback(
    (event: Readonly<Plotly.PlotMouseEvent>) => {
      const point = event.points[0];
      if (!point) return;
      const feature = features[point.y as number];
      if (feature) {
        router.push(`/explorer?feature=${encodeURIComponent(feature)}`);
      }
    },
    [features, router]
  );

  if (features.length === 0 || horizons.length === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-8 text-center">
        <p className="text-zinc-500 text-sm">No heatmap data available yet.</p>
        <p className="text-zinc-600 text-xs mt-1">
          Run agent cycles to generate hypothesis data.
        </p>
      </div>
    );
  }

  // Compute reasonable height based on feature count
  const plotHeight = Math.max(400, features.length * 18 + 100);

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
      <Plot
        data={[
          {
            z,
            x: horizons,
            y: features,
            type: "heatmap",
            colorscale: [
              [0, "#dc2626"],
              [0.25, "#f87171"],
              [0.5, "#1c1c1c"],
              [0.75, "#60a5fa"],
              [1, "#2563eb"],
            ],
            zmin: -0.1,
            zmax: 0.1,
            colorbar: {
              title: { text: "IC", font: { color: "#a1a1aa", size: 11 } },
              tickfont: { color: "#a1a1aa", size: 10 },
              bgcolor: "rgba(0,0,0,0)",
              bordercolor: "#3f3f46",
              len: 0.6,
            },
            hoverongaps: false,
            hovertemplate:
              "<b>%{y}</b><br>Horizon: %{x}<br>IC: %{z:.4f}<extra></extra>",
          },
        ]}
        layout={{
          width: undefined,
          height: plotHeight,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { family: "monospace", color: "#a1a1aa", size: 10 },
          margin: { l: 200, r: 80, t: 30, b: 60 },
          xaxis: {
            title: { text: "Horizon", font: { size: 11 } },
            tickfont: { size: 10 },
            side: "bottom",
            gridcolor: "#27272a",
          },
          yaxis: {
            tickfont: { size: 9 },
            autorange: "reversed",
            gridcolor: "#27272a",
          },
        }}
        config={{
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: [
            "select2d", "lasso2d", "autoScale2d",
          ],
          toImageButtonOptions: {
            format: "png",
            filename: "nat_ic_heatmap",
            height: plotHeight,
            width: 1200,
          },
          displaylogo: false,
        }}
        style={{ width: "100%" }}
        onClick={handleClick}
      />
    </div>
  );
}
