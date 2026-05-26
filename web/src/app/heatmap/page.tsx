"use client";

import { useEffect, useState } from "react";
import { IcHeatmap } from "@/components/ic-heatmap";
import { getHeatmap } from "@/lib/api";
import type { HeatmapResponse } from "@/lib/api";

export default function HeatmapPage() {
  const [data, setData] = useState<HeatmapResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHeatmap()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-5 max-w-7xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">IC Heatmap</h2>
          <p className="text-xs text-zinc-500 mt-1">
            Feature x horizon information coefficient matrix
          </p>
        </div>
        {data && (
          <span className="text-xs text-zinc-500">
            {data.features.length} features x {data.horizons.length} horizons
          </span>
        )}
      </div>

      {error && (
        <div className="text-xs text-red-400 bg-red-900/30 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {loading ? (
        <div className="bg-zinc-900 rounded-lg border border-zinc-800 h-96 animate-pulse" />
      ) : data ? (
        <IcHeatmap data={data} />
      ) : null}
    </div>
  );
}
