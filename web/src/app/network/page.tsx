"use client";

import { useEffect, useState, useMemo, useCallback } from "react";
import { getNetwork, type NetworkResponse } from "@/lib/api";
import { FeatureNetwork } from "@/components/feature-network";

// ---------------------------------------------------------------------------
// Category labels for filter chips
// ---------------------------------------------------------------------------

const CATEGORY_LABELS: Record<string, string> = {
  spread: "Spread",
  depth: "Depth",
  imbalance: "Imbalance",
  flow: "Flow",
  volatility: "Volatility",
  entropy: "Entropy",
  trend: "Trend",
  illiquidity: "Illiquidity",
  toxicity: "Toxicity",
  whale: "Whale",
  liquidation: "Liquidation",
  concentration: "Concentration",
  context: "Context",
  raw: "Raw",
  regime: "Regime",
  cross_symbol: "Cross-Symbol",
  derived: "Derived",
  other: "Other",
};

const CATEGORY_COLORS: Record<string, string> = {
  spread: "#ef4444",
  depth: "#3b82f6",
  imbalance: "#f97316",
  flow: "#22c55e",
  volatility: "#a855f7",
  entropy: "#ec4899",
  trend: "#14b8a6",
  illiquidity: "#eab308",
  toxicity: "#dc2626",
  whale: "#6366f1",
  liquidation: "#f43f5e",
  concentration: "#84cc16",
  context: "#64748b",
  raw: "#94a3b8",
  regime: "#8b5cf6",
  cross_symbol: "#06b6d4",
  derived: "#d97706",
  other: "#6b7280",
};

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function NetworkPage() {
  const [data, setData] = useState<NetworkResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [miThreshold, setMiThreshold] = useState(0);
  const [activeCategories, setActiveCategories] = useState<Set<string>>(
    new Set()
  );

  useEffect(() => {
    getNetwork()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Derive available categories from data
  const categories = useMemo(() => {
    if (!data) return [];
    const cats = new Set(data.nodes.map((n) => n.category));
    return Array.from(cats).sort();
  }, [data]);

  const toggleCategory = useCallback((cat: string) => {
    setActiveCategories((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  }, []);

  const handleNodeClick = useCallback((featureId: string) => {
    window.open(`/explorer?feature=${encodeURIComponent(featureId)}`, "_blank");
  }, []);

  // Stats
  const stats = useMemo(() => {
    if (!data) return null;
    const nonZeroMi = data.nodes.filter((n) =>
      Object.values(n.mi).some((v) => v > 0)
    ).length;
    const costViable = data.nodes.filter((n) => n.cost_viable).length;
    const withHypotheses = data.nodes.filter(
      (n) => n.hypothesis_count > 0
    ).length;
    return { total: data.nodes.length, nonZeroMi, costViable, withHypotheses };
  }, [data]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-zinc-500">Loading network data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4">
        <h2 className="text-xl font-bold mb-2">Feature Network</h2>
        <div className="text-red-400 bg-red-950/30 border border-red-900 rounded p-3">
          {error}
        </div>
      </div>
    );
  }

  if (!data || data.nodes.length === 0) {
    return (
      <div className="p-4">
        <h2 className="text-xl font-bold mb-2">Feature Network</h2>
        <p className="text-zinc-500">
          No IT engine data available. Run the IT engine to generate feature
          interaction data.
        </p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Feature Network</h2>
          <p className="text-zinc-500 text-sm">
            {data.meta.symbol} &mdash; {data.meta.total_features} features,{" "}
            {data.edges.length} co-occurrence edges &mdash; updated{" "}
            {data.meta.last_updated}
          </p>
        </div>
        {stats && (
          <div className="flex gap-4 text-xs text-zinc-400">
            <span>{stats.nonZeroMi} with MI &gt; 0</span>
            <span>{stats.costViable} cost-viable</span>
            <span>{stats.withHypotheses} tested</span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-4">
        {/* MI threshold slider */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-zinc-400 whitespace-nowrap">
            MI threshold
          </label>
          <input
            type="range"
            min={0}
            max={0.1}
            step={0.001}
            value={miThreshold}
            onChange={(e) => setMiThreshold(parseFloat(e.target.value))}
            className="w-32"
            data-testid="mi-slider"
          />
          <span className="text-xs text-zinc-300 font-mono w-12">
            {miThreshold.toFixed(3)}
          </span>
        </div>

        {/* Category filters */}
        <div className="flex flex-wrap gap-1.5">
          {categories.map((cat) => {
            const active = activeCategories.has(cat);
            const color = CATEGORY_COLORS[cat] || "#6b7280";
            return (
              <button
                key={cat}
                onClick={() => toggleCategory(cat)}
                className={`px-2 py-0.5 rounded text-xs border transition-colors ${
                  active
                    ? "border-current text-white"
                    : "border-zinc-700 text-zinc-400 hover:border-zinc-500"
                }`}
                style={active ? { borderColor: color, color } : undefined}
              >
                {CATEGORY_LABELS[cat] || cat}
              </button>
            );
          })}
          {activeCategories.size > 0 && (
            <button
              onClick={() => setActiveCategories(new Set())}
              className="px-2 py-0.5 rounded text-xs text-zinc-500 hover:text-zinc-300"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-xs text-zinc-400">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full border-2 border-white inline-block" />{" "}
          IT-selected
        </span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full border-2 border-yellow-400 inline-block" />{" "}
          Cost-viable
        </span>
        <span className="text-zinc-600">|</span>
        <span>Node size = max MI</span>
        <span>Edge width = co-occurrence count</span>
      </div>

      {/* Graph */}
      <FeatureNetwork
        nodes={data.nodes}
        edges={data.edges}
        miThreshold={miThreshold}
        activeCategories={activeCategories}
        onNodeClick={handleNodeClick}
      />
    </div>
  );
}
