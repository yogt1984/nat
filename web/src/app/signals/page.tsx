"use client";

import { useEffect, useState } from "react";
import { listSignals } from "@/lib/api";
import type { Hypothesis } from "@/lib/api";
import { SignalTable } from "@/components/signal-table";
import { IcBarChart } from "@/components/ic-bar-chart";
import { WeightTreemap } from "@/components/weight-treemap";

export default function SignalsPage() {
  const [signals, setSignals] = useState<Hypothesis[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listSignals()
      .then(setSignals)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-5 max-w-7xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Signal Registry</h2>
          <p className="text-xs text-zinc-500 mt-1">
            Promoted signals with live performance monitoring
          </p>
        </div>
        {signals.length > 0 && (
          <span className="text-xs text-zinc-500">
            {signals.length} registered signal{signals.length !== 1 && "s"}
          </span>
        )}
      </div>

      {error && (
        <div className="text-xs text-red-400 bg-red-900/30 px-3 py-2 rounded">
          {error}
        </div>
      )}

      {loading ? (
        <div className="space-y-4">
          <div className="bg-zinc-900 rounded-lg border border-zinc-800 h-48 animate-pulse" />
          <div className="bg-zinc-900 rounded-lg border border-zinc-800 h-64 animate-pulse" />
        </div>
      ) : (
        <>
          <SignalTable signals={signals} />

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <IcBarChart signals={signals} />
            <WeightTreemap signals={signals} />
          </div>
        </>
      )}
    </div>
  );
}
