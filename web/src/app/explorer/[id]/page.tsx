"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { getHypothesis } from "@/lib/api";
import type { Hypothesis } from "@/lib/api";

export default function HypothesisDetailPage() {
  const params = useParams<{ id: string }>();
  const [hypothesis, setHypothesis] = useState<Hypothesis | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!params.id) return;
    getHypothesis(params.id)
      .then(setHypothesis)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"));
  }, [params.id]);

  if (error) {
    return (
      <div className="p-6">
        <p className="text-red-400">{error}</p>
      </div>
    );
  }

  if (!hypothesis) {
    return (
      <div className="p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 w-48 bg-zinc-800 rounded" />
          <div className="h-4 w-96 bg-zinc-800 rounded" />
        </div>
      </div>
    );
  }

  const STATUS_COLOR: Record<string, string> = {
    replicated: "text-emerald-400",
    no_effect: "text-red-400",
    no_replication: "text-red-400",
    redundant: "text-amber-400",
  };

  return (
    <div className="space-y-6 max-w-4xl">
      <div>
        <h2 className="text-xl font-bold">{hypothesis.id}</h2>
        <p className="text-zinc-400 mt-1">{hypothesis.claim}</p>
      </div>

      <div className="grid grid-cols-3 gap-4 text-xs">
        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-3">
          <span className="text-zinc-500">Agent</span>
          <p className="text-zinc-200 mt-1">{hypothesis.agent}</p>
        </div>
        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-3">
          <span className="text-zinc-500">Generator</span>
          <p className="text-zinc-200 mt-1">{hypothesis.generator}</p>
        </div>
        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-3">
          <span className="text-zinc-500">Status</span>
          <p className={`mt-1 font-medium ${STATUS_COLOR[hypothesis.status] || "text-zinc-200"}`}>
            {hypothesis.status}
          </p>
        </div>
      </div>

      {/* Gate results */}
      {hypothesis.gates.length > 0 && (
        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
          <h3 className="text-sm font-semibold mb-3">Gate Results</h3>
          <div className="space-y-2">
            {hypothesis.gates.map((gate, i) => (
              <div key={i} className="flex items-center gap-3 text-xs">
                <span
                  className={`w-2 h-2 rounded-full shrink-0 ${
                    gate.passed ? "bg-emerald-500" : "bg-red-500"
                  }`}
                />
                <span className="w-28 text-zinc-400 font-medium">{gate.name}</span>
                <span className="text-zinc-300 flex-1">{gate.message}</span>
                {gate.metric != null && (
                  <span className="text-zinc-500 font-mono">{gate.metric.toFixed(4)}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Timestamps */}
      <div className="text-xs text-zinc-500 space-y-1">
        <p>Created: {hypothesis.timestamps.created}</p>
        {hypothesis.timestamps.completed && (
          <p>Completed: {hypothesis.timestamps.completed}</p>
        )}
        {hypothesis.parent_id && <p>Parent: {hypothesis.parent_id}</p>}
      </div>
    </div>
  );
}
