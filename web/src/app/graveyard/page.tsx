"use client";

import { useEffect, useState } from "react";
import { listHypotheses } from "@/lib/api";
import type { Hypothesis } from "@/lib/api";
import { FailurePie } from "@/components/failure-pie";
import { GeneratorBars } from "@/components/generator-bars";
import { NearMissesTable, RecyclableTable } from "@/components/near-misses";

const FAILURE_STATUSES = [
  "no_effect",
  "no_replication",
  "redundant",
  "fdr_rejected",
  "command_error",
];

export default function GraveyardPage() {
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Fetch all failure statuses in parallel
    Promise.all(
      FAILURE_STATUSES.map((s) =>
        listHypotheses({ status: s, limit: 500 }).then((r) => r.items)
      )
    )
      .then((results) => setHypotheses(results.flat()))
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-5 max-w-7xl">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Graveyard</h2>
          <p className="text-xs text-zinc-500 mt-1">
            Failed hypothesis analysis and recyclable candidates
          </p>
        </div>
        {hypotheses.length > 0 && (
          <span className="text-xs text-zinc-500">
            {hypotheses.length} failed hypothes{hypotheses.length !== 1 ? "es" : "is"}
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
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <div className="bg-zinc-900 rounded-lg border border-zinc-800 h-72 animate-pulse" />
            <div className="bg-zinc-900 rounded-lg border border-zinc-800 h-72 animate-pulse" />
          </div>
          <div className="bg-zinc-900 rounded-lg border border-zinc-800 h-48 animate-pulse" />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
            <FailurePie hypotheses={hypotheses} />
            <GeneratorBars hypotheses={hypotheses} />
          </div>

          <NearMissesTable hypotheses={hypotheses} />
          <RecyclableTable hypotheses={hypotheses} />
        </>
      )}
    </div>
  );
}
