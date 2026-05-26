"use client";

import { useCallback, useEffect, useState } from "react";
import { FilterBar } from "@/components/filter-bar";
import { GateFunnel } from "@/components/gate-funnel";
import { HypothesisTable, Pagination } from "@/components/hypothesis-table";
import { listHypotheses } from "@/lib/api";
import type { Hypothesis } from "@/lib/api";

type SortKey = "id" | "agent" | "generator" | "status" | "ic" | "date";
type SortDir = "asc" | "desc";

const PAGE_SIZE = 30;

export default function ExplorerPage() {
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  // Filters
  const [agent, setAgent] = useState("");
  const [generator, setGenerator] = useState("");
  const [status, setStatus] = useState("");

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, string | number> = {
        offset,
        limit: PAGE_SIZE,
      };
      if (agent) params.agent = agent;
      if (generator) params.generator = generator;
      if (status) params.status = status;

      const res = await listHypotheses(params as Parameters<typeof listHypotheses>[0]);
      setHypotheses(res.items);
      setTotal(res.total);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, [offset, agent, generator, status]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Reset to first page when filters change
  const handleFilterChange = (setter: (v: string) => void) => (v: string) => {
    setter(v);
    setOffset(0);
  };

  const handleSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  return (
    <div className="space-y-5 max-w-7xl">
      <h2 className="text-xl font-bold">Hypothesis Explorer</h2>

      <FilterBar
        agent={agent}
        generator={generator}
        status={status}
        onAgentChange={handleFilterChange(setAgent)}
        onGeneratorChange={handleFilterChange(setGenerator)}
        onStatusChange={handleFilterChange(setStatus)}
      />

      {error && (
        <div className="text-xs text-red-400 bg-red-900/30 px-3 py-2 rounded">
          {error}
        </div>
      )}

      <div className="grid grid-cols-3 gap-5">
        <div className="col-span-2">
          <div className={loading ? "opacity-50" : ""}>
            <HypothesisTable
              hypotheses={hypotheses}
              sortKey={sortKey}
              sortDir={sortDir}
              onSort={handleSort}
            />
          </div>
          <Pagination
            total={total}
            offset={offset}
            limit={PAGE_SIZE}
            onPageChange={setOffset}
          />
        </div>
        <div>
          <GateFunnel hypotheses={hypotheses} />
        </div>
      </div>
    </div>
  );
}
