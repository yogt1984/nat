"use client";

import { useRouter } from "next/navigation";
import type { Hypothesis } from "@/lib/api";

type SortKey = "id" | "agent" | "generator" | "status" | "ic" | "date";
type SortDir = "asc" | "desc";

interface HypothesisTableProps {
  hypotheses: Hypothesis[];
  sortKey: SortKey;
  sortDir: SortDir;
  onSort: (key: SortKey) => void;
}

const STATUS_COLORS: Record<string, string> = {
  replicated: "text-emerald-400",
  discovery_passed: "text-blue-400",
  no_effect: "text-red-400",
  no_replication: "text-red-400",
  redundant: "text-amber-400",
  fdr_rejected: "text-red-400",
  command_error: "text-red-400",
};

export function HypothesisTable({ hypotheses, sortKey, sortDir, onSort }: HypothesisTableProps) {
  const router = useRouter();

  const sorted = [...hypotheses].sort((a, b) => {
    const dir = sortDir === "asc" ? 1 : -1;
    switch (sortKey) {
      case "id": return dir * a.id.localeCompare(b.id);
      case "agent": return dir * a.agent.localeCompare(b.agent);
      case "generator": return dir * a.generator.localeCompare(b.generator);
      case "status": return dir * a.status.localeCompare(b.status);
      case "ic": return dir * (extractIc(a) - extractIc(b));
      case "date": return dir * ((a.timestamps.completed || a.timestamps.created).localeCompare(
        b.timestamps.completed || b.timestamps.created));
      default: return 0;
    }
  });

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-zinc-800 text-zinc-500">
            <Th label="ID" sortKey="id" current={sortKey} dir={sortDir} onSort={onSort} />
            <Th label="Agent" sortKey="agent" current={sortKey} dir={sortDir} onSort={onSort} />
            <Th label="Generator" sortKey="generator" current={sortKey} dir={sortDir} onSort={onSort} />
            <th className="px-3 py-2.5 text-left font-medium">Claim</th>
            <Th label="IC" sortKey="ic" current={sortKey} dir={sortDir} onSort={onSort} />
            <Th label="Status" sortKey="status" current={sortKey} dir={sortDir} onSort={onSort} />
            <Th label="Date" sortKey="date" current={sortKey} dir={sortDir} onSort={onSort} />
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800">
          {sorted.length === 0 ? (
            <tr>
              <td colSpan={7} className="px-3 py-8 text-center text-zinc-500">
                No hypotheses found
              </td>
            </tr>
          ) : (
            sorted.map((h) => (
              <tr
                key={h.id}
                onClick={() => router.push(`/explorer/${h.id}`)}
                className="hover:bg-zinc-800/50 cursor-pointer transition-colors"
              >
                <td className="px-3 py-2 text-zinc-400 font-mono">{h.id.slice(0, 12)}</td>
                <td className="px-3 py-2 text-zinc-300">{h.agent}</td>
                <td className="px-3 py-2 text-zinc-300">{h.generator}</td>
                <td className="px-3 py-2 text-zinc-300 max-w-xs truncate">{h.claim}</td>
                <td className="px-3 py-2 text-zinc-200 font-mono">{formatIc(extractIc(h))}</td>
                <td className={`px-3 py-2 font-medium ${STATUS_COLORS[h.status] || "text-zinc-400"}`}>
                  {h.status}
                </td>
                <td className="px-3 py-2 text-zinc-500">
                  {formatDate(h.timestamps.completed || h.timestamps.created)}
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

function Th({
  label, sortKey, current, dir, onSort,
}: {
  label: string;
  sortKey: SortKey;
  current: SortKey;
  dir: SortDir;
  onSort: (k: SortKey) => void;
}) {
  const active = current === sortKey;
  return (
    <th
      className="px-3 py-2.5 text-left font-medium cursor-pointer hover:text-zinc-300 select-none"
      onClick={() => onSort(sortKey)}
    >
      {label}
      {active && <span className="ml-1">{dir === "asc" ? "\u25B2" : "\u25BC"}</span>}
    </th>
  );
}

interface PaginationProps {
  total: number;
  offset: number;
  limit: number;
  onPageChange: (offset: number) => void;
}

export function Pagination({ total, offset, limit, onPageChange }: PaginationProps) {
  const page = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(total / limit));
  const hasPrev = offset > 0;
  const hasNext = offset + limit < total;

  return (
    <div className="flex items-center justify-between text-xs text-zinc-400 mt-3">
      <span>{total} hypotheses</span>
      <div className="flex items-center gap-2">
        <button
          disabled={!hasPrev}
          onClick={() => onPageChange(Math.max(0, offset - limit))}
          className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          Prev
        </button>
        <span>
          {page} / {totalPages}
        </span>
        <button
          disabled={!hasNext}
          onClick={() => onPageChange(offset + limit)}
          className="px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          Next
        </button>
      </div>
    </div>
  );
}

function extractIc(h: Hypothesis): number {
  for (const g of h.gates || []) {
    if (g.name === "IC" || g.name === "discovery") {
      if (g.metric != null) return g.metric;
    }
  }
  return 0;
}

function formatIc(ic: number): string {
  if (ic === 0) return "-";
  return ic.toFixed(4);
}

function formatDate(iso: string): string {
  if (!iso) return "-";
  return new Date(iso).toLocaleDateString("en-CA");
}
