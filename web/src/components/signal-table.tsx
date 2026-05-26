"use client";

import Link from "next/link";
import type { Hypothesis } from "@/lib/api";

interface SignalTableProps {
  signals: Hypothesis[];
}

const STATUS_STYLE: Record<string, string> = {
  replicated: "bg-emerald-900/40 text-emerald-400 border-emerald-700",
  discovery_passed: "bg-blue-900/40 text-blue-400 border-blue-700",
};

function extractIc(h: Hypothesis): number | null {
  const g = h.gates.find(
    (g) => g.name === "discovery" || g.name === "ic_check"
  );
  return g?.metric ?? null;
}

function formatHorizon(s: number): string {
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.round(s / 60)}min`;
  if (s < 86400) return `${Math.round(s / 3600)}h`;
  return `${Math.round(s / 86400)}d`;
}

export function SignalTable({ signals }: SignalTableProps) {
  if (signals.length === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 text-center">
        <p className="text-zinc-500 text-sm">No registered signals yet.</p>
      </div>
    );
  }

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-zinc-800 text-zinc-500 text-left">
            <th className="px-3 py-2 font-medium">ID</th>
            <th className="px-3 py-2 font-medium">Agent</th>
            <th className="px-3 py-2 font-medium">Generator</th>
            <th className="px-3 py-2 font-medium text-right">IC</th>
            <th className="px-3 py-2 font-medium text-right">Horizon</th>
            <th className="px-3 py-2 font-medium text-right">Features</th>
            <th className="px-3 py-2 font-medium">Status</th>
            <th className="px-3 py-2 font-medium">Date</th>
          </tr>
        </thead>
        <tbody>
          {signals.map((s) => {
            const ic = extractIc(s);
            return (
              <tr
                key={s.id}
                className="border-b border-zinc-800/50 hover:bg-zinc-800/40"
              >
                <td className="px-3 py-2">
                  <Link
                    href={`/explorer/${s.id}`}
                    className="text-blue-400 hover:text-blue-300 font-mono"
                  >
                    {s.id.slice(0, 12)}
                  </Link>
                </td>
                <td className="px-3 py-2 text-zinc-300">{s.agent}</td>
                <td className="px-3 py-2 text-zinc-300">{s.generator}</td>
                <td className="px-3 py-2 text-right font-mono text-zinc-200">
                  {ic != null ? ic.toFixed(4) : "—"}
                </td>
                <td className="px-3 py-2 text-right font-mono text-zinc-400">
                  {s.horizon_s != null ? formatHorizon(s.horizon_s) : "—"}
                </td>
                <td className="px-3 py-2 text-right text-zinc-400">
                  {s.features.length}
                </td>
                <td className="px-3 py-2">
                  <span
                    className={`inline-block px-1.5 py-0.5 text-[10px] rounded border ${
                      STATUS_STYLE[s.status] || "bg-zinc-800 text-zinc-300 border-zinc-700"
                    }`}
                  >
                    {s.status}
                  </span>
                </td>
                <td className="px-3 py-2 text-zinc-500">
                  {s.timestamps.created?.slice(0, 10) || "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
