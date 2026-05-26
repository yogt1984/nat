"use client";

import Link from "next/link";
import type { Hypothesis } from "@/lib/api";

interface NearMissesProps {
  hypotheses: Hypothesis[];
}

function extractIc(h: Hypothesis): number | null {
  const g = h.gates.find((g) => g.name === "discovery" || g.name === "ic_check");
  return g?.metric ?? null;
}

/** Near misses: failed exactly 1 gate (passed all others). */
export function filterNearMisses(hypotheses: Hypothesis[]): Hypothesis[] {
  return hypotheses.filter((h) => {
    if (h.gates.length === 0) return false;
    const failCount = h.gates.filter((g) => !g.passed).length;
    return failCount === 1;
  });
}

/** Recyclable: failed on replication gates but had good IC. */
export function filterRecyclable(hypotheses: Hypothesis[]): Hypothesis[] {
  const REPLICATION_GATES = new Set(["temporal", "symbol"]);
  return hypotheses.filter((h) => {
    const failedGates = h.gates.filter((g) => !g.passed);
    if (failedGates.length === 0) return false;
    const allReplication = failedGates.every((g) => REPLICATION_GATES.has(g.name));
    if (!allReplication) return false;
    const ic = extractIc(h);
    return ic != null && ic > 0.02;
  });
}

export function NearMissesTable({ hypotheses }: NearMissesProps) {
  const nearMisses = filterNearMisses(hypotheses)
    .sort((a, b) => (extractIc(b) ?? 0) - (extractIc(a) ?? 0));

  if (nearMisses.length === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 text-center">
        <p className="text-zinc-500 text-sm">No near misses found.</p>
      </div>
    );
  }

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-x-auto">
      <h3 className="text-sm font-semibold px-4 pt-4 pb-2">
        Near Misses <span className="text-zinc-500 font-normal">({nearMisses.length})</span>
      </h3>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-zinc-800 text-zinc-500 text-left">
            <th className="px-3 py-2 font-medium">ID</th>
            <th className="px-3 py-2 font-medium">Generator</th>
            <th className="px-3 py-2 font-medium text-right">IC</th>
            <th className="px-3 py-2 font-medium">Failed Gate</th>
            <th className="px-3 py-2 font-medium">Reason</th>
          </tr>
        </thead>
        <tbody>
          {nearMisses.map((h) => {
            const ic = extractIc(h);
            const failedGate = h.gates.find((g) => !g.passed);
            return (
              <tr key={h.id} className="border-b border-zinc-800/50 hover:bg-zinc-800/40">
                <td className="px-3 py-2">
                  <Link href={`/explorer/${h.id}`} className="text-blue-400 hover:text-blue-300 font-mono">
                    {h.id.slice(0, 12)}
                  </Link>
                </td>
                <td className="px-3 py-2 text-zinc-300">{h.generator}</td>
                <td className="px-3 py-2 text-right font-mono text-zinc-200">
                  {ic != null ? ic.toFixed(4) : "—"}
                </td>
                <td className="px-3 py-2">
                  <span className="text-red-400 font-medium">{failedGate?.name ?? "—"}</span>
                </td>
                <td className="px-3 py-2 text-zinc-500 truncate max-w-xs">
                  {failedGate?.message ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function RecyclableTable({ hypotheses }: NearMissesProps) {
  const recyclable = filterRecyclable(hypotheses)
    .sort((a, b) => (extractIc(b) ?? 0) - (extractIc(a) ?? 0));

  if (recyclable.length === 0) {
    return (
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-6 text-center">
        <p className="text-zinc-500 text-sm">No recyclable candidates.</p>
      </div>
    );
  }

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-x-auto">
      <h3 className="text-sm font-semibold px-4 pt-4 pb-2">
        Recyclable Candidates <span className="text-zinc-500 font-normal">({recyclable.length})</span>
      </h3>
      <p className="text-[10px] text-zinc-600 px-4 pb-2">
        Failed replication but IC &gt; 0.02 — candidates for parameter tuning
      </p>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-zinc-800 text-zinc-500 text-left">
            <th className="px-3 py-2 font-medium">ID</th>
            <th className="px-3 py-2 font-medium">Generator</th>
            <th className="px-3 py-2 font-medium text-right">IC</th>
            <th className="px-3 py-2 font-medium">Failed Gate</th>
            <th className="px-3 py-2 font-medium">Features</th>
          </tr>
        </thead>
        <tbody>
          {recyclable.map((h) => {
            const ic = extractIc(h);
            const failedGate = h.gates.find((g) => !g.passed);
            return (
              <tr key={h.id} className="border-b border-zinc-800/50 hover:bg-zinc-800/40">
                <td className="px-3 py-2">
                  <Link href={`/explorer/${h.id}`} className="text-blue-400 hover:text-blue-300 font-mono">
                    {h.id.slice(0, 12)}
                  </Link>
                </td>
                <td className="px-3 py-2 text-zinc-300">{h.generator}</td>
                <td className="px-3 py-2 text-right font-mono text-emerald-400">
                  {ic != null ? ic.toFixed(4) : "—"}
                </td>
                <td className="px-3 py-2">
                  <span className="text-amber-400 font-medium">{failedGate?.name ?? "—"}</span>
                </td>
                <td className="px-3 py-2 text-zinc-500 font-mono text-[10px]">
                  {h.features.slice(0, 3).join(", ")}
                  {h.features.length > 3 && ` +${h.features.length - 3}`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
