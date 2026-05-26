"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getHypothesis } from "@/lib/api";
import type { Hypothesis } from "@/lib/api";
import { GateWaterfall } from "@/components/gate-waterfall";
import { MathPanel } from "@/components/math-panel";

const STATUS_STYLE: Record<string, string> = {
  replicated: "bg-emerald-900/40 text-emerald-400 border-emerald-700",
  discovery_passed: "bg-blue-900/40 text-blue-400 border-blue-700",
  no_effect: "bg-red-900/40 text-red-400 border-red-700",
  no_replication: "bg-red-900/40 text-red-400 border-red-700",
  redundant: "bg-amber-900/40 text-amber-400 border-amber-700",
  fdr_rejected: "bg-orange-900/40 text-orange-400 border-orange-700",
  command_error: "bg-red-900/40 text-red-400 border-red-700",
};

const AGENT_STYLE: Record<string, string> = {
  microstructure: "bg-violet-900/40 text-violet-400 border-violet-700",
  medium_freq: "bg-blue-900/40 text-blue-400 border-blue-700",
  macro: "bg-cyan-900/40 text-cyan-400 border-cyan-700",
};

function Badge({ label, style }: { label: string; style?: string }) {
  return (
    <span
      className={`inline-block px-2 py-0.5 text-xs font-medium rounded border ${
        style || "bg-zinc-800 text-zinc-300 border-zinc-700"
      }`}
    >
      {label}
    </span>
  );
}

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

  const handleExportPdf = useCallback(() => {
    if (!hypothesis) return;
    const originalTitle = document.title;
    document.title = `NAT-${hypothesis.id}`;
    window.print();
    document.title = originalTitle;
  }, [hypothesis]);

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
          <div className="h-48 bg-zinc-800 rounded" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Print header — hidden on screen, shown in print */}
      <div className="print-header hidden">
        <h1>NAT Research Report</h1>
        <p>Hypothesis {hypothesis.id} &mdash; Generated {new Date().toLocaleDateString()}</p>
      </div>

      {/* Back link */}
      <Link
        href="/explorer"
        className="text-xs text-zinc-500 hover:text-zinc-300 no-print"
      >
        &larr; Back to Explorer
      </Link>

      {/* Header */}
      <div>
        <div className="flex items-center gap-3 flex-wrap">
          <h2 className="text-xl font-bold font-mono">{hypothesis.id}</h2>
          <Badge
            label={hypothesis.status}
            style={STATUS_STYLE[hypothesis.status]}
          />
          <Badge
            label={hypothesis.agent}
            style={AGENT_STYLE[hypothesis.agent]}
          />
          <Badge label={hypothesis.generator} />
          <button
            onClick={handleExportPdf}
            className="no-print ml-auto px-3 py-1 text-xs font-medium rounded border border-zinc-700 text-zinc-400 hover:text-zinc-200 hover:border-zinc-500 transition-colors"
            title="Export as PDF"
          >
            Export PDF
          </button>
        </div>
        <p className="text-zinc-400 text-sm mt-2">{hypothesis.claim}</p>
      </div>

      {/* Metadata row */}
      <div className="flex gap-4 flex-wrap text-xs">
        {hypothesis.horizon_s != null && (
          <MetaChip label="Horizon" value={formatHorizon(hypothesis.horizon_s)} />
        )}
        {hypothesis.regime_gate && (
          <MetaChip label="Regime gate" value={hypothesis.regime_gate} />
        )}
        {hypothesis.features.length > 0 && (
          <MetaChip label="Features" value={String(hypothesis.features.length)} />
        )}
        {hypothesis.parent_id && (
          <MetaChip label="Parent" value={hypothesis.parent_id} link={`/explorer/${hypothesis.parent_id}`} />
        )}
      </div>

      {/* Two-column layout: waterfall + gates text */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Gate waterfall chart */}
        <GateWaterfall gates={hypothesis.gates} />

        {/* Gate results text */}
        {hypothesis.gates.length > 0 && (
          <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
            <h3 className="text-sm font-semibold mb-3">Gate Results</h3>
            <div className="space-y-2">
              {hypothesis.gates.map((gate, i) => (
                <div key={i} className="flex items-start gap-3 text-xs">
                  <span
                    className={`w-2 h-2 rounded-full shrink-0 mt-1 ${
                      gate.passed ? "bg-emerald-500" : "bg-red-500"
                    }`}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-zinc-300 font-medium">{gate.name}</span>
                      {gate.p_value != null && (
                        <span className="text-zinc-600 font-mono">
                          p={gate.p_value.toFixed(4)}
                        </span>
                      )}
                    </div>
                    <p className="text-zinc-500 mt-0.5">{gate.message}</p>
                    {gate.metric != null && gate.threshold != null && (
                      <p className="text-zinc-600 font-mono mt-0.5">
                        {gate.metric.toFixed(4)} vs {gate.threshold.toFixed(4)}
                      </p>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Math derivation */}
      {hypothesis.math && <MathPanel latex={hypothesis.math} />}

      {/* Features list */}
      {hypothesis.features.length > 0 && (
        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
          <h3 className="text-sm font-semibold mb-3">Features</h3>
          <div className="flex flex-wrap gap-1.5">
            {hypothesis.features.map((f) => (
              <span
                key={f}
                className="px-2 py-0.5 text-xs font-mono bg-zinc-800 text-zinc-300 rounded border border-zinc-700"
              >
                {f}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Thresholds */}
      {Object.keys(hypothesis.thresholds).length > 0 && (
        <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
          <h3 className="text-sm font-semibold mb-3">Thresholds</h3>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs">
            {Object.entries(hypothesis.thresholds).map(([k, v]) => (
              <div key={k} className="font-mono">
                <span className="text-zinc-500">{k}: </span>
                <span className="text-zinc-300">{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Timestamps */}
      <div className="text-xs text-zinc-500 space-y-1 border-t border-zinc-800 pt-4">
        <p>Created: {hypothesis.timestamps.created}</p>
        {hypothesis.timestamps.completed && (
          <p>Completed: {hypothesis.timestamps.completed}</p>
        )}
      </div>

      {/* Print footer — hidden on screen, shown in print */}
      <div className="print-footer hidden">
        NAT Research Platform &mdash; {hypothesis.id} &mdash; {hypothesis.agent}/{hypothesis.generator}
      </div>
    </div>
  );
}

function MetaChip({
  label,
  value,
  link,
}: {
  label: string;
  value: string;
  link?: string;
}) {
  const inner = (
    <span className="inline-flex items-center gap-1.5 bg-zinc-900 border border-zinc-800 rounded px-2.5 py-1">
      <span className="text-zinc-500">{label}</span>
      <span className="text-zinc-200 font-mono">{value}</span>
    </span>
  );
  if (link) {
    return (
      <Link href={link} className="hover:opacity-80">
        {inner}
      </Link>
    );
  }
  return inner;
}

function formatHorizon(s: number): string {
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.round(s / 60)}min`;
  if (s < 86400) return `${Math.round(s / 3600)}h`;
  return `${Math.round(s / 86400)}d`;
}
