"use client";

import type { Hypothesis } from "@/lib/api";
import type { ResearchEvent } from "@/lib/ws";

const STATUS_BADGE: Record<string, { bg: string; text: string; label: string }> = {
  replicated: { bg: "bg-emerald-900", text: "text-emerald-300", label: "PASS" },
  registered: { bg: "bg-emerald-900", text: "text-emerald-300", label: "PASS" },
  discovery_passed: { bg: "bg-blue-900", text: "text-blue-300", label: "DISC" },
  fdr_rejected: { bg: "bg-red-900", text: "text-red-300", label: "FDR" },
  no_effect: { bg: "bg-red-900", text: "text-red-300", label: "FAIL" },
  no_replication: { bg: "bg-red-900", text: "text-red-300", label: "FAIL" },
  redundant: { bg: "bg-amber-900", text: "text-amber-300", label: "DUP" },
  command_error: { bg: "bg-red-900", text: "text-red-300", label: "ERR" },
  testing: { bg: "bg-amber-900", text: "text-amber-300", label: "TEST" },
};

interface FeedItem {
  id: string;
  agent: string;
  claim: string;
  status: string;
  generator: string;
  timestamp: string | null;
  isNew?: boolean;
}

interface HypothesisFeedProps {
  hypotheses: Hypothesis[];
  liveEvents: ResearchEvent[];
}

export function HypothesisFeed({ hypotheses, liveEvents }: HypothesisFeedProps) {
  // Build feed from API data
  const items: FeedItem[] = hypotheses.map((h) => ({
    id: h.id,
    agent: h.agent,
    claim: h.claim,
    status: h.status,
    generator: h.generator,
    timestamp: h.timestamps.completed || h.timestamps.created,
  }));

  // Prepend live events
  const liveItems: FeedItem[] = liveEvents
    .filter((e) => e.event === "hypothesis_started" || e.event === "hypothesis_registered")
    .map((e) => {
      if (e.event === "hypothesis_started") {
        return {
          id: e.id,
          agent: e.agent,
          claim: e.claim,
          status: "testing",
          generator: e.generator,
          timestamp: null,
          isNew: true,
        };
      }
      // hypothesis_registered
      return {
        id: e.id,
        agent: e.agent,
        claim: "",
        status: "registered",
        generator: "",
        timestamp: null,
        isNew: true,
      };
    });

  const feed = [...liveItems, ...items].slice(0, 20);

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800">
      <div className="px-4 py-3 border-b border-zinc-800">
        <h3 className="text-sm font-semibold">Recent Hypotheses</h3>
      </div>
      <div className="divide-y divide-zinc-800 max-h-96 overflow-auto">
        {feed.length === 0 ? (
          <p className="px-4 py-6 text-sm text-zinc-500 text-center">
            No hypotheses yet
          </p>
        ) : (
          feed.map((item) => (
            <FeedRow key={item.id + item.status} item={item} />
          ))
        )}
      </div>
    </div>
  );
}

function FeedRow({ item }: { item: FeedItem }) {
  const badge = STATUS_BADGE[item.status] || STATUS_BADGE.no_effect!;

  return (
    <div
      className={`px-4 py-2.5 flex items-start gap-3 text-xs transition-colors ${
        item.isNew ? "bg-zinc-800/50 animate-fade-in" : ""
      }`}
    >
      <span
        className={`shrink-0 mt-0.5 px-1.5 py-0.5 rounded font-medium ${badge.bg} ${badge.text}`}
      >
        {badge.label}
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-zinc-300 truncate">{item.claim || item.id}</p>
        <p className="text-zinc-500 mt-0.5">
          {item.agent} / {item.generator}
          {item.timestamp && (
            <span className="ml-2">{new Date(item.timestamp).toLocaleTimeString()}</span>
          )}
        </p>
      </div>
    </div>
  );
}
