"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AgentCard } from "@/components/agent-card";
import { CycleRing } from "@/components/cycle-ring";
import { HypothesisFeed } from "@/components/hypothesis-feed";
import { StatsBar } from "@/components/stats-bar";
import { getStats, listHypotheses } from "@/lib/api";
import { useResearchWs } from "@/lib/ws";
import type { ResearchStats, Hypothesis } from "@/lib/api";
import type { ResearchEvent, CycleCompletedEvent } from "@/lib/ws";

const AGENTS = ["microstructure", "medium_freq", "macro"] as const;

interface AgentState {
  phase: string;
  cycleCount: number;
  lastCycleAt: string | null;
  tested: number;
}

export default function DashboardPage() {
  const [stats, setStats] = useState<ResearchStats | null>(null);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [liveEvents, setLiveEvents] = useState<ResearchEvent[]>([]);
  const [agentStates, setAgentStates] = useState<Record<string, AgentState>>({});
  const [flashingAgents, setFlashingAgents] = useState<Set<string>>(new Set());
  const prevPhasesRef = useRef<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);

  // Fetch initial data
  useEffect(() => {
    async function load() {
      try {
        const [s, h] = await Promise.all([
          getStats(),
          listHypotheses({ limit: 20 }),
        ]);
        setStats(s);
        setHypotheses(h.items);

        // Derive agent states from stats
        const states: Record<string, AgentState> = {};
        for (const agent of AGENTS) {
          states[agent] = {
            phase: "IDLE",
            cycleCount: 0,
            lastCycleAt: null,
            tested: s.by_agent?.[agent] || 0,
          };
        }
        setAgentStates(states);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load");
      }
    }
    load();
  }, []);

  // Handle live WebSocket events
  const onEvent = useCallback((event: ResearchEvent) => {
    setLiveEvents((prev) => [event, ...prev].slice(0, 50));

    if (event.event === "cycle_completed") {
      const e = event as CycleCompletedEvent;
      setAgentStates((prev) => ({
        ...prev,
        [e.agent]: {
          ...prev[e.agent],
          phase: "SLEEPING",
          cycleCount: e.cycle,
          lastCycleAt: new Date().toISOString(),
          tested: (prev[e.agent]?.tested || 0) + e.tested,
        },
      }));
    }

    if (event.event === "hypothesis_started") {
      setAgentStates((prev) => ({
        ...prev,
        [event.agent]: {
          ...prev[event.agent],
          phase: "EXECUTE",
        },
      }));
    }
  }, []);

  useResearchWs({ onEvent });

  // Detect phase transitions → flash the agent card for 2s
  useEffect(() => {
    const prev = prevPhasesRef.current;
    const toFlash: string[] = [];

    for (const agent of AGENTS) {
      const currentPhase = agentStates[agent]?.phase || "IDLE";
      if (prev[agent] && prev[agent] !== currentPhase) {
        toFlash.push(agent);
      }
      prev[agent] = currentPhase;
    }

    if (toFlash.length > 0) {
      setFlashingAgents((s) => {
        const next = new Set(s);
        toFlash.forEach((a) => next.add(a));
        return next;
      });
      const timer = setTimeout(() => {
        setFlashingAgents((s) => {
          const next = new Set(s);
          toFlash.forEach((a) => next.delete(a));
          return next;
        });
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [agentStates]);

  // Pick the most active agent for the ring
  const activeAgent = Object.entries(agentStates).find(
    ([, s]) => s.phase !== "IDLE" && s.phase !== "SLEEPING"
  );
  const ringPhase = activeAgent?.[1].phase || "SLEEP";

  return (
    <div className="space-y-6 max-w-6xl">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Dashboard</h2>
        {error && (
          <span className="text-xs text-red-400 bg-red-900/30 px-2 py-1 rounded">
            {error}
          </span>
        )}
      </div>

      {/* Aggregate stats */}
      <StatsBar stats={stats} />

      {/* Agent cards + cycle ring */}
      <div className="flex gap-6">
        <div className="flex-1 grid grid-cols-3 gap-4">
          {AGENTS.map((name) => {
            const s = agentStates[name];
            return (
              <AgentCard
                key={name}
                name={name}
                phase={s?.phase || "IDLE"}
                cycleCount={s?.cycleCount || 0}
                lastCycleAt={s?.lastCycleAt || null}
                tested={s?.tested || 0}
                flash={flashingAgents.has(name)}
              />
            );
          })}
        </div>
        <div className="shrink-0">
          <CycleRing currentPhase={ringPhase} />
        </div>
      </div>

      {/* Hypothesis feed */}
      <HypothesisFeed hypotheses={hypotheses} liveEvents={liveEvents} />
    </div>
  );
}
