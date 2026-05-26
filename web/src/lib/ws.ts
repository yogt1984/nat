/**
 * WebSocket hook for real-time research events.
 *
 * Connects to /ws/research (proxied to Axum backend).
 * Auto-reconnects with exponential backoff on disconnect.
 */

"use client";

import { useEffect, useRef, useState, useCallback } from "react";

// ---------------------------------------------------------------------------
// Event types from the research stream
// ---------------------------------------------------------------------------

export interface HypothesisStartedEvent {
  event: "hypothesis_started";
  id: string;
  agent: string;
  claim: string;
  generator: string;
  cycle_id?: string;
  hypothesis_id?: string;
}

export interface GatePassedEvent {
  event: "gate_passed";
  id: string;
  gate: string;
  msg: string;
  cycle_id?: string;
  hypothesis_id?: string;
}

export interface GateFailedEvent {
  event: "gate_failed";
  id: string;
  gate: string;
  reason: string;
  cycle_id?: string;
  hypothesis_id?: string;
}

export interface HypothesisRegisteredEvent {
  event: "hypothesis_registered";
  id: string;
  agent: string;
  ic: number | null;
  cycle_id?: string;
  hypothesis_id?: string;
}

export interface CycleCompletedEvent {
  event: "cycle_completed";
  agent: string;
  tested: number;
  passed: number;
  cycle: number;
  cycle_id?: string;
}

export type ResearchEvent =
  | HypothesisStartedEvent
  | GatePassedEvent
  | GateFailedEvent
  | HypothesisRegisteredEvent
  | CycleCompletedEvent;

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

interface UseResearchWsOptions {
  onEvent?: (event: ResearchEvent) => void;
  enabled?: boolean;
}

export function useResearchWs(options: UseResearchWsOptions = {}) {
  const { onEvent, enabled = true } = options;
  const [readyState, setReadyState] = useState<number>(WebSocket.CLOSED);
  const [lastEvent, setLastEvent] = useState<ResearchEvent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const retriesRef = useRef(0);
  const onEventRef = useRef(onEvent);
  onEventRef.current = onEvent;

  const connect = useCallback(() => {
    if (!enabled) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws/research`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setReadyState(WebSocket.OPEN);
      retriesRef.current = 0;
    };

    ws.onmessage = (msg) => {
      try {
        const data = JSON.parse(msg.data) as ResearchEvent;
        if (process.env.NODE_ENV === "development") {
          const cid = "cycle_id" in data ? data.cycle_id : "-";
          const hid = "hypothesis_id" in data ? data.hypothesis_id : ("id" in data ? data.id : "-");
          console.debug(`[ws] ${data.event} cycle=${cid} hyp=${hid}`);
        }
        setLastEvent(data);
        onEventRef.current?.(data);
      } catch {
        // ignore non-JSON messages
      }
    };

    ws.onclose = () => {
      setReadyState(WebSocket.CLOSED);
      wsRef.current = null;
      // Exponential backoff: 1s, 2s, 4s, 8s, max 30s
      const delay = Math.min(1000 * 2 ** retriesRef.current, 30_000);
      retriesRef.current++;
      setTimeout(connect, delay);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, [enabled]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [connect]);

  return { readyState, lastEvent };
}
