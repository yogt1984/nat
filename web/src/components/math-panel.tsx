"use client";

import { useEffect, useRef, useState } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

interface MathPanelProps {
  latex: string;
}

function RenderBlock({ tex }: { tex: string }) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!ref.current) return;
    try {
      katex.render(tex.trim(), ref.current, {
        displayMode: true,
        throwOnError: false,
        trust: true,
      });
    } catch {
      if (ref.current) {
        ref.current.textContent = tex;
      }
    }
  }, [tex]);

  return <div ref={ref} className="overflow-x-auto py-1" />;
}

export function MathPanel({ latex }: MathPanelProps) {
  const [expanded, setExpanded] = useState(false);

  if (!latex || latex.trim().length === 0) return null;

  // Split on double newlines — each block is a display equation
  const blocks = latex.split(/\n\n+/).filter((b) => b.trim().length > 0);
  const preview = blocks.slice(0, 2);
  const hasMore = blocks.length > 2;

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold">Mathematical Derivation</h3>
        {hasMore && (
          <button
            onClick={() => setExpanded((e) => !e)}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            {expanded ? "Collapse" : `Show all ${blocks.length} equations`}
          </button>
        )}
      </div>
      <div className="space-y-2 text-zinc-200">
        {(expanded ? blocks : preview).map((block, i) => (
          <RenderBlock key={i} tex={block} />
        ))}
        {!expanded && hasMore && (
          <p className="text-xs text-zinc-500 italic">
            +{blocks.length - 2} more equations
          </p>
        )}
      </div>
    </div>
  );
}
