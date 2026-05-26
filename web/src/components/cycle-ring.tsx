"use client";

const PHASES = ["MANIFEST", "GENERATE", "EXECUTE", "MONITOR", "SLEEP"];
const PHASE_COLORS = ["#f59e0b", "#f59e0b", "#3b82f6", "#8b5cf6", "#10b981"];

interface CycleRingProps {
  currentPhase: string;
  size?: number;
}

export function CycleRing({ currentPhase, size = 120 }: CycleRingProps) {
  const normalised = normalisePhase(currentPhase);
  const activeIdx = PHASES.indexOf(normalised);
  const r = (size - 16) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const segmentAngle = 360 / PHASES.length;

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {PHASES.map((phase, i) => {
          const startAngle = i * segmentAngle - 90;
          const endAngle = startAngle + segmentAngle - 4;
          const isActive = i <= activeIdx;
          const isCurrent = i === activeIdx;
          return (
            <path
              key={phase}
              d={describeArc(cx, cy, r, startAngle, endAngle)}
              fill="none"
              stroke={isActive ? PHASE_COLORS[i] : "#3f3f46"}
              strokeWidth={isCurrent ? 6 : 4}
              strokeLinecap="round"
              opacity={isActive ? 1 : 0.3}
            />
          );
        })}
        <text
          x={cx}
          y={cy}
          textAnchor="middle"
          dominantBaseline="middle"
          fill="#e4e4e7"
          fontSize={11}
          fontFamily="monospace"
        >
          {normalised}
        </text>
      </svg>
      <div className="flex gap-2 text-[10px] text-zinc-500">
        {PHASES.map((p, i) => (
          <span key={p} className={i <= activeIdx ? "text-zinc-300" : ""}>
            {p.slice(0, 3)}
          </span>
        ))}
      </div>
    </div>
  );
}

function normalisePhase(phase: string): string {
  const upper = phase.toUpperCase();
  if (upper === "SLEEPING" || upper === "IDLE") return "SLEEP";
  if (upper === "RUNNING") return "EXECUTE";
  if (PHASES.includes(upper)) return upper;
  return "SLEEP";
}

function polarToCartesian(cx: number, cy: number, r: number, deg: number) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx: number, cy: number, r: number, startAngle: number, endAngle: number) {
  const start = polarToCartesian(cx, cy, r, endAngle);
  const end = polarToCartesian(cx, cy, r, startAngle);
  const largeArc = endAngle - startAngle > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArc} 0 ${end.x} ${end.y}`;
}
