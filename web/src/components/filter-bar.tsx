"use client";

interface FilterBarProps {
  agent: string;
  generator: string;
  status: string;
  onAgentChange: (v: string) => void;
  onGeneratorChange: (v: string) => void;
  onStatusChange: (v: string) => void;
}

const AGENTS = ["", "microstructure", "medium_freq", "macro"];
const GENERATORS = [
  "", "systematic", "spectral", "regime", "cross_asset", "recycler",
  "ensemble", "momentum", "vol_breakout", "flow_cluster",
  "funding_meanrev", "oi_divergence", "whale_momentum", "it_discovery",
];
const STATUSES = [
  "", "replicated", "discovery_passed", "no_effect",
  "no_replication", "redundant", "fdr_rejected", "command_error",
];

export function FilterBar({
  agent, generator, status,
  onAgentChange, onGeneratorChange, onStatusChange,
}: FilterBarProps) {
  return (
    <div className="flex gap-3 flex-wrap">
      <Select label="Agent" value={agent} options={AGENTS} onChange={onAgentChange} />
      <Select label="Generator" value={generator} options={GENERATORS} onChange={onGeneratorChange} />
      <Select label="Status" value={status} options={STATUSES} onChange={onStatusChange} />
    </div>
  );
}

function Select({
  label, value, options, onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-zinc-400">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-blue-500"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt || "All"}
          </option>
        ))}
      </select>
    </label>
  );
}
