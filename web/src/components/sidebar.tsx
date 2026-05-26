"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useResearchWs } from "@/lib/ws";

const NAV_ITEMS = [
  { href: "/", label: "Dashboard" },
  { href: "/explorer", label: "Explorer" },
  { href: "/signals", label: "Signals" },
  { href: "/heatmap", label: "Heatmap" },
  { href: "/math", label: "Math Lab" },
  { href: "/graveyard", label: "Graveyard" },
  { href: "/network", label: "Network" },
];

export function Sidebar() {
  const pathname = usePathname();
  const { readyState } = useResearchWs();

  return (
    <aside className="w-56 shrink-0 border-r border-zinc-800 bg-zinc-900 flex flex-col">
      <div className="p-4 border-b border-zinc-800">
        <h1 className="text-lg font-bold tracking-tight">NAT Research</h1>
        <p className="text-xs text-zinc-500 mt-1">Alpha Signal Discovery</p>
      </div>

      <nav className="flex-1 py-3">
        {NAV_ITEMS.map(({ href, label }) => {
          const active =
            href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`block px-4 py-2 text-sm transition-colors ${
                active
                  ? "bg-zinc-800 text-white border-l-2 border-blue-500"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/50 border-l-2 border-transparent"
              }`}
            >
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-zinc-800">
        <ConnectionStatus readyState={readyState} />
      </div>
    </aside>
  );
}

function ConnectionStatus({ readyState }: { readyState: number }) {
  const isOpen = readyState === WebSocket.OPEN;
  const isConnecting = readyState === WebSocket.CONNECTING;

  const dotColor = isOpen
    ? "bg-emerald-500"
    : isConnecting
      ? "bg-amber-500"
      : "bg-red-500";

  const label = isOpen
    ? "Connected"
    : isConnecting
      ? "Connecting..."
      : "Disconnected";

  return (
    <div className="flex items-center gap-2 text-xs text-zinc-500">
      <span
        className={`w-2 h-2 rounded-full ${dotColor} ${isOpen ? "animate-pulse" : ""}`}
        data-testid="ws-status-dot"
      />
      <span data-testid="ws-status-label">{label}</span>
    </div>
  );
}
