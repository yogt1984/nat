"use client";

import { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import type { NetworkNode, NetworkEdge } from "@/lib/api";

// ---------------------------------------------------------------------------
// Category colors
// ---------------------------------------------------------------------------

const CATEGORY_COLORS: Record<string, string> = {
  spread: "#ef4444",
  depth: "#3b82f6",
  imbalance: "#f97316",
  flow: "#22c55e",
  volatility: "#a855f7",
  entropy: "#ec4899",
  trend: "#14b8a6",
  illiquidity: "#eab308",
  toxicity: "#dc2626",
  whale: "#6366f1",
  liquidation: "#f43f5e",
  concentration: "#84cc16",
  context: "#64748b",
  raw: "#94a3b8",
  regime: "#8b5cf6",
  cross_symbol: "#06b6d4",
  derived: "#d97706",
  other: "#6b7280",
};

// ---------------------------------------------------------------------------
// Types for D3 simulation
// ---------------------------------------------------------------------------

interface SimNode extends d3.SimulationNodeDatum {
  id: string;
  category: string;
  maxMi: number;
  interaction: number;
  costViable: boolean;
  hypothesisCount: number;
  selected: boolean;
  mi: Record<string, number>;
  cmi: Record<string, number>;
}

interface SimEdge extends d3.SimulationLinkDatum<SimNode> {
  weight: number;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface FeatureNetworkProps {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  miThreshold: number;
  activeCategories: Set<string>;
  onNodeClick?: (featureId: string) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function FeatureNetwork({
  nodes,
  edges,
  miThreshold,
  activeCategories,
  onNodeClick,
}: FeatureNetworkProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<SimNode, SimEdge> | null>(null);
  const onNodeClickRef = useRef(onNodeClick);
  onNodeClickRef.current = onNodeClick;

  // Build filtered data
  const getFilteredData = useCallback(() => {
    const filteredNodes: SimNode[] = nodes
      .filter((n) => {
        const maxMi = Math.max(0, ...Object.values(n.mi));
        if (maxMi < miThreshold && n.hypothesis_count === 0) return false;
        if (activeCategories.size > 0 && !activeCategories.has(n.category))
          return false;
        return true;
      })
      .map((n) => ({
        id: n.id,
        category: n.category,
        maxMi: Math.max(0, ...Object.values(n.mi)),
        interaction: n.interaction,
        costViable: n.cost_viable,
        hypothesisCount: n.hypothesis_count,
        selected: n.selected,
        mi: n.mi,
        cmi: n.cmi,
      }));

    const nodeIds = new Set(filteredNodes.map((n) => n.id));
    const filteredEdges: SimEdge[] = edges
      .filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target))
      .map((e) => ({ source: e.source, target: e.target, weight: e.weight }));

    return { filteredNodes, filteredEdges };
  }, [nodes, edges, miThreshold, activeCategories]);

  useEffect(() => {
    if (!svgRef.current) return;
    const svgEl = svgRef.current;
    const svg = d3.select(svgEl);

    const width = svgEl.clientWidth || 800;
    const height = svgEl.clientHeight || 600;

    svg.selectAll("*").remove();

    const { filteredNodes, filteredEdges } = getFilteredData();
    if (filteredNodes.length === 0) return;

    // Container for zoom
    const g = svg.append("g");

    // Zoom behavior
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 5])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    // Scales
    const maxMiVal = d3.max(filteredNodes, (d) => d.maxMi) || 0.01;
    const radiusScale = d3
      .scaleSqrt()
      .domain([0, maxMiVal])
      .range([4, 20]);

    // Edges
    const link = g
      .append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(filteredEdges)
      .join("line")
      .attr("stroke", "#475569")
      .attr("stroke-opacity", 0.4)
      .attr("stroke-width", (d) => Math.min(d.weight * 1.5, 6));

    // Nodes
    const node = g
      .append("g")
      .attr("class", "nodes")
      .selectAll<SVGCircleElement, SimNode>("circle")
      .data(filteredNodes)
      .join("circle")
      .attr("r", (d) => radiusScale(d.maxMi))
      .attr("fill", (d) => CATEGORY_COLORS[d.category] || "#6b7280")
      .attr("stroke", (d) => (d.selected ? "#fff" : d.costViable ? "#fbbf24" : "none"))
      .attr("stroke-width", (d) => (d.selected ? 2.5 : d.costViable ? 1.5 : 0))
      .attr("cursor", "pointer")
      .on("click", (_, d) => {
        onNodeClickRef.current?.(d.id);
      });

    // Labels
    const label = g
      .append("g")
      .attr("class", "labels")
      .selectAll("text")
      .data(filteredNodes)
      .join("text")
      .text((d) => d.id.replace(/_/g, " "))
      .attr("font-size", 9)
      .attr("fill", "#cbd5e1")
      .attr("text-anchor", "middle")
      .attr("dy", (d) => radiusScale(d.maxMi) + 12)
      .attr("pointer-events", "none");

    // Tooltip
    const tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "nat-network-tooltip")
      .style("position", "absolute")
      .style("background", "#1e293b")
      .style("border", "1px solid #334155")
      .style("border-radius", "6px")
      .style("padding", "8px 12px")
      .style("font-size", "12px")
      .style("color", "#e2e8f0")
      .style("pointer-events", "none")
      .style("opacity", 0)
      .style("z-index", 1000);

    node
      .on("mouseover", (event, d) => {
        const miStr = Object.entries(d.mi)
          .map(([k, v]) => `${k}: ${v.toFixed(4)}`)
          .join(", ");
        tooltip
          .html(
            `<b>${d.id}</b><br/>` +
              `Category: ${d.category}<br/>` +
              `MI: ${miStr}<br/>` +
              `Interaction: ${d.interaction.toFixed(4)}<br/>` +
              `Hypotheses: ${d.hypothesisCount}` +
              (d.costViable ? "<br/>Cost viable" : "") +
              (d.selected ? "<br/>IT-selected" : "")
          )
          .style("opacity", 1)
          .style("left", event.pageX + 12 + "px")
          .style("top", event.pageY - 10 + "px");
      })
      .on("mousemove", (event) => {
        tooltip
          .style("left", event.pageX + 12 + "px")
          .style("top", event.pageY - 10 + "px");
      })
      .on("mouseout", () => {
        tooltip.style("opacity", 0);
      });

    // Force simulation
    const simulation = d3
      .forceSimulation<SimNode>(filteredNodes)
      .force(
        "link",
        d3
          .forceLink<SimNode, SimEdge>(filteredEdges)
          .id((d) => d.id)
          .distance(100)
      )
      .force("charge", d3.forceManyBody().strength(-80))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide<SimNode>().radius((d) => radiusScale(d.maxMi) + 4))
      .on("tick", () => {
        link
          .attr("x1", (d) => ((d.source as SimNode).x ?? 0))
          .attr("y1", (d) => ((d.source as SimNode).y ?? 0))
          .attr("x2", (d) => ((d.target as SimNode).x ?? 0))
          .attr("y2", (d) => ((d.target as SimNode).y ?? 0));
        node.attr("cx", (d) => d.x ?? 0).attr("cy", (d) => d.y ?? 0);
        label.attr("x", (d) => d.x ?? 0).attr("y", (d) => d.y ?? 0);
      });

    simulationRef.current = simulation;

    // Drag behavior
    const drag = d3
      .drag<SVGCircleElement, SimNode>()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });
    node.call(drag);

    return () => {
      simulation.stop();
      tooltip.remove();
    };
  }, [getFilteredData]);

  return (
    <svg
      ref={svgRef}
      className="w-full bg-zinc-900/50 rounded-lg border border-zinc-800"
      style={{ height: "calc(100vh - 280px)", minHeight: 400 }}
      data-testid="network-svg"
    />
  );
}
