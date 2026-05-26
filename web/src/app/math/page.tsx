"use client";

import { useState, useMemo } from "react";
import { MathPanel } from "@/components/math-panel";

// ---------------------------------------------------------------------------
// Section data — static math content derived from FEATURES.md + agent gates
// ---------------------------------------------------------------------------

interface MathSection {
  id: string;
  title: string;
  category: "features" | "gates" | "sizing" | "methods";
  description: string;
  latex: string;
}

const SECTIONS: MathSection[] = [
  // ── Feature categories ──────────────────────────────────────────────────
  {
    id: "raw",
    title: "Raw Order Book (10)",
    category: "features",
    description: "Direct L2 order book measurements. Gatheral & Oomen (2010).",
    latex: String.raw`\text{midprice} = \frac{P_{\text{bid}}^{(1)} + P_{\text{ask}}^{(1)}}{2}

\text{spread}_{\text{bps}} = \frac{P_{\text{ask}}^{(1)} - P_{\text{bid}}^{(1)}}{\text{midprice}} \times 10\,000

\text{microprice} = \frac{V_{\text{ask}}^{(1)} \cdot P_{\text{bid}}^{(1)} + V_{\text{bid}}^{(1)} \cdot P_{\text{ask}}^{(1)}}{V_{\text{bid}}^{(1)} + V_{\text{ask}}^{(1)}}

\text{depth}_k = \sum_{i=1}^{k} V_i \quad (k \in \{5, 10\})`,
  },
  {
    id: "imbalance",
    title: "Order Book Imbalance (8)",
    category: "features",
    description: "Asymmetry metrics. Cont, Stoikov & Talreja (2010).",
    latex: String.raw`\text{imbalance}_k = \frac{\sum_{i=1}^{k} V_{\text{bid}}^{(i)} - \sum_{i=1}^{k} V_{\text{ask}}^{(i)}}{\sum_{i=1}^{k} V_{\text{bid}}^{(i)} + \sum_{i=1}^{k} V_{\text{ask}}^{(i)}} \in [-1, 1]

\text{pressure}_{\text{bid}} = \sum_{i} V_{\text{bid}}^{(i)} \cdot \frac{1}{1 + d_i / 10} \quad (d_i \text{ in bps from mid})

\text{pressure}_{\text{ask}} = \sum_{i} V_{\text{ask}}^{(i)} \cdot \frac{1}{1 + d_i / 10}`,
  },
  {
    id: "flow",
    title: "Trade Flow (12)",
    category: "features",
    description: "Trade arrival patterns and aggressor dynamics.",
    latex: String.raw`\text{aggressor\_ratio}_w = \frac{\sum_{t \in w} V_t \cdot \mathbf{1}[\text{buy}]}{\sum_{t \in w} V_t} \in [0,1]

\text{VWAP}_w = \frac{\sum_{t \in w} P_t \cdot V_t}{\sum_{t \in w} V_t}

\text{VWAP\_deviation} = \frac{\text{VWAP}_w - P_{\text{last}}}{P_{\text{last}}}

\text{intensity} = \text{EMA}\!\left(\frac{N_{\text{trades}}}{\Delta t},\; \alpha = 0.3\right)`,
  },
  {
    id: "volatility",
    title: "Volatility (9)",
    category: "features",
    description: "Realized and range-based vol. Parkinson (1980), Garman & Klass (1980).",
    latex: String.raw`\sigma_{\text{realized}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} r_i^2}

\sigma_{\text{Parkinson}} = \frac{\ln(H/L)}{\sqrt{4 \ln 2}}

\sigma_{\text{GK}} = \sqrt{0.5 \cdot \bigl[\ln(H/L)\bigr]^2 - (2\ln 2 - 1)\cdot\bigl[\ln(C/O)\bigr]^2}

\text{vol\_ratio} = \frac{\sigma_{1\text{m}}}{\sigma_{5\text{m}}} \quad (>1 = \text{accelerating})

z_{\text{vol}} = \text{clamp}\!\left(\frac{\sigma_{1\text{m}} - \bar{\sigma}_{1\text{h}}}{\hat{\sigma}_{1\text{h}}},\; -10,\; 10\right)`,
  },
  {
    id: "entropy",
    title: "Entropy (24)",
    category: "features",
    description: "Information content and predictability. Bandt & Pompe (2002), Shannon (1948).",
    latex: String.raw`H_{\text{perm}}(m,\tau) = -\sum_{\pi \in S_m} p(\pi) \ln p(\pi) \bigg/ \ln(m!)

H_{\text{Shannon}} = -\sum_{i=1}^{B} p_i \ln p_i

H_{\text{tick}} = -\sum_{d \in \{\uparrow,\downarrow,\rightarrow\}} p_d \ln p_d \in [0, \ln 3]

H_{\text{vol\text{-}tick}} = -\sum_{d} \hat{p}_d \ln \hat{p}_d, \quad \hat{p}_d = \frac{\sum_{t: d_t=d} V_t}{\sum_t V_t}

\Delta H = H_t - H_{t-\tau} \quad (\text{sharp drop} = \text{regime onset})`,
  },
  {
    id: "trend",
    title: "Trend (15)",
    category: "features",
    description: "Trend detection and persistence. Jegadeesh & Titman (1993), Mandelbrot (1971).",
    latex: String.raw`\text{momentum}_w = \hat{\beta}_1, \quad P_t = \beta_0 + \beta_1 t + \varepsilon_t \;\text{(OLS over } w \text{ ticks)}

R^2_w = 1 - \frac{\sum (P_t - \hat{P}_t)^2}{\sum (P_t - \bar{P})^2}

\text{monotonicity}_w = \frac{\max(N_\uparrow, N_\downarrow)}{N_\uparrow + N_\downarrow} \in [0.5,\, 1]

H_{\text{Hurst}} = \frac{\ln(R/S)}{\ln(N)} \quad \begin{cases} < 0.5 & \text{mean-reverting} \\ = 0.5 & \text{random walk} \\ > 0.5 & \text{trending} \end{cases}

\text{MA\_crossover} = \text{EMA}(10) - \text{EMA}(50)`,
  },
  {
    id: "illiquidity",
    title: "Illiquidity (12)",
    category: "features",
    description: "Market impact measures. Kyle (1985), Amihud (2002), Hasbrouck (2009).",
    latex: String.raw`\lambda_{\text{Kyle}} = \frac{\text{Cov}(\Delta P,\; \text{signed\_vol})}{\text{Var}(\text{signed\_vol})}

\text{Amihud} = \frac{\sum |r_i|}{\sum V_i} \times 10^6

\lambda_{\text{Hasbrouck}} = \hat{\beta}_1, \quad \Delta P_t = \beta_0 + \beta_1 \cdot \text{sign}(V_t)\sqrt{|V_t|} + \varepsilon_t

s_{\text{Roll}} = 2\sqrt{-\text{Cov}(\Delta P_t, \Delta P_{t-1})}

\text{composite} = \frac{1}{3}\bigl[\tilde{\lambda}_{\text{Kyle}} + \tilde{\lambda}_{\text{Amihud}} + \tilde{\lambda}_{\text{Hasbrouck}}\bigr]`,
  },
  {
    id: "toxicity",
    title: "Toxicity (10)",
    category: "features",
    description: "Informed trading detection. Easley et al. (2012), VPIN model.",
    latex: String.raw`\text{VPIN}_n = \frac{1}{n}\sum_{b=1}^{n} \frac{|V_b^{\text{buy}} - V_b^{\text{sell}}|}{V_b}

s_{\text{effective}} = 2 \cdot \overline{|P_{\text{trade}} - \text{VWAP}|}

s_{\text{realized}} = \overline{d_t \cdot (P_t - P_{t+5}) \cdot 2}, \quad d_t = \text{sign}(\text{trade})

\text{adverse\_selection} = s_{\text{effective}} - s_{\text{realized}}

\text{toxic\_index} = w_1 \cdot \text{VPIN} + w_2 \cdot \tilde{s}_{\text{adv}} + w_3 \cdot |\text{flow\_imbalance}|`,
  },
  {
    id: "derived",
    title: "Derived Composites (15)",
    category: "features",
    description: "Cross-feature interactions: entropy×trend, regime classification, informed flow.",
    latex: String.raw`\text{entropy\_trend} = H_{\text{tick}} \cdot (1 - \text{mono}_{60})

\text{trend\_strength} = \text{sgn}(\text{mom}) \cdot (\text{mono} - 0.5) \times 2 \cdot (1 - H_{\text{tick}})

\text{regime\_type} = \sigma \cdot (1 - 2 H_{\text{tick}}) \quad \begin{cases} > 0 & \text{breakout} \\ < 0 & \text{chaos} \end{cases}

\text{illiq\_trend} = \lambda_{\text{Kyle}} \cdot |\text{mom}_{60}| \times 1000

\text{regime\_indicator} = \text{clamp}\bigl(\text{MR} - \text{TR} - \text{FF},\; -1,\; 1\bigr)`,
  },
  {
    id: "concentration",
    title: "Concentration (15)",
    category: "features",
    description: "Position crowding: Herfindahl, Gini, Theil indices.",
    latex: String.raw`\text{HHI} = \sum_{i=1}^{N} s_i^2, \quad s_i = \frac{\text{pos}_i}{\sum_j \text{pos}_j}

G = \frac{\sum_{i=1}^{N}\sum_{j=1}^{N} |x_i - x_j|}{2N \sum_i x_i}

T = \sum_{i=1}^{N} s_i \ln\!\left(\frac{s_i}{1/N}\right)

\text{top}_k = \sum_{i=1}^{k} s_{(i)} \quad (k \in \{5, 10, 20, 50\})`,
  },

  // ── Gate protocol ───────────────────────────────────────────────────────
  {
    id: "gate-discovery",
    title: "Gate 1: Discovery",
    category: "gates",
    description: "IC and differential IC check. Minimum effect size to proceed.",
    latex: String.raw`\text{IC} = \text{corr}\bigl(f_t,\; r_{t \to t+h}\bigr) > \text{IC}_{\min}

\Delta\text{IC} = \text{IC}_{\text{gated}} - \text{IC}_{\text{ungated}} > \Delta\text{IC}_{\min}

\text{coverage} = \frac{|\{t : f_t \neq \text{NaN}\}|}{T} > c_{\min}

\text{Default thresholds: } \text{IC}_{\min} = 0.10,\; \Delta\text{IC}_{\min} = 0.05,\; c_{\min} = 0.10`,
  },
  {
    id: "gate-cost",
    title: "Gate 2: Cost",
    category: "gates",
    description: "Net Sharpe after transaction costs (microstructure agent only).",
    latex: String.raw`\text{cost}_{\text{round-trip}} = 2 \times \text{fee}_{\text{bps}} \times 10^{-4}

\text{Sharpe}_{\text{net}} = \frac{\bar{r}_{\text{net}}}{\sigma_{r_{\text{net}}}} \cdot \sqrt{252}

r_{\text{net},t} = r_{\text{gross},t} - \mathbf{1}[\text{trade}_t] \cdot \text{cost}_{\text{rt}}

\text{Pass: } \text{Sharpe}_{\text{net}} \geq 0.5 \quad (\text{taker fee} = 3.5\,\text{bps one-way})`,
  },
  {
    id: "gate-temporal",
    title: "Gate 3: Temporal Replication",
    category: "gates",
    description: "Out-of-sample hold-out validation across time.",
    latex: String.raw`\text{Split: } \mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{OOS}}

\text{IC}_{\text{OOS}} = \text{corr}\bigl(f_t, r_{t \to t+h}\bigr)\Big|_{t \in \mathcal{D}_{\text{OOS}}}

\text{sign\_consistency} = \frac{|\{d : \text{sgn}(\text{IC}_d) = \text{sgn}(\text{IC}_{\text{in-sample}})\}|}{D_{\text{OOS}}}

\text{Pass: } D_{\text{OOS}} \geq 2,\; \text{sign\_consistency} \geq 0.80`,
  },
  {
    id: "gate-symbol",
    title: "Gate 4: Symbol Replication",
    category: "gates",
    description: "Cross-asset generalization check.",
    latex: String.raw`\text{Symbols: } \{s_1, \dots, s_K\} \quad (K \geq 2)

\text{IC}_{s_k} = \text{corr}\bigl(f_t^{(s_k)}, r_{t \to t+h}^{(s_k)}\bigr)

\text{Pass: } |\{k : \text{IC}_{s_k} > 0\}| \geq 2`,
  },
  {
    id: "gate-dedup",
    title: "Gate 5: Correlation Dedup",
    category: "gates",
    description: "Reject hypotheses that duplicate existing registered signals.",
    latex: String.raw`\rho_{ij} = \text{corr}\bigl(\mathbf{f}_i,\, \mathbf{f}_j\bigr) \quad \text{for all } j \in \mathcal{S}_{\text{registered}}

\text{Pass: } \max_j |\rho_{ij}| < \rho_{\text{thresh}} \quad (\rho_{\text{thresh}} = 0.70)`,
  },
  {
    id: "fdr",
    title: "FDR Control",
    category: "gates",
    description: "Benjamini–Hochberg procedure applied at end of each cycle.",
    latex: String.raw`\text{Order p-values: } p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}

k^* = \max\!\left\{k : p_{(k)} \leq \frac{k}{m} \cdot q\right\}

\text{Reject } H_0^{(i)} \text{ for } i = 1, \dots, k^*

\text{Default: } q = 0.05 \;\Rightarrow\; \text{FDR} \leq 5\%`,
  },

  // ── Position sizing ─────────────────────────────────────────────────────
  {
    id: "position-sizing",
    title: "Cost-Aware Position Sizing",
    category: "sizing",
    description: "Trade filter: only enter when E[gain] > 1.5x cost. Alpha pipeline Step 3.",
    latex: String.raw`E[\text{gain}] = |\hat{z}_t| \cdot \hat{\sigma}_r \cdot \text{IC}

\text{Trade filter: } E[\text{gain}] > \kappa \cdot c_{\text{rt}}, \quad \kappa = 1.5

p_t = \begin{cases} \text{sgn}(\hat{z}_t) \cdot \min(|\hat{z}_t|, 1) & \text{if filter passes} \\ p_{t-1} & \text{otherwise (hold)} \end{cases}

\text{Gate G3: trade reduction} \geq 50\%,\; \text{mean hold} > 2\text{h}`,
  },
  {
    id: "promotion",
    title: "Paper → Live Promotion",
    category: "sizing",
    description: "Criteria for promoting a paper-traded signal to live execution.",
    latex: String.raw`\text{Sharpe}_{\text{paper}} \geq 1.5 \quad (\text{over } 7 \text{ days})

\frac{\text{IC}_{\text{realized}}}{\text{IC}_{\text{backtest}}} \geq 0.8

\text{max\_drawdown} \leq 2\%

\text{All three conditions must hold simultaneously.}`,
  },

  // ── Methods ─────────────────────────────────────────────────────────────
  {
    id: "it-engine",
    title: "IT Engine: Mutual Information",
    category: "methods",
    description: "KSG estimator for MI, CMI, interaction info, transfer entropy.",
    latex: String.raw`I(X; Y) = \psi(k) - \langle\psi(n_x + 1) + \psi(n_y + 1)\rangle + \psi(N)

I(X; Y \mid Z) = I(X; Y, Z) - I(X; Z)

\text{II}(X; Y; Z) = I(X; Y) - I(X; Y \mid Z)

T_{X \to Y} = I(X_t;\, Y_{t+1} \mid Y_t)

\text{Greedy selection: pick } f^* = \arg\max_f I(f;\, r) - \lambda \cdot \max_{f' \in S} I(f;\, f')`,
  },
  {
    id: "convolver",
    title: "Heatmap Convolver",
    category: "methods",
    description: "Probabilistic cascade model for liquidity events.",
    latex: String.raw`\hat{y}_i = \sigma\!\left(\sum_{j} w_j \cdot K(x_i, x_j)\right)

K(x, x') = \exp\!\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)

P(\text{cascade} \mid \mathbf{x}) = \hat{y} > \tau \quad (\tau = 0.03)

\text{Validation: AUC} \geq 0.65,\; \text{lift} \geq 2.0,\; \text{net IC} \geq 0.02`,
  },
  {
    id: "risk-parity",
    title: "Risk Parity Portfolio",
    category: "methods",
    description: "Meta-agent portfolio assembly with inverse-volatility weighting.",
    latex: String.raw`w_i = \frac{1/\sigma_i}{\sum_j 1/\sigma_j}

\sigma_{\text{port}}^2 = \mathbf{w}^\top \Sigma \mathbf{w}

\text{Sharpe}_{\text{port}} = \frac{\mathbf{w}^\top \boldsymbol{\mu}}{\sigma_{\text{port}}} \cdot \sqrt{252}

\text{Constraint: } |\rho_{ij}| < 0.70 \;\forall\; i \neq j \;\text{(dedup gate)}`,
  },
];

const CATEGORY_LABELS: Record<string, string> = {
  features: "Feature Definitions",
  gates: "5-Gate Protocol",
  sizing: "Position Sizing & Promotion",
  methods: "Analytical Methods",
};

const CATEGORY_ORDER = ["features", "gates", "sizing", "methods"];

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function CollapsibleSection({
  section,
  isOpen,
  onToggle,
}: {
  section: MathSection;
  isOpen: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="border border-zinc-800 rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 py-3 bg-zinc-900 hover:bg-zinc-800/80 transition-colors text-left"
      >
        <div>
          <span className="text-sm font-semibold text-zinc-200">
            {section.title}
          </span>
          <p className="text-xs text-zinc-500 mt-0.5">{section.description}</p>
        </div>
        <span className="text-zinc-500 text-xs ml-4 shrink-0">
          {isOpen ? "−" : "+"}
        </span>
      </button>
      {isOpen && (
        <div className="px-4 py-3 border-t border-zinc-800">
          <MathPanel latex={section.latex} />
        </div>
      )}
    </div>
  );
}

export default function MathLabPage() {
  const [query, setQuery] = useState("");
  const [openSections, setOpenSections] = useState<Set<string>>(new Set());
  const [activeCategory, setActiveCategory] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    return SECTIONS.filter((s) => {
      if (activeCategory && s.category !== activeCategory) return false;
      if (!q) return true;
      return (
        s.title.toLowerCase().includes(q) ||
        s.description.toLowerCase().includes(q) ||
        s.id.includes(q)
      );
    });
  }, [query, activeCategory]);

  const grouped = useMemo(() => {
    const groups: Record<string, MathSection[]> = {};
    for (const s of filtered) {
      (groups[s.category] ??= []).push(s);
    }
    return groups;
  }, [filtered]);

  const toggle = (id: string) =>
    setOpenSections((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const expandAll = () =>
    setOpenSections(new Set(filtered.map((s) => s.id)));
  const collapseAll = () => setOpenSections(new Set());

  return (
    <div className="space-y-5 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Math Lab</h2>
          <p className="text-xs text-zinc-500 mt-1">
            Mathematical foundations — {SECTIONS.length} formulations across{" "}
            {CATEGORY_ORDER.length} categories
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={expandAll}
            className="text-xs text-zinc-500 hover:text-zinc-300"
          >
            Expand all
          </button>
          <span className="text-zinc-700">|</span>
          <button
            onClick={collapseAll}
            className="text-xs text-zinc-500 hover:text-zinc-300"
          >
            Collapse all
          </button>
        </div>
      </div>

      {/* Search + category filter */}
      <div className="flex items-center gap-3">
        <input
          type="text"
          placeholder="Search formulas..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="flex-1 bg-zinc-900 border border-zinc-800 rounded px-3 py-1.5 text-sm text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-600"
        />
        <div className="flex gap-1.5">
          <button
            onClick={() => setActiveCategory(null)}
            className={`px-2.5 py-1 rounded text-xs transition-colors ${
              activeCategory === null
                ? "bg-blue-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            All
          </button>
          {CATEGORY_ORDER.map((cat) => (
            <button
              key={cat}
              onClick={() =>
                setActiveCategory(activeCategory === cat ? null : cat)
              }
              className={`px-2.5 py-1 rounded text-xs transition-colors ${
                activeCategory === cat
                  ? "bg-blue-600 text-white"
                  : "bg-zinc-800 text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {CATEGORY_LABELS[cat]}
            </button>
          ))}
        </div>
      </div>

      {/* Sections grouped by category */}
      {filtered.length === 0 ? (
        <p className="text-sm text-zinc-500 py-8 text-center">
          No formulas match &ldquo;{query}&rdquo;
        </p>
      ) : (
        CATEGORY_ORDER.filter((cat) => grouped[cat]).map((cat) => (
          <div key={cat}>
            <h3 className="text-sm font-semibold text-zinc-400 mb-2 uppercase tracking-wide">
              {CATEGORY_LABELS[cat]}
            </h3>
            <div className="space-y-2">
              {grouped[cat].map((section) => (
                <CollapsibleSection
                  key={section.id}
                  section={section}
                  isOpen={openSections.has(section.id)}
                  onToggle={() => toggle(section.id)}
                />
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  );
}
