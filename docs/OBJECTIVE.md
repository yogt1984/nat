# NAT — Project Objective

NAT is an autonomous platform that continuously turns Hyperliquid perpetual-futures
microstructure into live trading edge — ingesting features, measuring where information about
future price action lives, and graduating the resulting algorithms from test to production.

## The loop
1. **Ingest** — continuously and reliably stream microstructure features.
2. **Discover** — on a schedule, run processes that quantify the *mutual information* between features and future price action.
3. **Cluster** — continuously group features by shared structure and information, tracking how those clusters form and drift across different periods.
4. **Visualize** — render findings and feature clusters, period by period, as interactive 3D mesh graphs (React + React Three Fiber / Three.js with custom GLSL shaders).
5. **Generate** — synthesize candidate algorithms from the signals those processes surface.
6. **Validate** — put each algorithm through a battery of tests (backtest → walk-forward → paper trading).
7. **Deploy** — promote graduated algorithms to live trading.

---

*How we build toward this: see [`METHODOLOGY.md`](METHODOLOGY.md).*
