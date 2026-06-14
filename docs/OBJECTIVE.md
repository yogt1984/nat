# NAT — Project Objective

NAT is an autonomous platform that continuously turns Hyperliquid perpetual-futures
microstructure into live trading edge — ingesting features, measuring where information about
future price action lives, and graduating the resulting algorithms from test to production.

## The loop
1. **Ingest** — continuously and reliably stream microstructure features.
2. **Discover** — on a schedule, run processes that quantify the *mutual information* between features and future price action.
3. **Visualize** — render findings as interactive 3D mesh graphs (React + React Three Fiber / Three.js with custom GLSL shaders).
4. **Generate** — synthesize candidate algorithms from the signals those processes surface.
5. **Validate** — put each algorithm through a battery of tests (backtest → walk-forward → paper trading).
6. **Deploy** — promote graduated algorithms to live trading.
