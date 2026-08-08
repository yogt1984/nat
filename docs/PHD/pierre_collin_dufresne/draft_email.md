# Draft — outreach email (execution-overlay preprint focus)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 270 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
codebase offer ✓ · empirical-improvements paragraph ✓.

---

**Subject:** Prospective PhD inquiry — an out-of-sample failure mode of adverse-selection measurement

Dear Professor Collin-Dufresne,

I am writing to express my interest in pursuing a PhD under your supervision at EPFL. My background
is in embedded systems engineering and quantitative trading; over the past year I built a real-time
research platform for cryptocurrency perpetual futures that computes 236 order-book and flow
features at 100 ms cadence on BTC, ETH and SOL.

Its central finding may interest you. Order-book imbalance predicts 1–5 s mid-price direction with
rank IC ≈ 0.45, uniformly across all three symbols and volatility regimes. Yet the signal is
structurally unmonetizable: conditioning on directionally-consistent maker fills collapses the IC
to ≈ 0.03, while conditioning on *any* fill raises it to 0.52 — isolating adverse selection, rather
than signal decay, as the binding mechanism. I read this as an out-of-sample failure mode of
standard informed-trading proxies, close in spirit to "Do Prices Reveal the Presence of Informed
Trading?": the information is visibly in the book, but the act of trading on it selects exactly the
states where it has already been spent.

A preprint documenting this — the full predictive census of the book (207 features, three
symbols), the empirical orthogonality of its directional and volatility channels, and the
fill-conditioning decomposition above as its mechanism section — is available at [SSRN link] (PDF
attached). I am currently completing a one-month out-of-time replication: fresh conditional-IC
estimates, drift measurement (the signal's half-life is shortening as the venue matures), and a
queue-position fill simulation on raw trade data. I would be glad to share results, the full
codebase, or the dataset itself.

Would you be open to a short call to discuss whether this could develop into a doctoral project?

Best regards,
Onat Yılmaz
yogt1984@gmail.com

---

## Post-send tracking (mirror into README.md log)

- date_sent:
- response_date / response_type:
- next_action:
