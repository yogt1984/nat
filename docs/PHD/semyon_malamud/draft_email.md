# Draft — outreach email (learned factors → pricing kernels)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 280 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> EPFL, Tier 1. Apply to EDFI by **Jan 15** regardless of reply.

---

**Subject:** Prospective PhD inquiry — discovered microstructure factors, and how few of them are real

Dear Professor Malamud,

I am writing to express my interest in pursuing a PhD at EPFL. My background is in embedded systems
engineering and quantitative trading; over the past year I built a real-time research platform for
cryptocurrency perpetual futures computing 236 order-book and flow features at 100 ms cadence on
BTC, ETH and SOL.

*Artificial Intelligence Pricing Theory* frames the question I have been circling empirically: what
happens to a pricing kernel when the factor set is learned rather than specified. My contribution is
from the measurement end, at a timescale where the data is abundant enough to be unforgiving. Two
findings seem relevant. First, the 236 features collapse to about eight independent directional
directions plus a disjoint volatility family — the effective breadth of the whole book is far smaller
than its dimension, and I have been developing the machinery to report that number honestly rather
than optimistically, since the binding correlation is the stress-regime one and not the calm-regime
one.

Second, and more sobering: the strongest factor is economically inert. Conditioning on
directionally-correct passive fills collapses its information coefficient from 0.45 to about 0.03.
A learned factor can be robust, universal, and unpriceable at once — which strikes me as a
constraint any AI pricing theory eventually has to absorb.

The preprints are at [SSRN link] (lead PDF attached, with a one-page summary of all eight). The
out-of-time replication of the lead result is still running.

Before anything larger: might I ask for an endorsement to post to arXiv under `q-fin.TR`? I am glad
to share the full codebase or the dataset.

Best regards,
Yigit Onat
yionat@gmail.com

---

## Post-send tracking (mirror into README.md log)

- date_sent:
- response_date / response_type:
- next_action:
