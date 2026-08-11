# Draft — outreach email (optimal stopping / approximation)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 275 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> ETH: hires directly, no deadline. The strongest bridge is optimal stopping — the execution
> problem below is literally one — so lead with that rather than with approximation bounds.

---

**Subject:** Prospective PhD inquiry — an optimal stopping problem where waiting destroys the signal

Dear Professor Cheridito,

I am writing to express my interest in pursuing a PhD under your supervision at ETH Zürich. My
background is in embedded systems engineering and quantitative trading; over the past year I built a
real-time research platform for cryptocurrency perpetual futures computing 236 features at 100 ms
cadence on three symbols.

The problem I would most like to work on is an optimal stopping problem with an adversarial twist.
A short-horizon directional signal is strong — rank IC ≈ 0.45 at 1–5 seconds, universal across
symbols and volatility regimes — and decays with an Ornstein–Uhlenbeck half-life of 5–7 seconds. One
may either cross the spread immediately and pay, or post passively and wait. But waiting is not free
in the usual sense: conditioning on the states in which a passive order is filled *and* the signal
was directionally correct collapses the IC to ≈ 0.03, while conditioning on any fill at all raises it
to 0.52. The stopping time and the payoff are coupled through the counterparty's decision to trade
against you.

I have measurements, a simulator, and no theory. Your work on deep optimal stopping is the closest
thing I have found to a way of attacking this, and I would rather learn it than keep approximating.

The preprints are at [SSRN link] (lead PDF attached, plus a one-page summary of the wider work,
including a pattern-recovery paper whose kernels are significant in sample and not out of sample).

Before anything larger: might I ask for an endorsement to post to arXiv under `q-fin.TR`? I am glad
to share the codebase or the dataset.

Best regards,
Yigit Onat
yionat@gmail.com

---

## Post-send tracking (mirror into README.md log)

- date_sent:
- response_date / response_type:
- next_action:
