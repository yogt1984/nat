# Draft — outreach email (optimal transport / robustness)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 275 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> ETH: hires directly, no deadline.

---

**Subject:** Prospective PhD inquiry — regime change in order-book distributions, and a robustness failure

Dear Professor Acciaio,

I am writing to express my interest in pursuing a PhD under your supervision at ETH Zürich. My
background is in embedded systems engineering and quantitative trading; over the past year I built a
real-time research platform for cryptocurrency perpetual futures computing 236 features at 100 ms
cadence on BTC, ETH and SOL.

Two things in it point toward your work. The first is that the natural object here is a
*distribution*, not a point: the order book at an instant is a measure over price levels, and what I
currently call a regime is a crude threshold on summary statistics of that measure. Comparing books
by a transport distance rather than by hand-chosen features seems obviously more principled, and I
have no training in it.

The second is a robustness failure that I think is structural rather than statistical. A directional
signal with rank IC ≈ 0.45 collapses to ≈ 0.03 once one conditions on the states where a passive
order would actually be filled correctly — the fill event is adversarially correlated with being
wrong. The empirical measure under which the strategy is evaluated is not the measure under which it
trades, which is precisely the kind of model risk your robust-finance work formalises.

The preprints are at [SSRN link] (lead PDF attached, with a one-page summary of the wider work);
the lead result's out-of-time replication is still running.

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
