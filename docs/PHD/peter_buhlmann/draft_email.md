# Draft — outreach email (replicability / FDR focus)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 290 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> **Lead preprint changed 2026-08-11** to `lob_predictive_structure` — measurement-first, and it does
> not carry the retired combiner. Of all fourteen contacts this is the one where the *negative*
> results are the pitch, not a caveat.

---

**Subject:** Prospective PhD inquiry — an FDR-controlled discovery pipeline that refuted its own findings

Dear Professor Bühlmann,

I am writing to express my interest in pursuing a PhD under your supervision at ETH Zürich. My
background is in embedded systems engineering and quantitative trading; over the past year I built a
real-time research platform for cryptocurrency perpetual futures that computes 236 order-book and
flow features at 100 ms cadence on BTC, ETH and SOL.

I am writing to you specifically because the platform's most useful output has been negative. Every
candidate signal passes a five-gate replication protocol — discovery with an incremental-IC
requirement, a cost floor, temporal replication, cross-symbol replication, and correlation
deduplication — under Benjamini–Hochberg control with an acceptance threshold that adapts as the
registry grows. Applied honestly, that apparatus refuted all five of the strategies the platform had
previously identified as winners, and I have since retired a signal-combination result on the same
grounds. A companion paper on pattern discovery finds six kernels significant after FDR in sample,
of which none survives out-of-sample and two flip sign.

This is stability selection meeting a domain where the multiple-testing burden is effectively
unbounded and every practitioner has an incentive not to look. Your work on post-selection inference
is what I keep reaching for and what I am least equipped to do properly on my own.

The preprint documenting the underlying measurements — a 236-feature census of the order book, and
an adverse-selection result that bounds what any of it can earn — is at [SSRN link] (PDF attached,
with a one-page summary of the wider work). Its own out-of-time replication is still running; I will
report it whichever way it lands.

Before anything larger: I am seeking an endorsement to post this to arXiv under `q-fin.TR`. If you
judge the work sound enough to be worth one, I would be grateful — and glad to share the full
codebase or the dataset.

Best regards,
Yigit Onat
yionat@gmail.com

---

## Post-send tracking (mirror into README.md log)

- date_sent:
- response_date / response_type:
- next_action:
