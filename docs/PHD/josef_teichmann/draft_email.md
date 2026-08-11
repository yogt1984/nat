# Draft — outreach email (learned bases / latent structure)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 285 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> ETH: the professor hires directly, there is no central deadline — **a reply is the whole game.**
> Do not claim the convolver kernels as a result; the OOS failure is stated below deliberately.

---

**Subject:** Prospective PhD inquiry — learned bases for order-book dynamics, and an honest negative

Dear Professor Teichmann,

I am writing to express my interest in pursuing a PhD under your supervision at ETH Zürich. My
background is in embedded systems engineering and quantitative trading; over the past year I built a
real-time research platform for cryptocurrency perpetual futures computing 236 order-book and flow
features at 100 ms cadence.

The methodological question I keep returning to is one your deep-hedging work approaches from the
other side: how much structure should be imposed, and how much discovered? My attempt is a pipeline
that defines market events *analytically* — breakouts, false breakouts, traps — so they remain
interpretable, extracts aligned OHLCV windows, decomposes them into four channels, and lets an SVD
discover the dominant shapes. The property I care about is that the SVD sees price geometry only and
never touches forward returns; basis discovery and label fitting are separated by construction, so
the usual objection that the shape was fitted to the outcome does not apply. Returns enter only at a
Benjamini–Hochberg-controlled gating step.

I should say plainly that the empirical result is negative: six kernels clear FDR in sample, none
survives out-of-sample, and two flip sign. I regard the separation property as the contribution and
the kernels as not yet a result — and the natural next question, extending discovered bases to
continuous-time dynamics rather than fixed windows, is exactly what I cannot do alone.

The preprint is at [SSRN link] (PDF attached, with a one-page summary of the wider work, including a
microstructure result on adverse selection that bounds this whole line of work).

Before anything larger: might I ask for an endorsement to post it to arXiv under `q-fin.TR`? I am
happy to share the full codebase or the dataset.

Best regards,
Yigit Onat
yionat@gmail.com

---

## Post-send tracking (mirror into README.md log)

- date_sent:
- response_date / response_type:
- next_action:
