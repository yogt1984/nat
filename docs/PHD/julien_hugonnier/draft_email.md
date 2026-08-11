# Draft — outreach email (perp pricing → empirical microstructure)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 285 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> **Lead preprint changed 2026-08-11** to `lob_predictive_structure`. It already cites
> Ackerer–Hugonnier–Jermann (2025) — the bridge is in the bibliography, not manufactured.
> EPFL: apply to EDFI by **Jan 15** regardless of whether this is answered.

---

**Subject:** Prospective PhD inquiry — empirical microstructure of perpetuals, and where the predictability stops

Dear Professor Hugonnier,

I am writing to express my interest in pursuing a PhD at EPFL. My background is in embedded systems
engineering and quantitative trading; over the past year I built a real-time research platform for
cryptocurrency perpetual futures computing 236 order-book and flow features at 100 ms cadence on
BTC, ETH and SOL.

Your *Perpetual Futures Pricing* provides the no-arbitrage structure that my work lacks, and I cite
it in the preprint below. What I can contribute from the other direction is measurement. On three
symbols and 2.17 million ticks each, the book carries two empirically orthogonal predictive channels
— a directional family led by order-book imbalance (rank IC up to 0.47 at 1–5 s) and a volatility
family of arrival-intensity and toxicity measures (IC 0.35) — with each carrying essentially none of
the other's information. The directional signal is spectrally localised to 0.005–0.1 Hz with an
Ornstein–Uhlenbeck half-life of 5–7 seconds, and its half-life is visibly shortening as the venue
matures.

The result I would most like to discuss is a bound rather than a signal: conditioning on
directionally-consistent maker fills collapses that IC from 0.45 to about 0.03, while conditioning
on the presence of any fill raises it to 0.52. Adverse selection, not signal decay, is what limits
monetization — which seems to me a question that wants pricing theory and not more features.

The preprint is at [SSRN link] (PDF attached, plus a one-page summary of the wider work); its
out-of-time replication is still running.

Before anything larger: might I ask for an endorsement to post it to arXiv under `q-fin.TR`? I am
also glad to share the full codebase or the dataset.

Best regards,
Yigit Onat
yionat@gmail.com

---

## Post-send tracking (mirror into README.md log)

- date_sent:
- response_date / response_type:
- next_action:
