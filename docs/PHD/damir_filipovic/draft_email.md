# Draft — outreach email (funding term structure / learned state variables)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 275 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> EPFL Swissquote Chair. Apply to EDFI by **Jan 15** regardless of reply.

---

**Subject:** Prospective PhD inquiry — hourly funding as an observable term structure in perpetuals

Dear Professor Filipović,

I am writing to express my interest in pursuing a PhD at EPFL. My background is in embedded systems
engineering and quantitative trading; over the past year I built a real-time research platform for
cryptocurrency perpetual futures computing 236 order-book and flow features at 100 ms cadence.

Perpetual futures seem to me an unusually clean laboratory for the questions your chair works on.
Funding settles *hourly* and is observable, so the carry term is first-order rather than a rounding
error at the horizons I study; the venue publishes the oracle price from which it constructs its own
mark, so the premium between them is directly measurable rather than inferred; and on one venue the
entire per-position liquidation structure is public, which makes the state of the market observable
in a way it is not on a traditional exchange.

What I have built is measurement infrastructure and a set of empirical regularities — two orthogonal
predictive channels in the book, spectrally localised with a 5–7 second mean-reversion half-life,
and a sharp bound on their monetization from adverse selection. What I have not built is a model:
the state variables are learned and unstructured, and I would like to learn to do that part properly.

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
