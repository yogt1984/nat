# Draft — outreach email (spiked matrix models / denoising)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 280 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> EPFL: apply to EDFI by **Jan 15** regardless of reply. Needs an SFI co-advisor —
> the guide pairs him with Collin-Dufresne; mention that only if he engages.

---

**Subject:** Prospective PhD inquiry — spiked models for order-book feature matrices

Dear Professor Krzakala,

I am writing to express my interest in pursuing a PhD at EPFL. My background is in embedded systems
engineering and quantitative trading; over the past year I built a real-time research platform for
cryptocurrency perpetual futures computing a 236-dimensional feature vector per symbol every 100 ms.

The object I keep producing is a tall, heavily correlated feature matrix, and the question I keep
failing to answer properly is how many of its directions are real. Empirically, 236 features
compress to roughly eight independent directional axes, and a second, disjoint family carries
volatility information — but "roughly eight" is a number I obtained by thresholding, not by any
principled separation of spike from bulk. I have implemented Marchenko–Pastur denoising to count
factors above the noise bulk, and I am aware that doing this correctly at finite aspect ratio, with
serially dependent rows and heavy tails, is a different problem from the textbook case.

That gap is why I am writing to you rather than to a finance group alone. Your work on spiked models
and optimal spectral methods is the theory my measurements are groping toward, and I would rather
learn to do it properly than keep tuning a threshold.

A preprint documenting the measurements — the feature census, the orthogonality of the two channels,
and an adverse-selection bound on what any of it can earn — is at [SSRN link] (PDF attached, with a
one-page summary of the wider work, including a paper that develops the effective-degrees-of-freedom
machinery).

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
