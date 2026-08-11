# Draft — outreach email (information-theoretic limits / matched filtering)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 285 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> ETH D-ITET: hires directly, no deadline. The pitch here is signal processing, not finance —
> lead with the template-bank framing, which is the honest description of the method.

---

**Subject:** Prospective PhD inquiry — recovery limits for template banks in noisy order-book data

Dear Professor Bölcskei,

I am writing to express my interest in pursuing a PhD under your supervision at ETH Zürich. My
background is in embedded systems engineering and quantitative trading; over the past year I built a
real-time research platform for cryptocurrency perpetual futures computing 236 features at 100 ms
cadence, with end-to-end feature latency under 80 ms at the 99th percentile.

One component is, in signal-processing terms, a template bank. Analytically defined market events
supply the alignment; aligned windows are decomposed into four channels; an SVD over the aligned
matrix yields the dominant shapes; and online detection is cosine similarity against the surviving
templates. It is the FINDCHIRP construction transplanted from gravitational-wave detection to price
geometry, with the basis learned rather than derived from a physical model — and the multiple-testing
control at the gating step is what I believe makes the transplant honest.

The empirical outcome is negative and I state it as such: six templates clear Benjamini–Hochberg in
sample, none survives out-of-sample, two flip sign. What I cannot answer, and what draws me to your
group, is whether that is a defect of my estimator or a statement about the problem — the
information-theoretic limits of recovering a template from data at this signal-to-noise ratio and
sample size. I have measurements and no theory of when recovery is possible at all.

The preprint is at [SSRN link] (PDF attached, with a one-page summary of the wider work).

Before anything larger: might I ask for an endorsement to post it to arXiv under `q-fin.TR`? I am
glad to share the full codebase or the dataset.

Best regards,
Yigit Onat
yionat@gmail.com

---

## Post-send tracking (mirror into README.md log)

- date_sent:
- response_date / response_type:
- next_action:
