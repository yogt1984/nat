# Draft — outreach email (detectability thresholds / phase transitions)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 275 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> EPFL: apply to EDFI by **Jan 15** regardless of reply. Needs an SFI co-advisor
> (guide pairs her with Malamud) — raise that only if she engages.

---

**Subject:** Prospective PhD inquiry — is this signal undetectable, or am I under-powered?

Dear Professor Zdeborová,

I am writing to express my interest in pursuing a PhD at EPFL. My background is in embedded systems
engineering and quantitative trading; over the past year I built a real-time research platform for
cryptocurrency perpetual futures computing 236 features at 100 ms cadence on three symbols.

I am writing because I keep hitting a question I recognise as yours and cannot answer. Running a
grid of 284 candidate predictability cells with a permutation null and Benjamini–Hochberg control,
I get zero discoveries — and the reason is instructive rather than disappointing: at 100 shuffles the
p-value floor makes it arithmetically impossible for any cell to clear BH at that grid size. So the
zero is a statement about my experimental design, not about the market. The honest fix is deeper
nulls on a smaller candidate set, which is a sample-complexity question.

Underneath it sits the question I actually want to work on: whether these signals sit above or below
a detectability threshold at all. A pattern-recovery component of the same platform finds six
templates significant in sample and none out of sample — exactly the behaviour I would expect near a
transition, and exactly what I have no theory to distinguish from a bad estimator.

The preprints are at [SSRN link] (lead PDF attached, plus a one-page summary of the wider work).

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
