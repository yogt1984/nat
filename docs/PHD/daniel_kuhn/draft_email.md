# Draft — outreach email (distributionally robust execution)

**Send after SSRN upload (P2) — replace the [SSRN link] placeholder first.**
Body ≈ 280 words. Checklist: specific paper ✓ · bridge ✓ · SSRN link ✓ · PDF attach ✓ ·
one-pager attach ✓ · codebase offer ✓ · one small ask ✓.

> EPFL: apply to EDFI by **Jan 15** regardless of reply. The bridge here is unusually clean —
> the adverse-selection barrier *is* a worst-case-distribution problem.

---

**Subject:** Prospective PhD inquiry — adverse selection as a worst-case fill distribution

Dear Professor Kuhn,

I am writing to express my interest in pursuing a PhD at EPFL. My background is in embedded systems
engineering and quantitative trading; over the past year I built a real-time research platform for
cryptocurrency perpetual futures computing 236 order-book and flow features at 100 ms cadence on
BTC, ETH and SOL.

Its central finding is a constraint, and I think it is a distributionally robust optimization problem
wearing a microstructure disguise. Order-book imbalance predicts 1–5 s direction with rank IC ≈ 0.45,
uniformly across symbols and volatility regimes. But conditioning on the states in which a passive
order would actually have been filled *directionally correctly* collapses that IC to ≈ 0.03, while
conditioning on the presence of any fill raises it to 0.52. The fill event is not exogenous: whether
you trade is correlated with whether you were right, adversarially so. Optimising execution against
the empirical fill distribution therefore optimises against the wrong measure, and the honest problem
is the worst case within an ambiguity set around it.

I have the measurements and a simulator; what I lack is the formalism to state the ambiguity set
correctly and the training to solve the resulting problem. That is why I am writing to your group
specifically rather than to a purely empirical one.

A preprint documenting the measurement is at [SSRN link] (PDF attached, with a one-page summary of
the wider work). Its out-of-time replication is still running and I will report it either way.

Before anything larger: might I ask for an endorsement to post it to arXiv under `q-fin.TR`? I am
glad to share the codebase or the dataset.

Best regards,
Yigit Onat
yionat@gmail.com

---

## Post-send tracking (mirror into README.md log)

- date_sent:
- response_date / response_type:
- next_action:
