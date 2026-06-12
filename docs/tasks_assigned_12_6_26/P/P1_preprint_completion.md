# P1.1–P1.4 — Preprint Completion

**Phase**: P1 — Preprint Completion
**Priority**: 4 (start while data accumulates)
**Status**: NOT STARTED
**Effort**: ~40h writing
**Depends on**: Existing data (Phases A–F already complete)

---

## Objective

Produce a camera-ready preprint titled "Event-Aligned Singular Value Decomposition for Data-Driven Pattern Kernel Discovery in Cryptocurrency Perpetual Futures" with spectral microstructure, regime-gated, and cross-symbol validation sections from the Spannung research arc.

## Context

The core findings exist across Phase A–F documentation. The preprint needs to:
1. Package SVD convolver results (6 significant kernels, turtle-soup/trap IC [0.20, 0.36])
2. Add Spannung spectral microstructure (Phase D) — the novel academic contribution
3. Add regime gating (Phase E) — ent_book_shape IC lift
4. Add cross-symbol validation (Phase F) — structural replication proof
5. Frame everything for academic audience (not quant trading audience)

Planned preprint: SSRN first, then arXiv q-fin.TR after endorsement.

## Prerequisites

- Spannung Phases A–F research complete (they are)
- Reproducible results with exact data fingerprints (ideally after Q1.4 provenance tracking)

## Scope

**In scope**:
- LaTeX document with standard academic structure
- SVD convolver core results (existing work)
- Spectral microstructure section (Phase D findings)
- Regime conditioning section (Phase E findings)
- Cross-symbol validation section (Phase F findings)
- Mathematical formulations: OU process, spectral PSD, Kalman filter design implications
- Figures: PSD plots, IC by frequency band, regime IC lift, cross-symbol walk-forward

**Out of scope**:
- Trading strategy details (withheld — proprietary edge)
- Exact regime thresholds and entry logic
- Live trading results (don't exist yet)
- Full Kalman filter implementation (proposed as future work)

## Steps

### P1.1 — SVD Convolver Core (~10h)
1. Draft introduction: why microstructure pattern discovery matters
2. Write methodology: event definition, SVD pipeline, FDR correction
3. Present results: 6 significant kernels from 158 candidates
4. Key table: kernel types, IC values, BH-adjusted p-values
5. Discussion: turtle-soup/trap signals stronger than breakout signals

### P1.2 — Spectral Microstructure Section (~10h)
6. Motivation: why frequency-domain analysis of order book imbalance
7. Method: PSD estimation, cross-spectral coherence, OU fitting
8. Results:
   - Brown noise spectrum (slope -1.86, Hurst H=0.43)
   - IC=0.45 entirely in ultra-low band (0.005–0.1 Hz)
   - OU half-life 5–7s, dominant coherence at 68s
9. Key figure: IC by frequency band (ultra-low vs mid vs high)
10. Implication: naive time-domain aggregation destroys signal

### P1.3 — Regime Gating Section (~8h)
11. Method: systematic screen of 17 features as regime conditions
12. Results: ent_book_shape lifts IC 0.45 → 0.55 (single) → 0.63–0.67 (combo)
13. Economic interpretation: structured book = informed positioning
14. IC-persistence Pareto frontier figure
15. Replication across IS/OOS

### P1.4 — Cross-Symbol Validation (~8h)
16. Method: 5-fold walk-forward on BTC, ETH, SOL
17. Results: all `_last` imbalance variants achieve KEEP verdict, ≥80% sign consistency
18. Liquidity ordering: SOL > ETH > BTC (less liquid = stronger mean-reversion)
19. `_last` vs `_mean`/`_std` split: instantaneous features replicate, aggregated degrade
20. Conclusion: structural microstructure phenomenon, not asset-specific

### Final (~4h)
21. Abstract, conclusion, references
22. Proofread, formatting, figure quality
23. Generate reproducibility appendix (git SHA, data dates, script versions)

## Acceptance Criteria

- [ ] LaTeX compiles without errors, PDF is well-formatted
- [ ] All key findings from Phases A–F are present with correct numbers
- [ ] IC values match source documents: 0.45 (raw), 0.55–0.67 (regime-gated), spectral band localization
- [ ] OU parameters cited correctly: H=0.43, τ½=5–7s, coherence at 0.015 Hz
- [ ] Cross-symbol validation shows KEEP verdict for all `_last` variants
- [ ] BH-FDR correction applied and reported for SVD kernel significance
- [ ] No proprietary trading details revealed (thresholds, entry logic, position sizing)
- [ ] References include all 10+ cited papers from `spannung_phd_brief.md`
- [ ] Reproducibility section: git SHA, data date range, script commands
- [ ] Paper is understandable by a quantitative finance PhD student

## Testing / Verification

```bash
# 1. LaTeX compiles
cd docs && pdflatex preprint.tex && bibtex preprint && pdflatex preprint.tex && pdflatex preprint.tex

# 2. Verify key numbers match source
python3 -c "
# Cross-check key claims against source data
claims = {
    'Raw IC': 0.45,
    'Regime-gated IC (single)': 0.55,
    'Regime-gated IC (combo)': 0.67,
    'Hurst exponent': 0.43,
    'OU half-life (s)': 6.0,  # midpoint of 5-7
    'Coherence frequency (Hz)': 0.015,
    'Spectral slope': -1.86,
    'Cross-symbol sign consistency': 0.80,
}
for claim, value in claims.items():
    print(f'{claim}: {value}')
print('Verify these match the LaTeX document manually')
"

# 3. No proprietary leaks
grep -i 'threshold\|entry_logic\|position_size\|stop_loss' docs/preprint.tex | wc -l
# Should be 0

# 4. Reference count
grep -c 'cite{' docs/preprint.tex
# Should be >= 10
```

## Key Files

- `docs/preprint.tex` — main LaTeX document (new)
- `docs/preprint_bib.bib` — bibliography (new)
- `docs/figures/` — generated figures (new)
- `docs/ideas/spannung.md` — Phase A–F source data
- `docs/ideas/spannung_phd_brief.md` — academic framing reference

## References

- Spannung research: `docs/ideas/spannung.md`
- PhD brief: `docs/ideas/spannung_phd_brief.md`
- Academic references: Cont/Stoikov (2010), Kyle (1985), Bouchaud (2004), Easley (2012), Avellaneda/Stoikov (2008)
