# P4 — PhD Application Submission

**Phase**: P4 — Application
**Priority**: Final PhD phase
**Status**: NOT STARTED
**Effort**: ~20h (application package preparation)
**Depends on**: P3 (at least 1 professor with active interest)

---

## Objective

Submit formal PhD applications to ETH Zürich and/or EPFL with a complete application package, targeting Sep 2027 entry.

## Context

ETH and EPFL have different application processes:
- **ETH**: Rolling admission for D-MATH PhD. Requires a professor willing to supervise before formal application. The professor essentially "accepts" you, then the department confirms.
- **EPFL EDFI**: Structured doctoral program in Finance (joint with University of Geneva and University of Lausanne). Fixed deadlines: Jan 15 and Mar 31 for Sep entry.

The ideal scenario: a professor commits informally during P3, then guides the formal application.

## Prerequisites

- P3 milestone met (at least 1 professor with active interest)
- Preprint published on SSRN (P2.1)
- Transcripts and references obtainable

## Scope

**In scope**:
- CV tailored for academic research position
- Research statement (2–4 pages) connecting past work to proposed PhD research
- Formal application to EPFL EDFI program
- Formal application to ETH D-MATH (if professor supports)
- Reference letters (2–3)
- Interview preparation

**Out of scope**:
- Applications outside ETH/EPFL (contingency only if P3 fails)
- Funding applications (PhD positions in Switzerland are typically funded)
- Visa/relocation logistics

## Steps

### P4.1 — Application Package (~12h)

1. **CV** (~2h):
   - Education, professional experience (embedded systems, quant trading)
   - Research: NAT platform, Spannung findings, preprint
   - Technical skills: Rust, Python, signal processing, ML
   - Publications: SSRN/arXiv links

2. **Research Statement** (~6h):
   - Section 1: Background and motivation (spectral microstructure gap in literature)
   - Section 2: Summary of completed work (SVD, spectral, regime, cross-symbol)
   - Section 3: Proposed PhD research directions:
     - Theoretical framework: spectral PSD → optimal Kalman filter design
     - Regime-dependent market making: adaptive quoting via OU parameters + entropy state
     - Information-theoretic characterization of order book states
     - Cross-asset information flow at spectral timescales
   - Section 4: Why this professor/group (personalized per application)
   - Section 5: Timeline (3–4 year PhD plan)

3. **Reference Letters** (~4h coordination):
   - Identify 2–3 referees (former professors, supervisors, collaborators)
   - Provide referees with CV, research statement, and preprint
   - Give 4+ weeks lead time before deadlines

### P4.2 — EPFL EDFI Round 1 (deadline: Jan 15, 2027)
4. Complete online application at epfl.ch/education/phd/edfi-finance/
5. Submit: CV, research statement, transcripts, GRE/GMAT (if required), references
6. Indicate preferred supervisor(s) from P3 responses

### P4.3 — ETH Application (aim: Jan–Feb 2027)
7. Contact supporting professor to initiate application
8. ETH process is professor-driven: professor nominates, department confirms
9. Submit formal paperwork as directed by professor/department

### P4.4 — EPFL Round 2 Backup (deadline: Mar 31, 2027)
10. If Round 1 unsuccessful, submit Round 2 application
11. Update research statement with any new results from Q2/Q3

### P4.5 — Interviews (~2h prep per interview)
12. Prepare 15-minute research presentation
13. Prepare for technical questions on:
    - SVD methodology and kernel selection
    - Spectral analysis and OU process fitting
    - Regime detection and entropy interpretation
    - Cross-symbol validation methodology
14. Prepare questions about the group's current research

## Acceptance Criteria

- [ ] CV completed, tailored for academic position (not industry)
- [ ] Research statement completed (2–4 pages), reviewed by at least 1 peer
- [ ] At least 2 reference letters secured with confirmed submission dates
- [ ] EPFL EDFI application submitted before Jan 15, 2027 deadline
- [ ] ETH application submitted (if professor supports) by Feb 2027
- [ ] **Milestone**: PhD offer received by Apr–May 2027

### Application quality checks:
- [ ] Research statement references the preprint with SSRN/arXiv link
- [ ] Research statement is tailored per application (different "why this group" section)
- [ ] CV includes quantitative metrics: 191 features, 236 computed, IC values
- [ ] No proprietary trading details in application materials
- [ ] All documents proofread for English quality

### If milestone fails:
- [ ] Apply to additional programs: Imperial, Oxford-Man, KTH, TU Delft
- [ ] Consider MSc-to-PhD bridge programs
- [ ] Revise research direction based on interview feedback
- [ ] Continue quant path — PhD is not the only viable outcome

## Testing / Verification

```
# 1. Application completeness checklist
# For each university:
#   - [ ] CV uploaded
#   - [ ] Research statement uploaded
#   - [ ] Transcripts uploaded
#   - [ ] References submitted (confirm with referees)
#   - [ ] Application fee paid
#   - [ ] Confirmation email received

# 2. Research statement review
#   - [ ] Reviewed by 1+ peer for clarity
#   - [ ] Checked for consistency with preprint claims
#   - [ ] No typos or formatting issues
#   - [ ] Personalized sections for each target university

# 3. Timeline verification
#   - Jan 15 2027: EPFL Round 1 submitted
#   - Jan-Feb 2027: ETH application submitted
#   - Mar 31 2027: EPFL Round 2 (if needed)
#   - Apr-May 2027: Decisions expected
```

## Key Files

- `docs/phd_application_guide.tex` — full guide with professor profiles
- `docs/phd_application_summary.tex` — one-page summary
- `docs/ideas/spannung_phd_brief.md` — research summary for inquiry
- `docs/preprint.pdf` — preprint to include with applications

## References

- EPFL EDFI program: epfl.ch/education/phd/edfi-finance/
- Application deadlines: Jan 15, Mar 31 (EPFL), rolling (ETH)
- Professor profiles: `docs/phd_application_guide.tex` Part II
