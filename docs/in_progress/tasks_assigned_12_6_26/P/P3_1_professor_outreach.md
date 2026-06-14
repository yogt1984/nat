# P3 — Professor Outreach (ETH Zürich & EPFL)

**Phase**: P3 — Professor Outreach
**Priority**: Follows P2 (need SSRN link first)
**Status**: NOT STARTED
**Effort**: ~12h (research + personalization + follow-up)
**Depends on**: P2.1 (SSRN link available)

---

## Objective

Contact 13 target professors across ETH Zürich (7) and EPFL (6) with personalized emails referencing the preprint. Achieve active interest from at least 2 professors by November 2026.

## Context

Target professors are selected for alignment with spectral microstructure, SVD-based pattern discovery, information theory, and cryptocurrency market research. Full profiles in `docs/phd_application_guide.tex`. Email template in the same document.

Three ideal co-supervision pairings identified:
1. Teichmann (ETH) + Bölcskei (ETH): ML in finance + signal processing theory
2. Collin-Dufresne (EPFL) + Krzakala (EPFL): microstructure + random matrix theory
3. Malamud (EPFL) + Zdeborova (EPFL): ML asset pricing + statistical physics

## Prerequisites

- P2.1 (SSRN link available for inclusion in emails)
- Preprint PDF attached to emails

## Scope

**In scope**:
- Personalized emails to 13 professors (Tier 1 first, Tier 2 after 2 weeks)
- Research alignment paragraph per professor (from guide profiles)
- Follow-up emails after 2 weeks if no response
- Track responses in a structured log
- Schedule video calls with interested professors

**Out of scope**:
- Formal application submission (P4)
- Professors outside ETH/EPFL (contingency only)
- Revising the preprint based on feedback (separate iteration)

## Steps

### P3.1 — Customize Tier 1 Emails (~4h)
1. Personalize for each Tier 1 professor:

| Professor | Institution | Pitch Angle |
|-----------|------------|-------------|
| Josef Teichmann | ETH D-MATH | Latent structure discovery in LOB dynamics, extending SVD to continuous-time stochastic models |
| Pierre Collin-Dufresne | EPFL SFI | Data-driven detection of informed trading via SVD, connecting to adverse selection theory |
| Julien Hugonnier | EPFL SFI | Empirical microstructure of crypto perpetuals grounded in his no-arbitrage pricing theory |
| Semyon Malamud | EPFL SFI | SVD-discovered microstructure factors as inputs to learned pricing kernels (AIPT connection) |
| Helmut Bölcskei | ETH D-ITET | Information-theoretic limits of pattern recovery from noisy order book data |

2. Each email includes:
   - Reference to a specific paper/course of the professor
   - Bridge from their work to your findings
   - SSRN link + attached PDF
   - Offer to share full codebase

### P3.2 — Send Tier 1 Emails (~1h)
3. Send 5 emails with read receipts or tracking
4. Log: date sent, professor, institution, pitch angle

### P3.3 — Tier 2 Emails (~4h, stagger 2 weeks after Tier 1)
5. Personalize for 8 Tier 2 professors:

| Professor | Institution | Pitch Angle |
|-----------|------------|-------------|
| Beatrice Acciaio | ETH D-MATH | Wasserstein-distance regime detection on LOB distributions |
| Patrick Cheridito | ETH D-MATH | Approximation-theoretic bounds on SVD-extracted features |
| Peter Bühlmann | ETH D-MATH | Stability and replicability guarantees for microstructure signals |
| Walter Farkas | ETH/UZH | Empirical microstructure of crypto perpetuals, data-driven risk |
| Markus Leippold | UZH/SFI | Multi-modal alpha extraction (text + microstructure signals) |
| Lenka Zdeborova | EPFL SPOC | Phase transitions in microstructure signal recovery |
| Florent Krzakala | EPFL IdePHICS | Spiked matrix models for LOB feature matrices |
| Daniel Kuhn | EPFL RAO | Distributionally robust execution for microstructure strategies |

### P3.4 — Explore Co-Supervision (~2h)
6. If multiple professors respond from same institution, propose joint framing
7. Three pre-identified pairings (from guide)
8. Draft joint thesis framing paragraph for each pairing

### P3.5 — Schedule Meetings (~1h)
9. Arrange video calls or campus visits with interested professors
10. Prepare 15-minute research presentation (slides)

## Acceptance Criteria

- [ ] 5 Tier 1 emails sent with personalized research alignment paragraph
- [ ] 8 Tier 2 emails sent (staggered 2 weeks after Tier 1)
- [ ] Each email references a specific paper or course of the professor
- [ ] SSRN link and PDF included in every email
- [ ] Response tracking log maintained: date_sent, professor, response_date, response_type, next_action
- [ ] Follow-up sent after 2 weeks for non-responses (max 1 follow-up)
- [ ] **Milestone**: At least 2 professors with active interest by Nov 2026
- [ ] At least 1 video call or in-person meeting scheduled with a responding professor

### If milestone fails:
- [ ] Broaden to universities outside ETH/EPFL: Imperial, Oxford, TU Delft, KTH
- [ ] Revise pitch angle based on any feedback received
- [ ] Consider attending a finance conference to make in-person connections

## Testing / Verification

```
# 1. Response tracking log (spreadsheet or structured file)
# Columns: professor, institution, tier, date_sent, response_date,
#           response_type (positive/neutral/negative/none), next_action

# 2. Email quality checklist per professor:
#    - [ ] References specific paper by name
#    - [ ] Bridge paragraph connects their work to your findings
#    - [ ] SSRN link included
#    - [ ] PDF attached
#    - [ ] Tone is professional, concise (< 300 words body)

# 3. Milestone check (Nov 2026):
#    Count professors with response_type = "positive"
#    Target: >= 2
```

## Key Files

- `docs/phd_application_guide.tex` — professor profiles and email template
- `docs/ideas/spannung_phd_brief.md` — academic framing of findings
- `docs/preprint.pdf` — paper to attach

## References

- Full professor profiles: `docs/phd_application_guide.tex` Part II
- Email template: `docs/phd_application_guide.tex` Part III
- Co-supervision pairings: `docs/phd_application_guide.tex` Part IV
