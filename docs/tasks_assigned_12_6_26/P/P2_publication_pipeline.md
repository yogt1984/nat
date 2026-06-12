# P2 — Publication Pipeline (SSRN + arXiv)

**Phase**: P2 — Publication
**Priority**: Follows P1 completion
**Status**: NOT STARTED
**Effort**: ~4h (submission mechanics)
**Depends on**: P1 (camera-ready preprint)

---

## Objective

Upload the preprint to SSRN (immediate, no barriers) and cross-post to arXiv (requires endorsement) to establish academic presence before professor outreach.

## Context

SSRN is the standard venue for finance preprints — professors check it regularly. arXiv covers the technical/quantitative audience. Having the paper on both platforms before emailing professors signals seriousness and provides a clean, citable link.

Strategy from `docs/phd_application_guide.tex`:
1. SSRN first (no barriers, approved in 1–3 business days)
2. Email professors with SSRN link
3. Obtain arXiv endorsement (possibly from a responding professor)
4. Cross-post to arXiv under q-fin.TR

## Prerequisites

- P1 complete (camera-ready PDF)

## Scope

**In scope**:
- SSRN account creation and paper upload
- SSRN e-journal submission (Capital Markets: Market Microstructure + Financial Engineering)
- arXiv account creation and endorsement request
- arXiv cross-posting once endorsed

**Out of scope**:
- Peer-reviewed journal submission (later — after PhD admission)
- Conference submissions
- Revisions based on feedback (handled separately)

## Steps

### P2.1 — SSRN Upload (~1h)
1. Create author profile at hq.ssrn.com (select "Independent Researcher")
2. Upload PDF with title, abstract, keywords, JEL codes
3. Keywords: order book microstructure, SVD, cryptocurrency, perpetual futures, spectral analysis
4. JEL codes: G14 (Information and Market Efficiency), G12 (Asset Pricing), C58 (Financial Econometrics)
5. Wait for SSRN staff review (1–3 business days, not peer review)
6. Record permanent URL: `ssrn.com/abstract=XXXXXXX`

### P2.2 — SSRN E-Journal Submission (~30min)
7. Submit to "Capital Markets: Market Microstructure" e-journal
8. Submit to "Financial Engineering" e-journal
9. These are SSRN classification channels that increase visibility

### P2.3 — arXiv Endorsement (~2h effort, variable wait)
10. Create arXiv account
11. Identify target category: `q-fin.TR` (Trading and Market Microstructure)
12. Alternative categories: `q-fin.ST`, `q-fin.CP`, `stat.ML`
13. Request endorsement:
    - Option A: Use arXiv's endorser suggestion system
    - Option B: Ask a responding professor from P3 to endorse
    - Option C: Find existing q-fin authors via paper citations
14. Endorsement is a spam filter, not peer review — should be obtainable

### P2.4 — arXiv Cross-Post (~30min)
15. Upload PDF to arXiv under q-fin.TR
16. Cross-list to q-fin.ST if appropriate
17. Update SSRN page with arXiv link
18. Update CV and email templates with both links

## Acceptance Criteria

- [ ] SSRN paper page live with permanent URL
- [ ] SSRN download count visible and tracking
- [ ] Paper submitted to at least 2 SSRN e-journals
- [ ] arXiv endorsement obtained (or clear path to obtaining it)
- [ ] arXiv paper live under q-fin.TR (or queued pending endorsement)
- [ ] Both URLs documented for use in P3 outreach emails
- [ ] Abstract, title, and keywords are optimized for discoverability

## Testing / Verification

```
# 1. SSRN page accessible
# Visit: ssrn.com/abstract=XXXXXXX
# Should show: title, abstract, download PDF button, author profile

# 2. SSRN e-journal listing
# Check "Capital Markets: Market Microstructure" e-journal for paper

# 3. arXiv listing (after endorsement)
# Visit: arxiv.org/abs/YYYY.NNNNN
# Should show: title, abstract, PDF link, q-fin.TR category

# 4. Google Scholar indexing (1-2 weeks after upload)
# Search: "Event-Aligned SVD Pattern Kernel Discovery"
# Should find SSRN and/or arXiv listing
```

## Key Files

- `docs/preprint.tex` — LaTeX source (from P1)
- `docs/preprint.pdf` — compiled PDF for upload
- `docs/phd_application_guide.tex` — publishing strategy reference

## References

- SSRN submission guide: `docs/phd_application_guide.tex` Part I
- Target categories and strategy documented in full in the guide
