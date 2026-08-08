# One-pagers

Dense single-page summaries of the eight P-branch preprints in `../`. **Personal reference cards, not
publication output** — they optimise for information per square centimetre, and each is hard-constrained
to exactly one A4 side.

| Card | Summarises | Body size | Page fill |
|---|---|---|---|
| `all_preprints_1p` | **all eight on one sheet**, plus the outreach map and known debts | `\footnotesize` | 90% |
| `convolver_1p` | `convolver_preprint` — event-aligned SVD | `\small` | 74% |
| `microstructure_alpha_1p` | `microstructure_alpha_preprint` — FDR-controlled discovery | `\small` | 90% |
| `liquidity_heatmap_1p` | `liquidity_heatmap_preprint` — liquidation geometry | `\small` | 88% |
| `prism_1p` | `prism_preprint` — Perception Pressure & Resonance | `\small` | 79% |
| `prism_signal_1p` | `prism_signal_preprint` — narrative as exogenous alpha | `\small` | 82% |
| `process_1p` | `process_preprint` — the process as an analytical unit | `\small` | 79% |
| `lob_predictive_structure_1p` | `lob_predictive_structure_preprint` — two orthogonal channels | `\small` | 84% |
| `nat2_exact_map_1p` | `nat2_exact_map_preprint` — exact or nothing | `\footnotesize` | 97% |

## What they carry that the abstracts do not

Each card foregrounds the **caveat that a skimming reader would miss**, marked `[!]`. Those are the
reason the cards exist:

- **convolver** — 6 kernels survive FDR, but *none* passes OOS robustness; two flip sign.
- **microstructure_alpha** — the combiner numbers rest on algorithms refuted by the Q4 kill gate;
  207 of 236 features were actually populated.
- **liquidity_heatmap / nat2** — the `0.0%` mapped fraction is confounded by snapshot ordering, not a
  refutation; and the 69% scoping probe is *not* the coverage number.
- **lob** — the OOS replication is in progress; the one systematically failing validation cell is
  temporal drift (−0.17 against a 0.10 tolerance).
- **process** — the first predictability surface returned 0 discoveries because BH cannot pass at the
  100-shuffle *p*-floor. A power problem, not a verdict.
- **prism** — sentiment labels are LLM-elicited, not human-annotated; clustering is lexical.
- **prism_signal** — the N_eff gain assumes calm-regime ρ̄; the binding one is under stress.

## Building

```bash
cd docs/research/onepagers
pdflatex -interaction=nonstopmode -halt-on-error <name>_1p.tex   # one pass is enough — no refs, no TOC
```

`onepager.sty` holds the shared layout. This TeX install has no `enumitem`/`titlesec`/`xcolor`/
`booktabs`/`microtype`, so the tight lists, section rules and fact tables are hand-rolled; only
`multicol` is external. Keep every card at **one page and zero overfull boxes** — both are checked when
these are regenerated.

Body size is per-card (`\small` or `\footnotesize`, chosen to fill the page without spilling to a second),
set on the line right after `\begin{document}`. If you add content and a card spills, drop it one size
rather than cutting the content.

## Caution

Every number on these cards is a **pointer, not a source**. Re-read the preprint — or, for `nat2`,
`~/nat2/data/ledger.jsonl` — before quoting anything in correspondence. The cards were built 2026-08-08
and do not update themselves.
