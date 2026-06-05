# ML Implementation Timeline

## Wave Schedule

| Phase | Duration | Deliverables | Gate |
|-------|----------|-------------|------|
| Wave 0 | 1 day | bar_level support, WelfordNormalizer, test fixtures | Smoke tests pass |
| Wave 1 | 5 days | CPD + Momentum + training pipeline | OOS Sharpe > 0.5 |
| Gate 1 | 1 day | Evaluate, decide proceed/stop | 4-case matrix |
| Wave 2 | 7 days | RSM + MR + Meta-labeling | 3+ algos positive |
| Gate 2 | 1 day | Evaluate, decide proceed/stop | 4-case matrix |
| Wave 3 | 5 days | Regime LGBM + KNN | Outperforms global |
| Wave 4 | Deferred | HMM + Stacking + Online | Trigger-based |
| **Total committed** | **7 days** | Wave 0 + Wave 1 | |
| **Total if all pass** | **21 days** | Full ML portfolio | |

## Key Milestones

- **Day 1:** Infrastructure complete (bar_level, normalizer, fixtures)
- **Day 6:** First useful ML signal (CPD or Momentum producing trades)
- **Day 7:** Wave 1 gate decision — proceed or stop all ML work
- **Day 15:** Most likely outcome: 2-3 algorithms with positive OOS alpha
- **Day 21:** Full ML portfolio if all gates pass

## Decision Points

Each gate produces one of four outcomes:

| Case | Condition | Action |
|------|-----------|--------|
| CASE_A | 3+ positive, uncorrelated | Proceed to next wave |
| CASE_B | 2 positive or correlated | Proceed with caution, extend training |
| CASE_C | 1 positive | Deploy winner, pause wave expansion |
| CASE_D | 0 positive | Stop ML work, dataset does not support ML alpha |

## Current Status

All code for Waves 0-3 is implemented. Algorithms await training on accumulated data.
Deferred algorithms (Wave 4) have automated trigger monitoring via `check_deferred_triggers.py`.
