# Synthetic default-risk model card

Version: 1.0.0; feature version: 1.0.0. Trained: 2026-10-10T06:57:41.691371+00:00 UTC.

## Intended use

Decision support for synthetic SME loan demonstrations. The model estimates a 12-month probability of default from point-in-time cash-flow, bill, bureau, and application features. Officers can review the score, evidence, and top adverse feature associations relative to the training mean. These reasons are not causal claims. A human officer retains the final decision; an LLM never decides credit.

## Model and score

Champion: logistic. The bank-friendly logistic candidate uses standardized features and L2 regularization; the challenger is a histogram gradient boosting classifier. The selected model is calibrated on validation. Score 600 corresponds to good-to-bad odds of 50:1; doubling those odds adds 20 points. Scores are clipped to 300–900. PD tiers: A <2%, B <5%, C <10%, D <20%, E ≥20%.

## Data, evaluation, and limits

All 3,000 businesses, statements, bureau records, and defaults are synthetic. Synthetic default labels are generated from latent business health and injected fraud; they do not validate performance on real borrowers. Business-disjoint train/validation/test partitions prevent entity leakage. Owner gender and age band are excluded from model features and reserved for fairness audit. Thin-file and thick-file test results appear in [VALIDATION.md](VALIDATION.md).

The DSCR proxy lacks observed existing-loan payments and can overstate capacity. Profile fields are unversioned; historical profile edits cannot be reconstructed. The six-month transaction window may miss annual seasonality. Calibration and tier thresholds require real-world validation and governance before any lending use. Missing bureau data is explicitly flagged; a score of 600 is an imputation, not a real bureau score.
