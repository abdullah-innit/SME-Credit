# Synthetic model validation

Generated: 2026-10-10T06:57:41.691371+00:00 UTC. Data SHA-256: `2abcf7bb74c0f7a29dc214660cea5b2db0919e3334333d24828ea7e16068abc2`.
This is a demonstration on synthetic businesses and synthetic defaults, not a claim of real-world performance.

## Split and selection

Business-disjoint stratified split with seed 42: train 1800, validation 600, test 600. Standardization is fit on train only. Both estimators are fit on train only. Platt and isotonic calibrators are fit on validation; validation Brier chooses each calibrator. Logistic regression is the prespecified champion; the boosting model is a challenger. The test set is evaluated once after selection. Validation Brier is optimistic because it also selects the calibrator.

Champion: **logistic**. Champion validation Brier: 0.081.

| Model | Test segment | N | Defaults | ROC-AUC | PR-AUC | KS | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| logistic | all | 600 | 63 | 0.785 | 0.301 | 0.439 | 0.081 |
| logistic | thin_file | 212 | 22 | 0.775 | 0.239 | 0.472 | 0.086 |
| logistic | thick_file | 388 | 41 | 0.791 | 0.332 | 0.483 | 0.078 |
| hist_gradient_boosting | all | 600 | 63 | 0.741 | 0.258 | 0.352 | 0.087 |
| hist_gradient_boosting | thin_file | 212 | 22 | 0.683 | 0.213 | 0.255 | 0.093 |
| hist_gradient_boosting | thick_file | 388 | 41 | 0.770 | 0.303 | 0.415 | 0.084 |

## Champion calibration by risk decile

| Risk decile | N | Mean predicted PD | Observed default rate | Lift |
|---:|---:|---:|---:|---:|
| 1 | 60 | 0.376 | 0.350 | 3.333 |
| 2 | 60 | 0.263 | 0.233 | 2.222 |
| 3 | 60 | 0.109 | 0.150 | 1.429 |
| 4 | 60 | 0.079 | 0.083 | 0.794 |
| 5 | 60 | 0.068 | 0.067 | 0.635 |
| 6 | 60 | 0.068 | 0.033 | 0.317 |
| 7 | 60 | 0.067 | 0.067 | 0.635 |
| 8 | 60 | 0.025 | 0.017 | 0.159 |
| 9 | 60 | 0.018 | 0.033 | 0.317 |
| 10 | 60 | 0.002 | 0.017 | 0.159 |

## Traditional-rule baseline on the same test businesses

Requires at least 24 bureau-history months and collateral coverage at least 1.5×. Approval rate: 0.345; observed default rate among approved: 0.130 (207 approved). These are policy outcomes, not model discrimination metrics.
