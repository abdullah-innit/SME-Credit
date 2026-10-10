# Feature engineering contract

Version 1.0.0 defines 30 numeric model features in `backend/app/ml/registry.py`. The registry supplies English and Urdu names, units, direction, scenario bounds, effort, and time to effect. Scenario bounds are planning constraints, not clipping limits for observed values. Gender, age band, and women-led status remain outside the model feature vector for fairness analysis.

`compute_features(application_id)` reads the application and its business evidence, computes the values, and creates a new immutable `FeatureSnapshot`. When passed a caller-owned SQLAlchemy session, the caller commits or rolls back. `derive_features(...)` is the pure function used by scenario planning; overrides for requested amount, tenor, balance, digital inflow share, and known existing monthly debt service recompute dependent ratios.

## Point in time

The cutoff is the application's submission date, or its creation date for drafts. Transactions dated on or after that day are excluded because there is no within-day decision timestamp. The window is the current calendar month and preceding five calendar months, with zero activity in missing months. Bills are included only when due before the cutoff; payments after the cutoff count as unpaid then. Bureau records created after the exact submission timestamp are excluded; records at that timestamp are available. Business profile fields are currently unversioned, so a later edit to incorporation, employees, NTN, or collateral cannot be reconstructed historically; production deployment needs profile history or frozen application inputs.

## Imputation and interpretation

- No transaction history: cash-flow, channel, concentration, activity, and balance features become zero; `has_transaction_history=0`.
- No observed closing balance: balance ratios are zero; `has_balance_data=0`.
- No due bills: bill ratios and lateness become zero; `has_bill_history=0`.
- No bureau record or a null score: score becomes neutral 600. `has_bureau_record` distinguishes an absent record; delinquency and active loans become zero when absent. A present record with null score retains `has_bureau_record=1`.
- No existing debt-service amount exists in the current database schema. `compute_features` therefore sets `has_debt_service_data=0` and the DSCR proxy denominator contains only the proposed principal instalment. This may overstate repayment capacity where other loans exist. Scenario callers can supply an existing monthly service amount, which sets the indicator to one. No interest or fees are inferred.
- Ratios with a zero denominator become zero. They should be read with the corresponding presence and cash-flow indicators.

The three top counterparties are identified only by existing hashes. The activity denominator spans the observed calendar days from the later of incorporation or the window start through the day before submission.

A read-only benchmark on the 3,000-business synthetic SQLite database computed and flushed 3,000 snapshots in 27.9 seconds on the development machine, then rolled back. The runtime depends on hardware and database size.
