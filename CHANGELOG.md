# Changelog

All notable changes to SME Credit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — 2026-10-10

### Added
- `backend/app/ml/synth.py` deterministic synthetic SME generator CLI (`python -m app.ml.synth --n 3000 --seed 42`).
- Generated 3,000 synthetic SMEs across 8 sectors, Pakistani cities, and legal forms with realistic distributions (~20% women-led, ~35% thin-file, ~60% NTN-registered).
- Latent business-health variable driving revenues, seasonality, volatility, utility-bill punctuality, and digital-channel adoption.
- 24 months of daily transactions per SME with strict running balance mathematical reconciliation (`inflow - outflow = balance_after`).
- Realistic fraud pattern injection (~3.5%): round-trip circular flows, window-dressing spikes prior to application, tampered statements with arithmetic discrepancies.
- Three canonical demo personas established:
  1. Ayesha Malik (Ayesha's Boutique, Karachi) — thin-file, women-led artisan apparel with strong digital wallet cash-flow.
  2. Karim Traders (Lahore) — thick bureau file, hardware retail with seasonal dips, borderline cash-flow suitable for Path to Approval.
  3. FastFix Logistics (Rawalpindi) — window-dressing fraud pattern, flagged by integrity scrutiny.
- Demo persona statement artifacts: CSV and valid text-based PDF statements in `samples/`.
- Holdout evaluation isolation (`loan_evaluations` table) preventing target leakage into feature pipelines.
- Sanity tests in `backend/tests/test_synth.py` validating population statistics and zero target label leakage.
- Detailed documentation of schema, data dictionary, and generation assumptions in `docs/MODEL-DATA-GUIDE.md`.

### Validation
- Backend: 21 pytest tests passed (`tests/test_synth.py` included); Ruff check passed with 0 errors.
- Frontend: ESLint passed with 0 errors.
- Population checks: ~35.0% thin-file, ~20.0% women-led, ~60.0% NTN, ~10.0% default rate, 100% balance reconciliation.

## [SME-Credit2.0] — 2026-10-09 (Fixes)

### Fixed
- Business.is_synthetic now exists in the SQLAlchemy model, create/read schemas and ERD.
- Legacy hyphenated age bands normalize to canonical underscored values in schemas
  and ORM writes; OwnerAgeBand and BAND_* aliases share the single AgeBand enum.
- Added an idempotent SQLite is_synthetic upgrade that preserves existing rows.
- Added regressions for age/synthetic round-trips and non-destructive upgrades.

### Validation
- All 19 tests in this uploaded source pass; backend Ruff and frontend ESLint pass.
- Backend health and frontend page both returned HTTP 200.
- The uploaded source contains 17 original tests, not the 40 tests in the local
  terminal transcript. Local-only tests and duplicate files were not supplied.

## [SME-Credit2.0] — 2026-10-09

### Added
- Sixteen SQLAlchemy credit-domain records, integer PKR constraints, canonical enums,
  relationships, indexed foreign keys/dates, UTC timestamps, and retained evidence links.
- Pydantic v2 create/read contracts for every record, strict integer amounts, enum
  validation, sanitized JSON-key guards, SHA-256 digest checks and override reasons.
- SQLite foreign-key enforcement, transactional session scope, startup create_all,
  typed seed callback and idempotent synthetic seed CLI.
- Append-only ORM audit records and privacy guards at schema and ORM boundaries.
- docs/ERD.md and seventeen acceptance/regression tests, including a 100-transaction
  round-trip, exact ERD topology/columns, amount constraints, privacy and seed rollback.

### Validation
- Backend: 17 pytest tests passed; Ruff lint passed and changed Python files formatted.
- Frontend: existing npm lint passed; no interface change requested.
- Both servers started; backend health and frontend page returned HTTP 200.

### Limits
- create_all is bootstrap only; existing databases need a migration strategy.
- Service-layer authorization, consent validity and cross-business ownership checks
  remain required. Raw identities in free text must be sanitized by ingestion modules.
- ORM audit immutability is not a substitute for database access controls.

## [0.1.0] — 2026-10-09

### Added
- Project scaffold: backend (FastAPI) and frontend (Next.js) skeletons
- Backend: settings from `.env`, health endpoint, CORS, structured logging, versioned `/api/v1` router
- Frontend: Next.js App Router landing page calling health endpoint
- `.env.example` with all required configuration variables
- `facts.yaml` with sourced external facts (default `verified: false`)
- Scripts: `setup.bat`, `run-backend.bat`, `run-frontend.bat`, `run-all.bat`, `test.bat`
- Documentation: `AGENT_CONTEXT.md`, `MODEL-DATA-GUIDE.md`, `VALIDATION.md`, `README.md`
- `.gitignore` for Python, Node, IDE, OS, DB, and ML artifacts
- Pytest tests for health endpoint
- SQLAlchemy session setup with SQLite
