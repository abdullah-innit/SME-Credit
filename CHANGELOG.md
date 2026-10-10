# Changelog

All notable changes to SME Credit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [SME-Credit2.0] — 2026-10-09

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

## [Unreleased] — 2026-10-09

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
