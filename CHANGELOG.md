# Changelog

All notable changes to SME Credit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased] — 2026-10-10

### Added — Policy-grounded advisor with tools (2026-10-10)
- Added original illustrative policy knowledge base (backend/app/knowledge/documents/) with 21 markdown chunks covering eligibility (KB-ELIG-001 to 003), required documents (KB-DOCS-001 to 002), product terms (KB-TERM-001 to 003), scoring methodology (KB-SCOR-001 to 003), Path to Approval score improvement (KB-IMPR-001 to 003), FAQs (KB-FAQS-001 to 003), repayment tips (KB-REPAY-001 to 003), and a responsible lending charter (KB-RESP-001). All clearly labeled illustrative.
- Added native BM25 (Okapi) keyword retrieval index (backend/app/knowledge/retrieval.py) with multilingual tokenization for English and Urdu / Arabic script, plus an optional vector embeddings provider toggled via settings.EMBEDDINGS_ENABLED.
- Implemented two grounded personas in backend/app/services/advisor.py:
  - **Applicant Assistant**: English and Urdu (اردو/Roman Urdu), strictly isolated to the applicant's own facts bundle.
  - **Officer Copilot**: Underwriter copilot for policy questions, referral analysis ('why was this referred'), side-by-side application comparison, and tailored covenant/condition list drafting.
- Added structured tool calling (backend/app/services/advisor_tools.py): get_application_summary, run_what_if (integrating Prompt 8 counterfactual engine), explain_reason, and search_policy. Responses emit tool_calls: list[ToolCallRecord] so the UI can render 'AI used a tool' chips.
- Added hard safety guardrails: the assistant can NEVER approve, decline, or promise a loan; polite refusals for approval guarantees and out-of-scope queries (recipes, weather, crypto, trivia).
- Enforced strict tenant authorization: applicants cannot read other applicants' data, returning polite 403 authorization refusals.
- Added Server-Sent Events (SSE) streaming endpoint (POST /api/v1/advisor/chat/stream), session-scoped conversation memory with turn history and clear endpoints, and PII-safe logging masking CNICs, phone numbers, card numbers, IBANs, and API keys.
- Added deterministic offline mode ensuring full functionality without external API keys.
- Added 14 acceptance tests in backend/tests/test_advisor.py covering tool routing, chip payloads, refusals, citations, applicant tenant isolation, officer workflows, Urdu bilingual output, SSE streaming, session memory, and PII masking.

### Added — AI credit memo and plain-language explanations (2026-10-10)
- Added bank-facing credit memo generation service (`backend/app/services/memo.py`) covering 8 required sections: executive summary, business snapshot, cash-flow analysis, key risk drivers (feature waterfall), integrity findings, policy checks, recommendation & conditions, and questions for applicant interview.
- Grounded memo metrics directly in the facts bundle (score, PD, tier, features, reasons, policy checks, data-quality score) with Prompt 10 number checking enforced; cited fact IDs as structured footnotes.
- Added mandatory banner label `'AI-assisted draft — officer review required'` on all generated memos.
- Added applicant-facing explanations in plain English, Urdu (اردو), and Roman Urdu at Grade 6–8 reading level, with respectful tone, adverse-action reasons for decline/referral, and Path to Approval summaries.
- Added officer editing with versioning in `MemoRecord`, tracking author, timestamps, `'Human-edited — officer reviewed'` label, and unified diff between versions.
- Added on-demand regeneration supporting `concise` and `detailed` tone options.
- Added print-optimised HTML export with responsive styling and RTL Urdu support (`Noto Nastaliq Urdu` / fallback fonts), and PDF export via ReportLab.
- Added REST routes in `backend/app/api/v1/memo.py` (`/api/v1/applications/{id}/memo`, `/versions`, `/export/html`, `/export/pdf`, `/api/v1/applications/{id}/explanation/applicant`).
- Added comprehensive test suite (`backend/tests/test_memo.py`) verifying under-10-second latency, zero ungrounded numbers, plain-language Urdu/Roman Urdu content, officer edits with unified diffs, and HTML/PDF exports.

### Added — LLM service layer with safety guardrails (2026-10-10)
- Added single safe, testable gateway (`backend/app/llm/gateway.py`) with Anthropic as default provider (`claude-3-5-sonnet-20241022`), pluggable OpenAI-compatible provider, and MockProvider for deterministic testing.
- Added structured Pydantic schemas (`CreditMemoOutput`, `StatementExtractionOutput`, `AdvisorOutput`) with automatic repair retry on invalid JSON and typed exception hierarchy.
- Versioned prompts in `app/llm/prompts/` with SHA-256 registry; shared system prompt strictly enforcing non-negotiable rules: never decide credit outcomes, never invent numbers, cite fact IDs, refuse unrelated requests.
- Added number-grounding verification (`app/llm/grounding.py`) ensuring every numeric value in generated responses originates in the facts bundle or whitelisted arithmetic; failures trigger regeneration before falling back to deterministic safe fallback.
- Added prompt-injection sanitisation (`app/llm/sanitise.py`) stripping control phrases from free-text and CSV data, and masking API keys.
- Added SQLite-backed caching keyed by SHA-256 of prompt and facts bundle, exponential backoff, sliding-window rate limiting, and execution logging to `AiCallLog` (latency, tokens, estimated cost in USD).
- Added endpoints `GET /api/v1/admin/ai-usage`, `GET /api/v1/admin/ai-health`, and `GET /api/v1/health/llm`.
- Added 11 acceptance tests in `backend/tests/test_llm.py` covering success, repair retry, number-checker failure, timeout fallback, prompt-injection hygiene, offline mode, and API key masking.

### Added — Statement ingestion and transaction categorisation (2026-10-10)
- Added CSV, XLSX and text-PDF adapters, validated generic column mapping, optional Prompt 10 LLM hooks, privacy-safe normalisation, rule-first category confidence and duplicate detection.
- Added upload, status and synthetic sample endpoints with a 10 MB allow-list and stage reports.
- Added running-balance, coverage, gap and outlier checks; statement quality now enters scoring assurance and evidence changes trigger a refreshed score.
- Verified all six sample statements, corrupted-file reporting and edited-balance detection in parser and API tests.
### Added — Path to Approval counterfactuals (2026-10-10)
- Added bounded, actionability-checked plans with consistent derived features, real model and policy verification, bilingual steps, timelines and persisted results.
- Added fast what-if and lender-view API responses with before/after reasons and policy checks.
- Evaluated all 600 holdout applications: 485 of 522 referred or declined cases found a plan (92.9%); all 1,452 returned plans reapplied as APPROVE.
### Added — Scoring pipeline and policy decision API (2026-10-10)
- Added application submission, idempotent scoring, versioned and reloadable policy checks, recommended offers, expected loss and bank assurance responses.
- Persisted score, system recommendation and audit records; exposed stable failure codes and documented synthetic-data and integrity-hook limits in docs/POLICY.md.
- Added persona, policy edit and warm-route regression tests.
### Added — Explainability and adverse-action reasons
- Added reconciled local raw-log-odds waterfalls for the logistic champion and deterministic permutation fallback for the boosting challenger when SHAP is unavailable.
- Added bilingual reason templates for all 30 model features, global importance metadata, tiny-noise stability checks, and rule-based notices for declined or referred decisions.
- Application scoring and the new GET /api/v1/applications/{id}/explanation endpoint persist bilingual reasons in ScoreResult.
- Added template, reconciliation, persona, stability, notice, and endpoint persistence tests; documented interpretation in docs/EXPLAINABILITY.md.



### Added — Model training and scorecard
- Added business-disjoint stratified 60/20/20 training, validation, and test splits with a fixed seed.
- Trained a regularised logistic champion and histogram gradient boosting challenger; selected Platt or isotonic calibration using validation Brier.
- Added version-checked joblib artifacts, metadata with feature and data hashes, a configurable 300–900 scorecard with A–E PD tiers, and a traditional bureau/collateral baseline.
- Generated synthetic-only validation and model-card reports, including thin-file and thick-file holdout results.
- Added split, monotonic-risk, loader, score scaling, and scoring-latency tests.

### Validation — Model training
- Logistic champion on 600 untouched test businesses: ROC-AUC 0.785, PR-AUC 0.301, KS 0.439, Brier 0.081.
- Challenger test ROC-AUC 0.741. Training completed in under a minute on the development machine.



### Added — Feature engineering
- Added 30 point-in-time model features, a validated English/Urdu actionability registry, and versioned persisted feature snapshots.
- Added pure scenario recomputation for amount, tenor, balance, digital share, and known debt service; protected attributes stay outside model features.
- Documented imputation and the existing-debt-service data gap in docs/FEATURES.md.
- Added hand-calculated, no-look-ahead, missingness, registry, and snapshot persistence tests.

### Validation — Feature engineering
- 3,000 synthetic applications computed and flushed in 27.9 seconds in a rollback-only benchmark.



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
