# Validation Plan

> Placeholder — will be populated as features are built.

## Test Strategy

- **Unit tests:** pytest in `backend/tests/`
- **API tests:** FastAPI TestClient
- **Frontend tests:** to be configured
- **Linting:** ruff (backend), ESLint (frontend)

## Holdout Evaluation

_To be documented when the ML model and holdout set are ready._

## Fairness Audit

_To be documented when the fairness module is implemented._


## Relational contract verification (2026-10-09)

After dependency setup, from `backend` in PowerShell (under two minutes):

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m ruff check app tests
.\venv\Scripts\python.exe -m app.db.seed
```

Expected: 19 tests pass; lint passes; the seed prints a success message.
The seed is synthetic and idempotent. Servers retain their existing commands.

Later modules import ORM records from `app.db.models`, create/read contracts
from `app.schemas.records`, and all enum values from `app.enums`.
Use `get_db` for FastAPI routes, `session_scope` for committed work, and
`init_db(seed_hook=...)` for an explicit bootstrap callback. Do not auto-seed production.
Assign new JSON dictionaries/lists when changing evidence; nested in-place changes
are not tracked. UTC timestamps are naive when read from SQLite.

Authorization and same-business evidence/consent checks belong in later service
modules; FK integrity alone does not establish ownership. Free-text content and
uploaded filenames must be sanitized before persistence. Hash counterparty IDs
upstream; do not store their original values. Append-only ORM guards cannot stop
privileged raw SQL changes. Existing databases require migrations, not create_all.


## SME-Credit2.0 compatibility

Business.is_synthetic is persisted and returned in schemas (default true for demo).
AgeBand and OwnerAgeBand refer to one enum; AGE_35_44 and BAND_35_44 are aliases.
Legacy input 35-44 is normalized to stored value 35_44. Unknown values remain invalid.
The ERD includes is_synthetic and is tested against exact mapped columns and links.
Use a fresh extraction to avoid local duplicate tests. For an existing SQLite DB,
run `python -m app.db.migrate_synthetic` before seeding or starting services.
