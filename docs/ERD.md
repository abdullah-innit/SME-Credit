# SME Credit relational model

Money fields are nonnegative integer PKR. Amount sign is represented by direction.
SHA-256 lowercase digests replace counterparty identities. Evidence JSON must be sanitized.
All 16 records have id, created_at and updated_at; foreign keys use RESTRICT deletion.
Audit records are append-only through the ORM; SQL maintenance needs separate controls.

```mermaid
erDiagram
    User {
        int id PK
        datetime created_at
        datetime updated_at
        string role
    }
    Business {
        int id PK
        datetime created_at
        datetime updated_at
        int owner_id FK
        string name
        string sector
        string city
        string province
        string legal_form
        date incorporated_on
        int employees
        string owner_gender
        string owner_age_band
        bool is_women_led
        bool has_ntn
        bool is_synthetic
        int collateral_value_pkr
    }
    ConsentRecord {
        int id PK
        datetime created_at
        datetime updated_at
        int business_id FK
        int user_id FK
        string scope
        string policy_version
        datetime granted_at
        datetime revoked_at
    }
    Application {
        int id PK
        datetime created_at
        datetime updated_at
        int business_id FK
        string product
        int amount_requested_pkr
        int tenor_months
        string purpose
        string status
        datetime submitted_at
        int consent_id FK
    }
    Document {
        int id PK
        datetime created_at
        datetime updated_at
        int business_id FK
        string type
        string filename
        stringing sha256
        string parse_status
        json parse_report
    }
    Transaction {
        int id PK
        datetime created_at
        datetime updated_at
        int business_id FK
        date date
        int amount_pkr
        string direction
        string channel
        string category
        stringing counterparty_hash
        int balance_after
        int document_id FK
    }
    BillPayment {
        int id PK
        datetime created_at
        datetime updated_at
        int business_id FK
        int document_id FK
        string utility_type
        date due_date
        date paid_date
        int amount_pkr
    }
    BureauRecord {
        int id PK
        datetime created_at
        datetime updated_at
        int business_id FK
        int document_id FK
        int score
        int active_loans
        bool delinquent_flag
        int history_months
    }
    FeatureSnapshot {
        int id PK
        datetime created_at
        datetime updated_at
        int application_id FK
        string feature_version
        json values
    }
    ScoreResult {
        int id PK
        datetime created_at
        datetime updated_at
        int application_id FK
        int feature_snapshot_id FK
        string model_version
        float pd
        int score
        string tier
        json reasons
        json counterfactuals
    }
    IntegrityResult {
        int id PK
        datetime created_at
        datetime updated_at
        int application_id FK
        string severity
        string code
        json evidence
    }
    FraudFlag {
        int id PK
        datetime created_at
        datetime updated_at
        int application_id FK
        int integrity_result_id FK
        string severity
        string code
        json evidence
    }
    Decision {
        int id PK
        datetime created_at
        datetime updated_at
        int application_id FK
        string outcome
        int amount
        int tenor
        string rate_band
        int decided_by FK
        bool is_override
        string override_reason
        string policy_version
    }
    MemoRecord {
        int id PK
        datetime created_at
        datetime updated_at
        int application_id FK
        int decision_id FK
        int created_by FK
        string version
        string content
    }
    AiCallLog {
        int id PK
        datetime created_at
        datetime updated_at
        int application_id FK
        int actor FK
        string provider
        string model
        string task
        string status
        stringing input_hash
        stringing output_hash
        int duration_ms
        string prompt_version
        int tokens
        float cost_usd
    }
    AuditLog {
        int id PK
        datetime created_at
        datetime updated_at
        int actor FK
        string action
        string entity
        int entity_id
        json before
        json after
        datetime at
    }
    User ||--o{ Business : "owner_id"
    Business ||--o{ ConsentRecord : "business_id"
    User ||--o{ ConsentRecord : "user_id"
    Business ||--o{ Application : "business_id"
    ConsentRecord ||--o{ Application : "consent_id"
    Business ||--o{ Document : "business_id"
    Business ||--o{ Transaction : "business_id"
    Document |o--o{ Transaction : "document_id"
    Business ||--o{ BillPayment : "business_id"
    Document |o--o{ BillPayment : "document_id"
    Business ||--o{ BureauRecord : "business_id"
    Document |o--o{ BureauRecord : "document_id"
    Application ||--o{ FeatureSnapshot : "application_id"
    Application ||--o{ ScoreResult : "application_id"
    FeatureSnapshot ||--o{ ScoreResult : "feature_snapshot_id"
    Application ||--o{ IntegrityResult : "application_id"
    Application ||--o{ FraudFlag : "application_id"
    IntegrityResult |o--o{ FraudFlag : "integrity_result_id"
    Application ||--o{ Decision : "application_id"
    User ||--o{ Decision : "decided_by"
    Application ||--o{ MemoRecord : "application_id"
    Decision |o--o{ MemoRecord : "decision_id"
    User ||--o{ MemoRecord : "created_by"
    Application |o--o{ AiCallLog : "application_id"
    User |o--o{ AiCallLog : "actor"
    User |o--o{ AuditLog : "actor"
```

## Shared contracts

Use `app.enums`, `app.db.models`, and `app.schemas.records` in later modules.
Create contracts forbid extra fields and reject fractional/negative money, invalid enums,
and raw account/counterparty keys (including nested JSON). Read contracts support ORM objects.
Relationships are named after the FK without `_id`; inverse collections use plural record names (for example, `business.transactions`).
Consent and linked evidence must be checked against the application business by service code.
Free text must never contain raw account numbers or counterparty names; do not persist raw
LLM prompts or outputs. AiCallLog stores hashes and metadata only.

## Database lifecycle

From backend: `python -m app.db.seed` creates all tables and idempotently seeds synthetic data.
`init_db(seed_hook=...)` accepts a typed session callback in a committed transaction.
`get_db` provides a closing FastAPI dependency; `session_scope` commits or rolls back.
`create_all` bootstraps a new database; it is not a schema migration system.

SQLite returns naive timestamps; stored timestamps represent UTC. Reassign JSON values after
edits so SQLAlchemy tracks them; nested in-place JSON changes are not tracked automatically.
