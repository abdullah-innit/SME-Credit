# SME Credit — AI Credit-Scrutiny Platform

> **AICON'26 (NUST) · Build With AI · Financial Operations**
>
> ⚠️ All data is **synthetic** — no real customers or bank partnerships.

An AI-powered credit-scrutiny platform that helps small and medium businesses in Pakistan access formal credit by scoring them on real cash-flow behaviour instead of collateral and bureau history alone, while giving banks explainable, fraud-checked, auditable decisions.

## Quick Start (Windows)

```powershell
# 1 — Clone & enter
git clone <repo-url> "SME Credit"
cd "SME Credit"

# 2 — Copy env
copy .env.example .env

# 3 — One-command setup
.\scripts\setup.bat

# 4 — Run everything
.\scripts\run-all.bat
```

| Service  | URL                          |
|----------|------------------------------|
| Backend  | http://localhost:8000/docs    |
| Frontend | http://localhost:3000         |
| Health   | http://localhost:8000/api/v1/health |

## Project Structure

```
SME Credit/
├── backend/           FastAPI + Python 3.11+
│   ├── app/
│   │   ├── api/       Versioned REST endpoints
│   │   ├── core/      Config, logging, security
│   │   ├── db/        SQLAlchemy models & sessions
│   │   ├── ml/        ML models & pipelines
│   │   ├── llm/       LLM integration (extraction, memos)
│   │   ├── services/  Business logic
│   │   ├── config/    facts.yaml, policy rules
│   │   └── knowledge/ Domain knowledge for RAG
│   └── tests/
├── frontend/          Next.js (App Router, TypeScript, Tailwind)
├── data/              Synthetic datasets
├── samples/           Sample inputs for demo
├── docs/              Documentation
└── scripts/           Setup & run scripts (.bat / .ps1)
```

## Stack

- **Backend:** FastAPI · Python 3.11+ · SQLite via SQLAlchemy · scikit-learn
- **Frontend:** Next.js (App Router) · TypeScript · Tailwind CSS
- **ML:** scikit-learn default-risk model with SHAP explanations
- **LLM:** Configurable (OpenAI / offline fallback) — *never* makes credit decisions

## Standing Rules

1. Working MVP over completeness
2. Synthetic data only, always labelled
3. LLM extracts, explains, drafts — never decides
4. Secrets in `.env`; full offline mode available
5. Typed code, pydantic schemas, small modules

## License

Built for AICON'26 hackathon. See LICENSE for details.
