# Agent Context — SME Credit

> This file is the single source of truth for every AI assistant working on this codebase.
> Regenerated from Prompt 1. Do not edit manually.

---

## Product

An AI credit-scrutiny platform that helps small and medium businesses in Pakistan get the loans they need by scoring them on real cash-flow behaviour instead of only collateral and bureau history, while giving banks explainable, fraud-checked, auditable decisions.

## Judging Narrative (must be visible in the product and the demo)

**Problem → Data/Input → AI Component → Solution → Impact.**

- **Problem:** SMEs are underserved by formal credit because of thin credit files and collateral demands; banks face slow manual scrutiny and hidden risk.
- **Data/Input:** bank and wallet statements, business profile, bill-payment history, bureau record (all synthetic for the demo).
- **AI Component:** default-risk model, explanations, counterfactual Path to Approval, LLM extraction, credit memo and advisor, fraud/integrity checks, fairness audit.
- **Solution:** applicant portal plus lender dashboard.
- **Impact:** more creditworthy SMEs approved at controlled risk, decisions in seconds, SDG 8 and 9 alignment, all measured on a holdout set.

## Judging Criteria

AI Integration Depth, Problem-Solving Quality, Demo & Presentation, Creativity & SDG Impact, Industrial Impact. Every feature must serve at least one, and AI must be meaningful, not cosmetic.

## Stack

- **Frontend:** Next.js (App Router, TypeScript, Tailwind) in `/frontend`
- **Backend:** FastAPI on Python 3.11+ in `/backend`
- **Database:** SQLite through SQLAlchemy
- **ML:** scikit-learn
- **Tests:** pytest
- **Platform:** Windows (avoid Linux-only deps; provide `.bat` / `.ps1` helpers plus `RUN-WINDOWS.txt`)

## Standing Rules (all 20 prompts)

1. **Working MVP over completeness.** Every prompt must leave the app runnable.
2. **Synthetic data only, always labelled as such.** Never imply real customers or real bank partnerships. Never hardcode unverified statistics: keep every external fact in `backend/app/config/facts.yaml` with `source` and `verified` fields (default `verified: false`).
3. **The LLM never makes credit decisions.** It extracts, explains and drafts. Decisions come from the model plus the policy engine, and a human officer has the final say.
4. **Secrets only in `.env`; ship `.env.example`.** The app must run fully in `OFFLINE_MODE` with no API key (deterministic fallbacks) because venue Wi-Fi may fail.
5. **Typed code, pydantic schemas, small focused modules, no dead code, clear commit messages.**
6. **Every prompt ends with the Standard Closing Block.**

## Standard Closing Block

1. Run tests and lint and fix failures.
2. Start both servers and confirm they work.
3. Add a dated entry to `CHANGELOG.md`.
4. Reply with: files created or changed, how to verify in under two minutes, known issues, and what the next prompt may assume.

---

*Event:* AICON'26 (NUST), Build With AI module, Financial Operations domain.
