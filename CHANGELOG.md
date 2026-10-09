# Changelog

All notable changes to SME Credit are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

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
