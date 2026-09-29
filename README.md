# SME Credit Scrutiny (practice build)

Data -> AI -> impact: alternative-data credit assessment for SMEs with no formal credit history.

## Run it
Backend (terminal 1):
    cd backend
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-ant-...                  # optional; without it the committee runs in offline mode
    uvicorn main:app --reload --port 8000

Frontend (terminal 2):
    cd frontend
    npm install
    npm run dev                                          # http://localhost:3000

## What is in it
- backend/data_gen.py   synthetic SME data (outcomes are synthetic: say so in the demo)
- backend/model.py      logistic regression, exact per-factor explanations, path-to-approval search (the wow feature)
- backend/llm.py        Claude credit committee + ledger-photo extraction
- backend/main.py       API: /api/applicants, /api/whatif, /api/committee, /api/impact, /api/upload-csv, /api/extract, /api/score-record
- frontend/app/page.tsx dashboard: score dial, factor bars, draggable path to approval, committee

## Ideas to add before the real event
1. A UI for /api/extract (photo of a khata or statement -> features -> score). The endpoint already works.
2. A UI for /api/upload-csv (credit officer uploads a portfolio).
3. Swap the linear model for XGBoost + SHAP if you want non-linear effects (keep the path-to-approval search).
4. Real-ish data: replace data_gen.py with a public SME/credit dataset.

## Demo script (5 minutes)
1. Impact bar: one sentence on the problem and the numbers.
2. Pick a declined business with no credit file. Show the score and the factors.
3. Drag the path-to-approval slider until it flips to Approve.
4. Convene the committee. Close on the impact line.
