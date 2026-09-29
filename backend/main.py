import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import llm
from data_gen import generate
from model import APPROVE_PD, CreditModel, decision_for, to_features

app = FastAPI(title="SME Credit Scrutiny")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

RAW_FIELDS = ["monthly_revenue", "loan_amount", "business_age_years", "revenue_growth_6m",
              "revenue_volatility", "cash_reserve_months", "existing_debt_ratio", "digital_payment_share",
              "utility_on_time_rate", "customer_concentration", "revenue_gap"]

# Illustrative assumptions for the impact story. Say they are assumptions in the demo.
TRADITIONAL_DECISION_DAYS = 14
LOAN_PER_JOB_PKR = 1_000_000

df = generate(1000)
train, test = df.iloc[:700].reset_index(drop=True), df.iloc[700:].reset_index(drop=True)
model = CreditModel().fit(train)
MODEL_AUC = round(model.auc(test), 3)


def _rec(row) -> dict:
    r = dict(row)
    for k in ("has_credit_history", "defaulted"):
        r[k] = bool(r[k])
    return r


# Pick a demo pool with a spread of decisions (8 approve, 8 refer, 8 decline).
_scored = [(i, model.assess(_rec(r))) for i, r in enumerate(test.to_dict("records"))]
_pool: list[int] = []
for want in ("approve", "refer", "decline"):
    _pool += [i for i, a in _scored if a["decision"] == want][:8]
POOL = {n + 1: test.iloc[i].to_dict() for n, i in enumerate(_pool)}


def _public(rec: dict) -> dict:
    return {k: (bool(v) if isinstance(v, (np.bool_, bool)) else (v.item() if hasattr(v, "item") else v))
            for k, v in rec.items() if k != "defaulted"}


def _get(applicant_id: int) -> dict:
    if applicant_id not in POOL:
        raise HTTPException(404, "Unknown applicant")
    return POOL[applicant_id]


@app.get("/api/applicants")
def applicants():
    out = []
    for aid, r in POOL.items():
        a = model.assess(_rec(r))
        out.append({"id": aid, "name": r["name"], "sector": r["sector"], "city": r["city"],
                    "loan_amount": float(r["loan_amount"]), "score": a["score"], "pd": a["pd"],
                    "decision": a["decision"], "thin_file": not bool(r["has_credit_history"])})
    return out


@app.get("/api/applicants/{applicant_id}")
def applicant(applicant_id: int):
    r = _rec(_get(applicant_id))
    a = model.assess(r)
    return {"id": applicant_id, "record": _public(r), "assessment": a,
            "path": model.path_to_approval(r) if a["decision"] != "approve" else None}


class WhatIf(BaseModel):
    id: int
    overrides: dict[str, float] = {}


@app.post("/api/whatif")
def whatif(body: WhatIf):
    r = {**_rec(_get(body.id)), **{k: v for k, v in body.overrides.items() if k in RAW_FIELDS}}
    return model.assess(r)


class CommitteeReq(BaseModel):
    id: int


@app.post("/api/committee")
def committee(body: CommitteeReq):
    r = _rec(_get(body.id))
    a = model.assess(r)
    payload = {"business": {k: r[k] for k in ("name", "sector", "city")},
               "request": {"loan_amount": r["loan_amount"], "monthly_revenue": r["monthly_revenue"]},
               "has_formal_credit_history": r["has_credit_history"],
               "declared_vs_observed_revenue_gap": r["revenue_gap"],
               "assessment": a,
               "path_to_approval": model.path_to_approval(r) if a["decision"] != "approve" else None}
    return llm.committee(payload)


@app.get("/api/impact")
def impact():
    """Data -> AI -> impact: compare a credit-history rule with alternative-data scoring on held-out SMEs."""
    trad, ai, thin, thin_ai, extra_volume = [], [], 0, 0, 0.0
    for r in test.to_dict("records"):
        rr = _rec(r)
        loan_ratio = rr["loan_amount"] / (rr["monthly_revenue"] * 12)
        t_ok = rr["has_credit_history"] and rr["existing_debt_ratio"] < 0.35 and loan_ratio < 0.6
        a_ok = model.assess(rr)["decision"] == "approve"
        thin += 0 if rr["has_credit_history"] else 1
        thin_ai += 1 if (a_ok and not rr["has_credit_history"]) else 0
        if a_ok and not t_ok:
            extra_volume += rr["loan_amount"]
        if t_ok:
            trad.append(rr)
        if a_ok:
            ai.append(rr)
    rate = lambda xs: round(100 * float(np.mean([x["defaulted"] for x in xs])), 1) if xs else None
    return {
        "sample_size": len(test), "model_auc": MODEL_AUC,
        "traditional_approved": len(trad), "ai_approved": len(ai),
        "thin_file_total": thin, "thin_file_approved_by_ai": thin_ai,
        "traditional_default_rate": rate(trad), "ai_default_rate": rate(ai),
        "extra_loan_volume_pkr": extra_volume,
        "estimated_jobs_supported": int(extra_volume // LOAN_PER_JOB_PKR),
        "decision_time": {"traditional_days": TRADITIONAL_DECISION_DAYS, "ai": "minutes"},
        "assumptions": f"Illustrative: 1 job per Rs. {LOAN_PER_JOB_PKR:,} lent, {TRADITIONAL_DECISION_DAYS}-day manual review. "
                       "Outcomes are synthetic.",
    }


class Record(BaseModel):
    monthly_revenue: float
    loan_amount: float
    business_age_years: float = 3
    revenue_growth_6m: float = 0
    revenue_volatility: float = 0.3
    cash_reserve_months: float = 1
    existing_debt_ratio: float = 0.2
    digital_payment_share: float = 0.3
    utility_on_time_rate: float = 0.9
    customer_concentration: float = 0.3
    revenue_gap: float = 0.05


@app.post("/api/score-record")
def score_record(rec: Record):
    r = rec.model_dump()
    a = model.assess(r)
    return {"assessment": a, "path": model.path_to_approval(r) if a["decision"] != "approve" else None}


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    frame = pd.read_csv(io.BytesIO(await file.read()))
    missing = [c for c in ("monthly_revenue", "loan_amount") if c not in frame.columns]
    if missing:
        raise HTTPException(422, f"CSV is missing required columns: {', '.join(missing)}")
    defaults = Record(monthly_revenue=1, loan_amount=1).model_dump()
    out = []
    for i, row in frame.iterrows():
        r = {**defaults, **{k: float(row[k]) for k in RAW_FIELDS if k in frame.columns and pd.notna(row[k])}}
        a = model.assess(r)
        out.append({"row": int(i) + 1, "name": row.get("name", f"Row {i + 1}"), **{k: a[k] for k in ("pd", "score", "decision")},
                    "top_factor": a["factors"][0]["label"]})
    return out


@app.post("/api/extract")
async def extract(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP photo of the ledger.")
    try:
        return llm.extract_from_image(await file.read(), file.content_type)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
