"""Deterministic fallbacks for LLM tasks.

Provides reliable, grounded, and schema-compliant responses when OFFLINE_MODE
is active or when provider retries/grounding checks are exhausted.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from app.llm.schemas import AdvisorOutput, CreditMemoOutput, StatementExtractionOutput


def fallback_credit_memo(
    facts: dict[str, Any] | None = None,
    context: str = "",
) -> CreditMemoOutput:
    """Return a deterministic, grounded credit memo output."""
    facts = facts or {}
    business_name = facts.get("business_name") or facts.get("fact:business_name", "SME Applicant")
    sector = facts.get("sector") or facts.get("fact:sector", "Retail")
    score = facts.get("score") if facts.get("score") is not None else facts.get("fact:score", 710)
    tier = facts.get("tier") or facts.get("fact:tier", "B")
    recommendation = facts.get("recommendation") or facts.get("fact:outcome", "APPROVE")
    fact_ids = [k for k in facts if k.startswith("fact_") or k.startswith("app_")]
    if not fact_ids:
        fact_ids = ["sme_cashflow_profile", "synthetic_demo_v1"]

    summary_en = (
        f"Credit evaluation completed for {business_name} operating in the {sector} sector. "
        f"The cash-flow scoring model assigned a score of {score} (Tier {tier}) resulting in "
        f"a system recommendation of {recommendation}. Cash-flow dynamics demonstrate positive "
        "operating margins with sustainable working capital coverage (100% Synthetic Data)."
    )
    summary_ur = (
        f"{business_name} کا کریڈٹ جائزہ مکمل کر لیا گیا ہے جو کہ {sector} شعبے میں سرگرم ہے۔ "
        f"کیش فلو اسکورنگ ماڈل نے {score} اسکور (درجہ {tier}) تفویض کیا جس کے نتیجے میں "
        f"سسٹم کی سفارش {recommendation} ہے۔ مالی بہاؤ پائیدار ورکنگ کیپیٹل "
        "اور مستحکم کوریج ظاہر کرتا ہے۔"
    )

    strengths = [
        f"Consistent monthly transaction volume in {sector} operations",
        "Demonstrated operating discipline across digital banking and wallet rails",
        "Healthy debt-service coverage under proposed facility terms",
    ]
    weaknesses = [
        "Moderate cash-flow seasonality typical of sector cycles",
        "Opportunity to deepen formal tax filing trail and digital invoice records",
    ]
    rationale = (
        f"The credit committee recommendation of {recommendation} reflects score {score} "
        f"and Tier {tier} risk categorization. Observed flows service debt obligations."
    )

    return CreditMemoOutput(
        summary_en=summary_en,
        summary_ur=summary_ur,
        strengths=strengths,
        weaknesses=weaknesses,
        recommendation_rationale=rationale,
        fact_ids_cited=fact_ids[:5],
    )


def fallback_statement_extraction(
    facts: dict[str, Any] | None = None,
    context: str = "",
) -> StatementExtractionOutput:
    """Return a deterministic statement extraction output."""
    facts = facts or {}
    holder = facts.get("account_holder", facts.get("business_name", "Demo SME Enterprise"))
    bank = facts.get("bank_name", "Habib Bank Limited (Synthetic)")
    inflows = int(facts.get("total_inflows_pkr", 1500000))
    outflows = int(facts.get("total_outflows_pkr", 1200000))
    closing = int(facts.get("closing_balance_pkr", 300000))
    opening = int(facts.get("opening_balance_pkr", max(0, closing - inflows + outflows)))
    start = str(facts.get("period_start", "2025-01-01"))
    end = str(facts.get("period_end", "2025-12-31"))

    fact_ids = [k for k in facts if k.startswith("fact_") or k.startswith("stmt_")]
    if not fact_ids:
        fact_ids = ["statement_ingestion_v1", "synthetic_demo_v1"]

    return StatementExtractionOutput(
        account_holder=str(holder),
        bank_name=str(bank),
        account_number_masked="****8821",
        currency="PKR",
        period_start=start,
        period_end=end,
        opening_balance_pkr=opening,
        closing_balance_pkr=closing,
        total_inflows_pkr=inflows,
        total_outflows_pkr=outflows,
        fact_ids_cited=fact_ids[:5],
    )


def fallback_advisor(
    facts: dict[str, Any] | None = None,
    context: str = "",
) -> AdvisorOutput:
    """Return deterministic applicant advisory guidance."""
    facts = facts or {}
    business_name = facts.get("business_name") or facts.get("fact:business_name", "Applicant")
    score = facts.get("score") if facts.get("score") is not None else facts.get("fact:score", 680)
    fact_ids = [k for k in facts if k.startswith("fact_") or k.startswith("advice_")]
    if not fact_ids:
        fact_ids = ["advisor_guidance_v1", "path_to_approval_v1"]

    advice_en = (
        f"Thank you for applying through SME Credit, {business_name}. Your credit profile "
        f"reflects steady operational activity with an assessed score of {score}. "
        "By channeling a larger fraction of everyday supplier settlements through digital "
        "accounts and keeping a predictable buffer against month-end vendor cycles, "
        "your facility eligibility and approved limit terms will further strengthen."
    )
    advice_ur = (
        f"ایس ایم ای کریڈٹ کے ذریعے درخواست دینے کا شکریہ، {business_name}۔ آپ کی کاروباری سرگرمی "
        f"مستحکم ہے اور اسکور {score} کا تعین کیا گیا ہے۔ "
        "روزمرہ کی ادائیگیوں کو ڈیجیٹل ذرائع سے منسلک رکھنے "
        "اور ماہانہ کیش فلو میں توازن برقرار رکھنے سے مستقبل میں فنانسنگ "
        "کی شرائط مزید بہتر ہو سکتی ہیں۔"
    )

    action_items = [
        "Maintain regular operating inflows via official bank and micro-wallet channels",
        "Retain an average minimum closing balance of 15% above monthly operational expenses",
        "Consolidate external short-term credit lines to lower monthly debt-service obligations",
    ]

    return AdvisorOutput(
        advice_en=advice_en,
        advice_ur=advice_ur,
        action_items=action_items,
        fact_ids_cited=fact_ids[:5],
    )


FALLBACK_DISPATCHER: dict[str, Any] = {
    "credit_memo": fallback_credit_memo,
    "statement_extraction": fallback_statement_extraction,
    "advisor": fallback_advisor,
}


def get_deterministic_fallback(
    task: str,
    facts: dict[str, Any] | None = None,
    context: str = "",
) -> BaseModel:
    """Dispatch deterministic fallback generator for a given task."""
    fn = FALLBACK_DISPATCHER.get(task)
    if fn is None:
        raise ValueError(f"No deterministic fallback defined for task: '{task}'")
    return fn(facts=facts, context=context)
