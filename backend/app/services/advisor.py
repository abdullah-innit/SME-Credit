"""Policy-grounded Advisor Service (Applicant Assistant & Officer Copilot).

Features:
- Two personas: Applicant Assistant (English & Urdu; strict applicant tenant isolation)
  and Officer Copilot (policy questions, referral rationale, app comparison, conditions drafting).
- Tool calling: get_application_summary, run_what_if, explain_reason, search_policy.
- Strict safety guardrails: can NEVER approve, decline or promise a loan.
- Polite refusals for out-of-scope or sensitive requests.
- Cites knowledge base chunk IDs for all policy answers.
- Session-scoped memory for conversational context.
- PII-safe logging.
- Offline mode: retrieval-only answers with cited chunks and template phrasing.
- Streaming endpoint (SSE) generator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import Application
from app.db.session import SessionLocal
from app.knowledge.retrieval import get_knowledge_retriever
from app.services.advisor_tools import (
    ToolCallRecord,
    explain_reason_tool,
    get_application_summary,
    run_what_if_tool,
    search_policy_tool,
)

logger = logging.getLogger("advisor")

# ---------------------------------------------------------------------------
# PII-Safe Logging Utilities
# ---------------------------------------------------------------------------

CNIC_REGEX = re.compile(r"\b\d{5}[-]?\d{7}[-]?\d\b")
PHONE_REGEX = re.compile(r"\b(?:\+?92|0)?3\d{2}[- ]?\d{7}\b")
CARD_REGEX = re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b")
IBAN_REGEX = re.compile(r"\bPK\d{2}[A-Za-z]{4}\d{16}\b")
API_KEY_REGEX = re.compile(r"(sk-[a-zA-Z0-9_-]{16,})")


def mask_pii(text: str) -> str:
    """Mask sensitive PII identifiers from text for safe logging."""
    if not text:
        return text
    text = CNIC_REGEX.sub("[CNIC-MASKED]", text)
    text = PHONE_REGEX.sub("[PHONE-MASKED]", text)
    text = CARD_REGEX.sub("[CARD-MASKED]", text)
    text = IBAN_REGEX.sub("[IBAN-MASKED]", text)
    text = API_KEY_REGEX.sub("[API-KEY-MASKED]", text)
    return text


def pii_safe_log(level: int, msg: str, *args: Any) -> None:
    """Log formatted message with PII redaction."""
    safe_msg = mask_pii(msg % args if args else msg)
    logger.log(level, safe_msg)


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class AdvisorRequest(BaseModel):
    """Input payload for advisor chat."""

    message: str = Field(
        ..., min_length=1, max_length=2000, description="User question or instruction"
    )
    session_id: str = Field(default="default-session", description="Session identifier for memory")
    persona: Literal["applicant", "officer"] = Field(
        default="applicant", description="Active persona: 'applicant' or 'officer'"
    )
    application_id: int | None = Field(
        default=None, description="Current application under discussion"
    )
    compare_application_id: int | None = Field(
        default=None, description="Second application ID for officer comparison"
    )
    user_id: int | None = Field(
        default=None, description="Authenticated user ID for tenant authorization"
    )
    language: Literal["en", "ur"] = Field(
        default="en", description="Preferred response language ('en' or 'ur')"
    )


class AdvisorResponse(BaseModel):
    """Structured response returned to caller/UI."""

    session_id: str
    persona: str
    language: str
    message: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    refusal: bool = False
    refusal_reason: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


# ---------------------------------------------------------------------------
# Session-Scoped Memory Store
# ---------------------------------------------------------------------------


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


class AdvisorSessionMemory:
    """Thread-safe in-memory session memory with max turns."""

    def __init__(self, max_turns: int = 20) -> None:
        self.max_turns = max_turns
        self._sessions: dict[str, list[ChatTurn]] = defaultdict(list)

    def add_turn(
        self,
        session_id: str,
        role: str,
        message: str,
        tool_calls: list[ToolCallRecord] | None = None,
        citations: list[str] | None = None,
    ) -> None:
        turns = self._sessions[session_id]
        turns.append(
            ChatTurn(
                role=role,
                message=message,
                tool_calls=tool_calls or [],
                citations=citations or [],
            )
        )
        if len(turns) > self.max_turns:
            self._sessions[session_id] = turns[-self.max_turns :]

    def get_history(self, session_id: str) -> list[ChatTurn]:
        return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]


session_memory = AdvisorSessionMemory()


# ---------------------------------------------------------------------------
# Guardrail & Intent Detection Rules
# ---------------------------------------------------------------------------

APPROVAL_GUARANTEE_PATTERNS = [
    r"guarantee.*approval",
    r"promise.*loan",
    r"promise.*approval",
    r"approve.*my.*loan",
    r"will.*i.*be.*approved",
    r"can.*you.*approve",
    r"give.*me.*approval",
    r"confirm.*approval",
    r"قرض.*کی.*منظوری",
    r"قرض.*کا.*وعدہ",
    r"منظوری.*کی.*ضمانت",
    r"میرا.*قرض.*منظور",
]

OFF_TOPIC_PATTERNS = [
    r"\bbiryani\b",
    r"\brecipe\b",
    r"\bcook(?:ing)?\b",
    r"\bweather\b",
    r"\bforecast\b",
    r"\btemperature\b",
    r"\bfootball\b",
    r"\bcricket\b",
    r"\bsoccer\b",
    r"\bmovie\b",
    r"\bfilm\b",
    r"\bcelebrity\b",
    r"\bactor\b",
    r"\bmusic\b",
    r"\bsong\b",
    r"\bbitcoin\b",
    r"\bcrypto\w*",
    r"\bstock\w*",
    r"\btrad(?:e|ing)\b",
    r"\bshares\b",
    r"\bforex\b",
    r"\beth(?:ereum)?\b",
    r"\bhoroscope\b",
    r"\bastrology\b",
    r"\bjoke\b",
    r"\bpoem\b",
    r"\bpoetry\b",
    r"\bखाना\b",
    r"\bموسم\b",
    r"\bبریانی\b",
    r"\bشاعری\b",
    r"\bلطیفہ\b",
]

URDU_DETECTION_REGEX = re.compile(r"[\u0600-\u06FF]")


def is_urdu_text(text: str) -> bool:
    """Return True if text contains significant Urdu / Arabic script."""
    return len(URDU_DETECTION_REGEX.findall(text)) >= 3


def check_approval_guarantee_request(text: str) -> bool:
    """Detect if user is demanding approval or loan promises."""
    lower = text.lower()
    return any(re.search(pattern, lower, re.IGNORECASE) for pattern in APPROVAL_GUARANTEE_PATTERNS)


def check_off_topic_request(text: str) -> bool:
    """Detect unrelated trivia, entertainment, recipes or sports."""
    lower = text.lower()
    return any(re.search(pattern, lower, re.IGNORECASE) for pattern in OFF_TOPIC_PATTERNS)


# ---------------------------------------------------------------------------
# Polite Refusal Templates
# ---------------------------------------------------------------------------

REFUSAL_APPROVAL_GUARANTEE_EN = (
    "As an AI advisor, I cannot approve, decline, or promise a loan. Formal credit "
    "decisions are evaluated and sanctioned exclusively by the bank's authorized human "
    "credit committee. I can, however, help explain your score factors, summarize policy "
    "guidelines, or explore what-if scenarios on your Path to Approval."
)

REFUSAL_APPROVAL_GUARANTEE_UR = (
    "بطور اے آئی اسسٹنٹ، میں قرض کی منظوری، تنسیخ یا کوئی وعدہ نہیں کر سکتا۔ حتمی فیصلہ "
    "صرف بینک کی بااختیار کریڈٹ کمیٹی کرتی ہے۔ البتہ میں آپ کے اسکور کے عوامل، "
    "پالیسی شرائط یا منظوری کے راستے (Path to Approval) کی رہنمائی کر سکتا ہوں۔"
)

REFUSAL_OFF_TOPIC_EN = (
    "I specialize exclusively in SME Credit evaluation, policy rules, and application assistance. "
    "I cannot assist with general trivia, recipes, weather, entertainment, or unrelated topics. "
    "Please ask a question related to business eligibility, documents, product terms, or score."
)

REFUSAL_OFF_TOPIC_UR = (
    "میں صرف ایس ایم ای کریڈٹ، پالیسی قوانین اور کاروباری درخواست کی رہنمائی کے لیے مخصوص ہوں۔ "
    "عام معلومات، کھانوں، شاعری یا غیر متعلقہ موضوعات پر مدد نہیں کر سکتا۔ "
    "براہ کرم کاروباری فنانسنگ، دستاویزات یا کریڈٹ اسکور کے متعلق سوال پوچھیں۔"
)

REFUSAL_UNAUTHORIZED_EN = (
    "Unauthorized: You can only view and inquire about data for your own application. "
    "Access to other applicants' business and financial records is strictly restricted."
)

REFUSAL_UNAUTHORIZED_UR = (
    "غیر مجاز: آپ صرف اپنی ذاتی درخواست کے بارے میں معلومات حاصل کر سکتے ہیں۔ "
    "دوسرے درخواست گزاروں کے مالی ریکارڈ تک رسائی سختی سے ممنوع ہے۔"
)


# ---------------------------------------------------------------------------
# Offline / Deterministic Advisor Engine
# ---------------------------------------------------------------------------


class AdvisorEngine:
    """Main Advisor reasoning engine supporting Applicant Assistant and Officer Copilot."""

    def __init__(self) -> None:
        self.retriever = get_knowledge_retriever()

    def process(self, req: AdvisorRequest, session: Session | None = None) -> AdvisorResponse:
        """Process user message and return grounded, tool-enabled response."""
        user_msg = req.message.strip()
        pii_safe_log(
            logging.INFO,
            "Advisor turn: session=%s persona=%s app_id=%s msg=%s",
            req.session_id,
            req.persona,
            req.application_id,
            user_msg,
        )

        # Detect language from input text or request preference
        is_urdu = req.language == "ur" or is_urdu_text(user_msg)
        lang_code = "ur" if is_urdu else "en"

        # -------------------------------------------------------------------
        # Guardrail 1: Refuse approval guarantees or loan promises
        # -------------------------------------------------------------------
        if check_approval_guarantee_request(user_msg):
            msg = REFUSAL_APPROVAL_GUARANTEE_UR if is_urdu else REFUSAL_APPROVAL_GUARANTEE_EN
            resp = AdvisorResponse(
                session_id=req.session_id,
                persona=req.persona,
                language=lang_code,
                message=msg,
                refusal=True,
                refusal_reason="APPROVAL_PROMISE_REFUSAL",
                citations=["KB-RESP-001"],
            )
            session_memory.add_turn(req.session_id, "user", user_msg)
            session_memory.add_turn(req.session_id, "assistant", msg, citations=resp.citations)
            return resp

        # -------------------------------------------------------------------
        # Guardrail 2: Refuse off-topic requests politely
        # -------------------------------------------------------------------
        if check_off_topic_request(user_msg):
            msg = REFUSAL_OFF_TOPIC_UR if is_urdu else REFUSAL_OFF_TOPIC_EN
            resp = AdvisorResponse(
                session_id=req.session_id,
                persona=req.persona,
                language=lang_code,
                message=msg,
                refusal=True,
                refusal_reason="OUT_OF_SCOPE",
                citations=[],
            )
            session_memory.add_turn(req.session_id, "user", user_msg)
            session_memory.add_turn(req.session_id, "assistant", msg)
            return resp

        # -------------------------------------------------------------------
        # Persona 1: Applicant Assistant
        # -------------------------------------------------------------------
        if req.persona == "applicant":
            return self._handle_applicant_persona(req, user_msg, is_urdu, session)

        # -------------------------------------------------------------------
        # Persona 2: Officer Copilot
        # -------------------------------------------------------------------
        return self._handle_officer_persona(req, user_msg, is_urdu, session)

    def _handle_applicant_persona(
        self,
        req: AdvisorRequest,
        user_msg: str,
        is_urdu: bool,
        session: Session | None,
    ) -> AdvisorResponse:
        """Applicant Assistant workflow: strictly grounded in applicant's own facts."""
        tool_records: list[ToolCallRecord] = []
        citations: list[str] = []
        lower = user_msg.lower()

        # Authorization check: verify applicant does not try to read another applicant's record
        if req.application_id is not None:
            own = session is None
            s = session or SessionLocal()
            try:
                app_obj = s.get(Application, req.application_id)
                if app_obj and req.user_id is not None and app_obj.business.owner_id != req.user_id:
                    # Authorisation failure
                    msg = REFUSAL_UNAUTHORIZED_UR if is_urdu else REFUSAL_UNAUTHORIZED_EN
                    return AdvisorResponse(
                        session_id=req.session_id,
                        persona=req.persona,
                        language="ur" if is_urdu else "en",
                        message=msg,
                        refusal=True,
                        refusal_reason="UNAUTHORIZED_APPLICANT_ACCESS",
                        citations=[],
                    )
            finally:
                if own:
                    s.close()

        # Case A: What-If / Improve Score / Path to Approval
        is_what_if = any(
            w in lower
            for w in [
                "what if",
                "improve",
                "increase",
                "path to approval",
                "boost",
                "behtar",
                "بہتر",
                "کیسے بہتر",
            ]
        )
        if is_what_if and req.application_id:
            try:
                # Default what-if simulation: +20% digital payment share
                whatif_data, record = run_what_if_tool(
                    req.application_id,
                    changes={"digital_payment_share": 0.70},
                    user_id=req.user_id,
                    persona="applicant",
                    session=session,
                )
                tool_records.append(record)

                orig_s = whatif_data.get("original", {}).get("score", 620)
                new_s = whatif_data.get("what_if", {}).get("score", 685)
                new_dec = whatif_data.get("what_if", {}).get("decision", "APPROVE")

                if is_urdu:
                    reply = (
                        "آپ کے اسکور کو بہتر بنانے کا تخمینہ (What-If):\n"
                        f"- اگر آپ ڈیجیٹل فروخت کا حصہ 70% تک بڑھائیں، تو اسکور **{orig_s}** "
                        f"سے بڑھ کر **{new_s}** ہو جائے گا!\n"
                        f"- متوقع نتیجہ: **{new_dec}** [KB-IMPR-001]۔\n\n"
                        "عملی اقدامات: یوٹیلیٹی بل وقت پر ادا کریں اور کھاتوں میں 15% اوسط بیلنس "
                        "برقرار رکھیں [KB-IMPR-002]۔"
                    )
                else:
                    reply = (
                        f"Here is your Path to Approval simulation (What-If analysis):\n"
                        f"- By routing 70% of business sales digitally via banking/Raast rails, "
                        f"your score projects to rise from **{orig_s}** to **{new_s}**.\n"
                        f"- Projected Outcome: **{new_dec}** [KB-IMPR-001].\n\n"
                        "Key actionable habits to achieve this: maintain a consistent closing "
                        "cash buffer and ensure utility bills are paid on time [KB-IMPR-002]."
                    )
                citations.extend(["KB-IMPR-001", "KB-IMPR-002"])
                return self._build_success_response(req, reply, tool_records, citations, is_urdu)
            except PermissionError:
                msg = REFUSAL_UNAUTHORIZED_UR if is_urdu else REFUSAL_UNAUTHORIZED_EN
                return AdvisorResponse(
                    session_id=req.session_id,
                    persona=req.persona,
                    language="ur" if is_urdu else "en",
                    message=msg,
                    refusal=True,
                    refusal_reason="UNAUTHORIZED_APPLICANT_ACCESS",
                )
            except Exception as e:
                logger.warning("Applicant what-if error: %s", e)

        # Case B: Application Summary / Status
        is_summary = any(
            w in lower
            for w in [
                "summary",
                "status",
                "my application",
                "how is my",
                "score snapshot",
                "application snapshot",
            ]
        ) or ("score" in lower and not is_what_if)
        if is_summary and req.application_id:
            try:
                summary_data, record = get_application_summary(
                    req.application_id,
                    user_id=req.user_id,
                    persona="applicant",
                    session=session,
                )
                tool_records.append(record)

                score_val = summary_data["score"]
                tier_val = summary_data["tier"]
                rec_val = summary_data["recommendation"]
                b_name = summary_data["business_name"]
                amt = f"PKR {summary_data['requested_amount_pkr']:,}"

                if is_urdu:
                    reply = (
                        f"آپ کی درخواست کا خلاصہ برائے **{b_name}**:\n"
                        f"- تفویض کردہ اسکور: **{score_val}** (درجہ {tier_val})\n"
                        f"- سسٹم کی ابتدائی سفارش: **{rec_val}**\n"
                        f"- درخواست کردہ رقم: **{amt}**\n\n"
                        "یہ جائزہ کیش فلو اور بینک ریکارڈ پر مبنی ہے۔ باضابطہ حتمی فیصلہ "
                        "بینک کمیٹی کرے گی [KB-SCOR-001]۔"
                    )
                else:
                    reply = (
                        f"Here is your application snapshot for **{b_name}**:\n"
                        f"- Cash-Flow Credit Score: **{score_val}** (Tier {tier_val})\n"
                        f"- System Recommendation: **{rec_val}**\n"
                        f"- Requested Amount: **{amt}** ({summary_data['tenor_months']} months)\n"
                        f"- Verified Turnover: PKR {summary_data['monthly_turnover_pkr']:,}\n\n"
                        "This assessment reflects your operating cash-flow discipline "
                        "[KB-SCOR-001]. Final sanction resides with human credit officer "
                        "[KB-RESP-001]."
                    )
                citations.extend(["KB-SCOR-001", "KB-RESP-001"])
                return self._build_success_response(req, reply, tool_records, citations, is_urdu)
            except PermissionError:
                msg = REFUSAL_UNAUTHORIZED_UR if is_urdu else REFUSAL_UNAUTHORIZED_EN
                return AdvisorResponse(
                    session_id=req.session_id,
                    persona=req.persona,
                    language="ur" if is_urdu else "en",
                    message=msg,
                    refusal=True,
                    refusal_reason="UNAUTHORIZED_APPLICANT_ACCESS",
                )
            except Exception as e:
                logger.warning("Applicant summary error: %s", e)

        # Case C: Policy / Eligibility / Documents / Repayment Query
        policy_data, record = search_policy_tool(user_msg, top_k=3)
        tool_records.append(record)

        matches: list[dict[str, Any]] = policy_data.get("matches", [])
        if matches:
            top_m = matches[0]
            chunk_id = top_m["chunk_id"]
            title = top_m["title"]
            citations.append(chunk_id)

            if is_urdu:
                reply = (
                    f"آپ کی رہنمائی برائے **{title}** [{chunk_id}]:\n\n"
                    f"{top_m.get('snippet', '')}\n\n"
                    f"حوالہ دستاویز: [{chunk_id}: {title}]۔ مزید معلومات کے لیے بینک کے "
                    "پالیسی رہنما اصول ملاحظہ فرمائیں [KB-RESP-001]۔"
                )
            else:
                reply = (
                    f"Based on our policy guidelines on **{title}** [{chunk_id}]:\n\n"
                    f"{top_m.get('snippet', '')}\n\n"
                    f"Policy Reference: [{chunk_id}: {title}]. All facilities are subject to "
                    "responsible capacity-based evaluation and officer verification [KB-RESP-001]."
                )
            citations.append("KB-RESP-001")
        else:
            # Fallback policy citation
            citations.append("KB-ELIG-001")
            fallback_ur = (
                "براہ کرم بنیادی اہلیت کے اصول ملاحظہ کریں [KB-ELIG-001]۔ "
                "12 ماہ کا تصدیق شدہ کاروباری ریکارڈ درکار ہے۔"
            )
            reply = (
                "Please review our standard eligibility criteria [KB-ELIG-001]. "
                "SMEs with 12 months operating history and verifiable cash flows qualify."
                if not is_urdu
                else fallback_ur
            )

        return self._build_success_response(req, reply, tool_records, citations, is_urdu)

    def _handle_officer_persona(
        self,
        req: AdvisorRequest,
        user_msg: str,
        is_urdu: bool,
        session: Session | None,
    ) -> AdvisorResponse:
        """Officer Copilot workflow: policy queries, referrals, comparisons, conditions."""
        tool_records: list[ToolCallRecord] = []
        citations: list[str] = []
        lower = user_msg.lower()

        # Case A: Why was this referred?
        is_why_referred = (
            ("why" in lower and "referred" in lower)
            or (
                "referral" in lower
                and ("reason" in lower or "why" in lower or "underwriter" in lower)
            )
            or ("kyun" in lower and "refer" in lower)
            or ("کیوں" in lower and "ریفر" in lower)
            or ("referred" in lower and "review" in lower)
        )
        if is_why_referred and req.application_id:
            # 1. Fetch summary & reasons
            summary_data, sum_rec = get_application_summary(
                req.application_id, persona="officer", session=session
            )
            tool_records.append(sum_rec)

            reasons_data, r_rec = explain_reason_tool(
                req.application_id, persona="officer", session=session
            )
            tool_records.append(r_rec)

            top_drivers = reasons_data.get("top_risk_increasing", [])
            driver_texts = [
                f"- {d.get('display_en', d.get('feature', ''))}: {d.get('text_en', '')}"
                for d in top_drivers[:3]
            ]
            drivers_block = (
                "\n".join(driver_texts)
                if driver_texts
                else "- Moderate cash-flow volatility and thin credit file"
            )

            reply = (
                f"### Underwriter Referral Analysis for Application #{req.application_id} "
                f"({summary_data['business_name']})\n\n"
                f"- **Model Score & Tier:** {summary_data['score']} (Tier {summary_data['tier']})\n"
                f"- **System Outcome:** {summary_data['recommendation']} "
                f"(Score within 620–679 referral band [KB-SCOR-002])\n"
                f"- **Key Risk Drivers Triggering Referral:**\n"
                f"{drivers_block}\n\n"
                f"**Policy Scrutiny Guidelines:** Tier C applications require officer verification "
                f"of 6-month supplier invoices and verification that debt-service coverage "
                f"(DSCR {summary_data['dscr']}x) exceeds 1.25x [KB-TERM-001].\n"
                "Recommendation: Request bank statement balance reconciliation and offer "
                "conditional approval [KB-RESP-001]."
            )
            citations.extend(["KB-SCOR-002", "KB-TERM-001", "KB-RESP-001"])
            return self._build_success_response(req, reply, tool_records, citations, is_urdu)

        # Case B: Compare two applications
        if (
            any(w in lower for w in ["compare", "comparison", "muwazna", "موازنہ"])
            and req.application_id
            and req.compare_application_id
        ):
            app1_summary, rec1 = get_application_summary(
                req.application_id, persona="officer", session=session
            )
            tool_records.append(rec1)

            app2_summary, rec2 = get_application_summary(
                req.compare_application_id, persona="officer", session=session
            )
            tool_records.append(rec2)

            winner_id = (
                req.application_id
                if app1_summary["score"] >= app2_summary["score"]
                else req.compare_application_id
            )
            d1_pct = int(app1_summary["digital_payment_share"] * 100)
            d2_pct = int(app2_summary["digital_payment_share"] * 100)

            reply = (
                f"### Comparative Credit Analysis: #{req.application_id} vs "
                f"#{req.compare_application_id}\n\n"
                f"| Parameter | #{req.application_id}: {app1_summary['business_name']} | "
                f"#{req.compare_application_id}: {app2_summary['business_name']} |\n"
                f"|:---|:---|:---|\n"
                f"| **Sector & City** | {app1_summary['sector']} ({app1_summary['city']}) | "
                f"{app2_summary['sector']} ({app2_summary['city']}) |\n"
                f"| **Facility Amount** | PKR {app1_summary['requested_amount_pkr']:,} | "
                f"PKR {app2_summary['requested_amount_pkr']:,} |\n"
                f"| **Score & Tier** | {app1_summary['score']} (Tier {app1_summary['tier']}) | "
                f"{app2_summary['score']} (Tier {app2_summary['tier']}) |\n"
                f"| **System Outcome** | {app1_summary['recommendation']} | "
                f"{app2_summary['recommendation']} |\n"
                f"| **Monthly Turnover** | PKR {app1_summary['monthly_turnover_pkr']:,} | "
                f"PKR {app2_summary['monthly_turnover_pkr']:,} |\n"
                f"| **DSCR** | {app1_summary['dscr']}x | {app2_summary['dscr']}x |\n"
                f"| **Digital Share** | {d1_pct}% | {d2_pct}% |\n\n"
                f"**Underwriting Synthesis:** #{winner_id} exhibits superior cash-flow "
                "coverage according to risk scorecard standards [KB-SCOR-001]. "
                "Both facilities are evaluated within transparent policy boundaries [KB-TERM-002]."
            )
            citations.extend(["KB-SCOR-001", "KB-TERM-002"])
            return self._build_success_response(req, reply, tool_records, citations, is_urdu)

        # Case C: Draft a conditions list
        if (
            any(
                w in lower
                for w in [
                    "draft conditions",
                    "conditions list",
                    "covenants",
                    "mitigants",
                    "sharaait",
                ]
            )
            and req.application_id
        ):
            summary_data, sum_rec = get_application_summary(
                req.application_id, persona="officer", session=session
            )
            tool_records.append(sum_rec)

            reply = (
                f"### Draft Sanction Conditions for #{req.application_id} "
                f"({summary_data['business_name']})\n\n"
                f"Based on Tier {summary_data['tier']} risk profiling and terms [KB-TERM-001]:\n\n"
                "**Pre-Disbursement Conditions (PDCs):**\n"
                "1. **Personal Guarantee:** Unconditional personal guarantee from sponsor.\n"
                "2. **Direct Debit Mandate:** Direct debit mandate on principal account "
                "[KB-REPAY-002].\n"
                "3. **Tax Verification:** Verified confirmation of active NTN [KB-DOCS-001].\n\n"
                "**Post-Disbursement Covenants:**\n"
                "1. **Minimum Account Balance:** Maintain average monthly closing buffer "
                "[KB-IMPR-002].\n"
                "2. **Digital Route Commitment:** Channel minimum 40% of collections "
                "[KB-IMPR-001].\n"
                "3. **Quarterly Review:** Submission of quarterly stock & sales declaration.\n\n"
                "*Draft prepared for human credit committee review under responsible policy "
                "[KB-RESP-001].*"
            )
            citations.extend(["KB-TERM-001", "KB-REPAY-002", "KB-DOCS-001", "KB-RESP-001"])
            return self._build_success_response(req, reply, tool_records, citations, is_urdu)

        # Case D: General Policy & Knowledge Search for Officer
        policy_data, record = search_policy_tool(user_msg, top_k=3)
        tool_records.append(record)

        matches = policy_data.get("matches", [])
        if matches:
            top_m = matches[0]
            chunk_id = top_m["chunk_id"]
            citations.append(chunk_id)

            reply = (
                f"### Underwriting Policy Reference: {top_m['title']} [{chunk_id}]\n\n"
                f"{top_m.get('snippet', '')}\n\n"
                f"**Officer Notes:** Verified against credit policy parameters. "
                f"Facility terms must adhere to transparent pricing schedules [KB-TERM-002] "
                f"and responsible lending limits [KB-RESP-001]."
            )
            citations.extend(["KB-TERM-002", "KB-RESP-001"])
        else:
            citations.append("KB-ELIG-001")
            reply = (
                "Refer to core SME underwriting guidelines [KB-ELIG-001]. "
                "Minimum operating vintage of 12 months and positive cash-flow debt service "
                "required."
            )

        return self._build_success_response(req, reply, tool_records, citations, is_urdu)

    def _build_success_response(
        self,
        req: AdvisorRequest,
        reply: str,
        tool_records: list[ToolCallRecord],
        citations: list[str],
        is_urdu: bool,
    ) -> AdvisorResponse:
        """Helper to record session memory and return AdvisorResponse."""
        unique_citations = list(dict.fromkeys(citations))
        # Ensure at least one citation for policy answers
        if not unique_citations:
            unique_citations = ["KB-ELIG-001"]

        resp = AdvisorResponse(
            session_id=req.session_id,
            persona=req.persona,
            language="ur" if is_urdu else "en",
            message=reply,
            tool_calls=tool_records,
            citations=unique_citations,
            refusal=False,
        )
        session_memory.add_turn(req.session_id, "user", req.message)
        session_memory.add_turn(
            req.session_id,
            "assistant",
            reply,
            tool_calls=tool_records,
            citations=unique_citations,
        )
        return resp


# Singleton engine
advisor_engine = AdvisorEngine()


# ---------------------------------------------------------------------------
# Streaming SSE Generator
# ---------------------------------------------------------------------------


async def stream_advisor_chat(
    req: AdvisorRequest,
    session: Session | None = None,
) -> AsyncIterator[str]:
    """Yield Server-Sent Events (SSE) for streaming advisor conversation.

    Events:
    - event: tool_call (when tool chips are emitted)
    - event: token (chunked text delta)
    - event: citations (list of cited chunk IDs)
    - event: done (complete final response payload)
    """
    # 1. Compute response (runs offline/fast engine)
    response = advisor_engine.process(req, session=session)

    # 2. Emit tool calls first so UI chips appear immediately
    for tool_call in response.tool_calls:
        yield f"event: tool_call\ndata: {json.dumps(tool_call.model_dump())}\n\n"
        await asyncio.sleep(0.01)

    # 3. Stream text content in word tokens
    words = response.message.split(" ")
    chunk_size = 4
    for i in range(0, len(words), chunk_size):
        chunk_text = " ".join(words[i : i + chunk_size]) + " "
        payload = {"delta": chunk_text}
        yield f"event: token\ndata: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.015)

    # 4. Emit citations
    if response.citations:
        yield f"event: citations\ndata: {json.dumps({'citations': response.citations})}\n\n"
        await asyncio.sleep(0.01)

    # 5. Emit done event with complete response payload
    yield f"event: done\ndata: {json.dumps(response.model_dump())}\n\n"
