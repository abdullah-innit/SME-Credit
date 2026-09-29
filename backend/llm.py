"""Claude-powered steps: the credit committee and ledger extraction.

Both work without an API key (committee falls back to a templated summary) so the
demo never dies on stage. Set ANTHROPIC_API_KEY to turn the real thing on.
"""
import base64
import json
import os
import re

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")


def _client():
    key = os.getenv("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=key) if (key and anthropic) else None


def _parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


COMMITTEE_SYSTEM = """You simulate a small SME credit committee in Pakistan with three members:
- underwriter: argues the strongest honest case FOR lending, citing the supplied numbers.
- skeptic: argues the strongest honest case AGAINST, citing the supplied numbers.
- compliance: checks data-quality and policy red flags (for example a revenue gap between declared and bank-observed).
Use ONLY the facts supplied. Never invent figures. Keep each member to 2 or 3 sentences.
Then give the committee verdict. Return ONLY JSON with this shape:
{"underwriter": str, "skeptic": str, "compliance": str,
 "verdict": {"decision": "approve"|"refer"|"decline", "conditions": [str], "summary": str}}"""


def committee(payload: dict) -> dict:
    client = _client()
    if client:
        try:
            msg = client.messages.create(
                model=MODEL, max_tokens=900, system=COMMITTEE_SYSTEM,
                messages=[{"role": "user", "content": json.dumps(payload)}])
            out = _parse_json(msg.content[0].text)
            out["source"] = "claude"
            return out
        except Exception as e:  # fall through to the offline version
            err = str(e)
    else:
        err = "no ANTHROPIC_API_KEY set"
    a = payload["assessment"]
    helps = [f["label"] for f in a["factors"] if f["impact_pts"] < 0][:2]
    hurts = [f["label"] for f in a["factors"] if f["impact_pts"] > 0][:2]
    return {
        "underwriter": f"Strengths: {', '.join(helps) or 'none stand out'}.",
        "skeptic": f"Concerns: {', '.join(hurts) or 'none stand out'}. Estimated default risk is {a['pd'] * 100:.0f}%.",
        "compliance": "Offline mode: no automated policy review was run.",
        "verdict": {"decision": a["decision"], "conditions": [], "summary": "Score-based decision (offline fallback)."},
        "source": f"offline ({err})",
    }


EXTRACT_PROMPT = """This is a photo or scan of a small business's ledger, sales register, or bank/wallet statement.
Extract what you can read. Return ONLY JSON:
{"monthly_sales": [numbers in PKR, oldest first, up to 12],
 "digital_share_estimate": number 0-1 or null,
 "top_customer_share": number 0-1 or null,
 "notes": "one sentence on legibility or anything unclear"}
Use null when a field is not visible. Never guess numbers you cannot read."""


def extract_from_image(data: bytes, media_type: str) -> dict:
    client = _client()
    if not client:
        raise RuntimeError("Set ANTHROPIC_API_KEY to enable ledger extraction.")
    msg = client.messages.create(
        model=MODEL, max_tokens=800,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                         "data": base64.b64encode(data).decode()}},
            {"type": "text", "text": EXTRACT_PROMPT}]}])
    raw = _parse_json(msg.content[0].text)
    sales = [float(x) for x in (raw.get("monthly_sales") or []) if x]
    out = {"notes": raw.get("notes"), "months_read": len(sales)}
    if len(sales) >= 3:
        mean = sum(sales) / len(sales)
        var = sum((s - mean) ** 2 for s in sales) / len(sales)
        first, last = sales[:3], sales[-3:]
        out["monthly_revenue"] = round(mean, -3)
        out["revenue_volatility"] = round((var ** 0.5) / mean, 3)
        out["revenue_growth_6m"] = round(sum(last) / sum(first) - 1, 3)
    if raw.get("digital_share_estimate") is not None:
        out["digital_payment_share"] = raw["digital_share_estimate"]
    if raw.get("top_customer_share") is not None:
        out["customer_concentration"] = raw["top_customer_share"]
    return out
