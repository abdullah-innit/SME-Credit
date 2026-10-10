"""Acceptance tests for Prompt 12: Policy-grounded advisor with tools.

Verifies:
- Mock-provider / deterministic tool routing and refusals.
- Off-topic question is refused politely.
- 'Guarantee me approval' request is refused politely.
- Every policy answer cites at least one chunk ID.
- Applicant cannot read another applicant's data (authorization test).
- Officer copilot workflows: why was this referred, compare two apps, draft conditions.
- Bilingual (Urdu & English) responses.
- Streaming SSE endpoint.
- Session-scoped memory.
- PII-safe logging masking.
"""

from fastapi.testclient import TestClient

from app.knowledge.retrieval import get_knowledge_retriever
from app.main import app
from app.services.advisor import mask_pii

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test 1 & 2: Tool Routing and UI Tool Chips
# ---------------------------------------------------------------------------


def test_tool_routing_applicant_summary():
    """Verify applicant assistant routes get_application_summary and returns UI tool chip."""
    payload = {
        "message": "Can you give me a summary of my application status and score?",
        "session_id": "test-summary-session",
        "persona": "applicant",
        "application_id": 1,
        "user_id": 1,  # Owner of Application 1 (Ayesha)
        "language": "en",
    }
    res = client.post("/api/v1/advisor/chat", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert not data["refusal"]
    assert "tool_calls" in data
    assert len(data["tool_calls"]) >= 1

    summary_call = next(
        (tc for tc in data["tool_calls"] if tc["tool"] == "get_application_summary"), None
    )
    assert summary_call is not None, "get_application_summary tool was not routed"
    assert "summary" in summary_call
    assert "Ayesha" in summary_call["summary"] or "Application" in summary_call["summary"]

    # Verify response text contains score/tier facts
    assert "score" in data["message"].lower()
    assert len(data["citations"]) >= 1


def test_tool_routing_what_if():
    """Verify what-if simulation tool routing for score improvement."""
    payload = {
        "message": (
            "What if I improve my digital payments? "
            "How can I improve my score on the path to approval?"
        ),
        "session_id": "test-whatif-session",
        "persona": "applicant",
        "application_id": 1,
        "user_id": 1,
        "language": "en",
    }
    res = client.post("/api/v1/advisor/chat", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert not data["refusal"]
    whatif_call = next((tc for tc in data["tool_calls"] if tc["tool"] == "run_what_if"), None)
    assert whatif_call is not None, "run_what_if tool was not routed"
    assert "summary" in whatif_call
    assert any("KB-IMPR" in c for c in data["citations"])


# ---------------------------------------------------------------------------
# Test 3 & 4: Polite Refusals (Approval Guarantee & Off-Topic)
# ---------------------------------------------------------------------------


def test_polite_refusal_guarantee_approval():
    """Verify polite refusal when user asks for approval guarantee or loan promise."""
    guarantee_prompts = [
        "Guarantee me approval for this loan right now",
        "Can you approve my loan and promise I will get the money?",
        "Will you promise me loan approval?",
    ]
    for prompt in guarantee_prompts:
        payload = {
            "message": prompt,
            "session_id": "test-guarantee-session",
            "persona": "applicant",
            "application_id": 1,
            "user_id": 1,
            "language": "en",
        }
        res = client.post("/api/v1/advisor/chat", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()

        assert data["refusal"] is True
        assert data["refusal_reason"] == "APPROVAL_PROMISE_REFUSAL"
        # Must make clear AI cannot approve or promise loans
        assert (
            "cannot approve" in data["message"].lower()
            or "human credit committee" in data["message"].lower()
        )


def test_polite_refusal_off_topic():
    """Verify polite refusal when user asks unrelated / off-topic queries."""
    off_topic_queries = [
        "What is the best recipe for chicken biryani?",
        "Can you write a poem about the weather in Lahore?",
        "Give me some good cryptocurrency and stock trading tips",
    ]
    for query in off_topic_queries:
        payload = {
            "message": query,
            "session_id": "test-offtopic-session",
            "persona": "applicant",
            "language": "en",
        }
        res = client.post("/api/v1/advisor/chat", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()

        assert data["refusal"] is True
        assert data["refusal_reason"] == "OUT_OF_SCOPE"
        assert (
            "specialize exclusively in sme credit" in data["message"].lower()
            or "unrelated" in data["message"].lower()
        )


# ---------------------------------------------------------------------------
# Test 5: Every Policy Answer Cites at Least One Chunk ID
# ---------------------------------------------------------------------------


def test_every_policy_answer_cites_chunk_id():
    """Verify every policy answer cites at least one chunk ID."""
    policy_questions = [
        "What are the core eligibility criteria for an SME loan?",
        "Which documents do I need to submit with my application?",
        "What are the loan limits, pricing bands and repayment tenors?",
        "Can you share repayment tips and how to maintain cash flow?",
        "What is your responsible lending policy for borrowers?",
    ]
    for q in policy_questions:
        payload = {
            "message": q,
            "session_id": "test-citations-session",
            "persona": "applicant",
            "language": "en",
        }
        res = client.post("/api/v1/advisor/chat", json=payload)
        assert res.status_code == 200, res.text
        data = res.json()

        assert not data["refusal"]
        # Mandatory acceptance criterion: every policy answer cites at least one chunk ID
        assert len(data["citations"]) >= 1, f"No citations for question: '{q}'"
        first_citation = data["citations"][0]
        assert first_citation.startswith("KB-"), f"Invalid chunk citation format: {first_citation}"
        assert first_citation in data["message"], (
            f"Chunk citation {first_citation} missing from message text"
        )


# ---------------------------------------------------------------------------
# Test 6: Applicant Authorization Test (Cannot Read Another Applicant's Data)
# ---------------------------------------------------------------------------


def test_applicant_cannot_read_another_applicant_data():
    """Verify applicant user cannot access another applicant's application data."""
    # User 1 owns Application 1 (Ayesha). Application 2 is owned by User 2 (Karim).
    payload = {
        "message": "Give me the summary and score for application 2",
        "session_id": "test-auth-session",
        "persona": "applicant",
        "application_id": 2,  # Target is Application 2
        "user_id": 1,  # Authenticated as User 1
        "language": "en",
    }
    res = client.post("/api/v1/advisor/chat", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    # Must refuse or reject unauthorized access
    assert data["refusal"] is True
    assert data["refusal_reason"] == "UNAUTHORIZED_APPLICANT_ACCESS"
    assert "unauthorized" in data["message"].lower() or "restricted" in data["message"].lower()
    # Confirm Karim's data was NOT returned
    assert "Karim" not in data["message"]


# ---------------------------------------------------------------------------
# Test 7, 8, 9: Officer Copilot Persona Workflows
# ---------------------------------------------------------------------------


def test_officer_copilot_why_referred():
    """Verify officer copilot explains why an application was referred."""
    payload = {
        "message": "Why was this application referred for underwriter review?",
        "session_id": "test-officer-referral",
        "persona": "officer",
        "application_id": 1,
        "language": "en",
    }
    res = client.post("/api/v1/advisor/chat", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert not data["refusal"]
    assert len(data["tool_calls"]) >= 2  # get_application_summary & explain_reason
    assert "referral" in data["message"].lower() or "referred" in data["message"].lower()
    assert any("KB-SCOR-002" in c for c in data["citations"])


def test_officer_copilot_compare_two_applications():
    """Verify officer copilot compares two applications side by side."""
    payload = {
        "message": "Compare these two applications side by side for the credit committee",
        "session_id": "test-officer-compare",
        "persona": "officer",
        "application_id": 1,
        "compare_application_id": 2,
        "language": "en",
    }
    res = client.post("/api/v1/advisor/chat", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert not data["refusal"]
    assert len(data["tool_calls"]) >= 2
    # Verify table format in output comparing both businesses
    assert "Parameter" in data["message"]
    assert "Ayesha" in data["message"]
    assert "Karim" in data["message"]


def test_officer_copilot_draft_conditions_list():
    """Verify officer copilot drafts a condition covenants list."""
    payload = {
        "message": "Draft a conditions list and covenants for this working capital facility",
        "session_id": "test-officer-conditions",
        "persona": "officer",
        "application_id": 1,
        "language": "en",
    }
    res = client.post("/api/v1/advisor/chat", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert not data["refusal"]
    assert "Pre-Disbursement Conditions" in data["message"]
    assert "Personal Guarantee" in data["message"]
    assert len(data["citations"]) >= 1


# ---------------------------------------------------------------------------
# Test 10: Bilingual Urdu Support
# ---------------------------------------------------------------------------


def test_bilingual_urdu_response():
    """Verify advisor responds in Urdu when queried in Urdu."""
    payload = {
        "message": "قرض حاصل کرنے کے لیے بنیادی اہلیت اور شرائط کیا ہیں؟",
        "session_id": "test-urdu-session",
        "persona": "applicant",
        "language": "ur",
    }
    res = client.post("/api/v1/advisor/chat", json=payload)
    assert res.status_code == 200, res.text
    data = res.json()

    assert data["language"] == "ur"
    assert any(c in data["citations"] for c in ["KB-ELIG-001", "KB-TERM-001"])
    # Check for Urdu characters in response
    assert any("\u0600" <= ch <= "\u06ff" for ch in data["message"])


# ---------------------------------------------------------------------------
# Test 11: SSE Streaming Endpoint
# ---------------------------------------------------------------------------


def test_sse_streaming_endpoint():
    """Verify Server-Sent Events (SSE) stream emits tool_call, token, citations, and done."""
    payload = {
        "message": "What is the policy regarding early settlement and prepayment penalties?",
        "session_id": "test-sse-session",
        "persona": "applicant",
        "language": "en",
    }
    res = client.post("/api/v1/advisor/chat/stream", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    body = res.text
    assert "event: token" in body
    assert "event: done" in body
    assert "KB-TERM-003" in body or "KB-" in body


# ---------------------------------------------------------------------------
# Test 12: Session-Scoped Memory & History API
# ---------------------------------------------------------------------------


def test_session_scoped_memory_and_clear():
    """Verify conversation history persists across turns in session and clears on request."""
    session_id = "mem-test-session-123"

    # Turn 1
    client.post(
        "/api/v1/advisor/chat",
        json={"message": "What are required documents?", "session_id": session_id},
    )
    # Turn 2
    client.post(
        "/api/v1/advisor/chat",
        json={"message": "Can thin-file SMEs apply?", "session_id": session_id},
    )

    # Get history
    h_res = client.get(f"/api/v1/advisor/history/{session_id}")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert h_data["turn_count"] == 4  # 2 user turns + 2 assistant turns

    # Clear history
    del_res = client.delete(f"/api/v1/advisor/history/{session_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "cleared"

    h_res2 = client.get(f"/api/v1/advisor/history/{session_id}")
    assert h_res2.json()["turn_count"] == 0


# ---------------------------------------------------------------------------
# Test 13: PII-Safe Logging Redaction
# ---------------------------------------------------------------------------


def test_pii_safe_logging():
    """Verify mask_pii removes CNIC, phone, credit card, IBAN, and secret keys."""
    raw_text = (
        "Customer CNIC 42101-1234567-1, phone 0300-1234567, card 4111-2222-3333-4444, "
        "IBAN PK36SCBL0000001123456701, and API key sk-ant-api03-abcdefghijklmnop"
    )
    masked = mask_pii(raw_text)

    assert "42101-1234567-1" not in masked
    assert "[CNIC-MASKED]" in masked
    assert "0300-1234567" not in masked
    assert "[PHONE-MASKED]" in masked
    assert "4111-2222-3333-4444" not in masked
    assert "[CARD-MASKED]" in masked
    assert "PK36SCBL0000001123456701" not in masked
    assert "[IBAN-MASKED]" in masked
    assert "sk-ant-api03-abcdefghijklmnop" not in masked
    assert "[API-KEY-MASKED]" in masked


# ---------------------------------------------------------------------------
# Test 14: Knowledge Retrieval API & Embeddings Flag Toggle
# ---------------------------------------------------------------------------


def test_knowledge_endpoint_and_embeddings_toggle():
    """Verify /knowledge route and retriever embeddings flag."""
    # List all chunks
    res = client.get("/api/v1/advisor/knowledge")
    assert res.status_code == 200
    data = res.json()
    assert data["count"] >= 20

    # Search knowledge
    s_res = client.get("/api/v1/advisor/knowledge?query=responsible+lending")
    assert s_res.status_code == 200
    s_data = s_res.json()
    assert s_data["count"] >= 1
    assert any("KB-RESP" in r["chunk_id"] for r in s_data["results"])

    # Test retriever embeddings flag toggle
    retriever_bm25 = get_knowledge_retriever(use_embeddings=False)
    matches_bm25 = retriever_bm25.search("working capital terms", top_k=2)
    assert len(matches_bm25) >= 1

    retriever_emb = get_knowledge_retriever(use_embeddings=True)
    matches_emb = retriever_emb.search("working capital terms", top_k=2)
    assert len(matches_emb) >= 1
