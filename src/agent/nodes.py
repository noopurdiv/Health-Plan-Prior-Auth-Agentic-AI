import json
import os
import re
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv

from src.agent.state import AgentState, PolicyCitation, ReasoningStep
from src.rag.retriever import retrieve_policy

load_dotenv()

CONFIDENCE_THRESHOLD = 0.60
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SYSTEM_PROMPT = """You are a clinical prior authorization reasoning agent for a health plan.
Analyze the patient request ONLY against the provided policy excerpts and respond in JSON only.

Your response must be valid JSON with exactly these keys:
- "decision": one of "APPROVE", "FLAG", or "ESCALATE"
- "confidence": a float between 0.0 and 1.0 reflecting how well the policy excerpts support your decision
- "reasoning_steps": an array of objects with "label" and "detail" strings

Decision definitions:
- APPROVE: patient clearly meets all coverage criteria stated in the excerpts
- FLAG: patient may meet criteria but clinical review is recommended
- ESCALATE: insufficient evidence, patient does not meet criteria, or policy not found in excerpts

ANTI-HALLUCINATION RULES (mandatory):
- Base every statement ONLY on the retrieved policy excerpts provided. Do NOT invent policy text, section numbers, or criteria.
- Quote or paraphrase only what appears in the excerpts. If a criterion is not in the excerpts, do NOT assume it exists.
- If you cannot determine whether criteria are met from the excerpts alone, use the word "undetectable" in your reasoning detail and set decision to ESCALATE with confidence below 0.60.
- If age, diagnosis, or procedure coverage cannot be confirmed from excerpts, say "undetectable" for that criterion.
- Never fabricate NCD section numbers, page references, or coverage rules.

Confidence rules:
- confidence 0.85-1.0: clear explicit match in policy excerpts
- confidence 0.60-0.84: partial match, some criteria undetectable or ambiguous
- confidence below 0.60: always output decision "ESCALATE"
- If any key criterion is "undetectable", confidence must be below 0.60

Other rules:
- Cite the specific policy source name from the excerpts for each reasoning step
- If clinical notes are empty, note the absence of clinical context
- Be concise and use language understandable to a nurse reviewer
"""


def _sex_label(sex: str) -> str:
    s = sex.strip().upper()
    if s in ("F", "FEMALE"):
        return "female"
    if s in ("M", "MALE"):
        return "male"
    return sex.lower()


def _avg_retrieval_score(citations: list[PolicyCitation]) -> float:
    if not citations:
        return 0.0
    return sum(c["relevance_score"] for c in citations) / len(citations)


def _blend_confidence(llm_confidence: float, citations: list[PolicyCitation]) -> float:
    """Blend LLM confidence with retrieval relevance for a more realistic score."""
    retrieval = _avg_retrieval_score(citations)
    blended = 0.65 * llm_confidence + 0.35 * retrieval
    return round(min(1.0, max(0.0, blended)), 4)


def _retrieval_fallback_confidence(citations: list[PolicyCitation]) -> float:
    """When LLM fails but policy was retrieved, use retrieval score as fallback."""
    if not citations:
        return 0.0
    return round(min(0.72, _avg_retrieval_score(citations) * 0.85), 4)


def parse_request(state: AgentState) -> dict[str, Any]:
    sex = _sex_label(state["patient_sex"])
    notes_part = (
        state["clinical_notes"].strip()
        if state.get("clinical_notes", "").strip()
        else "no clinical notes provided"
    )

    parsed_query = (
        f"{state['patient_age']} year old {sex}, "
        f"{state['diagnosis_label']}, "
        f"{state['procedure_label']}, "
        f"ICD-10 {state['diagnosis_code']}, "
        f"{state['urgency']}, {notes_part}"
    )

    step: ReasoningStep = {
        "step": 1,
        "label": "Parsing Request",
        "detail": (
            f"Structured request for a {state['patient_age']}-year-old {sex} "
            f"seeking {state['procedure_label']} ({state['procedure_code'] or 'no CPT'}) "
            f"for {state['diagnosis_label']} (ICD-10: {state['diagnosis_code']}). "
            f"Urgency: {state['urgency']}. "
            f"{'Clinical notes provided.' if state.get('clinical_notes', '').strip() else 'No clinical notes provided — limited clinical context available.'}"
        ),
    }

    return {
        "parsed_query": parsed_query,
        "reasoning_steps": [step],
        "reasoning_trace": [step],
    }


def retrieve_policy_node(state: AgentState) -> dict[str, Any]:
    results = retrieve_policy(state["parsed_query"], n_results=5)
    filtered = [r for r in results if r["relevance_score"] >= 0.35]

    citations: list[PolicyCitation] = [
        {
            "source": r["source"],
            "source_file": r.get("source_file", ""),
            "page_number": r.get("page_number"),
            "text": r["text"],
            "relevance_score": r["relevance_score"],
        }
        for r in filtered
    ]

    if not citations:
        step: ReasoningStep = {
            "step": 2,
            "label": "Retrieving Policy",
            "detail": (
                "No matching coverage policy found in the knowledge base. "
                "Request automatically escalated for manual review."
            ),
        }
        existing = list(state.get("reasoning_steps", []))
        existing.append(step)
        return {
            "policy_citations": [],
            "reasoning_steps": existing,
            "reasoning_trace": existing,
            "error": "NO_POLICY_FOUND",
            "ai_decision": "ESCALATE",
            "confidence_score": 0.0,
        }

    sources = ", ".join(c["source"] for c in citations[:3])
    step = {
        "step": 2,
        "label": "Retrieving Policy",
        "detail": (
            f"Retrieved {len(citations)} relevant policy section(s) from the knowledge base, "
            f"including: {sources}."
        ),
    }
    existing = list(state.get("reasoning_steps", []))
    existing.append(step)

    return {
        "policy_citations": citations,
        "reasoning_steps": existing,
        "reasoning_trace": existing,
    }


def reason_and_decide(state: AgentState) -> dict[str, Any]:
    if state.get("error") == "NO_POLICY_FOUND":
        return {}

    citations = state.get("policy_citations", [])
    policy_blocks = []
    for i, citation in enumerate(citations, start=1):
        page = citation.get("page_number")
        page_str = f", page {page}" if page else ""
        policy_blocks.append(
            f"[Policy {i}] Source: {citation['source']}{page_str}\n{citation['text']}"
        )

    user_prompt = f"""Patient Request:
- Age: {state['patient_age']}
- Sex: {state['patient_sex']}
- Diagnosis: {state['diagnosis_label']} ({state['diagnosis_code']})
- Procedure: {state['procedure_label']} ({state['procedure_code'] or 'N/A'})
- Urgency: {state['urgency']}
- Clinical Notes: {state.get('clinical_notes') or '(none provided)'}

Retrieved Policy Excerpts (ONLY use these — do not reference anything else):
{chr(10).join(policy_blocks)}

Provide your JSON decision."""

    raw_response = ""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_key_here":
        step = {
            "step": 3,
            "label": "Applying Coverage Criteria",
            "detail": (
                "API key not configured. Policy excerpts were retrieved but automated reasoning "
                "could not run. Criteria match: undetectable without LLM analysis. Escalating for manual review."
            ),
        }
        existing = list(state.get("reasoning_steps", []))
        existing.append(step)
        fallback = _retrieval_fallback_confidence(citations)
        return {
            "ai_decision": "ESCALATE",
            "confidence_score": fallback,
            "reasoning_steps": existing,
            "reasoning_trace": existing,
            "raw_llm_response": None,
            "error": "NO_API_KEY",
        }

    try:
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw_response = message.content[0].text if message.content else ""
        parsed = _parse_llm_json(raw_response)
        decision = str(parsed.get("decision", "ESCALATE")).upper()
        confidence = float(parsed.get("confidence", 0.0))
        llm_steps = parsed.get("reasoning_steps", [])

        if decision not in ("APPROVE", "FLAG", "ESCALATE"):
            decision = "ESCALATE"
            confidence = min(confidence, 0.5)

        confidence = _blend_confidence(confidence, citations)

        if confidence < CONFIDENCE_THRESHOLD:
            decision = "ESCALATE"

        step_detail_parts = []
        if isinstance(llm_steps, list):
            for item in llm_steps:
                if isinstance(item, dict):
                    step_detail_parts.append(item.get("detail", ""))

        detail = " ".join(step_detail_parts) if step_detail_parts else (
            f"Applied coverage criteria against retrieved policies. Decision: {decision} "
            f"with confidence {confidence:.0%}."
        )

        step: ReasoningStep = {
            "step": 3,
            "label": "Applying Coverage Criteria",
            "detail": detail,
        }
        existing = list(state.get("reasoning_steps", []))
        existing.append(step)

        return {
            "ai_decision": decision,
            "confidence_score": confidence,
            "reasoning_steps": existing,
            "reasoning_trace": existing,
            "raw_llm_response": raw_response,
        }
    except Exception as exc:
        fallback = _retrieval_fallback_confidence(citations)
        step = {
            "step": 3,
            "label": "Applying Coverage Criteria",
            "detail": (
                f"Automated reasoning encountered an error ({exc}). "
                f"Retrieved policy excerpts are available for manual review. "
                f"Coverage criteria match: undetectable by AI — escalated for human review."
            ),
        }
        existing = list(state.get("reasoning_steps", []))
        existing.append(step)
        return {
            "ai_decision": "ESCALATE",
            "confidence_score": fallback,
            "reasoning_steps": existing,
            "reasoning_trace": existing,
            "raw_llm_response": raw_response,
            "error": str(exc),
        }


def log_decision(state: AgentState) -> dict[str, Any]:
    return {}


def _parse_llm_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)
    return json.loads(text)
