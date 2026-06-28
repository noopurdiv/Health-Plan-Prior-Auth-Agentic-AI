# ClearAuth LangGraph Agent Interaction

Visual reference for how the prior authorization agent runs, pauses, and resumes.

## Diagram (SVG)

![LangGraph agent interaction diagram](./langgraph_agent_diagram.svg)

## Mermaid source

```mermaid
flowchart TB
    subgraph Input["Prior Auth Request"]
        REQ["Patient · ICD-10 · CPT · Procedure\nClinical notes · Urgency"]
    end

    subgraph Graph["LangGraph Agent (MemorySaver checkpointer)"]
        direction TB
        P["① parse_request\nBuild retrieval query\nReasoning step 1"]
        R["② retrieve_policy\nChromaDB vector search\nPolicy citations · Step 2"]
        RE["③ reason_and_decide\nClaude JSON reasoning\nAPPROVE / FLAG / ESCALATE · Step 3"]
        INT{{"⏸ INTERRUPT\ninterrupt_before: log_decision"}}
        L["④ log_decision\nGraph completion node"]
    end

    subgraph External["External Systems"]
        CDB[(ChromaDB\nPolicy PDF chunks)]
        LLM[Claude API\nclaude-sonnet-4-6]
        SQL[(SQLite\nagent_analyses)]
    end

    subgraph HITL["Human-in-the-Loop"]
        UI[Reviewer Dashboard]
        H["Human decision\nApprove · Reject · Escalate"]
        API["POST /api/decision"]
    end

    REQ --> P
    P --> R
    R --> CDB
    CDB --> R
    R -->|no policy found| RE
    R -->|citations found| RE
    RE --> LLM
    LLM --> RE
    RE --> INT
    INT -->|analysis cached| SQL
    INT --> UI
    UI --> H
    H --> API
    API --> SQL
    API -.->|resume graph\nbackground thread| L
    L --> END_NODE([END])

    style INT fill:#EDE9FE,stroke:#6B46C1,stroke-width:2px
    style H fill:#6B46C1,stroke:#553C9A,color:#fff
    style RE fill:#FEF3C7,stroke:#6B46C1
```

## Node responsibilities

| Node | Reads from state | Writes to state |
|------|----------------|-----------------|
| `parse_request` | Patient request fields | `parsed_query`, reasoning step 1 |
| `retrieve_policy` | `parsed_query` | `policy_citations`, reasoning step 2; may set `error=NO_POLICY_FOUND` |
| `reason_and_decide` | Request + citations | `ai_decision`, `confidence_score`, reasoning step 3 |
| `log_decision` | Full state | (no-op; graph completion only) |

## Interrupt and resume

1. `agent_graph.invoke(initial_state, config)` runs until the interrupt before `log_decision`.
2. FastAPI persists the analysis to `agent_analyses` and stores the thread config in `pending_states`.
3. The dashboard shows the recommendation; the human submits a decision via `POST /api/decision`.
4. Human decision is written to `human_decisions` immediately (fast response).
5. A background thread calls `agent_graph.invoke(None, config)` to resume and complete `log_decision`.

## Business rules (outside graph)

Applied in `main.py` after the agent returns:

- **Emergent** urgency → force `ESCALATE`
- **Confidence &lt; 0.60** → force `ESCALATE`
- **Reject** requires a written reason
