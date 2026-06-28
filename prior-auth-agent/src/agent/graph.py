from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agent.nodes import log_decision, parse_request, reason_and_decide, retrieve_policy_node
from src.agent.state import AgentState


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("parse_request", parse_request)
    builder.add_node("retrieve_policy", retrieve_policy_node)
    builder.add_node("reason_and_decide", reason_and_decide)
    builder.add_node("log_decision", log_decision)

    builder.set_entry_point("parse_request")
    builder.add_edge("parse_request", "retrieve_policy")
    builder.add_edge("retrieve_policy", "reason_and_decide")
    builder.add_edge("reason_and_decide", "log_decision")
    builder.add_edge("log_decision", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["log_decision"])
