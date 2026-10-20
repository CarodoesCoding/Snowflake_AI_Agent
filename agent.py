from typing import TypedDict
from langgraph.graph import StateGraph, END
import time
from schema import get_schema
from database import execute_sql
from llm import (
    generate_sql,
    fix_sql,
    validate_response,
    generate_nl_answer,
    get_cost_summary,
    reset_cost_tracker,
)

MAX_RETRIES = 3


# State
class AgentState(TypedDict):
    question: str
    schema: str
    sql: str
    columns: list
    rows: list
    error: str | None
    answer: str
    retries: int
    validation_issue: str | None
    validation_type: str | None
    logs: list


# Nodes
def node_generate_sql(state: AgentState) -> AgentState:
    t0 = time.time()
    sql = generate_sql(state["question"], state["schema"])
    log_entry = {"node": "generate_sql", "time": round(time.time() - t0, 2)}
    return {**state, "sql": sql, "error": None, "logs": state["logs"] + [log_entry]}


def node_execute_sql(state: AgentState) -> AgentState:
    t0 = time.time()
    result = execute_sql(state["sql"])
    log_entry = {"node": "execute_sql", "time": round(time.time() - t0, 2)}
    if result["success"]:
        return {
            **state,
            "columns": result["columns"],
            "rows": result["rows"],
            "error": None,
            "logs": state["logs"] + [log_entry],
        }
    return {
        **state,
        "columns": [],
        "rows": [],
        "error": result["error"],
        "logs": state["logs"] + [log_entry],
    }


def node_fix_sql(state: AgentState) -> AgentState:
    t0 = time.time()
    fixed_sql = fix_sql(
        state["schema"], state["question"], state["sql"], state["error"]
    )
    log_entry = {
        "node": f"fix_sql (retry {state['retries'] + 1})",
        "time": round(time.time() - t0, 2),
    }
    return {
        **state,
        "sql": fixed_sql,
        "error": None,
        "retries": state["retries"] + 1,
        "logs": state["logs"] + [log_entry],
    }


def node_validate_response(state: AgentState) -> AgentState:
    t0 = time.time()
    result = validate_response(state["columns"], state["rows"])
    log_entry = {"node": "validate_response", "time": round(time.time() - t0, 2)}
    if result["ok"]:
        return {**state, "validation_issue": None, "logs": state["logs"] + [log_entry]}
    # error nur bei wrong_type gesetzt (echtes SQL-Problem),
    # nicht bei empty/all_null
    is_sql_issue = result["type"] == "wrong_type"
    return {
        **state,
        "validation_issue": result["issue"],
        "validation_type": result["type"],
        "error": result["issue"] if is_sql_issue else state["error"],
        "logs": state["logs"] + [log_entry],
    }


def node_answer(state: AgentState) -> AgentState:
    t0 = time.time()
    answer = generate_nl_answer(
        state["question"], state["columns"], state["rows"], schema=state["schema"]
    )
    log_entry = {"node": "generate_nl_answer", "time": round(time.time() - t0, 2)}
    return {**state, "answer": answer, "logs": state["logs"] + [log_entry]}


def node_error_answer(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": f"Leider konnte keine gültige SQL generiert werden.\nFehler: {state['error']}",
    }


def node_no_data_answer(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": "Die Abfrage war erfolgreich, hat aber keine Ergebnisse geliefert. Möglicherweise existieren für diesen Zeitraum oder Filter keine Daten.",
    }


# Routing
def route_after_execute(state: AgentState) -> str:
    if state["error"] is None:
        return "validate_response"
    if state["retries"] < MAX_RETRIES:
        return "fix"
    return "error_answer"


def route_after_validate_response(state: AgentState) -> str:
    if state["validation_issue"] is None:
        return "answer"
    if state["validation_type"] in ("empty", "all_null"):
        return "no_data"  # Datenproblem -> kein Retry
    if state["retries"] < MAX_RETRIES:
        return "fix"  # wrong_type -> SQL korrigieren
    return "error_answer"


# Graph
def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("generate_sql", node_generate_sql)
    graph.add_node("execute_sql", node_execute_sql)
    graph.add_node("fix_sql", node_fix_sql)
    graph.add_node("validate_response", node_validate_response)
    graph.add_node("answer", node_answer)
    graph.add_node("error_answer", node_error_answer)
    graph.add_node("no_data", node_no_data_answer)

    graph.set_entry_point("generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {
            "validate_response": "validate_response",
            "fix": "fix_sql",
            "error_answer": "error_answer",
        },
    )
    graph.add_conditional_edges(
        "validate_response",
        route_after_validate_response,
        {
            "answer": "answer",
            "fix": "fix_sql",
            "error_answer": "error_answer",
            "no_data": "no_data",
        },
    )
    graph.add_edge("fix_sql", "execute_sql")
    graph.add_edge("answer", END)
    graph.add_edge("error_answer", END)
    graph.add_edge("no_data", END)

    return graph.compile()


_schema = get_schema()
agent = build_agent()


# Main
def run(question: str) -> str:
    reset_cost_tracker()
    result = agent.invoke(
        {
            "question": question,
            "schema": _schema,
            "sql": "",
            "columns": [],
            "rows": [],
            "error": None,
            "answer": "",
            "retries": 0,
            "validation_issue": None,
            "logs": [],
        }
    )
    print(f"Kosten: ${get_cost_summary()['total_cost']:.6f}")
    return {
        "answer": result["answer"],
        "sql": result["sql"],
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "retries": result["retries"],
        "logs": result["logs"],
        "costs": get_cost_summary(),
    }


# test
if __name__ == "__main__":
    antwort = run("Aus welcher Region kommen die meisten unique Hörer?")
    print(antwort)
