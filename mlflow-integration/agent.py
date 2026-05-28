"""Plan and Execute Agent using LangGraph + vLLM (OpenAI-compatible endpoint)."""

from __future__ import annotations

import os
from typing import Any, Generator, TypedDict

import mlflow
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

# MLflowトレーシングを有効化
mlflow.set_tracking_uri(os.environ["MLFLOW_URL"])
mlflow.set_experiment("LangGraph_Agent")
mlflow.langchain.autolog()

# LLM setup
def _build_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.environ["VLLM_BASE_URL"],
        api_key=os.environ.get("VLLM_API_KEY", "dummy"),
        model=os.environ["VLLM_MODEL"],
        temperature=temperature,
        streaming=True,
    )

# Graph state
class AgentState(TypedDict):
    """State shared across all nodes in the graph."""
    input: str                     # original user query
    plan: list[str]                # ordered list of sub-tasks
    current_step: int              # index into plan
    step_results: list[str]        # result for each completed step
    final_answer: str              # synthesised final answer
    events: list[dict[str, Any]]   # stream of UI events

# Node: Planner
PLAN_SYSTEM = """あなたは優秀なプランナーです。
ユーザーの質問を解決するために必要なステップを、番号付きリストで作成してください。
各ステップは具体的で独立した作業単位にしてください。
出力は以下の形式の番号付きリストのみとし、余分な説明は不要です:
1. [ステップ1の内容]
2. [ステップ2の内容]
...
"""
def planner_node(state: AgentState) -> AgentState:
    llm = _build_llm()
    messages = [
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=f"次の質問/タスクを解決するプランを作成してください:\n\n{state['input']}"),
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    steps: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if line and line[0].isdigit() and "." in line:
            # strip "1. " prefix
            step = line.split(".", 1)[1].strip()
            if step:
                steps.append(step)

    if not steps:
        steps = [raw]  # fallback: treat entire response as single step

    events = state.get("events", [])
    events.append({"type": "plan", "content": steps})

    return {
        **state,
        "plan": steps,
        "current_step": 0,
        "step_results": [],
        "events": events,
    }


# Node: Executor
EXEC_SYSTEM = """あなたは有能な実行者です。
与えられたタスクを丁寧に実行し、詳細な結果を返してください。
これまでの作業結果も参考にして、一貫した回答を心がけてください。
"""
def executor_node(state: AgentState) -> AgentState:
    llm = _build_llm(temperature=0.3)
    step_idx = state["current_step"]
    step = state["plan"][step_idx]

    # Build context from previous steps
    prev_context = ""
    for i, res in enumerate(state.get("step_results", [])):
        prev_context += f"\n[ステップ {i+1}: {state['plan'][i]}]\n結果: {res}\n"

    user_content = (
        f"元の質問: {state['input']}\n"
        f"\n現在のタスク (ステップ {step_idx+1}/{len(state['plan'])}): {step}"
    )
    if prev_context:
        user_content += f"\n\n--- これまでの作業結果 ---{prev_context}"

    messages = [
        SystemMessage(content=EXEC_SYSTEM),
        HumanMessage(content=user_content),
    ]
    response = llm.invoke(messages)
    result = response.content.strip()

    step_results = state.get("step_results", []) + [result]
    events = state.get("events", [])
    events.append({
        "type": "step_result",
        "step_index": step_idx,
        "step": step,
        "result": result,
    })

    return {
        **state,
        "current_step": step_idx + 1,
        "step_results": step_results,
        "events": events,
    }

# Node: Synthesizer
SYNTH_SYSTEM = """あなたは優秀なアナリストです。
各ステップの実行結果をまとめて、ユーザーの元の質問に対する
分かりやすく包括的な最終回答を作成してください。
"""
def synthesizer_node(state: AgentState) -> AgentState:
    llm = _build_llm(temperature=0.2)

    steps_summary = ""
    for i, (step, result) in enumerate(zip(state["plan"], state["step_results"])):
        steps_summary += f"\n[ステップ {i+1}] {step}\n結果: {result}\n"

    messages = [
        SystemMessage(content=SYNTH_SYSTEM),
        HumanMessage(
            content=(
                f"元の質問: {state['input']}\n\n"
                f"--- 各ステップの実行結果 ---{steps_summary}\n"
                "上記を統合して最終回答を作成してください。"
            )
        ),
    ]
    response = llm.invoke(messages)
    final_answer = response.content.strip()

    events = state.get("events", [])
    events.append({"type": "final_answer", "content": final_answer})

    return {**state, "final_answer": final_answer, "events": events}


# Routing
def should_continue(state: AgentState) -> str:
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    return "execute"

# Build graph
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("synthesizer", synthesizer_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")
    g.add_conditional_edges("executor", should_continue, {
        "execute": "executor",
        "synthesize": "synthesizer",
    })
    g.add_edge("synthesizer", END)

    return g.compile()

# Streaming helper
def run_agent_stream(query: str) -> Generator[dict[str, Any], None, None]:
    """Yield UI-event dicts as the agent progresses."""
    graph = build_graph()
    init_state: AgentState = {
        "input": query,
        "plan": [],
        "current_step": 0,
        "step_results": [],
        "final_answer": "",
        "events": [],
    }

    seen_events = 0
    for chunk in graph.stream(init_state, stream_mode="values"):
        new_events = chunk.get("events", [])[seen_events:]
        for event in new_events:
            yield event
        seen_events += len(new_events)
