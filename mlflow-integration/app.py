"""Streamlit UI for the Plan-and-Execute Agent."""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from agent import run_agent_stream

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Plan & Execute Agent",
    page_icon=None,
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
.plan-box {
    border-left: 4px solid #4a9eff;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 8px;
}
.step-box {
    background: #1a3a2a;
    border-left: 4px solid #4ade80;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 8px;
}
.step-header {
    color: #4ade80;
    font-weight: bold;
    font-size: 0.95rem;
    margin-bottom: 6px;
}
.final-box {
    padding: 16px 20px;
    border-radius: 6px;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: bold;
    margin-right: 6px;
}
.badge-plan  { background: #1e40af; color: #93c5fd; }
.badge-exec  { background: #14532d; color: #86efac; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Plan & Execute Agent")
st.caption(
    "LangGraph + LangChain + vLLM (Qwen3.6-35B-A3B) — "
    "複雑なタスクを自動的に計画・実行します"
)

# ---------------------------------------------------------------------------
# Sidebar: settings & history
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("設定")
    show_raw = st.checkbox("生のイベントログを表示", value=False)

    st.divider()
    st.header("使い方")
    st.markdown("""
1. テキストボックスに質問やタスクを入力
2. **実行** ボタンを押す
3. 思考の過程がリアルタイムで表示されます

**例:**
- 東京の観光スポットを調べて、3日間の旅行プランを作成して
- Pythonで機械学習パイプラインを設計して
- 気候変動の主な原因と対策を詳しく調べて
""")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

query = st.text_area(
    "質問 / タスクを入力してください",
    height=100,
    placeholder="例: 量子コンピュータの仕組みを解説し、現在の応用事例をまとめてください",
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("▶ 実行", type="primary", use_container_width=True)
with col2:
    clear_btn = st.button("クリア", use_container_width=False)

if clear_btn:
    st.session_state.pop("history", None)
    st.rerun()

# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------

def render_event(event: dict[str, Any], container: Any) -> None:
    etype = event.get("type")

    if etype == "plan":
        steps = event["content"]
        with container:
            st.markdown(
                '<span class="badge badge-plan">PLAN</span> '
                f'**{len(steps)} ステップのプランを作成しました**',
                unsafe_allow_html=True,
            )
            for i, s in enumerate(steps, 1):
                st.markdown(
                    f'<div class="plan-box">　<b>{i}.</b> {s}</div>',
                    unsafe_allow_html=True,
                )

    elif etype == "step_result":
        idx = event["step_index"] + 1
        step = event["step"]
        result = event["result"]
        with container:
            st.markdown(
                f'<span class="badge badge-exec">STEP {idx}</span> '
                f'**{step}**',
                unsafe_allow_html=True,
            )
            with st.expander(f"ステップ {idx} の実行結果", expanded=False):
                st.markdown(result)

    elif etype == "final_answer":
        with container:
            st.markdown(
                f'<div class="final-box">{event["content"]}</div>',
                unsafe_allow_html=True,
            )


if run_btn and query.strip():
    st.divider()

    # Layout: progress on top, events below
    progress_area = st.empty()
    status_area = st.empty()
    events_container = st.container()

    raw_log: list[dict] = []
    all_events: list[dict] = []
    plan_len = 0
    steps_done = 0
    start_ts = time.time()

    with st.spinner("エージェント実行中..."):
        try:
            for event in run_agent_stream(query.strip()):
                all_events.append(event)
                raw_log.append(event)

                etype = event.get("type")

                if etype == "plan":
                    plan_len = len(event["content"])
                    status_area.info(
                        f"プラン作成完了 — {plan_len} ステップを実行します"
                    )

                elif etype == "step_result":
                    steps_done += 1
                    pct = int(steps_done / plan_len * 100) if plan_len else 0
                    progress_area.progress(
                        pct / 100,
                        text=f"ステップ {steps_done}/{plan_len} 完了 ({pct}%)",
                    )

                elif etype == "final_answer":
                    elapsed = time.time() - start_ts
                    status_area.success(
                        f"完了 — {elapsed:.1f}秒"
                    )
                    progress_area.progress(1.0, text="完了")

                render_event(event, events_container)

        except Exception as exc:
            st.error(f"エラーが発生しました: {exc}")
            raise

    # Raw event log (sidebar toggle)
    if show_raw and raw_log:
        with st.expander("生のイベントログ"):
            st.json(raw_log)

    # Save to history
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "query": query.strip(),
        "events": all_events,
    })

elif run_btn and not query.strip():
    st.warning("質問またはタスクを入力してください。")

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

history: list[dict] = st.session_state.get("history", [])
if history:
    st.divider()
    with st.expander(f"実行履歴 ({len(history)} 件)", expanded=False):
        for i, item in enumerate(reversed(history), 1):
            st.markdown(f"**#{len(history)-i+1}** — {item['query'][:80]}...")
            for ev in item["events"]:
                if ev.get("type") == "final_answer":
                    st.markdown(ev["content"][:400] + ("..." if len(ev["content"]) > 400 else ""))
                    break
            st.divider()
