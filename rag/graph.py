# rag/graph.py
# Tool Use 반복문을 LangGraph "그래프"로 옮긴 것.
#
#   START ──▶ agent ──(도구 요청 있음)──▶ tools ──┐
#               ▲                                │
#               └────────────────────────────────┘
#             agent ──(도구 요청 없음 / 단계 초과)──▶ END
#
# 그래프 3요소
#   - State(상태): 노드들이 함께 보는 공책. 여기선 대화 메시지 목록 + Claude 호출 횟수
#   - Node(노드): 상태를 받아서 "바뀐 부분"만 돌려주는 함수 (agent, tools)
#   - Edge(엣지): 다음에 어느 노드로 갈지. route()가 조건부로 정한다
#
# 이전 PR의 for 반복문과 하는 일은 같다. 그래프로 바꾸면 좋은 점:
#   - 흐름이 그림(mermaid)으로 보인다
#   - "도구 실행 전에 멈추고 상담원 승인 받기"(interrupt)를 붙일 수 있다 ← 아래 tools_node
#
# 상담원 승인(human-in-the-loop) 흐름
#   tools 노드에서 book_test_drive 요청을 만나면 interrupt()로 그래프를 "일시정지"한다.
#   그때까지의 상태는 checkpointer(CHECKPOINTER)에 thread_id 이름으로 저장된다.
#   상담원이 승인/거절하면 Command(resume=...)로 같은 thread_id를 다시 실행 → 멈춘 곳부터 이어간다.
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from rag.tools import NEEDS_APPROVAL, TOOLS_BY_NAME, validate_booking

MAX_STEPS = 5          # Claude 호출 최대 횟수. 무한 반복과 비용 폭주를 막는 안전장치

# 멈춘 그래프의 상태를 보관하는 곳. 서버 메모리라서 재시작하면 대기 중인 승인은 사라진다 (데모용).
# 실제 서비스라면 PostgresSaver 같은 DB 저장소를 쓴다.
CHECKPOINTER = InMemorySaver()


class AgentState(TypedDict):
    # Annotated[..., add_messages]: 노드가 메시지를 돌려주면 "덮어쓰기"가 아니라 "뒤에 붙이기"
    # (Java로 치면 setter가 아니라 list.addAll)
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int         # 지금까지 Claude를 부른 횟수 (그냥 int라서 돌려주면 덮어씀)


def run_tool(call):
    """Claude가 요청한 도구 하나를 실행해서 ToolMessage로 돌려준다.
    도구에서 에러가 나도 서버를 멈추지 않고, 에러 내용을 Claude에게 보여 준다.
    → Claude가 "그 시간은 안 된대요, 다른 시간 고르세요"처럼 스스로 수습할 수 있다."""
    tool = TOOLS_BY_NAME.get(call["name"])
    if tool is None:
        return ToolMessage(f"없는 도구입니다: {call['name']}", tool_call_id=call["id"], status="error")
    try:
        # call 전체(name, args, id)를 넘기면 결과가 ToolMessage로 나온다 (tool_call_id 자동 연결)
        return tool.invoke(call)
    except Exception as e:
        return ToolMessage(f"도구 실행 실패: {e}", tool_call_id=call["id"], status="error")


def ask_approval(call):
    """승인이 필요한 도구면 그래프를 멈추고 상담원 답을 기다린다.
    돌려주는 값: None(그대로 실행) 또는 ToolMessage(실행하지 않고 Claude에게 보낼 결과)."""
    if call["name"] not in NEEDS_APPROVAL:
        return None
    args = call["args"]
    try:
        validate_booking(**args)               # 어차피 안 되는 예약이면 상담원에게 묻지 않음
    except Exception as e:
        return ToolMessage(f"도구 실행 실패: {e}", tool_call_id=call["id"], name=call["name"], status="error")

    # ── 여기서 그래프가 멈춘다 ──
    # 처음 실행: interrupt()가 값을 밖으로 내보내고 실행을 중단 (graph.invoke 결과에 __interrupt__로 나옴)
    # 다시 실행(resume): interrupt()가 상담원 답 {"approved": True/False}를 돌려준다
    decision = interrupt({"tool": call["name"], "args": args})
    if decision.get("approved"):
        return None
    return ToolMessage("상담원이 이 예약을 승인하지 않았습니다. 고객에게 알리고 다른 시간이나 지점을 제안하세요.",
                       tool_call_id=call["id"], name=call["name"], status="error")


def build_graph(llm, checkpointer=CHECKPOINTER):
    """llm: bind_tools까지 끝난 모델. 테스트에서는 가짜 LLM을 넣는다."""

    def agent_node(state: AgentState):
        ai = llm.invoke(state["messages"])
        return {"messages": [ai], "steps": state.get("steps", 0) + 1}   # 바뀐 부분만 돌려준다

    def tools_node(state: AgentState):
        # 마지막 메시지 = 방금 Claude가 보낸 도구 요청. 여러 개면 순서대로 모두 실행
        calls = state["messages"][-1].tool_calls
        # ① 승인부터 전부 받는다. 주의: resume하면 이 노드가 "처음부터 다시" 실행된다.
        #    그래서 예약 같은 실제 행동은 승인이 모두 끝난 ② 이후에만 해야 두 번 실행되지 않는다.
        decided = [ask_approval(call) for call in calls]
        # ② 실행: 거절/검증 실패한 건 그 결과를 쓰고, 나머지는 실제로 실행
        return {"messages": [msg if msg is not None else run_tool(call)
                             for call, msg in zip(calls, decided)]}

    def route(state: AgentState):
        # agent 다음에 어디로 갈지 정하는 조건부 엣지
        last = state["messages"][-1]
        if not last.tool_calls:
            return END                       # 도구를 안 부르면 = 최종 답
        if state["steps"] >= MAX_STEPS:
            return END                       # 단계 초과. run_agent()가 안내 문구로 바꾼다
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", END])
    graph.add_edge("tools", "agent")         # 도구 결과를 들고 다시 Claude에게
    return graph.compile(checkpointer=checkpointer)   # checkpointer가 있어야 멈췄다 이어갈 수 있다


if __name__ == "__main__":
    # 그래프 그림(mermaid) 출력. https://mermaid.live 에 붙여 넣으면 그림으로 보인다
    from rag.agent import get_agent_llm
    print(build_graph(get_agent_llm()).get_graph().draw_mermaid())
