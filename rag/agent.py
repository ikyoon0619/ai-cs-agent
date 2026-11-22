# rag/agent.py
# Tool Use 에이전트: Claude가 질문을 보고 필요한 도구를 골라 부르고, 결과를 읽고, 최종 답을 쓴다.
# 반복 흐름(agent ↔ tools)은 rag/graph.py의 LangGraph 그래프가 맡고,
# 이 파일은 프롬프트·대화 이력 준비와 최종 답 정리(출처, 사용한 도구)를 맡는다.
#
# 핵심: Claude는 함수를 "직접" 실행하지 않는다.
#       "check_inventory를 이 인자로 불러 줘"라고 요청(tool_call)만 하고, 실행은 우리 코드가 한다.
import re
import uuid
from datetime import datetime
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from langgraph.types import Command

from rag.graph import CHECKPOINTER, MAX_STEPS, build_graph   # noqa: F401  (MAX_STEPS는 테스트에서 씀)
from rag.llm import get_llm
from rag.tools import KST, TOOL_LABELS, TOOLS, TOOLS_BY_NAME

MAX_HISTORY = 10       # 이전 대화는 최근 10개 메시지만 보낸다 (토큰 비용 제한)
MAX_RESULT_CHARS = 1500   # 이전 대화에 다시 넣는 도구 결과는 이 길이까지만
WEEKDAYS = "월화수목금토일"

SYSTEM_PROMPT = """너는 쏘나타 디 엣지 판매 상담원이다. 오늘은 {today}({weekday}요일)이다.

도구 사용 규칙:
- 차량 사양, 트림, 옵션, 연비, 제원, 보증 같은 질문은 search_catalog로 찾은 내용만 근거로 답한다.
  근거로 쓴 문장 끝에 (p.27)처럼 페이지를 붙인다.
  찾아도 근거가 없으면 '카탈로그에서 찾을 수 없습니다'라고 답한다. 추측하지 않는다.
- 재고나 출고 질문은 check_inventory를 쓴다.
- 시승은 get_test_drive_slots로 가능한 시간을 먼저 보여 준다.
  book_test_drive는 고객이 지점, 날짜, 시간, 이름을 확정하고 예약해 달라고 한 뒤에만 부른다.
  예약은 상담원이 승인해야 확정된다. 거절되면 고객에게 알리고 다른 시간이나 지점을 제안한다.
  빠진 정보가 있으면 도구를 부르지 말고 고객에게 물어본다.
- 재고와 시승 정보는 데모용 가상 데이터라는 점을 답에 한 번 밝힌다.
- 쏘나타와 관계없는 질문은 정중히 거절한다.

답변은 한국어로, 3~6문장 정도로 짧게 쓴다."""


class ToolStep(BaseModel):
    # 도구 호출 한 번의 기록: 어떤 도구를, 어떤 인자로 불렀고, 무슨 결과가 나왔는지
    name: str = Field(max_length=40)
    args: dict = Field(default_factory=dict)
    result: str = Field(default="", max_length=4000)


class Turn(BaseModel):
    # 이전 대화 한 줄. API 요청과 Streamlit 화면이 같은 모양을 쓴다
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=2000)
    # 그 답을 만들 때 쓴 도구 호출과 결과 (assistant 답에만).
    # 글만 넘기면 Claude가 다음 질문에서 "아까 그 숫자는 근거 없이 말한 것"이라고 착각한다
    # (실제로 데모에서 겪은 버그). 그래서 도구 호출 → 결과 → 답 순서를 그대로 다시 보여 준다.
    tools: list[ToolStep] = Field(default_factory=list, max_length=10)


class PendingApproval(BaseModel):
    # 상담원 승인을 기다리며 멈춰 있는 요청. thread_id로 나중에 이어서 실행한다
    thread_id: str = Field(description="멈춘 그래프를 이어서 실행할 때 쓰는 id")
    tool: str = Field(description="승인이 필요한 도구 이름")
    args: dict = Field(description="도구에 넘길 인자 (지점, 날짜, 시간, 이름)")


class AgentAnswer(BaseModel):
    answer: str = Field(description="최종 답변")
    sources: list[str] = Field(default_factory=list, description="답에 인용한 카탈로그 페이지")
    tools_used: list[str] = Field(default_factory=list, description="이번 답에 사용한 도구 이름 (순서대로)")
    steps: list[ToolStep] = Field(default_factory=list,
                                  description="이번 답의 도구 호출 기록. 다음 질문 때 history의 tools로 돌려보낸다")
    pending: Optional[PendingApproval] = Field(default=None, description="승인 대기 중이면 그 내용, 아니면 null")


def get_agent_llm():
    # bind_tools: "이 도구들을 쓸 수 있어"라고 도구 설명서(JSON 스키마)를 요청에 붙인다.
    # 테스트에서는 이 함수를 가짜 LLM으로 바꿔치기한다.
    return get_llm().bind_tools(TOOLS)


def build_messages(question, history=()):
    now = datetime.now(KST)
    messages = [SystemMessage(SYSTEM_PROMPT.format(today=now.date().isoformat(),
                                                   weekday=WEEKDAYS[now.weekday()]))]
    for i, turn in enumerate(list(history)[-MAX_HISTORY:]):
        if not isinstance(turn, Turn):
            turn = Turn(**turn)              # dict로 와도 같은 검증을 거친다
        if turn.role == "user":
            messages.append(HumanMessage(turn.content))
            continue
        # 이전 답을 만들 때의 도구 호출 → 결과를 실제 대화 형식으로 다시 넣는다
        steps = [step for step in turn.tools if step.name in TOOLS_BY_NAME]   # 아는 도구만
        if steps:
            calls = [{"name": step.name, "args": step.args, "id": f"history_{i}_{j}", "type": "tool_call"}
                     for j, step in enumerate(steps)]
            messages.append(AIMessage("", tool_calls=calls))
            messages += [ToolMessage(step.result, tool_call_id=call["id"], name=step.name)
                         for step, call in zip(steps, calls)]
        messages.append(AIMessage(turn.content))
    messages.append(HumanMessage(question))
    return messages


def cited_pages(answer, docs):
    """답에 적힌 (p.27) 중에서 실제로 검색된 페이지만 출처로 인정한다 (지어낸 페이지 걸러냄)."""
    retrieved = {str(d.metadata.get("page_label")) for d in docs}
    pages = []
    for page in re.findall(r"p\.\s*(\d+)", answer):
        if page in retrieved and page not in pages:
            pages.append(page)
    return pages


def run_agent(question, history=()):
    # 질문마다 새 thread_id. 승인 때문에 멈추면 이 id로 나중에 이어서 실행한다
    thread_id = uuid.uuid4().hex
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_graph(get_agent_llm())
    # 그래프 실행: 처음 상태를 넣으면 END까지 돌거나, interrupt()에서 멈춘 상태를 돌려준다
    state = graph.invoke({"messages": build_messages(question, history), "steps": 0}, config)
    return summarize(thread_id, state)


def resume_agent(thread_id, approved):
    """상담원이 승인(approved=True) 또는 거절한 뒤, 멈춘 그래프를 이어서 실행한다."""
    config = {"configurable": {"thread_id": thread_id}}
    graph = build_graph(get_agent_llm())
    if not graph.get_state(config).interrupts:          # 이미 처리했거나 없는 id
        raise LookupError("승인을 기다리는 요청이 없습니다.")
    # Command(resume=값): 멈춘 interrupt()가 이 값을 돌려주면서 실행이 이어진다
    state = graph.invoke(Command(resume={"approved": approved}), config)
    return summarize(thread_id, state)


def summarize(thread_id, state):
    """그래프가 남긴 상태 → 화면/API에 줄 AgentAnswer"""
    messages = state["messages"]
    # 이번 질문(마지막 HumanMessage) 뒤의 메시지만 본다. 그 앞은 다시 넣어 준 이전 대화다
    start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage)) + 1
    current = messages[start:]

    # 이번 질문에서 "사용한 도구", "도구 호출 기록", "검색된 청크"를 모은다
    tools_used, retrieved_docs, steps = [], [], []
    calls_by_id = {}
    for msg in current:
        if isinstance(msg, AIMessage):
            tools_used += [call["name"] for call in msg.tool_calls]
            calls_by_id.update({call["id"]: call for call in msg.tool_calls})
        elif isinstance(msg, ToolMessage):
            call = calls_by_id.get(msg.tool_call_id, {})
            steps.append(ToolStep(name=msg.name or call.get("name", ""), args=call.get("args", {}),
                                  result=str(msg.content)[:MAX_RESULT_CHARS]))   # 다음 질문 비용을 위해 자름
            if msg.name == "search_catalog" and msg.artifact:
                retrieved_docs += msg.artifact

    last = messages[-1]
    unfinished = len(last.tool_calls) if isinstance(last, AIMessage) else 0
    done_tools = tools_used[:len(tools_used) - unfinished]   # 아직 실행 안 된 요청은 빼기

    interrupts = state.get("__interrupt__")
    if interrupts:                        # 상담원 승인 대기 중 (상태는 CHECKPOINTER에 남겨 둔다)
        info = interrupts[0].value        # ask_approval()이 interrupt()에 넘긴 {"tool", "args"}
        # steps는 비워 둔다: 승인 후 최종 답에 이번 질문의 도구 기록이 전부 담긴다 (중복 방지)
        return AgentAnswer(answer="예약 요청을 상담원에게 보냈어요. 승인되면 바로 확정해 드릴게요.",
                           tools_used=done_tools,
                           pending=PendingApproval(thread_id=thread_id, **info))

    CHECKPOINTER.delete_thread(thread_id)   # 끝난 대화는 메모리에서 지운다
    if unfinished:                        # MAX_STEPS에서 끊겨서 도구 요청이 남은 채로 끝남
        return AgentAnswer(answer="질문을 처리하는 데 단계가 너무 많이 필요합니다. 질문을 나눠서 다시 물어봐 주세요.",
                           tools_used=done_tools, steps=steps)
    answer = last.text.strip() or "답변을 만들지 못했습니다."
    return AgentAnswer(answer=answer, sources=cited_pages(answer, retrieved_docs),
                       tools_used=tools_used, steps=steps)


def tool_labels(names):
    """['search_catalog', 'search_catalog', 'check_inventory'] → ['카탈로그 검색', '재고 조회']"""
    labels = []
    for name in names:
        label = TOOL_LABELS.get(name, name)
        if label not in labels:
            labels.append(label)
    return labels


if __name__ == "__main__":
    for q in ["트림 종류 알려줘", "하이브리드 익스클루시브 재고 있어?", "내일 강남점 시승 가능한 시간 알려줘"]:
        result = run_agent(q)
        print("==== 질문:", q)
        print("도구:", result.tools_used)
        print("답변:", result.answer)
        print("출처:", result.sources)
        print()
